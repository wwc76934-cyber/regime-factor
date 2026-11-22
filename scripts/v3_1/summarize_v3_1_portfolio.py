"""Merge Step 3 ablations, three-benchmark results and operational gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BLOCKS = ("core","frequency","controls","turnover_sensitivity")
KEY = ["model","horizon","construction","eta","ema"]


def read_parts(root, name):
    return pd.concat([pd.read_csv(root/block/name) for block in BLOCKS],ignore_index=True)


def save(frame, path):
    frame.to_csv(path,index=False,encoding="utf-8-sig")


def run(root):
    names = ["model_portfolio_comparison_v3_1.csv","benchmark_comparison_v3_1.csv",
             "annual_benchmarks_v3_1.csv","beta_by_regime_v3_1.csv","unresolved_exposure.csv"]
    portfolio,base,annual,beta,issues = [read_parts(root,name) for name in names]
    issues["date"] = pd.to_datetime(issues.date,format="mixed")
    expected = set(portfolio.strategy)
    if len(portfolio)!=32 or len(expected)!=32 or base.groupby("strategy").benchmark.nunique().ne(3).any():
        raise ValueError("Incomplete or duplicated Step 3 experiment matrix")
    if set(base.strategy)!=expected or set(annual.strategy)!=expected or set(beta.strategy)!=expected:
        raise ValueError("Step 3 table strategy coverage differs")
    stuck = []
    end = pd.Timestamp("2025-12-31")
    for block in BLOCKS:
        for folder in (root/block/"runs").iterdir():
            h = pd.read_parquet(folder/"holdings.parquet",columns=["code","date","shares","locked_shares","value","stale"],
                                filters=[("stale","==",True)])
            for row in h.loc[h.date.eq(end)].itertuples(index=False):
                history = h.loc[h.code.eq(row.code)]
                stuck.append((folder.name,row.code,history.date.min(),row.date,len(history),row.shares,row.locked_shares,row.value))
    terminal = pd.DataFrame(stuck,columns=["strategy","code","first_stale_date","last_stale_date",
                                          "stale_sessions","shares","locked_shares","terminal_mark_value"])
    portfolio["terminal_stale_positions"] = portfolio.strategy.map(terminal.groupby("strategy").code.size()).fillna(0).astype(int)
    portfolio["ledger_verified"] = portfolio.unresolved_exposures.eq(0) & portfolio.terminal_stale_positions.eq(0)
    for table in [base,annual]:
        table["ledger_verified"] = table.strategy.map(portfolio.set_index("strategy").ledger_verified)
    pivot = base.pivot(index="strategy",columns="benchmark",values=["cumulative_excess_pp","benchmark_return"])
    h, f, e = (pivot["cumulative_excess_pp"][column] for column in ("H00300","FloatMV","CSI300_EW"))
    benchmark_bridge = pd.DataFrame({"H00300_excess_pp":h,"FloatMV_excess_pp":f,"CSI300_EW_excess_pp":e,
          "FloatMV_minus_H00300_pp":pivot["benchmark_return"]["FloatMV"]-pivot["benchmark_return"]["H00300"]}).reset_index()
    benchmark_bridge["positive_vs_all_three"] = benchmark_bridge[["H00300_excess_pp","FloatMV_excess_pp",
                                                                  "CSI300_EW_excess_pp"]].gt(0).all(axis=1)
    if not np.allclose(benchmark_bridge.FloatMV_excess_pp+benchmark_bridge.FloatMV_minus_H00300_pp,
                       benchmark_bridge.H00300_excess_pp,atol=1e-12):
        raise ValueError("Benchmark bridge does not reconcile")
    portfolio = portfolio.merge(benchmark_bridge[["strategy","positive_vs_all_three"]],on="strategy",validate="one_to_one")
    official_annual = annual.loc[annual.benchmark.eq("H00300")]
    later = official_annual.loc[official_annual.year.ge(2023)]
    wins = official_annual.cumulative_excess_pp.gt(0).groupby(official_annual.strategy).sum()
    portfolio["positive_official_excess_years"] = portfolio.strategy.map(wins)
    recent = later.groupby("strategy")[["net_return","benchmark_return"]].agg(lambda x:(1+x).prod()-1)
    portfolio["H00300_excess_2023_2025"] = portfolio.strategy.map(recent.net_return-recent.benchmark_return)
    columns = ["net_return","H00300_excess","weekly_equivalent_two_way_turnover","offensive_beta_hit",
               "H00300_ir","FloatMV_excess","CSI300_EW_excess","unresolved_exposures","terminal_stale_positions"]
    paired = portfolio.pivot(index=KEY,columns="fee_bps",values=columns)
    paired.columns = [f"{metric}_{bps}bp" for metric,bps in paired.columns]
    paired = paired.reset_index()
    paired["beats_H00300_both_fees"] = paired[["H00300_excess_10bp","H00300_excess_20bp"]].gt(0).all(axis=1)
    paired["turnover_within_limit_both_fees"] = paired[[f"weekly_equivalent_two_way_turnover_{bp}bp" for bp in (10,20)]].le(.15).all(axis=1)
    paired["offensive_beta_passes_both_fees"] = paired[["offensive_beta_hit_10bp","offensive_beta_hit_20bp"]].ge(.8).all(axis=1)
    paired["ledger_verified_both_fees"] = paired[["unresolved_exposures_10bp","unresolved_exposures_20bp",
                                                    "terminal_stale_positions_10bp","terminal_stale_positions_20bp"]].eq(0).all(axis=1)
    paired["passes_step3_checks"] = paired[["beats_H00300_both_fees","turnover_within_limit_both_fees",
                                           "offensive_beta_passes_both_fees","ledger_verified_both_fees"]].all(axis=1)
    buffered = portfolio.loc[portfolio.model.eq("fusion_50_50") & portfolio.construction.eq("dynamic")
                             & portfolio.horizon.isin([10,20]) & portfolio.ema.eq(1.)]
    eta_cmp = buffered.pivot(index=["horizon","fee_bps"],columns="eta",values=["offensive_beta_hit","H00300_excess",
                                                                                 "weekly_equivalent_two_way_turnover"])
    eta_cmp.columns = [f"{metric}_eta{eta:g}" for metric,eta in eta_cmp.columns]
    eta_cmp = eta_cmp.reset_index()
    eta_cmp["offensive_beta_hit_change"] = eta_cmp["offensive_beta_hit_eta0.96"]-eta_cmp.offensive_beta_hit_eta1
    issue_keys = issues.groupby(["code","date","category","reason"],dropna=False).strategy.nunique().rename("affected_strategies").reset_index()
    for name, frame in [*zip(names,[portfolio,base,annual,beta,issues]),
                        ("three_benchmark_bridge.csv",benchmark_bridge),("fee_pair_gate.csv",paired),
                        ("beta_buffer_ablation.csv",eta_cmp),("corporate_action_review_required.csv",issue_keys),
                        ("terminal_stale_positions.csv",terminal)]:
        save(frame,root/name)
    eligible = paired.loc[paired.passes_step3_checks & paired.horizon.isin([10,20])]
    status = {"status":"go" if not eligible.empty else "no_go","variants":len(portfolio),
              "corporate_action_keys_requiring_review":len(issue_keys),
              "terminal_stale_codes_requiring_review":int(terminal.code.nunique()),
              "all_ledgers_verified":bool(portfolio.ledger_verified.all()),
              "main_frequency_variants_meeting_all_step3_checks":len(eligible),"official_benchmark":"H00300"}
    (root/"step3_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(portfolio[["strategy","net_return","H00300_excess","FloatMV_excess","weekly_equivalent_two_way_turnover",
                     "offensive_beta_hit","unresolved_exposures"]].to_string(index=False))
    print(eta_cmp.to_string(index=False))
    print(json.dumps(status))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step3",type=Path,default=Path("research/v3_1/results/step3"))
    run(p.parse_args().step3)

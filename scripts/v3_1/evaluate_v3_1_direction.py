"""Compare nonnegative Ridge on reconciled ledgers and evaluate all freeze thresholds."""
import argparse
import json
from pathlib import Path

import pandas as pd

from optimize_v3_1_portfolio import FUSION, KEY, scores_by_variant
from prepare_v3_1_base import initialize
from report_v3_1_development import FRONTIER, frontier_name, frontier_run


def hit_rate(predictions, labels, decisions):
    scores,_ = scores_by_variant(predictions,labels,FUSION,.3)
    scored = scores.loc[scores.signal_date.isin(decisions)].merge(
        labels[KEY+["risk_adjusted_target"]],on=KEY,validate="one_to_one")
    hits=[]
    for _,week in scored.groupby("signal_date"):
        good=week.dropna(subset="risk_adjusted_target")
        if len(good)<50:
            continue
        top=week.sort_values(["score","code"],ascending=[False,True]).head(30)
        if top.risk_adjusted_target.count()<24:
            continue
        realized=good.sort_values(["risk_adjusted_target","code"],ascending=[False,True]).head(30)
        hits.append(len(set(top.code)&set(realized.code))/30.)
    if not hits:
        raise ValueError("No mature signal weeks")
    return sum(hits)/len(hits),len(hits)


def run(root, before, after, forecasts, out):
    v3,*_=initialize(root)
    source = root/"research/v3_1/results"
    attribution = pd.read_csv(out/"annual_attribution.csv")
    rows=[]
    for version,path,pred_dir in [("corrected",before,source/"step2"),("nonnegative_ridge",after,forecasts)]:
        for h,fee in FRONTIER:
            run_path=frontier_run(path,h,fee)
            summary=pd.read_csv(run_path/"benchmark_metrics.csv").set_index("benchmark")
            nav=pd.read_parquet(run_path/"nav.parquet")
            beta=pd.read_csv(run_path/"beta_rebalances.csv",parse_dates=["signal_date"])
            labels=pd.read_parquet(source/"step1/labels.parquet",filters=[("horizon","==",h)])
            prediction=pd.read_parquet(pred_dir/f"h{h}"/"predictions.parquet")
            hit,weeks=hit_rate(prediction,labels,beta.signal_date)
            metrics=summary.loc["H00300"]
            periods=attribution.loc[attribution.version.eq(version)&attribution.horizon.eq(h)&
                                    attribution.fee_bps.eq(fee)].set_index("period")
            actions=pd.read_csv(run_path/"unresolved_exposure.csv")
            holds=pd.read_parquet(run_path/"holdings.parquet",columns=["code","date","stale"],
                                  filters=[("stale","==",True)])
            stale=int(holds.date.eq(nav.date.max()).sum())
            rows.append({"version":version,"horizon":h,"fee_bps":fee,"strategy":frontier_name(h,fee),
                "net_return":metrics.net_return,"H00300_excess":metrics.cumulative_excess_pp,"H00300_ir":metrics.information_ratio,
                "FloatMV_excess":summary.at["FloatMV","cumulative_excess_pp"],"CSI300_EW_excess":summary.at["CSI300_EW","cumulative_excess_pp"],
                "turnover":nav.gross_traded.div(nav.opening_equity).sum()*5/len(nav),
                "offensive_beta_hit":beta.loc[beta.regime.eq("offensive"),"inside_official_regime_band"].mean(),
                "top30_signal_hit":hit,"top30_mature_signal_weeks":weeks,
                "unresolved_exposures":len(actions),"terminal_stale_positions":stale,
                "excess_2024":periods.at[2024,"h00300_excess"],"drawdown_2024":periods.at[2024,"portfolio_max_drawdown"],
                "excess_drawdown_2024":periods.at[2024,"max_excess_drawdown"],
                "excess_2025":periods.at[2025,"h00300_excess"],"excess_drawdown_2025":periods.at[2025,"max_excess_drawdown"]})
    table=pd.DataFrame(rows)
    checks=[]
    for (version,h),pair in table.groupby(["version","horizon"]):
        b=pair.set_index("fee_bps")
        tests=[
            ("10bp_cumulative_excess_gte_20pp",b.at[10,"H00300_excess"],b.at[10,"H00300_excess"]>=.20),
            ("20bp_cumulative_excess_gte_5pp",b.at[20,"H00300_excess"],b.at[20,"H00300_excess"]>=.05),
            ("10bp_ir_gte_0.60",b.at[10,"H00300_ir"],b.at[10,"H00300_ir"]>=.60),
            ("10bp_2025_beat_or_excess_drawdown_lte_3pct",b.at[10,"excess_2025"],
             b.at[10,"excess_2025"]>=0 or b.at[10,"excess_drawdown_2025"]<=.03),
            ("both_turnover_lte_15pct",b.turnover.max(),b.turnover.le(.15).all()),
            ("both_offensive_beta_gte_80pct",b.offensive_beta_hit.min(),b.offensive_beta_hit.ge(.8).all()),
            ("signal_hit_gt_10.5pct_and_verified_ledger",b.top30_signal_hit.min(),
             b.top30_signal_hit.gt(.105).all() and b.unresolved_exposures.eq(0).all() and b.terminal_stale_positions.eq(0).all()),
        ]
        checks.extend({"version":version,"horizon":h,"check":check,"observed":value,"passed":bool(ok)} for check,value,ok in tests)
    gate=pd.DataFrame(checks)
    result=gate.groupby(["version","horizon"]).passed.all()
    status={"status":"go" if result.any() else "no_go",
        "qualified":[{"version":v,"horizon":int(h)} for v,h in result[result].index],
        "tested_pairs":len(result),"freeze_authorized_by_results":bool(result.any())}
    v3.save(table,out/"direction_experiment_comparison.csv")
    v3.save(gate,out/"freeze_gate_audit.csv")
    (out/"freeze_gate_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(table[["version","horizon","fee_bps","H00300_ir","H00300_excess","excess_2024","drawdown_2024",
                 "excess_drawdown_2024","excess_2025","top30_signal_hit","offensive_beta_hit","turnover"]].to_string(index=False))
    print(json.dumps(status))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--forecasts",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    a=parser.parse_args()
    run(a.project_root.resolve(),a.before,a.after,a.forecasts,a.output or a.project_root/"research/v3_1/results/step4")

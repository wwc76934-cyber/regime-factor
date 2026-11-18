"""Isolate the effect of dynamic quotas with the same scores, fees and execution."""

import argparse
from decimal import Decimal
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

from evaluate_score_ema import verify_actions
from regime_allocator import allocation_table, select_targets, TIERS
from tiered_data import require
from v1_holdout_inputs import build_inputs, corporate_actions


NAV = "strategy date cash stock pending receivables prepaid equity fee gross opening stale".split()
MODELS = ["tiered_three_factor_ridge", "tiered_lightgbm_6f"]


def table(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def period_metrics(nav, benchmark):
    daily = pd.concat({"strategy": nav, "benchmark": benchmark}, axis=1).pct_change().iloc[1:]
    weekly = (1 + daily).groupby(daily.index.to_period("W-SUN")).prod() - 1
    active = daily["strategy"] - daily["benchmark"]
    relative = nav / benchmark
    return {"total_return": nav.iloc[-1]-1, "benchmark_return": benchmark.iloc[-1]-1,
            "annualized_excess": active.mean()*252, "information_ratio": active.mean()/active.std()*np.sqrt(252),
            "realized_beta": weekly["strategy"].cov(weekly["benchmark"])/weekly["benchmark"].var(),
            "beta_weeks": len(weekly), "max_drawdown": -(nav/nav.cummax()-1).min(),
            "relative_max_drawdown": -(relative/relative.cummax()-1).min()}


def run(root, source, scores_dir, output, overrides):
    sys.path.insert(0, str(root/"scripts"))
    from run_holdings_buffer_backtest import simulate

    output.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads((root/"configs/baseline.toml").read_text(encoding="utf-8-sig"))
    initial = Decimal(str(config["portfolio"]["initial_cash"]))
    low,high = config["research"]["bars_start"],config["research"]["development_end"]
    calendar = pd.read_parquet(root/f"data/raw/calendar/baostock_{low}_{high}.parquet")
    dev_days = pd.DatetimeIndex(calendar.loc[calendar.is_trading_day.eq("1"),"calendar_date"]).sort_values()
    dev_panel = pd.read_parquet(root/"data/processed/csi300_daily_panel.parquet")
    dev_events = pd.read_parquet(root/"data/processed/corporate_actions/csi300_backtest_actions.parquet")
    review = pd.read_parquet(root/"reports/portfolio/corporate_actions_remaining_review.parquet")
    dev_events, dev_review = verify_actions(dev_events,review)
    dev_index = pd.read_parquet(root/"data/raw/index/csi300_H00300_development.parquet").set_index("date")["close"].sort_index()
    cache = root/"data/raw/holdout_2025"
    days, _, _, panel, _, _, _, codes = build_inputs(cache)
    events, review, _ = corporate_actions(cache,codes,panel,days,overrides)
    post_index = pd.read_parquet(cache/"H00300.parquet").set_index("date")["close"].sort_index()
    states = pd.concat([pd.read_parquet(root/"data/processed/market/csi300_weekly_market_states.parquet"),
                        pd.read_parquet(source/"states_2025.parquet")],ignore_index=True)
    allocation = allocation_table(states,pd.read_parquet(source/"tier_reference_daily_nav.parquet"))
    table(allocation,output/"weekly_allocations.csv")
    scored = pd.read_parquet(scores_dir/"tiered_weekly_scores.parquet")
    development = scored["execution_date"].dt.year.le(2024)
    post = scored["execution_date"].dt.year.eq(2025)
    regimes = allocation.set_index("signal_date")["regime"]
    years = scored.loc[scored.strategy.eq(MODELS[0]) & scored.execution_date.between("2021-01-18","2025-12-31"),
                       ["signal_date","execution_date"]].drop_duplicates()
    table(years.assign(year=years.execution_date.dt.year,regime=years.signal_date.map(regimes))
          .groupby(["year","regime"]).size().rename("weeks").reset_index(),output/"regime_counts.csv")
    reports, annual, ledger, audits, pending = [],[],[],[],[]
    for sample, selected, quote_days, prices, actions, needs_review, index, first in [
        ("development", development, dev_days,dev_panel,dev_events,dev_review,dev_index,pd.Timestamp("2021-01-18")),
        ("retrospective_2025",post,days[days.year==2025],panel,events,review,post_index,pd.Timestamp("2025-01-06")),
    ]:
        stock_scores = scored.loc[selected & scored.strategy.isin(MODELS)]
        quote_days = quote_days[quote_days>=stock_scores.execution_date.min()] if sample=="development" else quote_days
        anchor = index.index[index.index<first][-1] if sample=="development" else pd.Timestamp("2024-12-31")
        evaluation_days = quote_days[quote_days>anchor].insert(0,anchor)
        benchmark = index.reindex(evaluation_days)
        require(benchmark.notna().all() and benchmark.gt(0).all(),"Benchmark does not match dates")
        benchmark = benchmark/benchmark.iloc[0]
        fields = ["date","code","raw_open","raw_close","preclose","tradestatus","isST"]
        bars = {day:{row.code:row for row in rows.itertuples(index=False)} for day,rows in
                prices.loc[prices.date.isin(quote_days)&prices.code.isin(stock_scores.code),fields].groupby("date")}
        for name, model_scores in stock_scores.groupby("strategy"):
            for mode in ["static","dynamic"]:
                for fee in [10,20]:
                    schedule, selector, audit = select_targets(model_scores,allocation,mode=="dynamic",first)
                    nav, _, _, _, unresolved = simulate(name,quote_days,bars,schedule,actions,needs_review,initial,
                        fee_rate=Decimal(fee)/Decimal(10000),select_targets=selector)
                    data = pd.DataFrame(nav,columns=NAV).set_index("date")
                    series = data["equity"].reindex(evaluation_days)
                    if sample!="development":
                        series.loc[anchor]=float(initial)
                    require(series.gt(0).all() and len(audit)==30*len(schedule),"Bad NAV or target ledger")
                    series=series/series.iloc[0]
                    turns=(data.loc[first:,"gross"]/data.loc[first:,"opening"]).dropna()
                    require(len(turns)==(202 if sample=="development" else 52),"Unexpected rebalance count")
                    meta = {"sample":sample,"model":name,"allocator":mode,"fee_bps":fee}
                    reports.append({**meta,**period_metrics(series,benchmark),"turnover":turns.mean(),
                        "fees_paid":data.loc[first:,"fee"].sum(),"unresolved_exposures":len(unresolved),
                        "stale_mark_days":int(data.loc[first:,"stale"].gt(0).sum())})
                    daily = pd.concat({"strategy":series,"benchmark":benchmark},axis=1).pct_change().dropna()
                    for year, year_days in daily.groupby(daily.index.year):
                        prior = evaluation_days[evaluation_days.get_loc(year_days.index[0])-1]
                        annual.append({**meta,"year":year,
                            **period_metrics(series.loc[prior:year_days.index[-1]],benchmark.loc[prior:year_days.index[-1]])})
                    ledger.append(pd.DataFrame({"date":evaluation_days,"nav":series.to_numpy(),
                                                "benchmark_nav":benchmark.to_numpy(),**meta}))
                    audits.append(pd.DataFrame(audit,columns=["date","regime","vol_tier","quota","buffer_cutoff","code",
                        "tier_rank","held_and_retained"]).assign(**meta))
                    pending.extend((*row, sample, mode, fee) for row in unresolved)
        print("Completed",sample,flush=True)
    table(pd.DataFrame(pending,columns=["model","date","code","opening_shares","pending_shares","category","reason",
        "sample","allocator","fee_bps"]),output/"allocator_unresolved_exposures.csv")
    summary = pd.DataFrame(reports)
    table(summary,output/"allocator_comparison.csv")
    table(pd.DataFrame(annual),output/"allocator_annual.csv")
    pd.concat(audits,ignore_index=True).to_parquet(output/"allocator_targets.parquet",index=False)
    pd.concat(ledger,ignore_index=True).to_parquet(output/"allocator_nav.parquet",index=False)
    print(summary.round(5).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--scores-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--verified-actions",type=Path,default=Path(__file__).with_name("verified_actions_2025.csv"))
    args=parser.parse_args()
    root=args.project_root.resolve()
    source=root/"research/v2_regime_tiered/results"
    run(root,source,args.scores_dir or source/"stage3_scores",args.output_dir or source/"stage3",args.verified_actions)

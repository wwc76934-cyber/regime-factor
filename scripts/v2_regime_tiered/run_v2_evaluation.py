"""Final v1/v2 matrix; 2025 is a retrospective comparison."""

import argparse
from decimal import Decimal
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

from evaluate_allocator import NAV, period_metrics, table
from evaluate_score_ema import verify_actions
from regime_allocator import allocation_table, select_targets
from tiered_data import require
from tiered_models import smooth
from v1_holdout_inputs import build_inputs, corporate_actions


V1 = {"state_ridge": "v1_state_ridge_buf45", "lightgbm_6f": "v1_lightgbm_6f_buf45"}
V2 = {"tiered_three_factor_ridge": "ridge", "tiered_lightgbm_6f": "lgb6"}
BENCHMARKS = ["H00300", "csi300_member_equal_weight"]
FEES = [0, 10, 20]


def read_rankings(path):
    return pd.read_parquet(path, columns=["strategy", "signal_date", "execution_date", "code", "score"])


def baseline_scores(dev_path, post_path):
    frames = []
    for file in (dev_path, post_path):
        ranked = read_rankings(file)
        frames.append(ranked.loc[ranked.strategy.isin(V1)])
    result = pd.concat(frames, ignore_index=True)
    require(not result.duplicated(["strategy", "signal_date", "code"]).any(), "Duplicate v1 ranking rows")
    tree = result.loc[result.strategy.eq("lightgbm_6f")]
    ema = smooth(tree).assign(strategy="lightgbm_6f_ema05")
    live = ema.execution_date.ge(pd.Timestamp("2021-01-18"))
    ema["score"] = ema["score_ema05"].where(live, ema["score"])
    return pd.concat([result, ema.drop(columns="score_ema05")], ignore_index=True)


def run(root, data_dir, v2_file, dev_file, post_file, output, overrides):
    sys.path.insert(0, str(root/"scripts"))
    from build_csi300_equal_weight import build_benchmark
    from run_holdings_buffer_backtest import make_buffer_selector, simulate

    cfg = tomllib.loads((root/"configs/baseline.toml").read_text(encoding="utf-8-sig"))
    initial = Decimal(str(cfg["portfolio"]["initial_cash"]))
    start, end = cfg["research"]["bars_start"], cfg["research"]["development_end"]
    processed, raw = root/"data/processed", root/"data/raw"
    calendar = pd.read_parquet(raw/f"calendar/baostock_{start}_{end}.parquet")
    days_dev = pd.DatetimeIndex(calendar.loc[calendar.is_trading_day.eq("1"), "calendar_date"]).sort_values()
    panel_dev = pd.read_parquet(processed/"csi300_daily_panel.parquet")
    review = pd.read_parquet(root/"reports/portfolio/corporate_actions_remaining_review.parquet")
    events_dev, review_dev = verify_actions(pd.read_parquet(
        processed/"corporate_actions/csi300_backtest_actions.parquet"), review)
    cache = raw/"holdout_2025"
    days, schedule, members, panel_post, _, _, _, codes = build_inputs(cache)
    days_post = days[days.year == 2025]
    events_post, review_post, _ = corporate_actions(cache, codes, panel_post, days, overrides)
    schedule = schedule.loc[schedule.execution_date.dt.year.eq(2025)]
    equal_post = build_benchmark(panel_post, members.loc[members.signal_date.isin(schedule.signal_date)],
                                 schedule, days_post.insert(0, pd.Timestamp("2024-12-31")))
    equal_dev = pd.read_parquet(processed/"benchmark/csi300_weekly_equal_weight.parquet")
    states = pd.concat([pd.read_parquet(processed/"market/csi300_weekly_market_states.parquet"),
                        pd.read_parquet(data_dir/"states_2025.parquet")], ignore_index=True)
    allocations = allocation_table(states, pd.read_parquet(data_dir/"tier_reference_daily_nav.parquet"))
    v1 = baseline_scores(dev_file, post_file)
    v2 = pd.read_parquet(v2_file)
    output.mkdir(parents=True, exist_ok=True)
    summaries, annual, nav_rows, issues = [], [], [], []
    for year, days, panel, events, review, equal, index_file, anchor, first in [
        (2024, days_dev, panel_dev, events_dev, review_dev, equal_dev, raw/"index/csi300_H00300_development.parquet",
         pd.Timestamp("2021-01-15"), pd.Timestamp("2021-01-18")),
        (2025, days_post, panel_post, events_post, review_post, equal_post, cache/"H00300.parquet",
         pd.Timestamp("2024-12-31"), days_post[0]),
    ]:
        start_year = 2020 if year == 2024 else 2025
        sample = "development" if year == 2024 else "retrospective_2025"
        data_v1 = v1.loc[v1.execution_date.dt.year.between(start_year, year)]
        data_v2 = v2.loc[v2.execution_date.dt.year.between(start_year, year) & v2.strategy.isin(V2)]
        days = days[days >= data_v1.execution_date.min()] if year == 2024 else days
        evaluation_days = days[days > anchor].insert(0, anchor)
        index = pd.read_parquet(index_file).set_index("date")["close"].reindex(evaluation_days)
        require(index.notna().all() and index.gt(0).all(), "H00300 mismatch")
        benchmark = index/index.iloc[0]
        equal = equal.set_index("date")["nav"].reindex(evaluation_days)
        require(equal.notna().all() and equal.gt(0).all(), "Equal-weight benchmark mismatch")
        codes = set(data_v1.code) | set(data_v2.code)
        columns = ["date","code","raw_open","raw_close","preclose","tradestatus","isST"]
        bars = {date: {row.code: row for row in group.itertuples(index=False)} for date, group in
                panel.loc[panel.date.isin(days) & panel.code.isin(codes), columns].groupby("date")}
        def record(name, fee, nav, turnover=np.nan, paid=np.nan, stale=0):
            meta = {"sample": sample, "strategy": name, "fee_bps": fee}
            with np.errstate(invalid="ignore"):
                result = period_metrics(nav, benchmark)
            summaries.append({**meta, **result, "cumulative_excess_pp": 100*(nav.iloc[-1]-benchmark.iloc[-1]),
                              "turnover": turnover, "fees_paid": paid, "stale_mark_days": stale})
            for calendar_year in sorted(set(evaluation_days[1:].year)):
                loc = np.flatnonzero(evaluation_days.year == calendar_year)
                loc = loc[loc > 0]
                if not len(loc):
                    continue
                subset = evaluation_days[loc[0]-1:loc[-1]+1]
                with np.errstate(invalid="ignore"):
                    r = period_metrics(nav.loc[subset]/nav.loc[subset[0]], benchmark.loc[subset]/benchmark.loc[subset[0]])
                annual.append({**meta, "year": calendar_year, **r,
                               "cumulative_excess_pp": (r["total_return"]-r["benchmark_return"])*100})
            nav_rows.append(pd.DataFrame({"date": evaluation_days, "nav": nav.to_numpy(), **meta}))

        record(BENCHMARKS[0], None, benchmark)
        record(BENCHMARKS[1], None, equal/equal.iloc[0])
        variants = []
        fitted = pd.Timestamp("2021-01-18")
        for name, rows in data_v1.groupby("strategy"):
            ranked = rows.sort_values(["execution_date","score","code"], ascending=[True,False,True]).copy()
            ranked["rank"] = ranked.groupby("execution_date").cumcount()+1
            ranked["fit_status"] = np.where(ranked.execution_date.ge(fitted),"fitted","warmup")
            targets = {date: group.set_index("code")["rank"].to_dict() for date, group in
                       ranked.loc[ranked["rank"].le(30)].groupby("execution_date")}
            label = V1.get(name,"v1_lightgbm_6f_ema05")
            variants.append((label,targets,lambda n=label,r=ranked: make_buffer_selector(n,r,45,[])))
        for name, rows in data_v2.groupby("strategy"):
            for dynamic in (False, True):
                targets, _, _ = select_targets(rows, allocations, dynamic, max(fitted,first))
                mode = "dynamic_tiered" if dynamic else "static_10_10_10"
                variants.append((f"v2_{mode}_{V2[name]}",targets,
                                 lambda r=rows,d=dynamic: select_targets(r,allocations,d,max(fitted,first))[1]))
        for name, targets, make_selector in variants:
            require(len(targets)==(255 if year==2024 else 52) and
                    all(len(book)==30 for book in targets.values()), f"Invalid target grid: {name}")
            for fee in FEES:
                records, _, _, _, pending = simulate(name,days,bars,targets,events,review,initial,
                    fee_rate=Decimal(fee)/Decimal(10000),select_targets=make_selector())
                issues.extend((*item,sample,fee) for item in pending)
                nav = pd.DataFrame(records,columns=NAV).set_index("date")
                equity = nav.equity.reindex(evaluation_days)
                if year==2025:
                    equity.loc[anchor]=float(initial)
                require(equity.notna().all() and equity.gt(0).all(), "Invalid NAV")
                period = nav.loc[first:]
                record(name,fee,equity/equity.iloc[0],(period.gross/period.opening).mean(),
                       period.fee.sum(),int(period.stale.gt(0).sum()))
            print(sample,name,flush=True)
    table(pd.DataFrame(issues,columns=["strategy","date","code","opening_shares","pending_shares",
        "category","reason","sample","fee_bps"]),output/"v2_unresolved_exposures.csv")
    require(not issues, "Resolve held corporate action exposures before publishing")
    all_rows = pd.DataFrame(summaries)
    history = pd.DataFrame(annual)
    require(len(all_rows)==46 and len(history)==115, "Incomplete final matrix")
    for frame, keys in [(all_rows,["sample","strategy"]), (history,["sample","strategy","year"])]:
        gross = frame.loc[frame.fee_bps.eq(0),keys+["total_return"]].rename(columns={"total_return":"gross_return"})
        frame["gross_return"] = frame.merge(gross,on=keys,how="left",validate="many_to_one")["gross_return"]
        frame.loc[frame.strategy.isin(BENCHMARKS),"gross_return"] = frame.loc[
            frame.strategy.isin(BENCHMARKS),"total_return"]
    table(all_rows.loc[all_rows["sample"].eq("development")],output/"v2_dev_2021_2024_summary.csv")
    table(all_rows.loc[all_rows["sample"].eq("retrospective_2025")],output/"v2_holdout_2025_summary.csv")
    table(history,output/"v2_yearly_returns_2021_2025.csv")
    pd.concat(nav_rows,ignore_index=True).to_parquet(output/"final_nav.parquet",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--v1-dev-rankings",type=Path)
    parser.add_argument("--v1-2025-rankings",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    root=args.project_root.resolve()
    data=root/"research/v2_regime_tiered/results"
    cached = data/"stage4/inputs"
    dev = cached/"v1_development.parquet"
    post = cached/"v1_2025.parquet"
    run(root,data,data/"stage3_scores/tiered_weekly_scores.parquet",
        args.v1_dev_rankings or (dev if dev.exists() else root/"reports/three_factor_experiments/data/full_rankings.parquet"),
        args.v1_2025_rankings or (post if post.exists() else root/"reports/holdout_2025/data/full_rankings.parquet"),
        args.output_dir or data/"stage4",Path(__file__).with_name("verified_actions_2025.csv"))

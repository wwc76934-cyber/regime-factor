import sys
import tomllib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_factor_regimes import icir, require, write_table


NAV_FIELDS = ("strategy date cash stock_value pending_stock_value receivables rights_prepaid equity fee "
              "gross_traded opening_equity stale_marks").split()
EXPOSURE_FIELDS = "strategy date code opening_shares pending_shares category reason".split()


def run(root, output):
    sys.path.insert(0, str(root / "scripts"))
    from compare_market_state_ridge import benchmark_series
    from run_holdings_buffer_backtest import RIGHTS_KEY, make_buffer_selector, simulate

    config = tomllib.loads((root / "configs/baseline.toml").read_text(encoding="utf-8-sig"))
    start = config["research"]["bars_start"]
    end = pd.Timestamp(config["research"]["development_end"])
    require(end == pd.Timestamp("2024-12-31"), "This experiment is development-only")
    top = config["portfolio"]["top_n"]
    initial_cash = Decimal(str(config["portfolio"]["initial_cash"]))
    rankings = pd.read_parquet(output / "data/verified_full_rankings.parquet")
    targets = pd.read_parquet(output / "data/verified_targets.parquet")
    require(targets.groupby(["strategy", "execution_date"]).size().eq(top).all()
            and np.allclose(targets["target_weight"], 1 / top), "Invalid equal-weight Top 30")
    date_path = root / f"data/raw/calendar/baostock_{start}_{end:%Y-%m-%d}.parquet"
    cal = pd.read_parquet(date_path)
    days = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["is_trading_day"].astype(str).eq("1"), "calendar_date"]))
    days = days[(days >= targets["execution_date"].min()) & (days <= end)].sort_values()
    require(days.is_unique and targets["execution_date"].isin(days).all(), "Invalid trading calendar")
    processed = root / "data/processed"
    panel = pd.read_parquet(processed / "csi300_daily_panel.parquet", columns=[
        "date", "code", "raw_open", "raw_close", "preclose", "tradestatus", "isST"])
    panel = panel.loc[panel["date"].isin(days) & panel["code"].isin(targets["code"])]
    require(not panel.duplicated(["date", "code"]).any(), "Duplicate quotes")
    bars = {date: {item.code: item for item in frame.itertuples(index=False)}
            for date, frame in panel.groupby("date", sort=False)}
    events = pd.read_parquet(processed / "corporate_actions/csi300_backtest_actions.parquet")
    review = pd.read_parquet(root / "reports/portfolio/corporate_actions_remaining_review.parquet")
    for field in ["record_date", "ex_date", "pay_date", "stock_listing_date"]:
        events[field] = pd.to_datetime(events[field], errors="raise")
    review["ex_date"] = pd.to_datetime(review["ex_date"], errors="raise")
    require(not events.duplicated(["code", "ex_date"]).any() and not (
        events["code"].eq(RIGHTS_KEY[0]) & events["ex_date"].eq(RIGHTS_KEY[1])).any(), "Duplicate corporate actions")

    first_execution = rankings.loc[rankings["fit_status"].eq("fitted"), "execution_date"].min()
    anchor = days[days < first_execution][-1]
    require(first_execution == pd.Timestamp("2021-01-18") and days[-1] == end, "Unexpected evaluation window")
    common_days = days[days >= anchor]
    index = root / "data/raw/index"
    benchmarks = pd.DataFrame(index=common_days)
    benchmarks["csi300_price"] = benchmark_series(
        index / f"baostock_sh.000300_{start}_{end:%Y-%m-%d}.parquet", "close", common_days, "sh.000300")
    benchmarks["csi300_total_return"] = benchmark_series(
        index / "csi300_H00300_development.parquet", "close", common_days, "H00300")
    benchmarks["csi300_member_equal_weight"] = benchmark_series(
        processed / "benchmark/csi300_weekly_equal_weight.parquet", "nav", common_days)
    benchmarks = benchmarks.div(benchmarks.iloc[0])
    write_table(benchmarks.rename_axis("date").reset_index(), output / "reports/benchmark_nav.parquet")
    write_table((benchmarks.iloc[-1] - 1).rename("total_return").rename_axis("benchmark").reset_index(),
                output / "reports/benchmark_returns.csv")
    weekly = pd.read_parquet(output / "reports/model_weekly_rank_ic.parquet")
    weekly = weekly.loc[weekly["fit_status"].eq("fitted")]
    prediction = weekly.groupby("strategy")["rank_ic"].agg(rank_ic="mean", rank_icir=icir, ic_weeks="count")

    stats, relative, ledgers, exposures, buffer_checks, yearly = [], [], [], [], [], []
    for strategy, group in targets.groupby("strategy", sort=False):
        ranking = rankings.loc[rankings["strategy"].eq(strategy)]
        schedule = {day: week.set_index("code")["rank"].to_dict() for day, week in group.groupby("execution_date")}
        require(set(ranking["execution_date"]) == set(schedule), "Ranking and target dates differ")
        for fee in [10, 20]:
            for buffer in [False, True]:
                chosen = []
                selector = make_buffer_selector(strategy, ranking, 45, chosen) if buffer else None
                records, _, _, _, unresolved = simulate(
                    strategy, days, bars, schedule, events, review, initial_cash,
                    fee_rate=Decimal(fee) / Decimal(10000), select_targets=selector)
                meta = {"strategy": strategy, "fee_bps": fee, "buffer_rank": 45 if buffer else 30}
                nav = pd.DataFrame(records, columns=NAV_FIELDS).set_index("date")
                require(nav.index.equals(days) and nav["equity"].gt(0).all(), "Invalid backtest NAV")
                evaluation = nav.loc[common_days].copy()
                evaluation["nav"] = evaluation["equity"] / evaluation.at[anchor, "equity"]
                daily = evaluation["nav"].pct_change().iloc[1:]
                turnover_days = evaluation.loc[evaluation["opening_equity"].notna()]
                fees = evaluation.loc[first_execution:, "fee"].sum()
                flag = "unresolved_corporate_action" if unresolved else "ok"
                stats.append({**meta, "start": first_execution, "end": end, "days": len(daily),
                              "rebalance_days": len(turnover_days), "status": flag, "unresolved_exposures": len(unresolved),
                              "total_return": evaluation["nav"].iat[-1] - 1,
                              "max_drawdown": -(evaluation["nav"] / evaluation["nav"].cummax() - 1).min(),
                              "mean_two_way_turnover": (turnover_days["gross_traded"] / turnover_days["opening_equity"]).mean(),
                              "fees_paid": fees, "fees_over_anchor_equity": fees / evaluation.at[anchor, "equity"],
                              **prediction.loc[strategy].to_dict()})
                for benchmark in benchmarks:
                    index_nav = benchmarks[benchmark]
                    active = daily - index_nav.pct_change().iloc[1:]
                    rel = evaluation["nav"] / index_nav
                    tracking = active.std(ddof=1) * np.sqrt(252)
                    annual = active.mean() * 252
                    relative.append({**meta, "status": flag, "benchmark": benchmark,
                                     "cumulative_excess_pp": (evaluation["nav"].iat[-1] - index_nav.iat[-1]) * 100,
                                     "annualized_excess_return": annual,
                                     "relative_max_drawdown": -(rel / rel.cummax() - 1).min(),
                                     "annualized_tracking_error": tracking,
                                     "information_ratio": annual / tracking if tracking else np.nan})
                for year, growth in (1 + daily).groupby(daily.index.year).prod().items():
                    yearly.append({**meta, "year": year, "total_return": growth - 1, "status": flag})
                ledgers.append(evaluation.reset_index().assign(fee_bps=fee, buffer_rank=meta["buffer_rank"]))
                exposures.append(pd.DataFrame(unresolved, columns=EXPOSURE_FIELDS)
                                 .assign(fee_bps=fee, buffer_rank=meta["buffer_rank"]))
                if buffer:
                    check = pd.DataFrame(chosen, columns=[
                        "strategy", "date", "code", "model_rank", "already_held", "outside_original_top"])
                    outside = check.loc[check["outside_original_top"]]
                    require(outside["already_held"].all() and outside["model_rank"].le(45).all(),
                            "Buffer kept a new or out-of-range stock")
                    buffer_checks.append(check.assign(fee_bps=fee))
        print("Step 4 | four execution variants completed:", strategy, flush=True)

    summary = pd.DataFrame(stats)
    excess = pd.DataFrame(relative)
    all_exposures = pd.concat(exposures, ignore_index=True)
    require(len(summary) == rankings["strategy"].nunique() * 4 and
            len(excess) == len(summary) * len(benchmarks.columns), "Incomplete execution matrix")
    write_table(summary, output / "reports/execution_ablation.csv")
    write_table(excess, output / "reports/benchmark_excess_metrics.csv")
    write_table(pd.DataFrame(yearly), output / "reports/execution_ablation_yearly.csv")
    write_table(pd.concat(ledgers, ignore_index=True), output / "reports/execution_daily_nav.parquet")
    write_table(all_exposures, output / "reports/execution_unresolved_exposures.csv")
    write_table(pd.concat(buffer_checks, ignore_index=True), output / "reports/execution_buffer_selections.parquet")
    print(summary[["strategy", "fee_bps", "buffer_rank", "total_return", "mean_two_way_turnover",
                   "fees_paid", "rank_ic", "unresolved_exposures"]].round(5).to_string(index=False), flush=True)
    if not all_exposures.empty:
        print(all_exposures.to_string(index=False), flush=True)

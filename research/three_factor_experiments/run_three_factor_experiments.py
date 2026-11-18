import argparse
import sys
import tomllib
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from build_three_factor_models import icir, read_development, require, run as build, write_table


NAV_COLUMNS = ("strategy date cash stock_value pending_stock_value receivables rights_prepaid equity fee "
               "gross_traded opening_equity stale_marks").split()
EXPOSURE_COLUMNS = "strategy date code opening_shares pending_shares category reason".split()
DIVIDEND = {
    "code": "sh.601985",
    "record_date": pd.Timestamp("2024-11-29"),
    "ex_date": pd.Timestamp("2024-12-02"),
    "pay_date": pd.Timestamp("2024-12-02"),
    "stock_listing_date": pd.NaT,
    "cash_per_share": "0.02",
    "bonus_shares_per_share": "0",
    "reserve_shares_per_share": "0",
    "event_type": "cash_only",
    "source_rows": 0,
    "source_category": "price_flag_only",
    "verification_note": "Company announcement 2024-082; same gross-cash convention as the engine",
    "source_url": "https://static.cninfo.com.cn/finalpage/2024-11-25/1221831318.PDF",
}


def apply_verified_dividend(events, review):
    key = events["code"].eq(DIVIDEND["code"]) & events["ex_date"].eq(DIVIDEND["ex_date"])
    pending = review["code"].eq(DIVIDEND["code"]) & review["ex_date"].eq(DIVIDEND["ex_date"])
    if key.any():
        for field in ["record_date", "pay_date"]:
            require(events.loc[key, field].eq(DIVIDEND[field]).all(), "Conflicting dividend date")
        for field in ["cash_per_share", "bonus_shares_per_share", "reserve_shares_per_share"]:
            require(all(Decimal(str(value)) == Decimal(DIVIDEND[field]) for value in events.loc[key, field]),
                    "Conflicting dividend value")
    else:
        events = pd.concat([events, pd.DataFrame([DIVIDEND])], ignore_index=True)
    return events, review.loc[~pending].copy()


def backtest(root, output):
    sys.path.insert(0, str(root / "scripts"))
    from compare_market_state_ridge import benchmark_series
    from run_holdings_buffer_backtest import RIGHTS_KEY, make_buffer_selector, simulate

    config = tomllib.loads((root / "configs/baseline.toml").read_text(encoding="utf-8-sig"))
    start, end = config["research"]["bars_start"], config["research"]["development_end"]
    initial = Decimal(str(config["portfolio"]["initial_cash"]))
    require(end == "2024-12-31" and config["portfolio"]["top_n"] == 30, "Unexpected development settings")
    processed = root / "data/processed"
    ranks = pd.read_parquet(output / "data/full_rankings.parquet")
    targets = pd.read_parquet(output / "data/targets.parquet")
    cal = pd.read_parquet(root / f"data/raw/calendar/baostock_{start}_{end}.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["is_trading_day"].astype(str).eq("1"), "calendar_date"]))
    days = days[(days >= targets["execution_date"].min()) & (days <= pd.Timestamp(end))].sort_values()
    require(days.is_unique and targets["execution_date"].isin(days).all()
            and targets.groupby(["strategy", "execution_date"]).size().eq(30).all(), "Invalid calendar or targets")
    panel = read_development(processed / "csi300_daily_panel.parquet",
                             ["date", "code", "raw_open", "raw_close", "preclose", "tradestatus", "isST"],
                             date_field="date")
    panel = panel.loc[panel["date"].isin(days) & panel["code"].isin(targets["code"])]
    require(not panel.duplicated(["date", "code"]).any(), "Duplicate quotes")
    bars = {date: {item.code: item for item in frame.itertuples(index=False)}
            for date, frame in panel.groupby("date", sort=False)}
    events = pd.read_parquet(processed / "corporate_actions/csi300_backtest_actions.parquet")
    review = pd.read_parquet(root / "reports/portfolio/corporate_actions_remaining_review.parquet")
    for field in ["record_date", "ex_date", "pay_date", "stock_listing_date"]:
        events[field] = pd.to_datetime(events[field], errors="raise")
    review["ex_date"] = pd.to_datetime(review["ex_date"], errors="raise")
    events = events.loc[events["ex_date"].isin(days)]
    review = review.loc[review["ex_date"].isin(days)]
    events, review = apply_verified_dividend(events, review)
    write_table(pd.DataFrame([DIVIDEND]), output / "reports/verified_dividend.csv")
    require(not events.duplicated(["code", "ex_date"]).any()
            and not (events["code"].eq(RIGHTS_KEY[0]) & events["ex_date"].eq(RIGHTS_KEY[1])).any(),
            "Duplicate corporate actions")
    first = ranks.loc[ranks["fit_status"].eq("fitted"), "execution_date"].min()
    anchor = days[days < first][-1]
    common = days[days >= anchor]
    require(first == pd.Timestamp("2021-01-18") and len(common) == 960, "Changed evaluation window")
    index = root / "data/raw/index"
    benchmarks = pd.DataFrame({
        "csi300_price": benchmark_series(index / f"baostock_sh.000300_{start}_{end}.parquet",
                                        "close", common, "sh.000300"),
        "csi300_total_return": benchmark_series(index / "csi300_H00300_development.parquet",
                                               "close", common, "H00300"),
        "csi300_member_equal_weight": benchmark_series(processed / "benchmark/csi300_weekly_equal_weight.parquet",
                                                      "nav", common),
    }, index=common)
    benchmarks = benchmarks.div(benchmarks.iloc[0])
    write_table((benchmarks.iloc[-1] - 1).rename("total_return").rename_axis("benchmark").reset_index(),
                output / "reports/benchmark_returns.csv")
    weekly = pd.read_parquet(output / "reports/model_weekly_rank_ic.parquet")
    weekly = weekly.loc[weekly["fit_status"].eq("fitted")]
    prediction = weekly.groupby("strategy")["rank_ic"].agg(rank_ic="mean", rank_icir=icir, ic_weeks="count")
    tie_source, tie_check = "lightgbm_6f", "lightgbm_6f_reverse_ties"
    tree_rows = ranks.loc[ranks["strategy"].eq(tie_source)].copy()
    fitted = tree_rows.loc[tree_rows["fit_status"].eq("fitted")].sort_values(
        ["signal_date", "score", "code"], ascending=[True, False, False])
    fitted["rank"] = fitted.groupby("signal_date").cumcount() + 1
    reverse = pd.concat([tree_rows.loc[tree_rows["fit_status"].ne("fitted")], fitted]).assign(strategy=tie_check)
    ranks = pd.concat([ranks, reverse], ignore_index=True)
    targets = pd.concat([targets, reverse.loc[reverse["rank"].le(30)].assign(target_weight=1 / 30)],
                        ignore_index=True)
    prediction.loc[tie_check] = prediction.loc[tie_source]
    cutoffs = ranks.loc[ranks["fit_status"].eq("fitted") & ranks["rank"].isin([30, 31])].pivot(
        index=["strategy", "signal_date"], columns="rank", values="score")
    ties = cutoffs[30].eq(cutoffs[31]).groupby("strategy").agg(tied_top30_weeks="sum", fitted_weeks="size")
    write_table(ties.reset_index(), output / "reports/top30_score_ties.csv")

    stats, relative, ledgers, exposures, yearly, dividend_audit = [], [], [], [], [], []
    for name, group in targets.groupby("strategy", sort=True):
        ranking = ranks.loc[ranks["strategy"].eq(name)]
        schedule = {day: week.set_index("code")["rank"].to_dict() for day, week in group.groupby("execution_date")}
        buffer_levels = [30, 45, 50, 55] if name == "state_ridge" else [30, 45]
        for keep in buffer_levels:
            for fee in [10, 20]:
                audit = []
                selector = make_buffer_selector(name, ranking, keep, audit) if keep > 30 else None
                records, _, _, actions, unresolved = simulate(
                    name, days, bars, schedule, events, review, initial, fee_rate=Decimal(fee) / Decimal(10000),
                    select_targets=selector)
                require(all((held and rank <= keep) for _, _, _, rank, held, outside in audit if outside),
                        "Buffer kept a new or out-of-range stock")
                nav = pd.DataFrame(records, columns=NAV_COLUMNS).set_index("date")
                require(nav.index.equals(days) and nav["equity"].gt(0).all(), "Invalid backtest NAV")
                eva = nav.loc[common].copy()
                eva["nav"] = eva["equity"] / eva.at[anchor, "equity"]
                daily = eva["nav"].pct_change().iloc[1:]
                turns = eva.loc[eva["opening_equity"].notna()]
                meta = {"strategy": name, "fee_bps": fee, "buffer_rank": keep}
                for _, date, code, ex_date, stage, amount, *_ in actions:
                    if code == DIVIDEND["code"] and ex_date == DIVIDEND["ex_date"]:
                        dividend_audit.append({**meta, "date": date, "code": code, "stage": stage, "amount": amount})
                stats.append({**meta, "unresolved_exposures": len(unresolved), "days": len(daily),
                              "rebalance_days": len(turns), "total_return": eva["nav"].iat[-1] - 1,
                              "max_drawdown": -(eva["nav"] / eva["nav"].cummax() - 1).min(),
                              "mean_two_way_turnover": (turns["gross_traded"] / turns["opening_equity"]).mean(),
                              "fees_paid": eva.loc[first:, "fee"].sum(), **prediction.loc[name].to_dict()})
                for benchmark in benchmarks:
                    active = daily - benchmarks[benchmark].pct_change().iloc[1:]
                    tracking = active.std(ddof=1) * np.sqrt(252)
                    rel = eva["nav"] / benchmarks[benchmark]
                    relative.append({**meta, "benchmark": benchmark,
                                     "annualized_excess_return": active.mean() * 252,
                                     "relative_max_drawdown": -(rel / rel.cummax() - 1).min(),
                                     "information_ratio": active.mean() * 252 / tracking if tracking else np.nan})
                for year, growth in (1 + daily).groupby(daily.index.year).prod().items():
                    yearly.append({**meta, "year": year, "total_return": growth - 1})
                ledgers.append(eva.reset_index().assign(fee_bps=fee, buffer_rank=keep))
                exposures.append(pd.DataFrame(unresolved, columns=EXPOSURE_COLUMNS)
                                 .assign(fee_bps=fee, buffer_rank=keep))
        print(f"Backtested {name}: {len(buffer_levels) * 2} configurations", flush=True)
    summary = pd.DataFrame(stats)
    require(len(summary) == 44 and summary["days"].eq(959).all() and summary["rebalance_days"].eq(202).all(),
            "Incomplete backtest matrix")
    write_table(summary, output / "reports/execution_ablation.csv")
    write_table(pd.DataFrame(relative), output / "reports/benchmark_excess_metrics.csv")
    write_table(pd.DataFrame(yearly), output / "reports/yearly_returns.csv")
    write_table(pd.concat(ledgers, ignore_index=True), output / "reports/daily_nav.parquet")
    write_table(pd.concat(exposures, ignore_index=True), output / "reports/unresolved_exposures.csv")
    write_table(pd.DataFrame(dividend_audit), output / "reports/verified_dividend_ledger.csv")
    print(summary.round(5).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run development-only three-factor Ridge and CPU LightGBM controls.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "reports/three_factor_experiments").resolve()
    require(output not in {root, root / "reports", root / "data", root / "configs"}, "Choose an experiment directory")
    started = perf_counter()
    build(root, output)
    backtest(root, output)
    print(f"Completed in {perf_counter() - started:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()

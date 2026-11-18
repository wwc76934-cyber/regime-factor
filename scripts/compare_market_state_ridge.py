"""Compare walk-forward controls and CSI 300 benchmarks on identical dates."""

import argparse
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


STRATEGIES = ["equal_weight", "rolling_ridge", "state_ridge"]
SESSIONS_PER_YEAR = 252


def benchmark_series(path, column, days, expected_code=None):
    cols = ["date", column] + (["code"] if expected_code is not None else [])
    data = pd.read_parquet(path, columns=cols)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any() or not data["date"].is_unique:
        raise ValueError(f"Missing or duplicate benchmark dates: {path}")
    if expected_code is not None and not data["code"].eq(expected_code).fillna(False).all():
        raise ValueError(f"Unexpected benchmark code in {path}")
    values = data.set_index("date")[column]
    missing = days.difference(values.index)
    if len(missing):
        raise ValueError(f"{path.name} missing {len(missing)} evaluation dates: {missing[:5].tolist()}")
    values = pd.to_numeric(values.reindex(days), errors="raise")
    if not (np.isfinite(values) & values.gt(0)).all():
        raise ValueError(f"Missing or invalid benchmark levels: {path}")
    return values


def compare_benchmarks(root, equity, anchor):
    config = tomllib.loads((root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig"))
    research = config["research"]
    start, end = research["bars_start"], research["development_end"]
    index = root / "data" / "raw" / "index"
    processed = root / "data" / "processed" / "benchmark"

    levels = equity.loc[anchor:, STRATEGIES].copy()
    levels["csi300_price"] = benchmark_series(
        index / f"baostock_sh.000300_{start}_{end}.parquet", "close", levels.index, "sh.000300"
    )
    levels["csi300_total_return"] = benchmark_series(
        index / "csi300_H00300_development.parquet", "close", levels.index, "H00300"
    )
    levels["csi300_member_equal_weight"] = benchmark_series(
        processed / "csi300_weekly_equal_weight.parquet", "nav", levels.index,
    )

    nav = levels.div(levels.iloc[0], axis="columns")
    returns = nav.pct_change().iloc[1:]
    benchmarks = [name for name in nav if name not in STRATEGIES]
    summary = pd.DataFrame({
        "total_return": nav[benchmarks].iloc[-1] - 1,
        "max_drawdown": -(nav[benchmarks] / nav[benchmarks].cummax() - 1).min(),
    }).rename_axis("benchmark").reset_index()

    rows, paths = [], []
    for benchmark in benchmarks:
        for strategy in STRATEGIES:
            active = returns[strategy] - returns[benchmark]
            relative = nav[strategy] / nav[benchmark]
            relative_dd = relative / relative.cummax() - 1
            annualized_active = active.mean() * SESSIONS_PER_YEAR
            tracking_error = active.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
            rows.append({
                "benchmark": benchmark,
                "strategy": strategy,
                "days": len(active),
                "cumulative_return_gap_pp": (nav[strategy].iloc[-1] - nav[benchmark].iloc[-1]) * 100,
                "annualized_excess_return": annualized_active,
                "relative_max_drawdown": -relative_dd.min(),
                "annualized_tracking_error": tracking_error,
                "information_ratio": annualized_active / tracking_error if tracking_error > 0 else np.nan,
            })
            paths.append(pd.DataFrame({
                "date": relative.index,
                "benchmark": benchmark,
                "strategy": strategy,
                "relative_nav": relative.to_numpy(),
                "relative_drawdown": relative_dd.to_numpy(),
                "daily_active_return": active.reindex(relative.index).to_numpy(),
            }))

    return nav.rename_axis("date").reset_index(), summary, pd.DataFrame(rows), pd.concat(paths, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root
    portfolio = root / "reports" / "portfolio"
    report = args.report_dir or portfolio / "walk_forward_three_variants"
    inputs = root / "data" / "processed" / "portfolio" / "walk_forward"

    unresolved = pd.read_parquet(report / "unresolved_exposure.parquet")
    if not unresolved.empty:
        raise ValueError("Resolve the logged corporate-action exposures before comparing performance")

    fits = pd.read_parquet(inputs / "walk_forward_weight_schedule.parquet")
    first_signal = fits.loc[fits["fit_status"].eq("fitted"), "signal_date"].min()
    targets = pd.read_parquet(args.targets or (inputs / "walk_forward_three_variant_targets.parquet"))
    start_dates = targets.loc[targets["signal_date"].eq(first_signal), "execution_date"].unique()
    if len(start_dates) != 1:
        raise ValueError("First fitted signal must have one common execution date")
    start = pd.Timestamp(start_dates[0])

    nav = pd.read_parquet(report / "daily_nav.parquet").sort_values(["strategy", "date"])
    if nav.duplicated(["strategy", "date"]).any() or set(nav["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected strategies or duplicated ledger dates")
    dates = nav.pivot(index="date", columns="strategy", values="equity")
    if not np.isfinite(dates.to_numpy()).all() or not dates.gt(0).all().all():
        raise ValueError("Invalid equity or nonmatching dates")
    before = dates.loc[dates.index < start]
    if before.empty or not all(
        np.array_equal(before["equal_weight"], before[name]) for name in STRATEGIES[1:]
    ):
        raise ValueError("All strategies must match before the first fitted execution")
    anchor = before.index[-1]

    previous = pd.read_parquet(portfolio / "walk_forward_controls" / "daily_nav.parquet")
    check = previous.pivot(index="date", columns="strategy", values="equity").sort_index()
    if not dates.index.equals(check.index) or not np.allclose(
        dates[STRATEGIES[:2]], check[STRATEGIES[:2]], rtol=0, atol=0.005
    ):
        raise ValueError("The two controls do not reproduce the previous run")

    rows = []
    comparison = []
    for name, group in nav.groupby("strategy", sort=False):
        group = group.set_index("date")
        daily = group.loc[group.index >= start].copy()
        daily["nav"] = daily["equity"] / group.at[anchor, "equity"]
        daily["drawdown"] = daily["nav"] / daily["nav"].cummax().clip(lower=1) - 1
        daily["daily_return"] = group["equity"].pct_change().reindex(daily.index)
        rebalance = daily["opening_equity"].notna()
        turnover = daily.loc[rebalance, "gross_traded"] / daily.loc[rebalance, "opening_equity"]
        rows.append({
            "strategy": name,
            "days": len(daily),
            "rebalance_days": int(rebalance.sum()),
            "total_return": daily["nav"].iloc[-1] - 1,
            "max_drawdown": -daily["drawdown"].min(),
            "mean_two_way_turnover": turnover.mean(),
            "fees_paid": daily["fee"].sum(),
        })
        comparison.append(daily[["nav", "drawdown", "daily_return"]].assign(strategy=name).reset_index())

    summary = pd.DataFrame(rows).set_index("strategy").reindex(STRATEGIES)
    daily = pd.concat(comparison, ignore_index=True)
    annual = (
        daily.assign(year=daily["date"].dt.year)
        .groupby(["year", "strategy"])["daily_return"]
        .agg(lambda values: (1 + values).prod() - 1)
        .unstack("strategy").reindex(columns=STRATEGIES)
    )
    annual["state_minus_rolling_pp"] = (annual["state_ridge"] - annual["rolling_ridge"]) * 100
    benchmark_nav, benchmark_summary, excess, relative = compare_benchmarks(root, dates, anchor)
    output = args.output_dir or report
    output.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_parquet(output / "post_warmup_summary.parquet", index=False)
    daily.to_parquet(output / "post_warmup_daily.parquet", index=False)
    annual.reset_index().to_parquet(output / "post_warmup_yearly.parquet", index=False)
    benchmark_nav.to_parquet(output / "benchmark_comparison_nav.parquet", index=False)
    benchmark_summary.to_parquet(output / "benchmark_summary.parquet", index=False)
    excess.to_parquet(output / "benchmark_excess_metrics.parquet", index=False)
    relative.to_parquet(output / "benchmark_relative_daily.parquet", index=False)
    print(f"First fitted signal: {first_signal:%Y-%m-%d} | Execution: {start:%Y-%m-%d} | Anchor close: {anchor:%Y-%m-%d}")
    print(summary.to_string(formatters={
        "total_return": "{:+.2%}".format, "max_drawdown": "{:.2%}".format,
        "mean_two_way_turnover": "{:.2%}".format, "fees_paid": "{:,.2f}".format,
    }))
    print("\nReturns by calendar year (first year is partial):")
    print(annual.to_string(formatters={
        "equal_weight": "{:+.2%}".format, "rolling_ridge": "{:+.2%}".format,
        "state_ridge": "{:+.2%}".format, "state_minus_rolling_pp": "{:+.2f}".format,
    }))
    print("\nBenchmark results:")
    print(benchmark_summary.to_string(index=False, formatters={
        "total_return": "{:+.2%}".format, "max_drawdown": "{:.2%}".format,
    }))
    print("\nRelative performance by benchmark:")
    print(excess.to_string(index=False, formatters={
        "cumulative_return_gap_pp": "{:+.2f}".format,
        "annualized_excess_return": "{:+.2%}".format,
        "relative_max_drawdown": "{:.2%}".format,
        "annualized_tracking_error": "{:.2%}".format,
        "information_ratio": "{:+.2f}".format,
    }))
    print("\nOutput:", output)


if __name__ == "__main__":
    main()

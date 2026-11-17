"""Compare walk-forward controls from the first fitted execution date."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRATEGIES = ["equal_weight", "rolling_ridge", "state_ridge"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--targets", type=Path)
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
    summary.reset_index().to_parquet(report / "post_warmup_summary.parquet", index=False)
    daily.to_parquet(report / "post_warmup_daily.parquet", index=False)
    annual.reset_index().to_parquet(report / "post_warmup_yearly.parquet", index=False)
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


if __name__ == "__main__":
    main()

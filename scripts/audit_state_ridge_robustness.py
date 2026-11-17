"""Audit the concentration and modeled fee sensitivity of the state Ridge results."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


STRATEGIES = ["equal_weight", "rolling_ridge", "state_ridge"]
FEES_BPS = (15, 20)


def equity_table(path):
    daily = pd.read_parquet(path / "daily_nav.parquet", columns=["date", "strategy", "equity"])
    if set(daily["strategy"]) != set(STRATEGIES):
        raise ValueError(f"Unexpected strategies: {path}")
    table = daily.pivot(index="date", columns="strategy", values="equity").sort_index()
    table = table[STRATEGIES]
    if not np.isfinite(table.to_numpy()).all() or not table.gt(0).all().all():
        raise ValueError(f"Missing or invalid daily equity: {path}")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "reports" / "portfolio" / "walk_forward_three_variants"
    output = (args.output_dir or source / "robustness").resolve()
    output.mkdir(parents=True, exist_ok=True)

    walk = root / "data" / "processed" / "portfolio" / "walk_forward"
    fit = pd.read_parquet(walk / "walk_forward_weight_schedule.parquet")
    first = fit.loc[fit["fit_status"].eq("fitted"), "signal_date"].min()
    target_path = walk / "walk_forward_three_variant_targets.parquet"
    targets = pd.read_parquet(target_path, columns=["signal_date", "execution_date"])
    dates = targets.loc[targets["signal_date"].eq(first), "execution_date"].unique()
    if len(dates) != 1:
        raise ValueError("Expected one first fitted execution date")
    start = pd.Timestamp(dates[0])

    baseline = equity_table(source)
    pre_start = baseline.index[baseline.index < start]
    if len(pre_start) == 0:
        raise ValueError("Missing pre-execution close")
    anchor = pre_start[-1]

    # Month-on-month returns use the previous month's actual closing equity.
    monthly_close = baseline.groupby(baseline.index.to_period("M")).last()
    monthly = monthly_close.pct_change().loc["2024", ["rolling_ridge", "state_ridge"]].copy()
    monthly["state_minus_rolling_pp"] = (monthly["state_ridge"] - monthly["rolling_ridge"]) * 100
    monthly.index = monthly.index.astype(str)
    monthly.rename_axis("month").reset_index().to_parquet(
        output / "state_ridge_2024_monthly.parquet", index=False
    )

    engine = Path(__file__).resolve().with_name("run_development_backtest_with_fees.py")
    paths = {10: source}
    for bps in FEES_BPS:
        destination = output / f"fee_{bps:02d}_bps"
        print(f"Running {bps} bps per side...", flush=True)
        run = subprocess.run([
            sys.executable, "-X", "utf8", str(engine),
            "--project-root", str(root),
            "--targets", str(target_path),
            "--output-dir", str(destination),
            "--fee-bps", str(bps),
        ], capture_output=True, encoding="utf-8")
        (output / f"fee_{bps:02d}_run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{bps} bps run failed. See {output / f'fee_{bps:02d}_run.log'}")
        paths[bps] = destination

    results = []
    for bps, path in paths.items():
        equity = equity_table(path)
        if not equity.index.equals(baseline.index):
            raise ValueError(f"Mismatched trading days for fee {bps} bps")
        exposures = pd.read_parquet(path / "unresolved_exposure.parquet")
        nav = equity.loc[start:] / equity.loc[anchor]
        drawdown = nav / nav.cummax().clip(lower=1) - 1
        fees = pd.read_parquet(path / "daily_nav.parquet", columns=["date", "strategy", "fee"])
        fees = fees.loc[fees["date"].ge(start)].groupby("strategy")["fee"].sum()
        for strategy in STRATEGIES:
            results.append({
                "fee_bps": bps, "strategy": strategy,
                "total_return": nav[strategy].iloc[-1] - 1,
                "max_drawdown": -drawdown[strategy].min(),
                "fees_paid": fees[strategy],
                "unresolved_exposure": int(exposures["strategy"].eq(strategy).sum()),
            })

    result = pd.DataFrame(results)
    primary = result.pivot(index="fee_bps", columns="strategy", values="total_return")
    primary = primary[STRATEGIES]
    primary["state_minus_rolling_pp"] = (primary["state_ridge"] - primary["rolling_ridge"]) * 100
    pending = result.groupby("fee_bps")["unresolved_exposure"].sum()
    primary["unresolved_exposure"] = pending
    result.to_parquet(output / "fee_sensitivity_detail.parquet", index=False)
    primary.reset_index().to_parquet(output / "fee_sensitivity_summary.parquet", index=False)
    formats = {name: "{:+.2%}".format for name in STRATEGIES}
    formats["state_minus_rolling_pp"] = "{:+.2f}".format
    print(f"\nEvaluation: {start:%Y-%m-%d} to {baseline.index.max():%Y-%m-%d}")
    print("\n2024 monthly returns:")
    print(monthly.to_string(formatters=formats))
    print("\nFee sensitivity (bps on each buy and sell):")
    print(primary.to_string(formatters=formats))
    if pending.gt(0).any():
        print("Scenarios with unresolved exposures are provisional. See the corresponding ledgers.")
    print("\nOutput:", output)


if __name__ == "__main__":
    main()

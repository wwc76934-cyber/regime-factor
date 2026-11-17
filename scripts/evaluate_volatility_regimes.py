"""用筛选段固定阈值，比较 VOL_60 在高低市场波动周的表现。"""

from pathlib import Path

import numpy as np
import pandas as pd


FACTOR = "VOL_60"
SPLITS = ("screening", "validation")
REGIMES = ("low", "high")


def prepare_weeks(factor_report, market_states):
    weekly = factor_report.loc[
        factor_report["factor"].eq(FACTOR)
        & factor_report["evaluation_split"].isin(SPLITS),
        ["signal_date", "evaluation_split", "rank_ic", "q5_minus_q1"],
    ].copy()

    states = market_states[["signal_date", "market_volatility_20d"]].copy()
    for frame in (weekly, states):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
        if frame["signal_date"].isna().any() or not frame["signal_date"].is_unique:
            raise ValueError("信号日期缺失或重复")

    joined = weekly.merge(
        states, on="signal_date", how="left", validate="one_to_one"
    )
    if set(joined["evaluation_split"]) != set(SPLITS):
        raise ValueError("缺少筛选段或验证段的周度因子结果")

    vol = joined["market_volatility_20d"]
    if not np.isfinite(vol).all() or vol.lt(0).any():
        raise ValueError("存在未匹配的信号周或无效的市场波动率")

    screening = joined.loc[joined["evaluation_split"].eq("screening")]
    threshold = float(screening["market_volatility_20d"].median())

    # 等于阈值的周归入高波动组，两段使用同一数值。
    joined["volatility_regime"] = np.where(vol.ge(threshold), "high", "low")
    joined["low_vol_ic"] = -joined["rank_ic"]
    joined["low_minus_high_bps"] = -joined["q5_minus_q1"] * 10_000

    return joined.sort_values("signal_date"), threshold


def summarize(weekly):
    summary = weekly.groupby(
        ["evaluation_split", "volatility_regime"],
        sort=False,
    ).agg(
        weeks=("signal_date", "size"),
        ic_weeks=("low_vol_ic", "count"),
        spread_weeks=("low_minus_high_bps", "count"),
        mean_low_vol_ic=("low_vol_ic", "mean"),
        mean_low_minus_high_bps=("low_minus_high_bps", "mean"),
    )

    order = pd.MultiIndex.from_product(
        [SPLITS, REGIMES],
        names=["evaluation_split", "volatility_regime"],
    )
    return summary.reindex(order)


def compare_regimes(summary):
    low = summary.xs("low", level="volatility_regime")
    high = summary.xs("high", level="volatility_regime")

    return pd.DataFrame({
        "low_volatility_weeks": low["weeks"],
        "high_volatility_weeks": high["weeks"],
        "high_minus_low_ic": (
            high["mean_low_vol_ic"] - low["mean_low_vol_ic"]
        ),
        "high_minus_low_spread_bps": (
            high["mean_low_minus_high_bps"]
            - low["mean_low_minus_high_bps"]
        ),
    })


def main():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "factor_evaluation"
    market_dir = root / "data" / "processed" / "market"

    factor_report = pd.read_parquet(report_dir / "single_factor_weekly.parquet")
    market_states = pd.read_parquet(market_dir / "csi300_weekly_market_states.parquet")

    weekly, threshold = prepare_weeks(factor_report, market_states)
    weekly["volatility_threshold"] = threshold
    summary = summarize(weekly)
    differences = compare_regimes(summary)

    output = report_dir / "volatility_regimes"
    output.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(output / "volatility_regime_weekly.parquet", index=False)
    summary.reset_index().to_parquet(output / "volatility_regime_summary.parquet", index=False)
    differences.reset_index().to_parquet(
        output / "volatility_regime_differences.parquet", index=False
    )

    print(f"筛选段确定的日波动率阈值：{threshold:.4%}")
    print("\n各时期和状态下的低波因子表现")
    print(summary.to_string(formatters={
        "mean_low_vol_ic": "{:+.4f}".format,
        "mean_low_minus_high_bps": "{:+.1f}".format,
    }))

    print("\n高波动周的因子表现减去低波动周")
    print(differences.to_string(formatters={
        "high_minus_low_ic": "{:+.4f}".format,
        "high_minus_low_spread_bps": "{:+.1f}".format,
    }))
    print(f"\n输出目录：{output}")


if __name__ == "__main__":
    main()
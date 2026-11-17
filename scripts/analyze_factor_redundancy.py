"""在筛选段计算候选因子之间的周度截面秩相关。"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MIN_COMMON_STOCKS = 50
HIGH_CORRELATION = 0.75


def weekly_correlations(candidates, factors):
    pairs = list(combinations(factors, 2))
    rows = []

    for signal_date, week in candidates.groupby("signal_date", sort=True):
        usable = week["signal_bar_usable"].fillna(False)
        values = week.loc[usable, factors].replace([np.inf, -np.inf], np.nan)

        # corr 按因子对使用共同有效的股票，并在该样本上计算 Spearman。
        correlations = values.corr(
            method="spearman",
            min_periods=MIN_COMMON_STOCKS,
        )

        # 矩阵乘积得到每个因子对的共同有效股票数。
        available = values.notna().astype("int64")
        common_counts = available.T @ available

        for left, right in pairs:
            rho = correlations.at[left, right]
            rows.append({
                "signal_date": signal_date,
                "factor_a": left,
                "factor_b": right,
                "common_stocks": int(common_counts.at[left, right]),
                "rank_corr": rho,
            })

    return pd.DataFrame(rows)


def summarize_pairs(weekly):
    weekly = weekly.copy()
    weekly["abs_rank_corr"] = weekly["rank_corr"].abs()
    weekly["high_corr"] = (
        weekly["abs_rank_corr"].ge(HIGH_CORRELATION)
        .astype(float)
        .where(weekly["rank_corr"].notna())
    )

    return (
        weekly.groupby(["factor_a", "factor_b"], as_index=False)
        .agg(
            signal_weeks=("signal_date", "size"),
            valid_weeks=("rank_corr", "count"),
            median_common_stocks=("common_stocks", "median"),
            median_rank_corr=("rank_corr", "median"),
            median_abs_rank_corr=("abs_rank_corr", "median"),
            high_corr_week_share=("high_corr", "mean"),
        )
        .sort_values(
            ["median_abs_rank_corr", "high_corr_week_share"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def main():
    root = Path(__file__).resolve().parents[1]
    factor_dir = root / "data" / "processed" / "factors"
    report_dir = root / "reports" / "factor_evaluation"

    registry = pd.read_parquet(factor_dir / "candidate_factor_registry.parquet")
    factors = registry.loc[registry["status"].eq("candidate"), "factor"].tolist()

    candidates = pd.read_parquet(
        factor_dir / "csi300_weekly_candidates.parquet",
        columns=["signal_date", "code", "signal_bar_usable", *factors],
    )
    split_dates = pd.read_parquet(
        factor_dir / "csi300_weekly_forward_labels.parquet",
        columns=["signal_date", "evaluation_split"],
    ).drop_duplicates()

    if not split_dates["signal_date"].is_unique:
        raise ValueError("同一个信号日出现了多个评价分段")

    screening_dates = split_dates.loc[
        split_dates["evaluation_split"].eq("screening"), "signal_date"
    ]
    screening = candidates.loc[candidates["signal_date"].isin(screening_dates)]
    if screening["signal_date"].nunique() != len(screening_dates):
        raise ValueError("部分筛选段信号日缺少候选因子记录")
    if screening.empty:
        raise ValueError("没有筛选段因子记录")

    weekly = weekly_correlations(screening, factors)
    summary = summarize_pairs(weekly)

    # 方向用有符号中位数展示。是否列为重点取决于绝对相关的中位数。
    matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
    for factor in factors:
        matrix.at[factor, factor] = 1.0
    for row in summary.itertuples(index=False):
        matrix.at[row.factor_a, row.factor_b] = row.median_rank_corr
        matrix.at[row.factor_b, row.factor_a] = row.median_rank_corr

    report_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(report_dir / "redundancy_weekly.parquet", index=False)
    summary.to_parquet(report_dir / "redundancy_pairs.parquet", index=False)
    matrix.rename_axis("factor").reset_index().to_parquet(
        report_dir / "redundancy_signed_matrix.parquet", index=False
    )

    flagged = summary.loc[summary["median_abs_rank_corr"].ge(HIGH_CORRELATION)]
    print(f"筛选段周数：{len(screening_dates)}，因子数：{len(factors)}")
    print(f"因子对数：{len(summary)}，周度相关记录：{len(weekly):,}")
    print(f"达到初筛门槛的因子对：{len(flagged)}")

    print("\n绝对秩相关中位数最高的 15 对")
    columns = [
        "factor_a", "factor_b", "valid_weeks", "median_common_stocks",
        "median_rank_corr", "median_abs_rank_corr", "high_corr_week_share",
    ]
    print(summary[columns].head(15).to_string(index=False, formatters={
        "median_rank_corr": "{:+.3f}".format,
        "median_abs_rank_corr": "{:.3f}".format,
        "high_corr_week_share": "{:.1%}".format,
    }))
    print(f"\n输出目录：{report_dir}")


if __name__ == "__main__":
    main()
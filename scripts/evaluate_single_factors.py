"""分阶段计算候选因子的周度 Rank IC 与五分组收益。"""

from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]
MIN_STOCKS_PER_WEEK = 50
QUINTILES = range(1, 6)


def evaluate_week(week, factor):
    # 每个因子单独确定有效样本。
    usable = week["evaluation_ready"].fillna(False).astype(bool)
    usable &= np.isfinite(week[factor])
    usable &= np.isfinite(week["forward_return"])

    sample = week.loc[usable, [factor, "forward_return"]]
    n = len(sample)
    row = {
        "members": len(week),
        "observations": n,
        "coverage": n / len(week),
        "rank_ic": np.nan,
        "q5_minus_q1": np.nan,
        "status": "ok",
    }
    for group in QUINTILES:
        row[f"q{group}_return"] = np.nan
        row[f"q{group}_count"] = 0

    if n < MIN_STOCKS_PER_WEEK:
        row["status"] = "too_few_observations"
        return row

    factor_rank = sample[factor].rank(method="average")
    return_rank = sample["forward_return"].rank(method="average")

    if factor_rank.nunique() < 2:
        row["status"] = "constant_factor"
        return row

    # Spearman 相关系数等于两列秩的 Pearson 相关系数。
    if return_rank.nunique() >= 2:
        row["rank_ic"] = factor_rank.corr(return_rank)
    else:
        row["status"] = "constant_forward_return"

    # 相同因子值取平均秩，因此相同值会落进同一个组。
    # 因子并列较多时，各组人数可能不同，个别组也可能为空。
    percentile = (factor_rank - 0.5) / n
    groups = np.floor(5 * percentile).astype(int) + 1
    grouped = sample["forward_return"].groupby(groups).agg(["mean", "size"])

    for group in QUINTILES:
        if group in grouped.index:
            row[f"q{group}_return"] = grouped.at[group, "mean"]
            row[f"q{group}_count"] = int(grouped.at[group, "size"])

    if row["q1_count"] and row["q5_count"]:
        row["q5_minus_q1"] = row["q5_return"] - row["q1_return"]

    return row


def summarize(weekly, keys):
    grouped = weekly.groupby(keys, sort=False)
    summary = grouped.agg(
        signal_weeks=("signal_date", "size"),
        ic_weeks=("rank_ic", "count"),
        mean_coverage=("coverage", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        std_rank_ic=("rank_ic", "std"),
        positive_ic_weeks=("rank_ic", lambda values: values.gt(0).sum()),
        spread_weeks=("q5_minus_q1", "count"),
        mean_q5_minus_q1=("q5_minus_q1", "mean"),
    )

    summary["icir"] = (
        summary["mean_rank_ic"]
        / summary["std_rank_ic"].where(summary["std_rank_ic"].gt(0))
    )
    summary["positive_ic_ratio"] = (
        summary["positive_ic_weeks"]
        / summary["ic_weeks"].where(summary["ic_weeks"].gt(0))
    )

    group_columns = [f"q{group}_return" for group in QUINTILES]
    summary = summary.join(grouped[group_columns].mean())
    return summary.reset_index()


def main():
    root = Path(__file__).resolve().parents[1]
    factor_dir = root / "data" / "processed" / "factors"

    registry = pd.read_parquet(factor_dir / "candidate_factor_registry.parquet")
    factors = registry.loc[registry["status"].eq("candidate"), "factor"].tolist()

    candidates = pd.read_parquet(
        factor_dir / "csi300_weekly_candidates.parquet",
        columns=KEYS + factors,
    )
    labels = pd.read_parquet(
        factor_dir / "csi300_weekly_forward_labels.parquet",
        columns=KEYS + [
            "forward_return", "evaluation_ready", "evaluation_split",
        ],
    )

    research = candidates.merge(
        labels,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not research["_merge"].eq("both").all():
        raise ValueError("候选因子表和收益标签表的日期与证券键不完全一致")

    research = research.loc[
        research["evaluation_split"].isin(["screening", "validation"])
    ]

    weekly_rows = []
    for (split, signal_date), week in research.groupby(
        ["evaluation_split", "signal_date"], sort=True
    ):
        for factor in factors:
            row = evaluate_week(week, factor)
            row["evaluation_split"] = split
            row["signal_date"] = signal_date
            row["factor"] = factor
            weekly_rows.append(row)

    weekly = pd.DataFrame(weekly_rows)
    weekly["year"] = weekly["signal_date"].dt.year

    by_split = summarize(weekly, ["evaluation_split", "factor"])
    by_year = summarize(weekly, ["evaluation_split", "year", "factor"])

    output = root / "reports" / "factor_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(output / "single_factor_weekly.parquet", index=False)
    by_split.to_parquet(output / "single_factor_by_split.parquet", index=False)
    by_year.to_parquet(output / "single_factor_by_year.parquet", index=False)

    # 只按筛选段的 |IC| 排展示顺序，不据此自动决定保留因子。
    screening = by_split.loc[by_split["evaluation_split"].eq("screening")]
    factor_order = (
        screening.assign(abs_ic=screening["mean_rank_ic"].abs())
        .sort_values(["abs_ic", "factor"], ascending=[False, True])["factor"]
        .tolist()
    )

    print(f"因子数：{len(factors)}，因子×周记录：{len(weekly):,}")
    for split, title in [("screening", "筛选段"), ("validation", "验证段")]:
        table = (
            by_split.loc[by_split["evaluation_split"].eq(split)]
            .set_index("factor")
            .reindex(factor_order)
        )
        table["spread_bps"] = table["mean_q5_minus_q1"] * 10_000
        columns = ["ic_weeks", "mean_coverage", "mean_rank_ic", "icir", "spread_bps"]
        print(f"\n{title}（因子按筛选段 |IC| 排序）")
        print(table[columns].head(10).to_string(formatters={
            "mean_coverage": "{:.2%}".format,
            "mean_rank_ic": "{:+.4f}".format,
            "icir": "{:+.3f}".format,
            "spread_bps": "{:+.1f}".format,
        }))

    print(f"\n输出目录：{output}")


if __name__ == "__main__":
    main()
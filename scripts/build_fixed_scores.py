"""生成周度固定因子评分与去掉 BIAS_20 的对照名单。"""

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]
FACTORS = ["RET_5", "VOL_60", "CORR_PV_10", "BIAS_20"]


def build_scores(candidates, top_n):
    if candidates.empty or candidates[KEYS].isna().any().any():
        raise ValueError("候选因子表为空或有缺失的日期、证券代码")
    if candidates.duplicated(KEYS).any():
        raise ValueError("候选因子表存在重复的日期和证券代码")
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n 必须是正整数")

    scores = candidates.copy()
    finite = np.isfinite(scores[FACTORS].to_numpy(dtype=float)).all(axis=1)
    scores["eligible"] = (
        scores["signal_bar_usable"].astype("boolean").fillna(False) & finite
    )
    eligible = scores.loc[scores["eligible"], KEYS + FACTORS].copy()

    # 四项因子都以原始值较低为优先，取负后，较优的股票百分位较高。
    percentiles = (-eligible[FACTORS]).groupby(eligible["signal_date"]).rank(
        method="average", pct=True
    ).add_prefix("pct_")
    eligible = eligible.join(percentiles)

    eligible["score_fixed"] = percentiles.mean(axis=1)
    eligible["score_without_bias"] = percentiles.drop(
        columns="pct_BIAS_20"
    ).mean(axis=1)

    score_columns = [*percentiles.columns, "score_fixed", "score_without_bias"]
    scores = scores.merge(
        eligible[KEYS + score_columns], on=KEYS, how="left", validate="one_to_one"
    )

    for variant in ("fixed", "without_bias"):
        score_column = f"score_{variant}"
        rank_column = f"rank_{variant}"
        selected_column = f"selected_{variant}"

        ordered = eligible[KEYS + [score_column]].sort_values(
            ["signal_date", score_column, "code"], ascending=[True, False, True]
        )
        ordered[rank_column] = ordered.groupby("signal_date").cumcount() + 1
        scores = scores.merge(
            ordered[KEYS + [rank_column]], on=KEYS, how="left", validate="one_to_one"
        )
        scores[rank_column] = scores[rank_column].astype("Int64")
        scores[selected_column] = scores[rank_column].le(top_n).fillna(False)

    return scores.sort_values(
        ["signal_date", "rank_fixed", "code"]
    ).reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    top_n = config["portfolio"]["top_n"]
    factor_dir = root / "data" / "processed" / "factors"

    candidates = pd.read_parquet(
        factor_dir / "csi300_weekly_candidates.parquet",
        columns=KEYS + ["signal_bar_usable", *FACTORS],
    )
    scores = build_scores(candidates, top_n)
    output = factor_dir / "csi300_weekly_fixed_scores.parquet"
    scores.to_parquet(output, index=False)

    weekly = scores.assign(
        overlap=scores["selected_fixed"] & scores["selected_without_bias"]
    ).groupby("signal_date").agg(
        eligible=("eligible", "sum"),
        fixed=("selected_fixed", "sum"),
        without_bias=("selected_without_bias", "sum"),
        overlap=("overlap", "sum"),
    )

    print(f"信号周数：{len(weekly)}，成分记录：{len(scores):,}")
    print(f"合格股票最少的一周：{weekly['eligible'].min()} 只")
    print(f"四因子版未选满 {top_n} 只的周数：{weekly['fixed'].lt(top_n).sum()}")
    print(f"三因子版未选满 {top_n} 只的周数：{weekly['without_bias'].lt(top_n).sum()}")
    print(f"两版每周平均共同入选：{weekly['overlap'].mean():.2f} 只")

    first = scores.loc[scores["signal_date"].eq(weekly.index[0])]
    print("\n首周四因子版前 5 名")
    print(
        first[KEYS + ["score_fixed", "rank_fixed"]]
        .head(5)
        .to_string(index=False)
    )
    print(f"\n输出文件：{output}")


if __name__ == "__main__":
    main()
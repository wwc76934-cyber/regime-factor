"""将周度选股结果转换为两套策略的等权目标持仓。"""

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


VARIANTS = ("fixed", "without_bias")


def build_targets(scores, schedule, top_n):
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n 必须是正整数")

    if scores.empty or scores[["signal_date", "code"]].isna().any().any():
        raise ValueError("评分表为空，或存在缺失的信号日、证券代码")
    if scores.duplicated(["signal_date", "code"]).any():
        raise ValueError("评分表存在重复的信号日和证券代码")

    schedule = schedule[["signal_date", "execution_date"]].copy()
    if schedule.empty or schedule.isna().any().any():
        raise ValueError("调仓表为空，或存在缺失日期")
    if not schedule["signal_date"].is_unique:
        raise ValueError("调仓表存在重复信号日")
    if not schedule["execution_date"].is_unique:
        raise ValueError("调仓表存在重复执行日")
    if not schedule["execution_date"].gt(schedule["signal_date"]).all():
        raise ValueError("执行日必须晚于信号日")

    if set(scores["signal_date"]) != set(schedule["signal_date"]):
        raise ValueError("评分表和调仓表的信号日不一致")

    strategy_targets = []
    for variant in VARIANTS:
        score_column = f"score_{variant}"
        rank_column = f"rank_{variant}"
        selected_column = f"selected_{variant}"

        chosen = scores.loc[
            scores[selected_column].fillna(False),
            ["signal_date", "code", score_column, rank_column],
        ].rename(columns={score_column: "score", rank_column: "rank"})

        counts = (
            chosen.groupby("signal_date").size()
            .reindex(schedule["signal_date"], fill_value=0)
        )
        wrong_count = counts.loc[counts.ne(top_n)]
        if not wrong_count.empty:
            raise ValueError(
                f"{variant} 部分信号日的入选数不等于 {top_n}："
                f"{wrong_count.head().to_dict()}"
            )

        if not np.isfinite(chosen["score"].to_numpy(dtype=float)).all():
            raise ValueError(f"{variant} 的入选股票存在无效评分")

        # 每周必须恰好由排名 1 到 top_n 的股票组成。
        for signal_date, week in chosen.groupby("signal_date"):
            if week["rank"].isna().any() or set(week["rank"]) != set(range(1, top_n + 1)):
                raise ValueError(f"{variant} {signal_date:%Y-%m-%d} 的入选排名异常")

        chosen = chosen.merge(
            schedule, on="signal_date", how="left", validate="many_to_one"
        )
        chosen["strategy"] = variant
        chosen["target_weight"] = 1.0 / top_n
        strategy_targets.append(chosen)

    targets = pd.concat(strategy_targets, ignore_index=True)
    weight_sums = targets.groupby(["strategy", "execution_date"])["target_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(), 1.0):
        raise ValueError("部分策略的目标权重之和不等于 1")

    columns = [
        "strategy", "signal_date", "execution_date",
        "code", "rank", "score", "target_weight",
    ]
    return targets[columns].sort_values(
        ["strategy", "execution_date", "rank"]
    ).reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    top_n = config["portfolio"]["top_n"]

    processed = root / "data" / "processed"
    scores = pd.read_parquet(
        processed / "factors" / "csi300_weekly_fixed_scores.parquet"
    )
    schedule = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet"
    )
    scores["signal_date"] = pd.to_datetime(scores["signal_date"], errors="raise")
    for column in ("signal_date", "execution_date"):
        schedule[column] = pd.to_datetime(schedule[column], errors="raise")

    targets = build_targets(scores, schedule, top_n)

    output = processed / "portfolio" / "csi300_weekly_target_positions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output, index=False)

    print(f"目标持仓记录：{len(targets):,}")
    summary = targets.groupby("strategy").agg(
        rebalance_weeks=("execution_date", "nunique"),
        target_rows=("code", "size"),
    )
    print(summary.to_string())

    first_signal = targets["signal_date"].min()
    first = targets.loc[targets["signal_date"].eq(first_signal)]
    print("\n首周两版各自的前 5 只")
    print(first.groupby("strategy").head(5).to_string(
        index=False, formatters={"target_weight": "{:.4%}".format}
    ))
    print(f"\n输出文件：{output}")


if __name__ == "__main__":
    main()
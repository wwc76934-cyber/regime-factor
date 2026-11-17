"""按周度执行日生成开盘到开盘的因子评价标签。"""

from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]


def build_holding_windows(schedule):
    schedule = schedule[["signal_date", "execution_date"]].copy()
    for column in schedule.columns:
        schedule[column] = pd.to_datetime(schedule[column], errors="raise")

    schedule = schedule.sort_values("signal_date").reset_index(drop=True)
    if schedule.isna().any().any():
        raise ValueError("调仓表包含空日期")
    if schedule["signal_date"].duplicated().any():
        raise ValueError("调仓表包含重复信号日")
    if not schedule["execution_date"].gt(schedule["signal_date"]).all():
        raise ValueError("执行日必须晚于信号日")
    if (
        not schedule["execution_date"].is_monotonic_increasing
        or not schedule["execution_date"].is_unique
    ):
        raise ValueError("执行日必须严格递增")

    schedule = schedule.rename(columns={"execution_date": "entry_date"})
    schedule["exit_date"] = schedule["entry_date"].shift(-1)
    return schedule


def join_open_price(labels, panel, side):
    date_column = f"{side}_date"
    price_column = f"{side}_open_adj"
    status_column = f"{side}_status"

    prices = panel.rename(columns={
        "date": date_column,
        "adj_open": price_column,
        "tradestatus": status_column,
        "adj_ohlc_ok": f"{side}_price_ok",
    })

    return labels.merge(
        prices,
        on=[date_column, "code"],
        how="left",
        validate="many_to_one",
        indicator=f"{side}_match",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    development_end = pd.Timestamp(config["research"]["development_end"])

    # 2020—2022 年用于初筛，2023—2024 年用于开发期验证。
    validation_start = pd.Timestamp("2023-01-01")

    processed = root / "data" / "processed"
    candidates = pd.read_parquet(
        processed / "factors" / "csi300_weekly_candidates.parquet",
        columns=KEYS + ["signal_bar_usable"],
    )
    candidates["signal_date"] = pd.to_datetime(
        candidates["signal_date"], errors="raise"
    )
    if candidates[KEYS].isna().any().any() or candidates.duplicated(KEYS).any():
        raise ValueError("候选因子表的信号日或证券代码缺失、重复")

    schedule = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet"
    )
    windows = build_holding_windows(schedule)
    labels = candidates.merge(
        windows, on="signal_date", how="left", validate="many_to_one"
    )
    if labels["entry_date"].isna().any():
        raise ValueError("部分因子记录找不到对应的执行日")

    panel = pd.read_parquet(
        processed / "csi300_daily_panel.parquet",
        columns=["date", "code", "adj_open", "tradestatus", "adj_ohlc_ok"],
    )
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    if panel[["date", "code"]].isna().any().any():
        raise ValueError("日线面板包含空日期或证券代码")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("日线面板包含重复的日期和证券组合")

    labels = join_open_price(labels, panel, "entry")
    labels = join_open_price(labels, panel, "exit")

    # 每条记录只报告第一个失败原因，便于汇总。
    reasons = [
        (labels["exit_date"].isna(), "no_next_execution"),
        (labels["entry_match"].ne("both"), "entry_bar_missing"),
        (labels["exit_match"].ne("both"), "exit_bar_missing"),
    ]

    for side in ("entry", "exit"):
        status = labels[f"{side}_status"].astype("string").str.strip()
        reasons.extend([
            (status.eq("0").fillna(False), f"{side}_suspended"),
            (~status.isin(["0", "1"]), f"{side}_status_unknown"),
        ])
        opening = labels[f"{side}_open_adj"]
        prices_ok = (
            labels[f"{side}_price_ok"].astype("boolean").fillna(False)
            & np.isfinite(opening)
            & opening.gt(0)
        )
        reasons.append((~prices_ok, f"{side}_adjusted_price_invalid"))

    labels["label_status"] = np.select(
        [condition.to_numpy(dtype=bool) for condition, _ in reasons],
        [reason for _, reason in reasons],
        default="ok",
    )
    valid = labels["label_status"].eq("ok")
    labels["forward_return"] = (
        labels["exit_open_adj"].where(valid)
        / labels["entry_open_adj"].where(valid)
        - 1
    )

    # 筛选段的标签必须在验证开始前结束。
    # 跨越分段边界的窗口保留在账本中，供审计。
    screening = (
        labels["signal_date"].lt(validation_start)
        & labels["exit_date"].lt(validation_start)
    )
    validation = (
        labels["signal_date"].ge(validation_start)
        & labels["exit_date"].le(development_end)
    )
    labels["evaluation_split"] = np.select(
        [screening, validation],
        ["screening", "validation"],
        default="outside_evaluation",
    )
    labels["evaluation_ready"] = (
        labels["signal_bar_usable"].astype("boolean").fillna(False)
        & valid
        & labels["evaluation_split"].ne("outside_evaluation")
    )

    labels = labels.drop(columns=["entry_match", "exit_match"])
    labels = labels.sort_values(KEYS).reset_index(drop=True)

    output = processed / "factors" / "csi300_weekly_forward_labels.parquet"
    labels.to_parquet(output, index=False)

    print(f"信号周数：{labels['signal_date'].nunique()}")
    print(f"总记录数：{len(labels):,}")
    print(f"有效收益标签：{int(valid.sum()):,}")
    print("\n标签状态统计")
    print(labels["label_status"].value_counts().to_string())

    summary = labels.groupby("evaluation_split").agg(
        weeks=("signal_date", "nunique"),
        member_rows=("code", "size"),
        evaluation_ready=("evaluation_ready", "sum"),
    )
    print("\n分段统计")
    print(summary.to_string())

    first_day = labels["signal_date"].min()
    sample_columns = KEYS + [
        "entry_date", "exit_date", "entry_open_adj", "exit_open_adj",
        "forward_return", "label_status",
    ]
    print("\n首个信号日的前 3 条有效标签")
    print(
        labels.loc[
            labels["signal_date"].eq(first_day) & labels["forward_return"].notna(),
            sample_columns,
        ].head(3).to_string(index=False)
    )
    print(f"\n输出文件：{output}")


if __name__ == "__main__":
    main()
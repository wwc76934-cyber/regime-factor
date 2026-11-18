"""Build a weekly equal-weight research benchmark for historical CSI 300 members."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_csi300_total_return import evaluation_dates


KEYS = ["date", "code"]


def build_benchmark(panel, members, schedule, days):
    if panel.duplicated(KEYS).any():
        raise ValueError("日线面板存在重复日期和证券代码")
    if members.duplicated(["signal_date", "code"]).any():
        raise ValueError("成分表存在重复日期和证券代码")
    if not schedule["signal_date"].is_unique or not schedule["execution_date"].is_unique:
        raise ValueError("调仓表存在重复日期")
    if not schedule["execution_date"].gt(schedule["signal_date"]).all():
        raise ValueError("执行日必须晚于信号日")
    if not schedule["execution_date"].isin(days).all():
        raise ValueError("调仓日期超出回测日历")

    by_signal = members.groupby("signal_date")["code"].agg(list).to_dict()
    if set(by_signal) != set(schedule["signal_date"]):
        raise ValueError("历史成分和调仓表的信号日期不一致")
    codes = pd.Index(sorted(members["code"].unique()))
    rows = schedule.itertuples(index=False)
    wanted = {
        row.execution_date: codes.get_indexer(by_signal[row.signal_date])
        for row in rows
    }
    if any(len(indices) != 300 or (indices < 0).any() for indices in wanted.values()):
        raise ValueError("部分调仓日没有 300 只有效成分代码")

    def matrix(column):
        return (
            panel.pivot(index="date", columns="code", values=column)
            .reindex(index=days, columns=codes)
            .to_numpy()
        )

    opens = matrix("adj_open").astype(float)
    closes = matrix("adj_close").astype(float)
    valid_bar = matrix("adj_ohlc_ok") == True
    active = matrix("tradestatus") == "1"

    quantities = np.zeros(len(codes))
    last_mark = np.full(len(codes), np.nan)
    cash = 1.0
    records = []

    for i, day in enumerate(days):
        open_ok = valid_bar[i] & np.isfinite(opens[i]) & (opens[i] > 0)
        close_ok = valid_bar[i] & np.isfinite(closes[i]) & (closes[i] > 0)
        can_trade = open_ok & active[i]
        last_mark[open_ok] = opens[i, open_ok]

        is_rebalance = day in wanted
        eligible_count = locked_count = 0
        if is_rebalance:
            held = quantities > 0
            locked = held & ~can_trade
            eligible = wanted[day][can_trade[wanted[day]]]

            locked_count = int(locked.sum())
            eligible_count = len(eligible)
            gross_before = cash + np.dot(quantities[held], last_mark[held])
            locked_value = np.dot(quantities[locked], last_mark[locked])
            investable = gross_before - locked_value

            # 旧仓中无法交易的部分原样结转。其余资产在可交易成分中等分。
            quantities[~locked] = 0
            cash = investable
            if eligible_count:
                quantities[eligible] = investable / eligible_count / opens[i, eligible]
                cash = 0.0

            held = quantities > 0
            gross_after = cash + np.dot(quantities[held], last_mark[held])
            if not np.isclose(gross_after, gross_before, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{day:%Y-%m-%d} 等权基准调仓前后资产不守恒")

        last_mark[close_ok] = closes[i, close_ok]
        held = quantities > 0
        nav = cash + np.dot(quantities[held], last_mark[held])
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"{day:%Y-%m-%d} 基准净值无效")
        records.append((day, nav, is_rebalance, eligible_count, locked_count,
                        int((held & ~close_ok).sum())))

    return pd.DataFrame(records, columns=[
        "date", "nav", "is_rebalance", "tradable_members_at_open",
        "locked_positions_at_open", "held_without_valid_close",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root
    processed = root / "data" / "processed"
    portfolio = root / "reports" / "portfolio" / "walk_forward_three_variants"
    output = args.output_dir or processed / "benchmark"
    output.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_parquet(portfolio / "daily_nav.parquet", columns=["date", "strategy"])
    days = pd.DatetimeIndex(pd.to_datetime(
        ledger.loc[ledger["strategy"].eq("equal_weight"), "date"]
    )).sort_values()
    if days.empty or days.hasnans or not days.is_unique:
        raise ValueError("回测交易日期为空、缺失或重复")

    schedule = pd.read_parquet(processed / "universe" / "csi300_weekly_schedule.parquet")
    members = pd.read_parquet(
        processed / "universe" / "csi300_weekly_members.parquet",
        columns=["signal_date", "code"],
    )
    panel = pd.read_parquet(processed / "csi300_daily_panel.parquet", columns=[
        "date", "code", "adj_open", "adj_close", "adj_ohlc_ok", "tradestatus",
    ])
    panel = panel.loc[panel["date"].isin(days)]

    benchmark = build_benchmark(panel, members, schedule, days)
    start = evaluation_dates(root)[1]
    pre_start = benchmark.loc[benchmark["date"].lt(start)]
    if pre_start.empty or start not in days:
        raise ValueError("无法确定评价首日或上一交易日")
    anchor = pre_start.iloc[-1]
    weekly = benchmark.loc[benchmark["is_rebalance"]]
    result = benchmark["nav"].iloc[-1] / anchor["nav"] - 1

    path = output / "csi300_weekly_equal_weight.parquet"
    benchmark.to_parquet(path, index=False)
    print(f"交易日：{len(benchmark):,} | 调仓日：{len(weekly)}")
    print(f"调仓日最少可交易成分：{weekly['tradable_members_at_open'].min()} 只")
    print(f"存在未能交易旧仓的调仓日：{weekly['locked_positions_at_open'].gt(0).sum()} 天")
    print(f"持仓缺少有效收盘价：{benchmark['held_without_valid_close'].sum()} 条证券日")
    print(f"评价起点：{start:%Y-%m-%d} | 归一化基准日：{anchor['date']:%Y-%m-%d}")
    print(f"评价区间累计收益：{result:+.2%}")
    print(f"输出：{path}")


if __name__ == "__main__":
    main()

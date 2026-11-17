from pathlib import Path
import tomllib

import baostock as bs
import pandas as pd

from fetch_universe import check_response, fetch_snapshot


def fetch_schedule(start, end):
    # 额外取一周日历，以判断区间末尾是否真的是当周最后一个交易日。
    calendar_end = (pd.Timestamp(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    response = bs.query_trade_dates(start_date=start, end_date=calendar_end)
    check_response(response, "查询交易日历")

    rows = []
    while response.next():
        rows.append(response.get_row_data())
    check_response(response, "读取交易日历")

    calendar = pd.DataFrame(rows, columns=response.fields)
    calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"])
    expected_dates = pd.date_range(start, calendar_end)
    actual_dates = pd.DatetimeIndex(calendar["calendar_date"]).sort_values()
    if not actual_dates.equals(expected_dates):
        raise ValueError("交易日历缺少日期或包含重复日期")

    trading_days = (
        calendar.loc[calendar["is_trading_day"] == "1", "calendar_date"]
        .sort_values()
        .reset_index(drop=True)
    )
    schedule = pd.DataFrame({
        "signal_date": trading_days,
        "execution_date": trading_days.shift(-1),
    })

    # 下一交易日落在另一周，说明当天是本周最后一个交易日。
    this_week = schedule["signal_date"].dt.to_period("W-SUN")
    next_week = schedule["execution_date"].dt.to_period("W-SUN")
    within_end = schedule["execution_date"].le(pd.Timestamp(end))
    schedule = schedule.loc[within_end & this_week.ne(next_week)].reset_index(drop=True)
    if schedule.empty:
        raise ValueError("配置区间没有可执行的周度调仓日")
    return schedule


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    start = config["research"]["development_start"]
    end = config["research"]["development_end"]

    raw_dir = root / "data" / "raw" / "universe" / "baostock" / "csi300"
    output_dir = root / "data" / "processed" / "universe"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_response(bs.login(), "登录 BaoStock")
    snapshots = []
    try:
        schedule = fetch_schedule(start, end)
        total = len(schedule)

        for number, signal_date in enumerate(schedule["signal_date"], start=1):
            query_date = signal_date.strftime("%Y-%m-%d")
            path = raw_dir / f"{query_date}.parquet"
            cached = path.exists()
            print(f"[{number}/{total}] {query_date} {'缓存' if cached else '查询'}", flush=True)

            snapshot = pd.read_parquet(path) if cached else fetch_snapshot(query_date)

            if len(snapshot) != 300 or not snapshot["code"].is_unique:
                raise ValueError(f"{query_date} 成分数或证券代码异常：{len(snapshot)} 行")
            if not snapshot["query_date"].eq(query_date).all():
                raise ValueError(f"{query_date} 快照的查询日期不匹配")

            # 这是异常检查。供应商日期本身仍需结合调样公告解释。
            update_dates = pd.to_datetime(snapshot["updateDate"], errors="raise")
            if update_dates.isna().any() or update_dates.gt(signal_date).any():
                raise ValueError(f"{query_date} 快照缺少更新日期或出现未来日期")

            if not cached:
                temporary_path = path.with_suffix(".tmp")
                snapshot.to_parquet(temporary_path, index=False)
                temporary_path.replace(path)

            snapshots.append(
                snapshot.assign(signal_date=signal_date, provider="baostock")
            )
    finally:
        bs.logout()

    members = pd.concat(snapshots, ignore_index=True)
    members = members.sort_values(["signal_date", "code"]).reset_index(drop=True)

    schedule.to_parquet(output_dir / "csi300_weekly_schedule.parquet", index=False)
    members.to_parquet(output_dir / "csi300_weekly_members.parquet", index=False)

    print(f"\n调仓周数：{len(schedule)}")
    print(f"成员记录：{len(members)}")
    print(f"历年成分股去重后：{members['code'].nunique()} 只")
    print("\n首三次调仓")
    print(schedule.head(3).to_string(index=False))
    print(f"\n保存目录：{output_dir}")


if __name__ == "__main__":
    main()
"""对照日线前收参考价和上一市场交易日的不复权收盘价。"""

from pathlib import Path
import tomllib

import pandas as pd


PRICE_TOLERANCE = 0.005


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    start = config["research"]["bars_start"]
    end = config["research"]["development_end"]
    processed = root / "data" / "processed"

    schedule = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet",
        columns=["execution_date"],
    )
    first_execution = pd.to_datetime(schedule["execution_date"]).min()

    calendar = pd.read_parquet(
        root / "data" / "raw" / "calendar" / f"baostock_{start}_{end}.parquet"
    )
    trading_days = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[
            calendar["is_trading_day"].astype("string").eq("1"),
            "calendar_date",
        ],
        errors="raise",
    )).sort_values()

    if trading_days.empty or trading_days.hasnans or not trading_days.is_unique:
        raise ValueError("交易日历为空，或包含缺失、重复日期")

    previous_day = pd.Series(trading_days[:-1], index=trading_days[1:])

    bars = pd.read_parquet(
        processed / "csi300_daily_panel.parquet",
        columns=["date", "code", "raw_close", "preclose", "raw_ohlc_ok"],
    )
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")
    if bars.duplicated(["date", "code"]).any():
        raise ValueError("日线面板存在重复日期和证券组合")
    if not bars["date"].isin(trading_days).all():
        raise ValueError("日线面板存在交易日历以外的日期")

    # 按整个市场的上一交易日连接。若个股那天缺记录，就保留缺失。
    current = bars.loc[bars["date"].ge(first_execution)].copy()
    current["previous_date"] = current["date"].map(previous_day)
    previous = bars[["date", "code", "raw_close", "raw_ohlc_ok"]].rename(
        columns={
            "date": "previous_date",
            "raw_close": "previous_raw_close",
            "raw_ohlc_ok": "previous_raw_ohlc_ok",
        }
    )
    compared = current.merge(
        previous, on=["previous_date", "code"], how="left", validate="one_to_one"
    )

    comparable = (
        compared["raw_ohlc_ok"].eq(True).fillna(False)
        & compared["previous_raw_ohlc_ok"].eq(True).fillna(False)
        & compared["preclose"].gt(0)
        & compared["previous_raw_close"].gt(0)
    )
    compared["reference_gap"] = compared["preclose"] - compared["previous_raw_close"]

    columns = [
        "date", "code", "previous_date", "previous_raw_close",
        "preclose", "reference_gap",
    ]
    flagged = compared.loc[
        comparable & compared["reference_gap"].abs().gt(PRICE_TOLERANCE), columns
    ].sort_values(["date", "code"])

    output = root / "reports" / "portfolio" / "price_reference_flags.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(output, index=False)

    print(f"首次执行日：{first_execution:%Y-%m-%d}")
    print(f"可比较记录：{int(comparable.sum()):,}")
    print(f"无法比较的记录：{int((~comparable).sum()):,}")
    print(f"待核验记录：{len(flagged):,}，涉及 {flagged['code'].nunique()} 只证券")
    print(flagged.head(5).to_string(index=False))
    print(f"\n输出文件：{output}")


if __name__ == "__main__":
    main()
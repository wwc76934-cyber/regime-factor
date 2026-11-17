from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


KEYS = ["date", "code"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def load_bars(folder, codes, prefix, expected_flag):
    frames = []

    for code in codes:
        path = folder / f"{code}.parquet"
        bars = pd.read_parquet(path)
        bars["date"] = pd.to_datetime(bars["date"], errors="raise")

        if not bars["code"].eq(code).all():
            raise ValueError(f"{path.name} 的证券代码不匹配")
        if not bars["adjustflag"].astype(str).str.strip().eq(expected_flag).all():
            raise ValueError(f"{path.name} 的复权标记不匹配")
        if bars["date"].isna().any() or bars.duplicated(KEYS).any():
            raise ValueError(f"{path.name} 的日期缺失或重复")

        numeric_columns = PRICE_COLUMNS.copy()
        if prefix == "raw":
            numeric_columns += ["preclose", "volume", "amount"]

        for column in numeric_columns:
            bars[column] = pd.to_numeric(bars[column], errors="raise")

        rename = {column: f"{prefix}_{column}" for column in PRICE_COLUMNS}
        frames.append(bars.drop(columns="adjustflag").rename(columns=rename))

    return pd.concat(frames, ignore_index=True)


def valid_ohlc(panel, prefix):
    columns = [f"{prefix}_{column}" for column in PRICE_COLUMNS]
    prices = panel[columns]
    finite = np.isfinite(prices.to_numpy(dtype=float)).all(axis=1)

    return (
        finite
        & prices.gt(0).all(axis=1)
        & prices[f"{prefix}_high"].eq(prices.max(axis=1))
        & prices[f"{prefix}_low"].eq(prices.min(axis=1))
    )


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    start = config["research"]["bars_start"]
    end = config["research"]["development_end"]

    processed = root / "data" / "processed"
    members = pd.read_parquet(
        processed / "universe" / "csi300_weekly_members.parquet",
        columns=["signal_date", "code"],
    )
    members["signal_date"] = pd.to_datetime(members["signal_date"])
    codes = sorted(members["code"].unique())

    cache = root / "data" / "raw" / "bars" / "baostock" / f"{start}_{end}"
    raw = load_bars(cache / "unadjusted", codes, "raw", "3")
    adjusted = load_bars(cache / "back_adjusted", codes, "adj", "1")

    # 外连接保留只出现在其中一份数据里的日期，供后续检查。
    panel = raw.merge(
        adjusted, on=KEYS, how="outer",
        validate="one_to_one", indicator="price_source",
    )
    panel["raw_ohlc_ok"] = valid_ohlc(panel, "raw")
    panel["adj_ohlc_ok"] = valid_ohlc(panel, "adj")
    panel["has_both"] = panel["price_source"].eq("both")
    panel["trading_status_normal"] = panel["tradestatus"].astype("string").eq("1").fillna(False)
    panel["signal_bar_usable"] = (
        panel["has_both"]
        & panel["raw_ohlc_ok"]
        & panel["adj_ohlc_ok"]
        & panel["trading_status_normal"]
    )
    panel = panel.sort_values(KEYS).reset_index(drop=True)

    # 以成分表为左表，确保没有行情的成分股也进入统计分母。
    checks = ["has_both", "raw_ohlc_ok", "adj_ohlc_ok", "signal_bar_usable"]
    signal_bars = panel[KEYS + checks].rename(columns={"date": "signal_date"})
    linked = members.merge(
        signal_bars, on=["signal_date", "code"], how="left", validate="one_to_one"
    )
    for column in checks:
        linked[column] = linked[column].eq(True)

    weekly = linked.groupby("signal_date", as_index=False).agg(
        members=("code", "size"),
        matched_both=("has_both", "sum"),
        usable_signal_bars=("signal_bar_usable", "sum"),
    )
    weekly["usable_ratio"] = weekly["usable_signal_bars"] / weekly["members"]

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(processed / "csi300_daily_panel.parquet", index=False)
    weekly.to_parquet(reports / "csi300_weekly_bar_coverage.parquet", index=False)

    issues = panel.loc[
        ~panel["has_both"] | ~panel["raw_ohlc_ok"] | ~panel["adj_ohlc_ok"],
        KEYS + ["price_source", "raw_ohlc_ok", "adj_ohlc_ok"],
    ]
    issues.to_parquet(reports / "csi300_daily_bar_issues.parquet", index=False)

    print(f"日线面板：{len(panel):,} 行，{len(codes)} 只证券")
    print("两种价格口径的日期匹配情况")
    print(panel["price_source"].value_counts().to_string())
    print(f"\n异常记录：{len(issues):,} 行")
    print(issues.head(10).to_string(index=False))

    print("\n覆盖率最低的 5 个信号日")
    print(
        weekly.nsmallest(5, "usable_ratio").to_string(
            index=False, formatters={"usable_ratio": "{:.2%}".format}
        )
    )


if __name__ == "__main__":
    main()
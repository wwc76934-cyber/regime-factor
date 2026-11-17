"""以信号日可见的信息构建周度沪深 300 市场状态特征。"""

from pathlib import Path
import tomllib

import baostock as bs
import numpy as np
import pandas as pd

from fetch_universe import check_response


INDEX_CODE = "sh.000300"


def fetch_index(start, end):
    check_response(bs.login(), "登录 BaoStock")
    try:
        response = bs.query_history_k_data_plus(
            INDEX_CODE,
            "date,code,close",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        check_response(response, "查询沪深 300 指数")

        rows = []
        while response.next():
            rows.append(response.get_row_data())
        check_response(response, "读取沪深 300 指数")

        return pd.DataFrame(rows, columns=response.fields)
    finally:
        bs.logout()


def load_index(path, start, end, trading_days):
    cached = path.is_file()
    history = pd.read_parquet(path) if cached else fetch_index(start, end)

    if history.empty or not history["code"].eq(INDEX_CODE).all():
        raise ValueError("指数数据为空，或返回的指数代码不一致")

    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history["close"] = pd.to_numeric(history["close"], errors="raise")
    if history["date"].isna().any() or not history["date"].is_unique:
        raise ValueError("指数行情的日期缺失或重复")
    if not (np.isfinite(history["close"]) & history["close"].gt(0)).all():
        raise ValueError("指数行情存在无效收盘价")

    history = history.sort_values("date").reset_index(drop=True)
    observed = pd.DatetimeIndex(history["date"])
    missing = trading_days.difference(observed)
    extra = observed.difference(trading_days)
    if len(missing) or len(extra):
        raise ValueError(
            f"指数行情与交易日历不一致。"
            f"缺失示例 {missing[:5].tolist()}，额外日期 {extra[:5].tolist()}"
        )

    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        history.to_parquet(temporary, index=False)
        temporary.replace(path)

    return history.set_index("date")["close"].reindex(trading_days)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    research = config["research"]
    start, end = research["bars_start"], research["development_end"]
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"

    calendar = pd.read_parquet(raw / "calendar" / f"baostock_{start}_{end}.parquet")
    trading_days = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[
            calendar["is_trading_day"].astype("string").eq("1"),
            "calendar_date",
        ],
        errors="raise",
    )).sort_values()
    if trading_days.empty or trading_days.hasnans or not trading_days.is_unique:
        raise ValueError("交易日历为空，或包含缺失、重复日期")

    index_path = raw / "index" / f"baostock_sh.000300_{start}_{end}.parquet"
    close = load_index(index_path, start, end, trading_days)

    daily_return = close / close.shift(1) - 1
    index_features = pd.DataFrame({
        "index_close": close,
        "market_return_20d": close / close.shift(20) - 1,
        "market_volatility_20d": daily_return.rolling(20, min_periods=20).std(),
    }).rename_axis("signal_date").reset_index()

    members = pd.read_parquet(
        processed / "factors" / "csi300_weekly_candidates.parquet",
        columns=["signal_date", "code", "RET_20"],
    )
    members["signal_date"] = pd.to_datetime(members["signal_date"], errors="raise")
    if members[["signal_date", "code"]].isna().any().any():
        raise ValueError("候选表存在缺失的信号日或证券代码")
    if members.duplicated(["signal_date", "code"]).any():
        raise ValueError("候选表存在重复信号日和证券代码")

    # 使用信号日的历史成分，分母是有有效 20 日收益的股票数。
    # 广度只需要价格历史，不要求该股同时具备四项可评分因子。
    members["valid_return"] = np.isfinite(members["RET_20"])
    members["positive_return"] = members["valid_return"] & members["RET_20"].gt(0)
    breadth = members.groupby("signal_date", as_index=False).agg(
        member_count=("code", "size"),
        breadth_observations=("valid_return", "sum"),
        advancing_count=("positive_return", "sum"),
    )
    breadth["market_breadth_20d"] = (
        breadth["advancing_count"]
        / breadth["breadth_observations"].where(breadth["breadth_observations"].gt(0))
    )

    states = breadth.merge(
        index_features, on="signal_date", how="left", validate="one_to_one"
    ).sort_values("signal_date")

    feature_columns = [
        "market_return_20d",
        "market_volatility_20d",
        "market_breadth_20d",
    ]
    if not np.isfinite(states[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("信号日缺少市场状态值，请检查指数预热期和成分股覆盖")

    output = processed / "market" / "csi300_weekly_market_states.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(output, index=False)

    print(f"指数日线：{len(close):,} 条")
    print(f"市场状态：{len(states)} 个信号周")
    print(f"广度最少有效股票数：{states['breadth_observations'].min()}")
    print("\n首五个信号日")
    columns = ["signal_date", *feature_columns, "breadth_observations"]
    print(states[columns].head().to_string(
        index=False,
        formatters={column: "{:.2%}".format for column in feature_columns},
    ))
    print(f"\n输出文件：{output}")


if __name__ == "__main__":
    main()
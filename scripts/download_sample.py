from pathlib import Path
import argparse
import akshare as ak
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="sh600000")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    output = (
        project_dir / "data" / "raw"
        / f"{args.symbol}_202401_tx_unadjusted.parquet"
    )

    bars = ak.stock_zh_a_hist_tx(
        symbol=args.symbol,
        start_date="20240102",
        end_date="20240131",
        adjust="",
    )
    
    if bars.empty:
        raise ValueError("接口返回了空数据")

    price_columns = ["open", "high", "low", "close"]
    bars = bars[["date", *price_columns, "volume"]].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="raise")

    for column in [*price_columns, "volume"]:
        bars[column] = pd.to_numeric(bars[column], errors="raise")

    bars = bars.sort_values("date").reset_index(drop=True)

    if bars["date"].isna().any() or bars["date"].duplicated().any():
        raise ValueError("交易日期存在缺失或重复")

    prices = bars[price_columns]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("OHLC 价格存在缺失或非法值")

    if (
        (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any()
        or (bars["low"] > bars[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("最高价或最低价与其他价格不一致")

    volume = bars["volume"]
    if not np.isfinite(volume.to_numpy()).all() or (volume < 0).any():
        raise ValueError("成交量存在缺失或非法值")

    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(output, index=False)

    saved = pd.read_parquet(output)
    print(f"已保存并重新读取 {len(saved)} 条日线")
    print(f"日期范围：{saved['date'].min():%Y-%m-%d} 至 {saved['date'].max():%Y-%m-%d}")
    print(saved.head(3).to_string(index=False))
    print(f"文件位置：{output}")


if __name__ == "__main__":
    main()
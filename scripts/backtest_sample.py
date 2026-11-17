from pathlib import Path

import pandas as pd

def simulate(bars, open_target, initial_cash=100_000.0, fee_rate=0.001, lot_size=100):
    cash = initial_cash
    shares = 0
    records = []

    for row, target in zip(bars.itertuples(index=False), open_target, strict=True):
        trade_shares = 0

        if target == 1 and shares == 0:
            # 买入时，现金必须同时覆盖成交金额和手续费。
            cost_per_lot = row.open * lot_size * (1 + fee_rate)
            trade_shares = int(cash / cost_per_lot) * lot_size
        elif target == 0 and shares > 0:
            trade_shares = -shares

        trade_value = trade_shares * row.open
        fee = abs(trade_value) * fee_rate
        cash -= trade_value + fee
        shares += trade_shares

        records.append({
            "trade_shares": trade_shares,
            "fill_price": row.open if trade_shares else None,
            "fee": fee,
            "shares": shares,
            "cash": cash,
            "equity": cash + shares * row.close,
        })

    return pd.DataFrame(records, index=bars.index)

def main():
    project_dir = Path(__file__).resolve().parents[1]
    source = project_dir / "data" / "raw" / "sh600000_202401_tx_unadjusted.parquet"
    report = project_dir / "reports" / "sample_backtest.parquet"

    bars = pd.read_parquet(source).sort_values("date").reset_index(drop=True)

    # 每一行的均线使用截至当天收盘的价格。
    bars["ma5"] = bars["close"].rolling(5).mean()
    bars["ma10"] = bars["close"].rolling(10).mean()
    bars["close_signal"] = (bars["ma5"] > bars["ma10"]).astype(int)

    # 把收盘信号向下移一行，成为下一交易日的开盘目标。
    bars["open_target"] = bars["close_signal"].shift(1, fill_value=0)

    initial_cash = 100_000.0

    # 第 10 个收盘价出现后，下一交易日开盘才是共同起点。
    ready_at_close = bars["ma10"].notna()
    benchmark_target = ready_at_close.shift(1, fill_value=False).astype(int)

    strategy = simulate(bars, bars["open_target"], initial_cash)
    benchmark = simulate(bars, benchmark_target, initial_cash)

    result = bars.join(strategy).join(benchmark.add_prefix("benchmark_"))
    report = report.with_name("sample_backtest_with_fees.parquet")
    report.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(report, index=False)

    trade_columns = ["trade_shares", "fill_price", "fee", "cash", "equity"]
    for name, ledger in [("均线策略", strategy), ("买入持有", benchmark)]:
        trades = bars[["date"]].join(ledger)
        trades = trades.loc[trades["trade_shares"] != 0]
        final_equity = ledger["equity"].iloc[-1]

        print(f"\n{name}首笔成交")
        print(trades[["date", *trade_columns]].head(1).to_string(index=False))
        print(f"期末资产：{final_equity:,.2f} 元")
        print(f"样本收益率：{final_equity / initial_cash - 1:.2%}")

    print(f"\n逐日账本：{report}")

if __name__ == "__main__":
    main()
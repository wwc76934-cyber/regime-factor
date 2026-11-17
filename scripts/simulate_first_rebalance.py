"""模拟两套组合的首次建仓，并保存成交明细和收盘账本。"""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


LOT_SIZE = 100
BUY_FEE_RATE = Decimal("0.001")
CENT = Decimal("0.01")
ZERO = Decimal("0.00")


def round_money(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def valid_price(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def simulate_opening(orders, initial_cash, top_n):
    cash = initial_cash
    records = []

    for row in orders.sort_values("rank").itertuples(index=False):
        # 等权预算包含手续费。每只股票使用同一个初始资金基准。
        budget = initial_cash / top_n
        shares = 0
        gross = ZERO
        fee = ZERO
        fill_price = None

        status = str(row.tradestatus).strip()

        if row.bar_match != "both":
            reason = "missing_bar"
        elif status == "0":
            reason = "suspended"
        elif status != "1":
            reason = "unknown_trade_status"
        elif not valid_price(row.raw_open):
            reason = "invalid_open_price"
        else:
            open_price = Decimal(str(row.raw_open))
            available = min(budget, cash)
            lot_cost = open_price * LOT_SIZE * (1 + BUY_FEE_RATE)
            shares = int(available // lot_cost) * LOT_SIZE

            if shares == 0:
                reason = "below_one_lot"
            else:
                fill_price = open_price
                gross = round_money(fill_price * shares)
                fee = round_money(gross * BUY_FEE_RATE)
                cash -= gross + fee
                reason = "filled"

        # 开盘成交确定后，再用收盘价估值。
        market_value = ZERO
        if shares > 0:
            if not valid_price(row.raw_close):
                raise ValueError(f"{row.code} 已成交，但缺少有效收盘价")
            market_value = round_money(Decimal(str(row.raw_close)) * shares)

        records.append({
            "strategy": row.strategy,
            "signal_date": row.signal_date,
            "execution_date": row.execution_date,
            "code": row.code,
            "rank": row.rank,
            "target_weight": row.target_weight,
            "target_budget": float(round_money(budget)),
            "simulated_shares": shares,
            "fill_price": float(fill_price) if fill_price is not None else None,
            "gross": float(gross),
            "fee": float(fee),
            "cash_after": float(cash),
            "closing_market_value": float(market_value),
            "status": reason,
        })

    return pd.DataFrame(records)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    initial_cash = Decimal(str(config["portfolio"]["initial_cash"]))
    top_n = config["portfolio"]["top_n"]

    processed = root / "data" / "processed"
    targets = pd.read_parquet(
        processed / "portfolio" / "csi300_weekly_target_positions.parquet"
    )
    first_date = targets["execution_date"].min()
    first = targets.loc[targets["execution_date"].eq(first_date)].copy()

    if first.empty or not first.groupby("strategy").size().eq(top_n).all():
        raise ValueError("首次建仓的目标持仓数量异常")

    panel = pd.read_parquet(
        processed / "csi300_daily_panel.parquet",
        columns=["date", "code", "raw_open", "raw_close", "tradestatus"],
    )
    bars = panel.loc[panel["date"].eq(first_date)].rename(
        columns={"date": "execution_date"}
    )
    orders = first.merge(
        bars,
        on=["execution_date", "code"],
        how="left",
        validate="many_to_one",
        indicator="bar_match",
    )

    ledger = pd.concat(
        [
            simulate_opening(group, initial_cash, top_n)
            for _, group in orders.groupby("strategy", sort=True)
        ],
        ignore_index=True,
    )

    summary = ledger.groupby("strategy").agg(
        target_count=("code", "size"),
        filled_count=("status", lambda statuses: statuses.eq("filled").sum()),
        gross=("gross", "sum"),
        fee=("fee", "sum"),
        closing_cash=("cash_after", "last"),
        closing_market_value=("closing_market_value", "sum"),
    )
    summary["closing_equity"] = summary["closing_cash"] + summary["closing_market_value"]

    if not np.allclose(
        summary["closing_cash"] + summary["gross"] + summary["fee"],
        float(initial_cash),
        rtol=0,
        atol=0.005,
    ):
        raise ValueError("现金对账失败")

    report_dir = root / "reports" / "portfolio"
    report_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(report_dir / "first_rebalance_ledger.parquet", index=False)
    summary.reset_index().to_parquet(
        report_dir / "first_rebalance_summary.parquet", index=False
    )

    print(f"首次执行日：{first_date:%Y-%m-%d}")
    print(summary.round(2).to_string())

    columns = [
        "strategy", "code", "target_budget", "simulated_shares",
        "fill_price", "gross", "fee", "status",
    ]
    print("\n两版各自第一笔")
    print(ledger.loc[ledger["rank"].eq(1), columns].to_string(index=False))

    print("\n未成交目标")
    print(ledger.loc[ledger["status"].ne("filled"), columns].to_string(index=False))
    print(f"\n输出目录：{report_dir}")


if __name__ == "__main__":
    main()
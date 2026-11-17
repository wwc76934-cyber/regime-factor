"""继承首次建仓的现金和股数，验证第二次周度调仓。"""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_first_rebalance import LOT_SIZE, ZERO, round_money


FEE_RATE = Decimal("0.001")


def simulate_rebalance(targets, opening_positions, opening_cash, bars):
    """模拟样本日的开盘调仓，返回逐股流水和收盘资产。"""
    strategy = targets["strategy"].iloc[0]
    execution_date = targets["execution_date"].iloc[0]
    rank = targets.set_index("code")["rank"].to_dict()
    codes = sorted(set(opening_positions) | set(rank))

    if not np.allclose(targets["target_weight"], 1 / len(targets)):
        raise ValueError(f"{strategy} 的目标持仓不是等权")

    quotes = bars.reindex(codes)
    prices = quotes[["raw_open", "raw_close"]].to_numpy(dtype=float)
    if not (
        np.isfinite(prices).all()
        and (prices > 0).all()
        and quotes["tradestatus"].astype("string").str.strip().eq("1").fillna(False).all()
    ):
        raise ValueError(f"{strategy} 样本日存在缺失行情或非正常交易状态")

    opening_prices = {
        code: Decimal(str(quotes.at[code, "raw_open"])) for code in codes
    }
    closing_prices = {
        code: Decimal(str(quotes.at[code, "raw_close"])) for code in codes
    }

    opening_equity = opening_cash + sum(
        round_money(opening_prices[code] * shares)
        for code, shares in opening_positions.items()
    )
    target_value = opening_equity / len(targets)

    target_shares = {
        code: int(target_value // (opening_prices[code] * LOT_SIZE)) * LOT_SIZE
        if code in rank else 0
        for code in codes
    }
    changes = {
        code: target_shares[code] - opening_positions.get(code, 0)
        for code in codes
    }

    sells = [code for code in codes if changes[code] < 0]
    buys = sorted(
        [code for code in codes if changes[code] > 0],
        key=rank.__getitem__,
    )
    unchanged = [code for code in codes if changes[code] == 0]

    cash = opening_cash
    records = []

    for code in sells + buys + unchanged:
        price = opening_prices[code]
        requested = changes[code]
        filled = requested

        if requested > 0:
            lot_cost = price * LOT_SIZE * (1 + FEE_RATE)
            affordable = int(cash // lot_cost) * LOT_SIZE
            filled = min(requested, affordable)

        # 买入时成交金额为正，卖出时为负，费用始终是支出。
        gross = round_money(price * filled)
        fee = round_money(abs(gross) * FEE_RATE)
        cash -= gross + fee
        shares_before = opening_positions.get(code, 0)
        shares_after = shares_before + filled

        if cash < ZERO or shares_after < 0:
            raise ValueError(f"{strategy} {code} 出现负现金或负持仓")

        reason = "filled"
        if requested != filled:
            reason = "cash_limited"
        elif requested == 0:
            reason = "below_one_lot" if code in rank and shares_after == 0 else "unchanged"

        closing_value = round_money(closing_prices[code] * shares_after)
        records.append({
            "strategy": strategy,
            "execution_date": execution_date,
            "code": code,
            "shares_before": shares_before,
            "target_shares": target_shares[code],
            "requested_change": requested,
            "filled_change": filled,
            "shares_after": shares_after,
            "open_price": float(price),
            "signed_gross": float(gross),
            "fee": float(fee),
            "cash_after_order": float(cash),
            "closing_market_value": float(closing_value),
            "status": reason,
        })

    ledger = pd.DataFrame(records)
    if not np.isclose(
        float(opening_cash) - ledger["signed_gross"].sum() - ledger["fee"].sum(),
        float(cash),
        rtol=0,
        atol=0.005,
    ):
        raise ValueError(f"{strategy} 现金对账失败")
    if not ledger["shares_before"].add(ledger["filled_change"]).eq(ledger["shares_after"]).all():
        raise ValueError(f"{strategy} 股数对账失败")

    market_value = ledger["closing_market_value"].sum()
    summary = {
        "strategy": strategy,
        "execution_date": execution_date,
        "opening_equity": float(opening_equity),
        "sell_orders_filled": int(ledger["filled_change"].lt(0).sum()),
        "buy_orders_filled": int(ledger["filled_change"].gt(0).sum()),
        "fee": ledger["fee"].sum(),
        "closing_cash": float(cash),
        "closing_market_value": market_value,
        "closing_equity": float(cash) + market_value,
    }
    return ledger, summary


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    reports = root / "reports" / "portfolio"

    targets = pd.read_parquet(
        processed / "portfolio" / "csi300_weekly_target_positions.parquet"
    )
    first_ledger = pd.read_parquet(reports / "first_rebalance_ledger.parquet")
    first_summary = pd.read_parquet(
        reports / "first_rebalance_summary.parquet"
    ).set_index("strategy")

    first_date = pd.Timestamp("2020-01-06")
    second_date = pd.Timestamp("2020-01-13")
    if not first_ledger["execution_date"].eq(first_date).all():
        raise ValueError("首次建仓账本的执行日期与本次样本不匹配")

    next_targets = targets.loc[targets["execution_date"].eq(second_date)]
    if next_targets.empty or set(next_targets["strategy"]) != set(first_summary.index):
        raise ValueError("第二次调仓目标与首次建仓账本的策略不匹配")

    panel = pd.read_parquet(
        processed / "csi300_daily_panel.parquet",
        columns=["date", "code", "raw_open", "raw_close", "tradestatus"],
    )
    bars = panel.loc[panel["date"].eq(second_date)].set_index(
        "code", verify_integrity=True
    )

    ledgers = []
    summaries = []
    for strategy, target in next_targets.groupby("strategy", sort=True):
        held = first_ledger.loc[
            first_ledger["strategy"].eq(strategy)
            & first_ledger["simulated_shares"].gt(0)
        ]
        positions = held.set_index("code", verify_integrity=True)[
            "simulated_shares"
        ].astype(int).to_dict()
        cash = Decimal(str(first_summary.at[strategy, "closing_cash"]))

        ledger, summary = simulate_rebalance(target, positions, cash, bars)
        ledgers.append(ledger)
        summaries.append(summary)

    combined = pd.concat(ledgers, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    combined.to_parquet(reports / "second_rebalance_ledger.parquet", index=False)
    summary_table.to_parquet(reports / "second_rebalance_summary.parquet", index=False)

    print(summary_table.round(2).to_string(index=False))
    print("\n两版 sh.600011 的调仓记录")
    columns = [
        "strategy", "code", "shares_before", "target_shares",
        "filled_change", "shares_after", "open_price", "fee",
    ]
    print(combined.loc[combined["code"].eq("sh.600011"), columns].to_string(index=False))
    print(f"\n输出目录：{reports}")


if __name__ == "__main__":
    main()
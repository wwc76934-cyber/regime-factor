"""用一笔真实分红验证登记、应收和到账三步记账。"""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import baostock as bs
import pandas as pd


CODE = "sh.600011"
OPERATE_DATE = "2020-07-06"
RECORD_DATE = "2020-07-03"
PAY_DATE = "2020-07-06"
CASH_PER_SHARE = Decimal("0.135")
RECORD_SHARES = 5_900
ZERO = Decimal("0.00")


def money(value):
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def check_response(result, action):
    if result.error_code != "0":
        raise RuntimeError(f"{action}失败：{result.error_code} {result.error_msg}")


def fetch_dividend():
    print("正在查询 BaoStock 分红记录…", flush=True)
    check_response(bs.login(), "登录 BaoStock")
    try:
        result = bs.query_dividend_data(
            code=CODE, year="2020", yearType="operate"
        )
        check_response(result, "查询分红")

        rows = []
        while result.next():
            rows.append(result.get_row_data())
        check_response(result, "读取分红")
        return pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()


def main():
    root = Path(__file__).resolve().parents[1]
    source = fetch_dividend()
    matched = source.loc[
        source["code"].eq(CODE)
        & source["dividOperateDate"].eq(OPERATE_DATE)
    ]
    if len(matched) != 1:
        raise ValueError(f"预期找到一条样本分红，实际找到 {len(matched)} 条")

    event = matched.iloc[0]
    cash_per_share = Decimal(event["dividCashPsBeforeTax"])
    if (
        event["dividRegistDate"] != RECORD_DATE
        or event["dividPayDate"] != PAY_DATE
        or cash_per_share != CASH_PER_SHARE
    ):
        raise ValueError(f"接口结果与已核对的公告口径不一致：{event.to_dict()}")

    panel = pd.read_parquet(
        root / "data" / "processed" / "csi300_daily_panel.parquet",
        columns=["date", "code", "raw_close", "preclose"],
    )
    dates = pd.to_datetime([RECORD_DATE, OPERATE_DATE])
    bars = panel.loc[panel["code"].eq(CODE) & panel["date"].isin(dates)]
    bars = bars.set_index("date").reindex(dates)
    if bars[["raw_close", "preclose"]].isna().any().any():
        raise ValueError("缺少登记日或除息日的行情")

    prior_close = Decimal(str(bars.iloc[0]["raw_close"]))
    ex_close = Decimal(str(bars.iloc[1]["raw_close"]))
    preclose = Decimal(str(bars.iloc[1]["preclose"]))

    # 测试账户只跟踪这笔持仓和它产生的股息。
    before_market_value = money(prior_close * RECORD_SHARES)
    after_market_value = money(ex_close * RECORD_SHARES)
    dividend = money(cash_per_share * RECORD_SHARES)

    records = []

    def record(day, stage, market_value, receivable, cash):
        records.append({
            "date": day,
            "code": CODE,
            "stage": stage,
            "shares": RECORD_SHARES,
            "market_value": float(market_value),
            "dividend_receivable": float(receivable),
            "cash": float(cash),
            "equity": float(market_value + receivable + cash),
        })

    record(RECORD_DATE, "record_date_close", before_market_value, ZERO, ZERO)

    # 以下两行使用同一个收盘价，单独核对派息只改变资产的构成。
    record(OPERATE_DATE, "ex_date_receivable", after_market_value, dividend, ZERO)
    record(PAY_DATE, "pay_date_settled", after_market_value, ZERO, dividend)

    ledger = pd.DataFrame(records)
    if ledger.at[1, "equity"] != ledger.at[2, "equity"]:
        raise ValueError("应收转现金后资产发生变化")

    output = root / "reports" / "portfolio"
    output.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output / "cash_dividend_sample_source.parquet", index=False)
    ledger.to_parquet(output / "cash_dividend_sample_ledger.parquet", index=False)

    print(f"登记日收盘价：{prior_close:.2f} 元")
    print(f"除息日前收参考价：{preclose:.2f} 元")
    print(f"公告每股现金红利（税前）：{cash_per_share} 元")
    print(ledger.to_string(index=False, float_format="{:.2f}".format))
    print(f"\n输出目录：{output}")


if __name__ == "__main__":
    main()
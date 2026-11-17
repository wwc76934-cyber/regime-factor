"""标准化公司行动，防止同一证券同一天重复记账。"""

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


AMOUNT_COLUMNS = {
    "cash_per_share": "dividCashPsBeforeTax",
    "bonus_shares_per_share": "dividStocksPs",
    "reserve_shares_per_share": "dividReserveToStockPs",
}

EVENT_COLUMNS = [
    "code", "ex_date", "record_date", "pay_date", "stock_listing_date",
    "cash_per_share", "bonus_shares_per_share", "reserve_shares_per_share",
    "event_type", "source_rows",
]


def text_values(column):
    return [value for value in column.dropna().astype(str).str.strip() if value]


def unique_date(group, column, required):
    values = sorted(set(text_values(group[column])))
    if len(values) > 1:
        raise ValueError(f"{column} 日期冲突：{values}")
    if not values:
        if required:
            raise ValueError(f"缺少 {column}")
        return pd.NaT
    return pd.to_datetime(values[0], format="%Y-%m-%d", errors="raise")


def unique_amount(group, column):
    values = set()
    for text in text_values(group[column]):
        try:
            number = Decimal(text)
        except InvalidOperation:
            raise ValueError(f"{column} 存在无法解析的值：{text}") from None
        if not number.is_finite() or number < 0:
            raise ValueError(f"{column} 存在无效值：{text}")
        values.add(number)

    if len(values) > 1:
        raise ValueError(f"{column} 金额或比例冲突：{sorted(values)}")

    return next(iter(values), Decimal("0"))


def normalize_event(code, ex_date, group):
    amounts = {
        name: unique_amount(group, column)
        for name, column in AMOUNT_COLUMNS.items()
    }
    has_cash = amounts["cash_per_share"] > 0
    has_stock = (
        amounts["bonus_shares_per_share"] + amounts["reserve_shares_per_share"] > 0
    )
    if not has_cash and not has_stock:
        raise ValueError("没有可识别的现金、送股或转增数值")

    record_date = unique_date(group, "dividRegistDate", required=True)
    pay_date = unique_date(group, "dividPayDate", required=True) if has_cash else pd.NaT
    listing_date = (
        unique_date(group, "dividStockMarketDate", required=True)
        if has_stock else pd.NaT
    )

    if record_date >= ex_date:
        raise ValueError("股权登记日未早于除权除息日")
    if has_cash and pay_date < ex_date:
        raise ValueError("派息日早于除权除息日")
    if has_stock and listing_date < ex_date:
        raise ValueError("新增股份上市日早于除权除息日")

    if has_cash and has_stock:
        event_type = "cash_and_stock"
    elif has_cash:
        event_type = "cash_only"
    else:
        event_type = "stock_only"

    # 金额和比例以十进制文本保存，账本可直接用 Decimal 读取。
    return {
        "code": code,
        "ex_date": ex_date,
        "record_date": record_date,
        "pay_date": pay_date,
        "stock_listing_date": listing_date,
        **{name: format(value, "f") for name, value in amounts.items()},
        "event_type": event_type,
        "source_rows": len(group),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    action_dir = root / "data" / "processed" / "corporate_actions"
    report_dir = root / "reports" / "portfolio"

    source = pd.read_parquet(action_dir / "csi300_baostock_actions.parquet")
    reconciliation = pd.read_parquet(
        report_dir / "corporate_action_reconciliation.parquet"
    )

    source["ex_date"] = pd.to_datetime(
        source["dividOperateDate"], format="%Y-%m-%d", errors="raise"
    )
    if source[["code", "ex_date"]].isna().any().any():
        raise ValueError("公司行动源表有缺失的证券代码或除权除息日")

    events = []
    review = []
    grouped = source.groupby(["code", "ex_date"], sort=True)

    for (code, ex_date), group in grouped:
        try:
            events.append(normalize_event(code, ex_date, group))
        except ValueError as error:
            review.append({
                "code": code,
                "ex_date": ex_date,
                "category": "source_event_review",
                "reason": str(error),
                "source_rows": len(group),
            })

    source_review_count = len(review)

    # 只有行情线索的日期单独登记，等待核验配股或其他事件。
    price_only = reconciliation.loc[
        reconciliation["match"].eq("price_flag_only"),
        ["code", "date"],
    ]
    for row in price_only.itertuples(index=False):
        review.append({
            "code": row.code,
            "ex_date": row.date,
            "category": "price_flag_only",
            "reason": "行情前收参考价存在差异，缺少匹配的分红送转事件",
            "source_rows": 0,
        })

    normalized = pd.DataFrame(events, columns=EVENT_COLUMNS)
    pending = pd.DataFrame(
        review, columns=["code", "ex_date", "category", "reason", "source_rows"]
    )
    if len(normalized) + source_review_count != grouped.ngroups:
        raise ValueError("标准化事件和源表待核验事件数量未对齐")
    if normalized.duplicated(["code", "ex_date"]).any():
        raise ValueError("标准化后仍存在重复事件")

    event_path = action_dir / "csi300_normalized_actions.parquet"
    review_path = report_dir / "corporate_action_review_queue.parquet"
    normalized.to_parquet(event_path, index=False)
    pending.to_parquet(review_path, index=False)

    print(f"公司行动源记录：{len(source):,}")
    print(f"源表事件键：{grouped.ngroups:,}")
    print(f"可入账事件：{len(normalized):,}")
    print(normalized["event_type"].value_counts().to_string())

    print(f"\n待核验事件：{len(pending):,}")
    print(pending["category"].value_counts().to_string())

    print("\n源表中需要核验的前 10 条：")
    print(pending.loc[pending["category"].eq("source_event_review")].head(10).to_string(
        index=False
    ))

    sample = normalized.loc[
        normalized["code"].eq("sh.600011")
        & normalized["ex_date"].eq(pd.Timestamp("2020-07-06"))
    ]
    print("\n华能国际样本：")
    print(sample.to_string(index=False))
    print(f"\n标准化事件表：{event_path}")
    print(f"待核验清单：{review_path}")


if __name__ == "__main__":
    main()
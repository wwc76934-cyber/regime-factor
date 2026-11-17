"""将公告核验通过的三笔现金分红补入回测事件表。"""

from pathlib import Path

import pandas as pd


OVERRIDES = [
    {
        "code": "sh.600989",
        "record_date": "2021-05-19",
        "ex_date": "2021-05-20",
        "pay_date": "2021-05-20",
        "cash_per_share": "0.32091",
        "source_category": "source_event_review",
        "verification_note": "差异化分红，采用无限售流通股适用金额",
        "source_url": (
            "https://dataclouds.cninfo.com.cn/shgonggao/2022/2022-03-10/"
            "15024e609fa211ec9819fa163e26e5de.pdf"
        ),
    },
    {
        "code": "sh.601966",
        "record_date": "2024-06-13",
        "ex_date": "2024-06-14",
        "pay_date": "2024-06-14",
        "cash_per_share": "0.377",
        "source_category": "source_event_review",
        "verification_note": "同日实施年度和一季度分红，公告合计为0.377元",
        "source_url": (
            "https://epaper.stcn.com/pic/202406/07/"
            "39b5d9d80fd564560fbcddfe5394f4bf.pdf"
        ),
    },
    {
        "code": "sh.600519",
        "record_date": "2023-12-19",
        "ex_date": "2023-12-20",
        "pay_date": "2023-12-20",
        "cash_per_share": "19.106",
        "source_category": "price_flag_only",
        "verification_note": "根据特别分红实施公告补录",
        "source_url": (
            "https://www.sse.com.cn/disclosure/listedinfo/announcement/c/new/"
            "2023-12-14/600519_20231214_WR1C.pdf"
        ),
    },
    {
        "code": "sz.000538",
        "record_date": "2024-11-22",
        "ex_date": "2024-11-25",
        "pay_date": "2024-11-25",
        "cash_per_share": "1.213",
        "source_category": "price_flag_only",
        "verification_note": "按2024年特别分红实施公告补录，每10股派12.13元",
        "source_url": "https://static.cninfo.com.cn/finalpage/2024-11-19/1221767436.PDF",
    },
]

KEYS = ["code", "ex_date"]


def main():
    root = Path(__file__).resolve().parents[1]
    action_dir = root / "data" / "processed" / "corporate_actions"
    report_dir = root / "reports" / "portfolio"

    base = pd.read_parquet(action_dir / "csi300_normalized_actions.parquet")
    pending = pd.read_parquet(report_dir / "corporate_action_review_queue.parquet")

    overrides = pd.DataFrame(OVERRIDES)
    for column in ["record_date", "ex_date", "pay_date"]:
        overrides[column] = pd.to_datetime(overrides[column], errors="raise")
    base["ex_date"] = pd.to_datetime(base["ex_date"], errors="raise")
    pending["ex_date"] = pd.to_datetime(pending["ex_date"], errors="raise")

    # 补录必须能对应到待核验清单中的原始问题。
    review_info = pending[KEYS + ["category", "source_rows"]]
    overrides = overrides.merge(
        review_info, on=KEYS, how="left", validate="one_to_one"
    )
    if not overrides["category"].eq(overrides["source_category"]).all():
        raise ValueError("补录事件与待核验清单中的事件类型不匹配")

    if not (
        overrides["record_date"].lt(overrides["ex_date"])
        & overrides["pay_date"].ge(overrides["ex_date"])
    ).all():
        raise ValueError("补录事件日期次序不正确")

    overrides["stock_listing_date"] = pd.NaT
    overrides["bonus_shares_per_share"] = "0"
    overrides["reserve_shares_per_share"] = "0"
    overrides["event_type"] = "cash_only"

    base["source_category"] = "baostock_normalized"
    base["verification_note"] = ""
    base["source_url"] = ""

    additions = overrides.drop(columns="category")
    ready = pd.concat([base, additions], ignore_index=True)
    if ready.duplicated(KEYS).any():
        raise ValueError("回测事件表存在重复证券和除权除息日")
    ready = ready.sort_values(["ex_date", "code"]).reset_index(drop=True)

    remaining = pending.merge(
        additions[KEYS], on=KEYS, how="left",
        validate="one_to_one", indicator=True,
    )
    remaining = remaining.loc[remaining["_merge"].eq("left_only")].drop(
        columns="_merge"
    )
    if len(remaining) + len(additions) != len(pending):
        raise ValueError("补录与剩余核验数量无法对账")

    event_path = action_dir / "csi300_backtest_actions.parquet"
    review_path = report_dir / "corporate_actions_remaining_review.parquet"
    ready.to_parquet(event_path, index=False)
    remaining.to_parquet(review_path, index=False)

    print(f"回测事件数：{len(ready):,}")
    print(f"公告核验补录：{len(additions)}")
    print(f"剩余待核验：{len(remaining)}")
    print(additions[[
        "code", "record_date", "ex_date", "pay_date", "cash_per_share",
    ]].to_string(index=False))
    print(f"\n回测事件表：{event_path}")
    print(f"剩余待核验表：{review_path}")


if __name__ == "__main__":
    main()
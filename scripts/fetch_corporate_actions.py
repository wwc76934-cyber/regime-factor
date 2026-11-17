"""下载历史分红送转记录，并与日线前收参考价差异对账。"""

import argparse
from pathlib import Path
import tomllib

import baostock as bs
import pandas as pd


REQUIRED_FIELDS = [
    "code",
    "dividRegistDate",
    "dividOperateDate",
    "dividPayDate",
    "dividStockMarketDate",
    "dividCashPsBeforeTax",
    "dividStocksPs",
    "dividReserveToStockPs",
]


def check_response(response, action):
    if response.error_code != "0":
        raise RuntimeError(f"{action}失败：{response.error_code} {response.error_msg}")


def fetch_actions(code, year):
    action = f"{code} {year} 年公司行动"
    response = bs.query_dividend_data(code=code, year=str(year), yearType="operate")
    check_response(response, f"查询 {action}")

    rows = []
    while response.next():
        rows.append(response.get_row_data())
    check_response(response, f"读取 {action}")

    if not rows:
        # 成功返回空表也写入缓存，表示查过这个证券和年份。
        return pd.DataFrame(columns=REQUIRED_FIELDS)

    data = pd.DataFrame(rows, columns=response.fields)
    missing = set(REQUIRED_FIELDS) - set(data.columns)
    if missing:
        raise ValueError(f"{action} 缺少字段：{sorted(missing)}")
    if not data["code"].eq(code).all():
        raise ValueError(f"{action} 的证券代码与请求不一致")

    # 原始日期和比例字段仍按接口返回值保存。
    return data


def write_parquet(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    data.to_parquet(temporary, index=False)
    temporary.replace(path)


def reconcile(actions, flags, start, end):
    # 对账单位是同一证券的同一个除权除息日。
    # 一个日期下有多条源记录时，保留记录数供后续检查。
    date_text = actions["dividOperateDate"].astype("string").str.strip()
    ex_date = pd.to_datetime(
        date_text.replace("", pd.NA), format="%Y-%m-%d", errors="raise"
    )
    events = actions.assign(date=ex_date)
    events = events.loc[events["date"].between(start, end)]
    event_keys = (
        events.groupby(["date", "code"], as_index=False)
        .size()
        .rename(columns={"size": "source_event_rows"})
    )

    result = flags.merge(
        event_keys,
        on=["date", "code"],
        how="outer",
        validate="one_to_one",
        indicator="match",
    )
    result["match"] = result["match"].map({
        "both": "matched",
        "left_only": "price_flag_only",
        "right_only": "action_only",
    })
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="本次最多查询多少个尚未缓存的证券×年份。0 表示全部。",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 必须大于或等于 0")

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    research = config["research"]
    start = pd.Timestamp(research["development_start"])
    end = pd.Timestamp(research["development_end"])

    processed = root / "data" / "processed"
    members = pd.read_parquet(
        processed / "universe" / "csi300_weekly_members.parquet",
        columns=["code"],
    )
    codes = sorted(members["code"].unique())
    years = range(start.year, end.year + 1)

    cache = root / "data" / "raw" / "corporate_actions" / "baostock"

    def cache_path(code, year):
        return cache / str(year) / f"{code}.parquet"

    tasks = [(code, year) for year in years for code in codes]
    pending = [
        (code, year) for code, year in tasks
        if not cache_path(code, year).is_file()
    ]
    batch = pending[:args.limit] if args.limit else pending

    print(
        f"证券数：{len(codes)}，总查询项：{len(tasks):,}，"
        f"待查询：{len(pending):,}，本次：{len(batch):,}",
        flush=True,
    )

    if batch:
        check_response(bs.login(), "登录 BaoStock")
        try:
            for number, (code, year) in enumerate(batch, start=1):
                print(f"[{number}/{len(batch)}] {code} {year}", flush=True)
                data = fetch_actions(code, year)
                write_parquet(data, cache_path(code, year))
        finally:
            bs.logout()

    remaining = sum(
        not cache_path(code, year).is_file() for code, year in tasks
    )
    print(f"已缓存：{len(tasks) - remaining:,} / {len(tasks):,}，剩余：{remaining:,}")
    if remaining:
        return

    frames = []
    for code, year in tasks:
        data = pd.read_parquet(cache_path(code, year))
        if not data.empty:
            frames.append(data.assign(query_year=year))

    if not frames:
        raise ValueError("全部查询完成，但未获得任何公司行动记录")

    actions = pd.concat(frames, ignore_index=True)

    report_dir = root / "reports" / "portfolio"
    flags = pd.read_parquet(report_dir / "price_reference_flags.parquet")
    flags["date"] = pd.to_datetime(flags["date"], errors="raise")

    reconciliation = reconcile(actions, flags, start, end)

    action_path = processed / "corporate_actions" / "csi300_baostock_actions.parquet"
    report_path = report_dir / "corporate_action_reconciliation.parquet"
    write_parquet(actions, action_path)
    write_parquet(reconciliation, report_path)

    print(f"\n公司行动源记录：{len(actions):,}")
    print("事件日期对账：")
    print(reconciliation["match"].value_counts().to_string())

    unmatched = reconciliation.loc[
        reconciliation["match"].eq("price_flag_only")
    ]
    print("\n仅出现在行情异常表的前 5 条：")
    print(unmatched[["date", "code", "reference_gap"]].head(5).to_string(index=False))
    print(f"\n公司行动表：{action_path}")
    print(f"对账报告：{report_path}")


if __name__ == "__main__":
    main()
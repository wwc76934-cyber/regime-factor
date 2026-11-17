import argparse
from pathlib import Path
import tomllib

import baostock as bs
import pandas as pd

from fetch_universe import check_response


RAW_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus,isST"
ADJUSTED_FIELDS = "date,code,open,high,low,close,adjustflag"


def fetch_bars(code, start, end, fields, adjust_flag):
    action = f"{code} 日线（adjustflag={adjust_flag}）"
    response = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjust_flag,
    )
    check_response(response, f"查询 {action}")

    rows = []
    while response.next():
        rows.append(response.get_row_data())
    check_response(response, f"读取 {action}")

    bars = pd.DataFrame(rows, columns=response.fields)
    if bars.empty:
        raise ValueError(f"{action} 返回空数据")

    dates = pd.to_datetime(bars["date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{action} 的日期缺失或重复")
    if not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError(f"{action} 返回了请求范围之外的日期")

    actual_codes = bars["code"].astype("string").str.strip()
    actual_flags = bars["adjustflag"].astype("string").str.strip()
    code_matches = actual_codes.eq(code).fillna(False)
    flag_matches = actual_flags.eq(adjust_flag).fillna(False)
    bad_rows = ~(code_matches & flag_matches)

    if bad_rows.any():
        invalid = bars.loc[bad_rows]
        known_exception = (
            code == "sh.688223"
            and adjust_flag == "1"
            and len(invalid) == 1
            and code_matches.loc[invalid.index[0]]
            and actual_flags.loc[invalid.index[0]] == "3"
            and dates.loc[invalid.index[0]] == pd.Timestamp("2022-01-26")
        )

        if not known_exception or len(bars) == 1:
            sample = invalid[["date", "code", "adjustflag"]].head(5)
            raise ValueError(
                f"{action} 返回值不匹配。"
                f"adjustflag 分布：{actual_flags.value_counts(dropna=False).to_dict()}；"
                f"异常行数：{len(invalid)}/{len(bars)}；"
                f"前 5 条异常：{sample.to_dict('records')}"
            )

        root = Path(__file__).resolve().parents[1]
        quarantine_dir = (
            root / "data" / "raw" / "bars" / "baostock"
            / f"{start}_{end}" / "quarantine"
        )
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        invalid.to_parquet(
            quarantine_dir / "sh.688223_2022-01-26.parquet",
            index=False,
        )

        bars = bars.loc[~bad_rows].copy()
        print(
            f"已隔离 {code} 2022-01-26 的异常记录，"
            f"保留 {len(bars)} 条后复权日线。",
            flush=True,
        )

    return bars.sort_values("date").reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="本次最多处理多少只尚未下载完整的股票。0 表示全部。",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 必须大于或等于 0")

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    start = config["research"]["bars_start"]
    end = config["research"]["development_end"]

    members_path = root / "data" / "processed" / "universe" / "csi300_weekly_members.parquet"
    codes = sorted(pd.read_parquet(members_path, columns=["code"])["code"].unique())

    cache_dir = root / "data" / "raw" / "bars" / "baostock" / f"{start}_{end}"
    variants = [
        ("unadjusted", RAW_FIELDS, "3"),
        ("back_adjusted", ADJUSTED_FIELDS, "1"),
    ]
    for name, _, _ in variants:
        (cache_dir / name).mkdir(parents=True, exist_ok=True)

    def complete(code):
        return all(
            (cache_dir / name / f"{code}.parquet").is_file()
            for name, _, _ in variants
        )

    pending = [code for code in codes if not complete(code)]
    batch = pending[:args.limit] if args.limit else pending

    print(f"总证券数：{len(codes)}，待处理：{len(pending)}，本次处理：{len(batch)}", flush=True)
    if batch:
        check_response(bs.login(), "登录 BaoStock")
        try:
            for number, code in enumerate(batch, start=1):
                for name, fields, flag in variants:
                    output = cache_dir / name / f"{code}.parquet"
                    if output.is_file():
                        continue

                    print(f"[{number}/{len(batch)}] {code} {name}", flush=True)
                    try:
                        bars = fetch_bars(code, start, end, fields, flag)
                    except ValueError as error:
                        print(f"数据校验失败，待补：{error}", flush=True)
                        continue
                    temporary = output.with_suffix(".tmp")
                    bars.to_parquet(temporary, index=False)
                    temporary.replace(output)
        finally:
            bs.logout()

    print(f"双口径已缓存：{sum(complete(code) for code in codes)} / {len(codes)} 只")
    print(f"保存目录：{cache_dir}")
    incomplete = [code for code in codes if not complete(code)]
    print(f"仍待补齐 {len(incomplete)} 只，前 20 只：{incomplete[:20]}")


if __name__ == "__main__":
    main()
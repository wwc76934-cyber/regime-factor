from pathlib import Path
import tomllib

import baostock as bs
import pandas as pd


def check_response(response, action):
    if response.error_code != "0":
        raise RuntimeError(f"{action}失败：{response.error_code} {response.error_msg}")


def fetch_snapshot(query_date):
    response = bs.query_hs300_stocks(date=query_date)
    check_response(response, f"查询 {query_date} 成分")

    rows = []
    while response.next():
        rows.append(response.get_row_data())
    check_response(response, f"读取 {query_date} 成分")

    snapshot = pd.DataFrame(rows, columns=response.fields)
    if snapshot.empty:
        raise ValueError(f"{query_date} 未返回成分股")

    valid_codes = snapshot["code"].str.fullmatch(r"(sh|sz)\.\d{6}", na=False)
    if not valid_codes.all() or snapshot["code"].duplicated().any():
        raise ValueError(f"{query_date} 包含无效或重复的证券代码")

    snapshot.insert(0, "query_date", query_date)
    return snapshot.sort_values("code").reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "configs" / "baseline.toml").read_text(encoding="utf-8"))
    dates = config["universe"]["snapshot_check_dates"]

    output_dir = root / "data" / "raw" / "universe" / "baostock" / "csi300"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在连接 BaoStock...", flush=True)
    check_response(bs.login(), "登录 BaoStock")

    snapshots = []
    try:
        for query_date in dates:
            print(f"正在查询 {query_date}...", flush=True)
            snapshot = fetch_snapshot(query_date)
            snapshot.to_parquet(output_dir / f"{query_date}.parquet", index=False)
            snapshots.append(snapshot)
            print(
                f"{query_date} | 成分数 {len(snapshot)} | "
                f"updateDate {snapshot['updateDate'].unique().tolist()}",
                flush=True,
            )
    finally:
        bs.logout()

    before = set(snapshots[0]["code"])
    after = set(snapshots[1]["code"])
    print(f"调入 {len(after - before)} 只：{sorted(after - before)}")
    print(f"调出 {len(before - after)} 只：{sorted(before - after)}")
    print(f"保存目录：{output_dir}")


if __name__ == "__main__":
    main()
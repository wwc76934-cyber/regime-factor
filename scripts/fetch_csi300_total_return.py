"""获取 H00300 全收益指数，并与已有回测的评价日期逐日核对。"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


INDEX_CODE = "H00300"
OFFICIAL_API = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
MIRROR_URL = (
    "https://raw.githubusercontent.com/YellowPancake/quant-autoresearch/"
    "4d8b42d377609c86b3071f72d9266c00e017ef50/"
    "examples/studies/total_return/data/china_total_return.csv"
)


def evaluation_dates(root):
    report = root / "reports" / "portfolio" / "walk_forward_three_variants"

    daily = pd.read_parquet(
        report / "daily_nav.parquet", columns=["strategy", "date"]
    )
    daily = daily.loc[daily["strategy"].eq("equal_weight")]
    full_dates = pd.DatetimeIndex(pd.to_datetime(daily["date"])).sort_values()
    if full_dates.empty or full_dates.hasnans or not full_dates.is_unique:
        raise ValueError("等权策略账本日期为空、缺失或重复")

    walk = root / "data" / "processed" / "portfolio" / "walk_forward"
    fits = pd.read_parquet(walk / "walk_forward_weight_schedule.parquet",
                           columns=["signal_date", "fit_status"])
    first_signal = fits.loc[fits["fit_status"].eq("fitted"), "signal_date"].min()
    targets = pd.read_parquet(walk / "walk_forward_three_variant_targets.parquet",
                              columns=["signal_date", "execution_date"])
    executions = targets.loc[targets["signal_date"].eq(first_signal), "execution_date"].unique()
    if len(executions) != 1:
        raise ValueError("首个拟合信号必须对应唯一执行日")
    first_eval = pd.Timestamp(executions[0])
    earlier = full_dates[full_dates < first_eval]
    if earlier.empty or first_eval not in full_dates:
        raise ValueError("无法确定评价起点和前一交易日")

    return full_dates[full_dates >= earlier[-1]]


def download(url):
    request = Request(url, headers={"User-Agent": "regime-factor-research/1.0"})
    with urlopen(request, timeout=20) as response:
        return response.read()


def read_source(args, days):
    if args.mirror_csv or args.mirror:
        raw = args.mirror_csv.read_bytes() if args.mirror_csv else download(MIRROR_URL)
        mirror = pd.read_csv(BytesIO(raw), usecols=["date", "csi300"], dtype={"date": str})
        history = mirror.rename(columns={"csi300": "close"}).assign(code=INDEX_CODE)
        source = str(args.mirror_csv.resolve()) if args.mirror_csv else MIRROR_URL
        return history, raw, source, "mirror"

    params = urlencode({
        "indexCode": INDEX_CODE,
        "startDate": days[0].strftime("%Y%m%d"),
        "endDate": days[-1].strftime("%Y%m%d"),
    })
    url = f"{OFFICIAL_API}?{params}"
    raw = download(url)
    rows = json.loads(raw).get("data")
    if not rows:
        raise ValueError("中证指数接口没有返回行情")

    history = pd.DataFrame(rows).rename(
        columns={"tradeDate": "date", "indexCode": "code"}
    )
    return history[["date", "code", "close"]], raw, url, "official_api"


def validate_history(history, expected):
    history = history.copy()
    # 兼容 20210115 和 2021-01-15 两种日期格式。
    date_text = history["date"].astype("string").str.strip()
    history["date"] = pd.to_datetime(
        date_text.str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="raise",
    )
    history = history.loc[history["date"].between(expected[0], expected[-1])].copy()
    history["close"] = pd.to_numeric(history["close"], errors="raise")

    if history.empty or not history["code"].astype("string").str.upper().eq(INDEX_CODE).all():
        raise ValueError("指数代码与 H00300 不一致，或请求区间内没有数据")
    if not history["date"].is_unique:
        raise ValueError("指数数据有重复日期")
    if not (np.isfinite(history["close"]) & history["close"].gt(0)).all():
        raise ValueError("指数收盘点位有缺失或非法值")

    observed = pd.DatetimeIndex(history["date"])
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"日期对不上：缺少 {len(missing)} 天，额外 {len(extra)} 天。"
            f"缺失示例：{missing[:5].strftime('%Y-%m-%d').tolist()}"
        )

    history["code"] = INDEX_CODE
    return history[["date", "code", "close"]].sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mirror", action="store_true", help="使用固定版本的公开镜像")
    source.add_argument("--mirror-csv", type=Path, help="读取手动下载的同一份镜像 CSV")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    days = evaluation_dates(root)
    history, raw, source_url, source_kind = read_source(args, days)
    history = validate_history(history, days)

    folder = root / "data" / "raw" / "index"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "csi300_H00300_development.parquet"
    temporary = path.with_suffix(".tmp")
    history.to_parquet(temporary, index=False)
    temporary.replace(path)

    metadata = {
        "index": INDEX_CODE,
        "source_kind": source_kind,
        "source": source_url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "start_date": f"{days[0]:%Y-%m-%d}",
        "end_date": f"{days[-1]:%Y-%m-%d}",
        "rows": len(history),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    growth = history["close"].iloc[-1] / history["close"].iloc[0] - 1
    print(f"来源：{source_kind}")
    print(f"日期：{days[0]:%Y-%m-%d} 至 {days[-1]:%Y-%m-%d}")
    print(f"H00300：{len(history)} 个收盘点位，与回测日期逐日匹配")
    print(f"区间累计全收益：{growth:+.2%}")
    print(f"输出：{path}")


if __name__ == "__main__":
    main()

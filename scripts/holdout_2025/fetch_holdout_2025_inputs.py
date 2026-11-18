import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time
import socket
from urllib.request import urlopen
import json

import pandas as pd


BAR_START = "2024-08-01"
END = "2025-12-31"
CAL_END = "2026-01-12"


class BoundedSocket(socket.socket):
    def recv(self, *args):
        data = super().recv(*args)
        if not data:
            raise ConnectionError("BaoStock closed the connection")
        return data


def bounded_network():
    socket.socket = BoundedSocket
    socket.setdefaulttimeout(15)


def put(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def get_rows(response):
    if response.error_code != "0":
        raise RuntimeError(response.error_msg)
    rows = response.get_data()
    if response.error_code != "0":
        raise RuntimeError(response.error_msg)
    return rows


def job(payload):
    bounded_network()
    project, output, codes = payload
    sys.path.insert(0, str(project / "scripts"))
    import baostock as bs
    from fetch_daily_bars import fetch_bars, RAW_FIELDS, ADJUSTED_FIELDS
    from fetch_corporate_actions import fetch_actions

    def login():
        response = bs.login()
        if response.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {response.error_code} {response.error_msg}")

    def pull_bars(code, adjusted):
        start = "2025-02-17" if adjusted and code == "sz.302132" else BAR_START
        fields, flag = (ADJUSTED_FIELDS, "1") if adjusted else (RAW_FIELDS, "3")
        return fetch_bars(code, start, END, fields, flag)

    connected = False
    try:
        errors = []
        for code in codes:
            for subdir, pull in [
                ("raw", lambda: pull_bars(code, False)),
                ("adjusted", lambda: pull_bars(code, True)),
                ("actions", lambda: fetch_actions(code, 2025)),
            ]:
                path = output / subdir / f"{code}.parquet"
                if path.exists():
                    continue
                for attempt in range(5):
                    try:
                        if not connected:
                            login()
                            connected = True
                        put(pull(), path)
                        break
                    except Exception as error:
                        connected = False
                        if attempt == 4:
                            errors.append((code, subdir, str(error)))
                        else:
                            time.sleep(attempt + 1)
        return errors
    finally:
        if connected:
            bs.logout()


def benchmark(output, mirror=None):
    candidates = [
        "https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=H00300&startDate=20241231&endDate=20251231",
    ]
    if mirror:
        frame = pd.read_csv(mirror, usecols=["date", "csi300"]).rename(columns={"csi300": "close"})
        frame["source"] = "user_supplied_mirror"
    else:
        from urllib.request import Request
        request = Request(candidates[0], headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"})
        with urlopen(request, timeout=35) as response:
            data = json.load(response)["data"]
        frame = pd.DataFrame(data).rename(columns={"tradeDate": "date", "close": "close"})
        if "date" not in frame or "close" not in frame:
            raise ValueError(f"Unexpected official index schema: {list(frame)}")
        frame["source"] = "official_csindex"
    digits = frame["date"].astype(str)
    frame["date"] = pd.to_datetime(digits.where(~digits.str.fullmatch(r"\d{8}"),
                                               pd.to_datetime(digits.where(digits.str.fullmatch(r"\d{8}")),
                                                              format="%Y%m%d").dt.strftime("%Y-%m-%d")))
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.loc[frame["date"].between("2024-12-31", END), ["date", "close", "source"]].sort_values("date")
    if frame["date"].duplicated().any() or len(frame) < 230 or not frame["close"].gt(0).all():
        raise ValueError("Index date/level validation failed")
    put(frame, output / "H00300.parquet")


def prepare(project, output, workers, mirror):
    bounded_network()
    sys.path.insert(0, str(project / "scripts"))
    import baostock as bs
    from fetch_universe import fetch_snapshot

    response = bs.login()
    if response.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {response.error_code} {response.error_msg}")
    try:
        calendar = get_rows(bs.query_trade_dates(start_date=BAR_START, end_date=CAL_END))
        calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"])
        if not pd.DatetimeIndex(calendar["calendar_date"]).equals(pd.date_range(BAR_START, CAL_END)):
            raise ValueError("Incomplete calendar")
        put(calendar, output / "calendar.parquet")
        days = calendar.loc[calendar["is_trading_day"].eq("1"), "calendar_date"].reset_index(drop=True)
        sched = pd.DataFrame({"signal_date": days, "execution_date": days.shift(-1)})
        sched = sched.loc[sched["signal_date"].dt.to_period("W-SUN").ne(
            sched["execution_date"].dt.to_period("W-SUN")) & sched["execution_date"].between("2025-01-01", "2026-01-09")
            & sched["signal_date"].le(END)]
        put(sched, output / "schedule.parquet")
        index = get_rows(bs.query_history_k_data_plus("sh.000300", "date,code,close",
                         start_date=BAR_START, end_date=END, frequency="d", adjustflag="3"))
        put(index, output / "price_index.parquet")
        codes = set()
        for signal in sched["signal_date"]:
            day = signal.strftime("%Y-%m-%d")
            path = output / "members" / f"{day}.parquet"
            current = pd.read_parquet(path) if path.exists() else fetch_snapshot(day)
            basis = pd.to_datetime(current["updateDate"], errors="raise")
            if len(current) != 300 or not current["code"].is_unique or not basis.le(signal).all():
                raise ValueError(f"Invalid point-in-time snapshot on {day}")
            if not path.exists():
                put(current.assign(signal_date=signal), path)
            codes.update(current["code"])
    finally:
        bs.logout()
    if not (output / "H00300.parquet").exists():
        try:
            benchmark(output, mirror)
        except Exception as error:
            print(f"H00300 unavailable yet: {error}", flush=True)
    codes = sorted(codes)
    print(f"Cached schedules and snapshots; downloading/checking {len(codes)} stock histories", flush=True)
    tasks = [(project, output, codes[i:i+15]) for i in range(0, len(codes), 15)]
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(job, tasks), start=1):
            errors.extend(result)
            print(f"Completed input batch {i}/{len(tasks)} with {len(result)} errors", flush=True)
    if errors:
        raise RuntimeError(str(errors))
    if not (output / "H00300.parquet").exists():
        benchmark(output, mirror)


def resume(project, output, workers, mirror):
    schedule = pd.read_parquet(output / "schedule.parquet")
    codes = sorted(set().union(*(set(pd.read_parquet(
        output / "members" / f"{day:%Y-%m-%d}.parquet", columns=["code"])["code"])
        for day in schedule["signal_date"])))
    missing = [code for code in codes if any(
        not (output / part / f"{code}.parquet").exists() for part in ("raw", "adjusted", "actions"))]
    print(f"Expected {len(codes)} stocks; missing files for {len(missing)}: {missing[:20]}", flush=True)
    if workers == 1:
        errors = [error for code in missing for error in job((project, output, [code]))]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            errors = [error for batch in pool.map(job, [(project, output, [code]) for code in missing])
                      for error in batch]
    if errors:
        raise RuntimeError(str(errors))
    if not (output / "H00300.parquet").exists():
        benchmark(output, mirror)
    print(f"Cache ready: {len(codes)} × 3 stock partitions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index-mirror-csv", type=Path)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse cached calendar and snapshots; only fill missing files")
    args = parser.parse_args()
    if args.index_only:
        benchmark(args.cache, args.index_mirror_csv)
    elif args.resume:
        resume(args.project_root.resolve(), args.cache.resolve(), args.workers, args.index_mirror_csv)
    else:
        prepare(args.project_root.resolve(), args.cache.resolve(), args.workers, args.index_mirror_csv)

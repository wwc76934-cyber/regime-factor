"""Step 1: user-run BaoStock collection and offline data quality audit."""

import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import pandas as pd


WARM = pd.Timestamp("2014-06-01")
FIRST = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2025-12-31")
FIELDS = ["turn", "peTTM", "pbMRQ"]
OHLC = ["open", "high", "low", "close"]
RAW = ["date", "code", *OHLC, "preclose", "volume", "amount", "adjustflag", *FIELDS, "tradestatus", "isST"]
ADJ = RAW
INDEX = ["date", "code", *OHLC]
SAMPLE = ["sh.600000", "sz.000001", "sh.601318"]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if path.suffix == ".parquet":
        data.to_parquet(tmp, index=False)
    else:
        data.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def read_response(response, label, empty=False):
    need(response.error_code == "0", f"{label}: {response.error_code} {response.error_msg}")
    rows = []
    while response.next():
        rows.append(response.get_row_data())
    need(response.error_code == "0", f"{label}: {response.error_code} {response.error_msg}")
    need(empty or rows, f"{label}: empty response")
    return pd.DataFrame(rows, columns=response.fields)


class BoundedSocket(socket.socket):
    def recv(self, *args):
        data = super().recv(*args)
        if not data:
            raise ConnectionError("BaoStock closed its connection")
        return data


class BaoSession:
    def __init__(self):
        import baostock
        import baostock.common.context as context

        self.api = baostock
        self.context = context
        self.connected = False
        socket.socket = BoundedSocket
        socket.setdefaulttimeout(20)

    def close(self):
        connection = getattr(self.context, "default_socket", None)
        if connection is not None:
            connection.close()
        self.connected = False

    def query(self, method, label, *args, empty=False, **kwargs):
        for attempt in range(3):
            try:
                if not self.connected:
                    response = self.api.login()
                    if response.error_code != "0":
                        raise ConnectionError(f"BaoStock login failed: {response.error_code} {response.error_msg}")
                    self.connected = True
                return read_response(getattr(self.api, method)(*args, **kwargs), label, empty)
            except (ValueError, OSError) as error:
                self.close()
                if attempt == 2:
                    raise RuntimeError(f"{label}: {error}") from error
                time.sleep(attempt + 1)


def daily(session, code, first, last, fields, flag, empty=True):
    label = f"{code}, {first} .. {last}, flag {flag}"
    frame = session.query("query_history_k_data_plus", label, code, ",".join(fields),
                          start_date=str(first.date()), end_date=str(last.date()),
                          frequency="d", adjustflag=flag, empty=empty)
    need(set(fields).issubset(frame.columns), f"Unexpected fields: {label}")
    dates = pd.to_datetime(frame["date"], errors="raise")
    need(frame["code"].eq(code).all() and dates.is_unique and dates.between(first, last).all(), label)
    return frame[fields].assign(date=dates).sort_values("date").reset_index(drop=True)


def check_snapshot(frame, signal):
    need({"code", "updateDate"}.issubset(frame.columns), "Missing member fields")
    need(len(frame) == 300 and frame["code"].is_unique and
         frame["code"].str.fullmatch(r"(sh|sz)\.\d{6}").all(), f"Invalid members on {signal.date()}")
    dates = pd.to_datetime(frame["updateDate"], errors="raise")
    need(dates.notna().all() and dates.le(signal).all(), f"Future or missing member update: {signal.date()}")
    return frame.assign(signal_date=signal)


def weekly_calendar(frame):
    dates = pd.to_datetime(frame["calendar_date"], errors="raise")
    need(pd.DatetimeIndex(dates).sort_values().equals(pd.date_range(WARM, END)),
         "Calendar must cover all civil dates in the configured range")
    days = dates.loc[frame["is_trading_day"].astype(str).eq("1")].sort_values().reset_index(drop=True)
    schedule = pd.DataFrame({"signal_date": days, "execution_date": days.shift(-1)})
    schedule = schedule.loc[schedule.signal_date.ge(FIRST) & schedule.execution_date.notna()
                            & schedule.signal_date.dt.to_period("W-SUN").ne(
                                schedule.execution_date.dt.to_period("W-SUN"))].reset_index(drop=True)
    need(schedule.signal_date.dt.year.min() == FIRST.year and schedule.execution_date.dt.year.max() == END.year,
         "Unexpected execution years")
    return pd.DatetimeIndex(days), schedule


def ranges(schedule, members):
    joined = members.merge(schedule, on="signal_date", validate="many_to_one")
    tasks = defaultdict(set)
    for code, week in joined.groupby("code"):
        lower = week["signal_date"].min() - timedelta(days=230)
        for year in range(max(WARM.year, lower.year), END.year + 1):
            tasks[year].add(code)
    return {year: sorted(codes) for year, codes in sorted(tasks.items())}


def year_bounds(year):
    return max(WARM, pd.Timestamp(year, 1, 1)), min(END, pd.Timestamp(year, 12, 31))


def probe(root, cache, output):
    session = BaoSession()
    diagnostics = []
    try:
        snap = session.query("query_hs300_stocks", "2015 snapshot", date="2015-01-09")
        check_snapshot(snap, pd.Timestamp("2015-01-09"))
        for period in [("2015-01-05", "2015-01-30"), ("2025-10-09", "2025-10-31")]:
            for code in SAMPLE:
                for basis, flag, fields in [("raw", "3", RAW), ("adjusted", "1", ADJ)]:
                    frame = daily(session, code, *map(pd.Timestamp, period), fields, flag, empty=False)
                    save(frame, cache / "probe" / f"{period[0]}_{code}_{basis}.parquet")
                    numeric = frame[FIELDS].apply(pd.to_numeric, errors="coerce")
                    row = {"start": period[0], "code": code, "basis": basis, "rows": len(frame),
                           "expected_adjustflag": flag, "matching_flag": int(frame.adjustflag.eq(flag).sum())}
                    row.update({f"{name}_non_null": int(np.isfinite(numeric[name]).sum()) for name in FIELDS})
                    diagnostics.append(row)
    finally:
        session.close()
    result = pd.DataFrame(diagnostics)
    save(result, output / "probe_fields.csv")
    print(result.to_string(index=False), flush=True)
    print("Probe complete; examine probe_fields.csv before bulk download.", flush=True)


def load_universe(cache):
    days, expected = weekly_calendar(pd.read_parquet(cache / "calendar.parquet"))
    schedule = pd.read_parquet(cache / "schedule.parquet")
    pd.testing.assert_frame_equal(expected, schedule, check_dtype=False)
    members = []
    for signal in schedule.signal_date:
        path = cache / "members" / f"{signal:%Y-%m-%d}.parquet"
        members.append(check_snapshot(pd.read_parquet(path), signal))
    return days, schedule, pd.concat(members, ignore_index=True)


def collect(cache, through_year, only):
    calendar = cache / "calendar.parquet"
    session = BaoSession()
    try:
        if not calendar.exists():
            cal = session.query("query_trade_dates", "development calendar",
                                start_date=str(WARM.date()), end_date=str(END.date()))
            weekly_calendar(cal)
            save(cal, calendar)
        _, schedule = weekly_calendar(pd.read_parquet(calendar))
        save(schedule, cache / "schedule.parquet")
        for pos, signal in enumerate(schedule.signal_date, 1):
            path = cache / "members" / f"{signal:%Y-%m-%d}.parquet"
            if not path.exists():
                frame = session.query("query_hs300_stocks", f"snapshot {signal.date()}", date=str(signal.date()))
                save(check_snapshot(frame, signal), path)
            if pos % 50 == 0:
                print(f"Snapshots {pos}/{len(schedule)}", flush=True)
        days, schedule, members = load_universe(cache)
        work = ranges(schedule, members)
        if only != "universe":
            for year, codes in work.items():
                if year > through_year:
                    break
                first, last = year_bounds(year)
                if only in {"all", "bars"}:
                    index = cache / "price_index" / f"{year}.parquet"
                    if not index.exists():
                        save(daily(session, "sh.000300", first, last, INDEX, "3", empty=False), index)
                for pos, code in enumerate(codes, 1):
                    if only in {"all", "bars"}:
                        for basis, flag, fields in [("raw", "3", RAW), ("adjusted", "1", ADJ)]:
                            path = cache / basis / str(year) / f"{code}.parquet"
                            if not path.exists():
                                save(daily(session, code, first, last, fields, flag), path)
                    if year >= FIRST.year and only in {"all", "actions"}:
                        path = cache / "actions" / str(year) / f"{code}.parquet"
                        if not path.exists():
                            frame = session.query("query_dividend_data", f"actions {code} {year}",
                                                  code=code, year=str(year), yearType="operate", empty=True)
                            need(frame.empty or frame["code"].eq(code).all(), f"Wrong action code: {code}")
                            save(frame, path)
                    if pos % 50 == 0 or pos == len(codes):
                        print(f"{year} {pos}/{len(codes)} stocks ({only})", flush=True)
        return days
    finally:
        session.close()


def collect_benchmark(cache, mirror, start):
    from fetch_csi300_total_return import read_source

    days, _, _ = load_universe(cache)
    need(pd.Timestamp("2016-12-01") <= start <= END, "Benchmark start must be within the development evaluation dates")
    days = days[days >= start]
    options = argparse.Namespace(mirror_csv=mirror, mirror=False)
    if mirror:
        frame, payload, source, kind = read_source(options, days)
    else:
        chunks, hashes = [], []
        for year in sorted(set(days.year)):
            chunk, data, _, kind = read_source(options, days[days.year == year])
            chunks.append(chunk)
            hashes.append(hashlib.sha256(data).hexdigest())
        frame, payload, source = pd.concat(chunks), "\n".join(hashes).encode(), "official_csindex_yearly"
    frame["date"] = pd.to_datetime(frame["date"].astype(str).str.replace("-", "", regex=False),
                                       format="%Y%m%d", errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.loc[frame.date.isin(days)].sort_values("date").reset_index(drop=True)
    need(frame.date.is_unique and frame.code.astype(str).str.upper().eq("H00300").all()
         and np.isfinite(frame.close).all() and frame.close.gt(0).all() and not frame.empty,
         "Invalid H00300 rows")
    need(pd.DatetimeIndex(frame.date).equals(days[days>=frame.date.iat[0]]),
         "H00300 has internal gaps or is missing the last development session")
    save(frame, cache / "H00300.parquet")
    (cache / "H00300_source.json").write_text(json.dumps({"source": source, "kind": kind,
        "source_digest": hashlib.sha256(payload).hexdigest(), "first_date": str(frame.date.iat[0].date()),
        "last_date": str(frame.date.iat[-1].date()), "rows": len(frame)}, indent=2), encoding="utf-8")


def audit(root, cache, output):
    from normalize_corporate_actions import normalize_event, EVENT_COLUMNS

    days, schedule, members = load_universe(cache)
    work = ranges(schedule, members)
    missing = []
    for year, codes in work.items():
        index = cache / "price_index" / f"{year}.parquet"
        if not index.exists():
            missing.append(str(index.relative_to(cache)))
        for code in codes:
            for kind in ["raw", "adjusted", *(["actions"] if year >= FIRST.year else [])]:
                path = cache / kind / str(year) / f"{code}.parquet"
                if not path.exists():
                    missing.append(str(path.relative_to(cache)))
    if not (cache / "H00300.parquet").exists():
        missing.append("H00300.parquet")
    save(pd.DataFrame({"missing_file": missing}), output / "missing_files.csv")
    need(not missing, f"{len(missing)} incomplete files; see missing_files.csv")
    quality, quotes, actions, review, anomalies = [], [], [], [], []
    sessions = {day: pos for pos, day in enumerate(days)}
    for year, codes in work.items():
        index = pd.read_parquet(cache / "price_index" / f"{year}.parquet")
        expected = days[days.year == year]
        need(pd.DatetimeIndex(index.date).sort_values().equals(expected) and index.code.eq("sh.000300").all()
             and index[OHLC].apply(pd.to_numeric, errors="coerce").gt(0).all().all(), f"Invalid price index {year}")
        for basis, flag in [("raw", "3"), ("adjusted", "1")]:
            frames = []
            for code in codes:
                frame = pd.read_parquet(cache / basis / str(year) / f"{code}.parquet")
                need(frame.code.eq(code).all() and frame.date.is_unique and frame.date.isin(expected).all(),
                     f"Invalid {basis} keys {year} {code}")
                if not frame.empty:
                    frames.append(frame)
            combined = pd.concat(frames, ignore_index=True)
            ok = combined.adjustflag.eq(flag)
            if (~ok).any():
                anomalies.append(combined.loc[~ok, ["date","code","adjustflag"]].assign(expected=flag, basis=basis))
            numeric = combined[FIELDS].apply(pd.to_numeric, errors="coerce")
            for field in FIELDS:
                value = numeric[field]
                finite = np.isfinite(value)
                eligible = ok & combined.tradestatus.astype(str).eq("1")
                sample = value[eligible & finite]
                quality.append({"year": year, "basis": basis, "field": field, "rows": len(combined),
                    "matching_flag_rows": int(ok.sum()), "active_rows": int(eligible.sum()),
                    "finite_active": len(sample), "active_coverage": len(sample)/int(eligible.sum()) if eligible.any() else np.nan,
                    "negative_active": int(sample.lt(0).sum()), "zero_active": int(sample.eq(0).sum()),
                    **{f"p{int(q*100):02}": sample.quantile(q) for q in [.01,.50,.99]}})
            valid = combined[OHLC].apply(pd.to_numeric, errors="coerce")
            usable = (ok & np.isfinite(valid).all(axis=1) & valid.gt(0).all(axis=1)
                      & valid["high"].ge(valid[["open","close","low"]].max(axis=1))
                      & valid["low"].le(valid[["open","close"]].min(axis=1))
                      & combined.tradestatus.astype(str).eq("1"))
            part = combined[["date","code","preclose","close","isST","tradestatus"]].copy()
            part["usable"] = usable
            part["basis"] = basis
            quotes.append(part)
        if year >= FIRST.year:
            for code in codes:
                frame = pd.read_parquet(cache / "actions" / str(year) / f"{code}.parquet")
                if frame.empty:
                    continue
                frame["ex_date"] = pd.to_datetime(frame["dividOperateDate"], errors="coerce")
                for day, group in frame.groupby("ex_date", dropna=False):
                    if pd.notna(day) and day > END:
                        continue
                    if pd.isna(day) or day.year != year:
                        review.append((code, day, "action_date_unverified", str(year)))
                        continue
                    try:
                        actions.append(normalize_event(code, day, group))
                    except ValueError as error:
                        review.append((code, day, "action_terms_unverified", str(error)))
    all_quotes = pd.concat(quotes, ignore_index=True)
    links = all_quotes.pivot(index=["date","code"], columns="basis", values="usable").reset_index()
    links["eligible_both"] = links.adjusted.eq(True) & links.raw.eq(True)
    linked = members.merge(links[["date","code","eligible_both"]].rename(columns={"date":"signal_date"}),
                           how="left", on=["signal_date","code"], validate="one_to_one")
    weekly = linked.assign(eligible_both=linked.eligible_both.eq(True)).groupby("signal_date").agg(
        members=("code","size"), eligible_both=("eligible_both","sum")).reset_index()
    weekly["year"] = weekly.signal_date.dt.year
    year_counts = weekly.groupby("year").agg(weeks=("signal_date","size"), min_eligible=("eligible_both","min"),
                  median_eligible=("eligible_both","median"), weeks_ge_280=("eligible_both",lambda x: x.ge(280).sum())).reset_index()
    raw = all_quotes.loc[all_quotes.basis.eq("raw")].sort_values(["code","date"]).copy()
    raw["prior_close"] = raw.groupby("code")["close"].shift()
    raw["gap"] = pd.to_numeric(raw["preclose"],errors="coerce")-pd.to_numeric(raw["prior_close"],errors="coerce")
    raw["session"] = raw.date.map(sessions)
    consecutive = raw.groupby("code")["session"].diff().eq(1)
    valid_previous = raw.groupby("code")["usable"].shift().eq(True)
    flags = raw.loc[consecutive & valid_previous & raw.usable & raw.date.ge(FIRST) & raw.gap.abs().gt(.005),
                    ["date","code","gap"]].rename(columns={"date":"ex_date"})
    events = pd.DataFrame(actions, columns=EVENT_COLUMNS)
    need(not events.duplicated(["code","ex_date"]).any(), "Duplicate normalized actions")
    flagged = set(zip(flags.code,flags.ex_date))
    known = set(zip(events.code,events.ex_date)) | {(code,day) for code,day,_,_ in review}
    for code, day in sorted(flagged-known):
        review.append((code,day,"price_reference_only","Verify against exchange records before execution"))
    benchmark = pd.read_parquet(cache / "H00300.parquet").set_index("date")["close"].sort_index()
    require_days=days[days >= max(pd.Timestamp("2016-12-01"),benchmark.index[0])]
    need(benchmark.index.equals(require_days) and np.isfinite(benchmark).all() and benchmark.gt(0).all(),
         "H00300 comparison calendar mismatch")
    for name, value in [("field_yearly",pd.DataFrame(quality)),("signal_coverage",weekly),("signal_yearly",year_counts),
                        ("adjustflag_quarantine",pd.concat(anomalies,ignore_index=True) if anomalies else
                         pd.DataFrame(columns=["date","code","adjustflag","expected","basis"])),
                        ("price_reference_flags",flags),("action_review",pd.DataFrame(review,
                         columns=["code","ex_date","category","reason"]))]:
        save(value, output / f"{name}.csv")
    save(events, output / "normalized_actions.parquet")
    print(year_counts.to_string(index=False))
    print(f"H00300 comparable from {benchmark.index[0].date()}; {len(events)} actions + {len(review)} review rows")
    print("Step 1 audit written. Exposure review is still required before portfolio simulation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe","download","benchmark","audit"])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--through-year", type=int, choices=range(2015, 2026), default=2025)
    parser.add_argument("--only", choices=["all","universe","bars","actions"], default="all")
    parser.add_argument("--benchmark-csv", type=Path, help="Verified H00300 CSV with date,csi300")
    parser.add_argument("--benchmark-start", type=pd.Timestamp, default=pd.Timestamp("2016-12-01"),
                        help="Explicit later H00300 start if earlier data cannot be independently verified")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "scripts"))
    cache = args.cache or root / "data/raw/v3/development"
    output = args.output_dir or root / "research/v3/results/step1"
    if args.action == "probe":
        probe(root, cache, output)
    elif args.action == "download":
        collect(cache, args.through_year, args.only)
    elif args.action == "benchmark":
        collect_benchmark(cache, args.benchmark_csv, args.benchmark_start)
    else:
        audit(root, cache, output)

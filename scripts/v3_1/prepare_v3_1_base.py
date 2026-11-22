"""Audit the V3 development cache and build V3.1 benchmarks. Network actions are explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile


FIELDS = ["psTTM", "pcfNcfTTM"]
DEVELOPMENT_LAST = pd.Timestamp("2025-12-31")
INDEX_FIRST = pd.Timestamp("2015-01-01")
EVAL_FIRST = pd.Timestamp("2017-01-01")
FLOAT_WINDOW = 20
FLOAT_MIN = 10


def initialize(root):
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "scripts" / "v3"))
    import fetch_and_audit_v3_data as v3

    source = root / "data/raw/v3/development"
    days, schedule, members = v3.load_universe(source)
    if days.max() != DEVELOPMENT_LAST or members.signal_date.max() > DEVELOPMENT_LAST:
        raise ValueError("V3.1 development must end in 2025")
    return v3, source, days, schedule, members


def valuation_tasks(members):
    return sorted({(day.year, code) for day, code in zip(members.signal_date, members.code, strict=True)})


def official_history(source, days):
    frame = pd.read_parquet(source / "H00300.parquet")
    from fetch_csi300_total_return import validate_history

    last = pd.to_datetime(frame["date"]).min()
    coverage = days[days >= last]
    return validate_history(frame, coverage)


def get_development_index(v3, source, days, cache, mirror):
    from fetch_csi300_total_return import read_source, validate_history

    existing = official_history(source, days)
    add = days[(days >= INDEX_FIRST) & (days < existing.date.iat[0])]
    if not len(add):
        return
    path = cache / "H00300_early.parquet"
    if path.exists() and mirror is None:
        validate_history(pd.read_parquet(path), add)
        return
    options = argparse.Namespace(mirror_csv=mirror, mirror=False)
    year_groups = [add] if mirror else [add[add.year == year] for year in sorted(set(add.year))]
    chunks = []
    provenance = []
    for dates in year_groups:
        frame, payload, label, kind = read_source(options, dates)
        chunk = validate_history(frame, dates)
        chunks.append(chunk)
        provenance.append({"source": str(mirror.name) if mirror else label, "kind": kind,
                           "first": str(dates[0].date()), "last": str(dates[-1].date()),
                           "sha256": hashlib.sha256(payload).hexdigest()})
    v3.save(pd.concat(chunks, ignore_index=True), path)
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"H00300 historical gap filled: {len(add)} sessions", flush=True)


def probe(v3, cache, out):
    session = v3.BaoSession()
    results = []
    try:
        for start, end in [("2015-01-05", "2015-01-30"), ("2025-10-09", "2025-10-31")]:
            for code in ["sh.600000", "sh.601318", "sz.000001"]:
                frame = collect_valuation(v3, session, code, pd.Timestamp(start), pd.Timestamp(end))
                values = frame[FIELDS].apply(pd.to_numeric, errors="coerce")
                known = values.apply(np.isfinite).sum()
                positive = values.where(values.apply(np.isfinite)).gt(0).sum()
                results.append({"start": start, "code": code, "rows": len(frame),
                                **{f"{key}_finite": int(known[key]) for key in FIELDS},
                                **{f"{key}_positive": int(positive[key]) for key in FIELDS}})
                v3.save(frame, cache / "probe" / f"{start}_{code}.parquet")
    finally:
        session.close()
    audit = pd.DataFrame(results)
    v3.save(audit, out / "valuation_probe.csv")
    print(audit.to_string(index=False))


def collect_valuation(v3, session, code, start, end):
    frame = v3.daily(session, code, start, end, ["date", "code", "adjustflag", *FIELDS], "3")
    if not frame["adjustflag"].astype(str).eq("3").all():
        raise ValueError(f"Unexpected adjustment flag for valuation fields: {code}, {start.year}")
    return frame


def collect_valuations(v3, source, members, cache):
    tasks = valuation_tasks(members)
    remaining = [(year, code) for year, code in tasks
                 if not (cache / "valuations" / str(year) / f"{code}.parquet").exists()
                 and not set(FIELDS).issubset(ParquetFile(source / "raw" / str(year) / f"{code}.parquet").schema_arrow.names)]
    if not remaining:
        print(f"Valuation partitions ready: {len(tasks)}", flush=True)
        return
    by_code = {}
    for year, code in remaining:
        by_code.setdefault(code, []).append(year)
    session = v3.BaoSession()
    try:
        written = 0
        for pos, (code, years) in enumerate(by_code.items(), 1):
            start, end = pd.Timestamp(min(years), 1, 1), pd.Timestamp(max(years), 12, 31)
            frame = collect_valuation(v3, session, code, start, end)
            for year in years:
                raw = pd.read_parquet(source / "raw" / str(year) / f"{code}.parquet",
                                      columns=["date", "adjustflag", "tradestatus"])
                active = raw.loc[raw.tradestatus.astype(str).eq("1") & raw.adjustflag.astype(str).eq("3"), "date"]
                block = frame.loc[frame.date.dt.year.eq(year)]
                if not active.isin(block.date).all():
                    # Some wide-history queries can have incomplete calendar coverage.
                    block = collect_valuation(v3, session, code, pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
                if not active.isin(block.date).all():
                    raise ValueError(f"Valuation response dates do not match the active raw calendar: {code}, {year}")
                v3.save(block, cache / "valuations" / str(year) / f"{code}.parquet")
                written += 1
            if pos % 25 == 0 or pos == len(by_code):
                print(f"Valuation download: {pos}/{len(by_code)} stocks, {written}/{len(remaining)} partitions", flush=True)
        print(f"Valuation partitions ready: {len(tasks)}", flush=True)
    finally:
        session.close()


def read_bars(v3, source, cache, schedule, members, days):
    tasks = v3.ranges(schedule, members)
    required = dict()
    for (year, code), group in members.groupby([members.signal_date.dt.year, "code"]):
        required[(year, code)] = pd.DatetimeIndex(group.signal_date)
    raw_blocks, adj_blocks, values = [], [], []
    for year, codes in tasks.items():
        raw_parts, adj_parts = [], []
        sessions = days[days.year == year]
        for code in codes:
            fname = f"{code}.parquet"
            raw = ParquetFile(source / "raw" / str(year) / fname)
            available = set(raw.schema_arrow.names) & set(FIELDS)
            bars = raw.read(columns=["date", "code", "amount", "turn", "adjustflag", "tradestatus",
                                     *sorted(available)]).to_pandas()
            bars = bars.loc[bars.adjustflag.astype(str).eq("3")].copy()
            if not bars.date.is_unique or not bars.date.isin(sessions).all() or not bars.code.eq(code).all():
                raise ValueError(f"Invalid raw partition {year}/{code}")
            raw_parts.append(bars[["date", "code", "amount", "turn", "tradestatus"]])
            if (year, code) in required:
                selected = bars.set_index("date").reindex(required[(year, code)])
                part = pd.DataFrame({"trade_active": selected.tradestatus.astype(str).eq("1")}, index=selected.index)
                missing = set(FIELDS) - available
                supplemental = cache / "valuations" / str(year) / fname
                supplied = None
                if missing and supplemental.exists():
                    supplied = pd.read_parquet(supplemental)
                    if (not supplied.date.is_unique or not supplied.date.isin(sessions).all()
                            or not supplied.code.eq(code).all()
                            or not supplied.adjustflag.astype(str).eq("3").all()):
                        raise ValueError(f"Invalid valuation supplement {year}/{code}")
                    supplied = supplied.set_index("date").reindex(part.index)
                for col in FIELDS:
                    src = selected if col in available else supplied
                    part[col] = pd.to_numeric(src[col], errors="coerce") if src is not None else np.nan
                    part[f"{col}_supplied"] = src is not None
                values.append(part.rename_axis("signal_date").reset_index().assign(code=code))
            adj = pd.read_parquet(source / "adjusted" / str(year) / fname,
                                  columns=["date", "code", "open", "close", "adjustflag"])
            if not adj.date.is_unique or not adj.date.isin(sessions).all() or not adj.code.eq(code).all():
                raise ValueError(f"Invalid adjusted partition {year}/{code}")
            adj_parts.append(adj.loc[adj.adjustflag.astype(str).eq("1"), ["date", "code", "open", "close"]])
        raw_blocks.append(pd.concat(raw_parts, ignore_index=True))
        adj_blocks.append(pd.concat(adj_parts, ignore_index=True))
        print(f"Cached bars checked: {year}", flush=True)
    return pd.concat(raw_blocks, ignore_index=True), pd.concat(adj_blocks, ignore_index=True), pd.concat(values, ignore_index=True)


def proxy_weights(raw, members, days):
    amount, turn = (pd.to_numeric(raw[field], errors="coerce") for field in ["amount", "turn"])
    active = raw.tradestatus.astype(str).eq("1")
    valid = active & amount.gt(0) & turn.gt(0) & turn.le(100) & np.isfinite(amount) & np.isfinite(turn)
    snapshot = raw[["date", "code"]].assign(cap=(amount / (turn / 100)).where(valid))
    daily = snapshot.pivot(index="date", columns="code", values="cap").reindex(days)
    smooth = daily.rolling(FLOAT_WINDOW, min_periods=FLOAT_MIN).median()
    wanted = pd.MultiIndex.from_frame(members[["signal_date", "code"]])
    weekly = smooth.loc[wanted.levels[0]].rename_axis(index="signal_date", columns="code").stack()
    result = members[["signal_date", "code"]].copy()
    result["float_mv"] = weekly.reindex(wanted).to_numpy()
    result["above_zero"] = result.float_mv.gt(0)
    total = result.float_mv.groupby(result.signal_date).transform("sum")
    if not total.gt(0).all():
        raise ValueError("At least one signal has no usable FloatMV estimates")
    result["weight_float_mv"] = result.float_mv.fillna(0) / total
    result["weight_equal"] = 1 / result.groupby("signal_date").code.transform("size")
    return result


def benchmark_nav(weights, quotes, schedule, days):
    """Build zero-cost fractional-share research benchmarks from adjusted opens and closes."""
    codes = sorted(weights.code.unique())
    price = {}
    for field in ["open", "close"]:
        number = pd.to_numeric(quotes[field], errors="coerce")
        values = quotes[["date", "code"]].assign(price=number.where(np.isfinite(number) & number.gt(0)))
        price[field] = values.pivot(index="date", columns="code", values="price").reindex(index=days, columns=codes)
    rows = []
    shares = np.zeros((2, len(codes)))
    equity = np.ones(2)
    rebalance = schedule.set_index("execution_date").signal_date
    weekly = {day: group.set_index("code").reindex(codes)[["weight_float_mv", "weight_equal"]]
              .fillna(0).to_numpy().T for day, group in weights.groupby("signal_date")}
    latest_marks = price["close"].loc[price["close"].index < rebalance.index.min()].ffill().iloc[-1].to_numpy()
    for day in days[days >= rebalance.index.min()]:
        fresh_open = price["open"].loc[day].to_numpy()
        fresh_close = price["close"].loc[day].to_numpy()
        # Every fallback mark is available as of the timestamp being valued.
        adjusted_open = np.where(np.isfinite(fresh_open), fresh_open, latest_marks)
        excluded = 0
        dropped_weight = np.zeros(2)
        if day in rebalance:
            target = weekly[rebalance[day]].copy()
            unavailable = ~np.isfinite(adjusted_open)
            excluded = int((unavailable & (target.max(axis=0) > 0)).sum())
            dropped_weight = target[:, unavailable].sum(axis=1)
            target[:, unavailable] = 0
            total = target.sum(axis=1)
            if not (total > 0).all():
                raise ValueError(f"No opening marks for benchmark members on {day.date()}")
            target /= total[:, None]
            if shares.any():
                equity = shares @ np.nan_to_num(adjusted_open)
            shares = target * equity[:, None] / np.nan_to_num(adjusted_open, nan=1)
        valuation = np.where(np.isfinite(fresh_close), fresh_close, adjusted_open)
        if not (np.isfinite(valuation) | (shares.max(axis=0) == 0)).all():
            raise ValueError(f"Benchmark position without a mark on {day.date()}")
        latest_marks = valuation
        equity = shares @ np.nan_to_num(valuation)
        held = shares.max(axis=0) > 0
        stale_close = int((held & ~np.isfinite(fresh_close)).sum())
        stale_open = int((held & ~np.isfinite(fresh_open)).sum()) if day in rebalance else 0
        rows.append((day, *equity, day in rebalance, stale_open, stale_close, excluded, *dropped_weight))
    return pd.DataFrame(rows, columns=["date", "float_mv_nav", "equal_weight_nav", "rebalance",
                                        "stale_open_members", "stale_close_members", "unmarked_target_members",
                                        "unmarked_float_mv_weight", "unmarked_equal_weight"])


def audit_benchmarks(v3, weights, adj, official, schedule, days, out):
    bench = benchmark_nav(weights, adj, schedule, days)
    joined = bench.merge(official[["date", "close"]].rename(columns={"close": "h00300"}), on="date", how="left")
    anchor = joined.loc[joined.date.lt(EVAL_FIRST) & joined.h00300.notna()].tail(1)
    period = pd.concat([anchor, joined.loc[joined.date.ge(EVAL_FIRST)]], ignore_index=True)
    if len(anchor) != 1 or period.h00300.isna().any():
        raise ValueError("H00300 does not cover the evaluation period plus its anchor")
    daily = period.set_index("date")[["float_mv_nav", "equal_weight_nav", "h00300"]].pct_change().iloc[1:]
    rows = []
    for year in ["all", *range(2017, 2026)]:
        r = daily if year == "all" else daily.loc[daily.index.year == year]
        for col in ["float_mv_nav", "equal_weight_nav"]:
            excess = r[col] - r["h00300"]
            rows.append({"year": year, "benchmark": col, "sessions": len(r), "correlation": r[col].corr(r.h00300),
                         "annualized_tracking_error": np.sqrt(252) * excess.std(),
                         "benchmark_return": (1 + r[col]).prod() - 1, "h00300_return": (1 + r.h00300).prod() - 1})
    audit = pd.DataFrame(rows)
    v3.save(audit, out / "benchmark_tracking.csv")
    v3.save(joined, out / "benchmark_nav.parquet")
    return audit


def weight_coverage(weights):
    weekly = weights.groupby("signal_date", as_index=False).agg(
        members=("code", "size"), float_mv_members=("above_zero", "sum"), max_weight=("weight_float_mv", "max"),
        concentration=("weight_float_mv", lambda x: (x * x).sum()),
        above_hard_bound=("weight_float_mv", lambda x: x.gt(.09 + 1e-12).sum()))
    weekly["coverage"] = weekly.float_mv_members / weekly.members
    weekly["year"] = weekly.signal_date.dt.year
    annual = weekly.groupby("year", as_index=False).agg(weeks=("signal_date", "size"),
        mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"), max_weight=("max_weight", "max"),
        incompatible_weight_weeks=("above_hard_bound", lambda x: x.gt(0).sum()))
    return weekly, annual


def audit(v3, source, days, schedule, members, cache, out):
    from fetch_csi300_total_return import validate_history

    official = official_history(source, days)
    early_days = days[(days >= INDEX_FIRST) & (days < official.date.iat[0])]
    early_file = cache / "H00300_early.parquet"
    early_ready = not len(early_days) or early_file.exists()
    if len(early_days) and early_ready:
        vetted = validate_history(pd.read_parquet(early_file), early_days)
        official = pd.concat([vetted, official], ignore_index=True)
    raw, adj, values = read_bars(v3, source, cache, schedule, members, days)
    weights = proxy_weights(raw, members, days)
    weekly, annual = weight_coverage(weights)
    v3.save(weights, out / "benchmark_weights.parquet")
    v3.save(weekly, out / "float_mv_coverage_weekly.csv")
    v3.save(annual, out / "float_mv_coverage_by_year.csv")
    tracking = audit_benchmarks(v3, weights, adj, official, schedule, days, out)

    val_audit = []
    values["year"] = values.signal_date.dt.year
    for year, part in values.groupby("year"):
        for col in FIELDS:
            finite = np.isfinite(part[col]) & part.trade_active
            val_audit.append({"year": year, "field": col, "members": len(part),
                              "requested_member_rows": int(part[f"{col}_supplied"].sum()),
                              "active_member_rows": int(part.trade_active.sum()),
                              "finite_active": int(finite.sum()), "positive_active": int((finite & part[col].gt(0)).sum()),
                              "negative_active": int((finite & part[col].lt(0)).sum())})
    v3.save(pd.DataFrame(val_audit), out / "valuation_coverage.csv")
    v3.save(values, out / "valuation_panel.parquet")
    v3.save(official, out / "H00300_development.parquet")
    planned = valuation_tasks(members)
    supplied = values.groupby(["year", "code"])[[f"{name}_supplied" for name in FIELDS]].all().all(axis=1)
    pending = [key for key in planned if not supplied.get(key, False)]
    v3.save(pd.DataFrame(pending, columns=["year", "code"]), out / "valuation_missing_partitions.csv")
    status = {"development_max_date": str(days.max().date()), "early_h00300_missing_sessions": 0 if early_ready else len(early_days),
              "valuation_partitions_missing": len(pending), "valuation_partitions_expected": len(planned),
              "data_collection_required": not early_ready or bool(pending)}
    (out / "step0_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(annual.to_string(index=False))
    print(tracking.loc[tracking.year.eq("all")].to_string(index=False))
    print(json.dumps(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "probe", "fetch-index", "fetch-valuations"])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-csv", type=Path,
                        help="Verified historical H00300 CSV with date,csi300")
    args = parser.parse_args()
    root = args.project_root.resolve()
    v3, source, days, schedule, members = initialize(root)
    cache = args.cache or root / "data/raw/v3_1/development"
    out = args.output or root / "research/v3_1/results/step0"
    if args.action == "audit":
        audit(v3, source, days, schedule, members, cache, out)
    elif args.action == "probe":
        probe(v3, cache, out)
    elif args.action == "fetch-index":
        get_development_index(v3, source, days, cache, args.benchmark_csv)
    else:
        collect_valuations(v3, source, members, cache)


if __name__ == "__main__":
    main()

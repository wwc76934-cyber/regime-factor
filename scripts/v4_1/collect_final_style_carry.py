"""Recover the historical style cohorts at the year boundary; collect only their missing marks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from collect_final_2026 import ANCHOR, FIRST, LAST, ROOT, market, save_json, seal

sys.path.insert(0, str(ROOT / "scripts/v3_1"))
from build_v3_1_factors_and_labels import neutralized_valuation
from build_v3_factors_and_labels import stock_history
from v3_1_style_state import LOOKBACK


def trading_days(path):
    frame = pd.read_parquet(path)
    return pd.DatetimeIndex(pd.to_datetime(frame.loc[frame.is_trading_day.astype(str).eq("1"), "calendar_date"]))


def carry_plan(root):
    registration = seal(root)
    cache = root / "data/raw/v4_1/final_2026"
    dev = root / "data/raw/v3/development"
    ready = json.loads((cache / "collection_complete.json").read_text(encoding="utf-8"))
    if ready["configuration_sha256"] != registration["configuration_sha256"]:
        raise ValueError("Cache and freeze differ")
    collected = set(ready["stock_codes"])
    days = trading_days(dev / "calendar.parquet").union(trading_days(cache / "calendar.parquet"))
    history = days[days.year == 2025]
    member_files = {pd.Timestamp(f.stem): f for f in (dev / "members").glob("2025-*.parquet")}
    roster_dates = pd.DatetimeIndex(sorted(member_files))
    anchor_signal = roster_dates[-1]
    signals = [anchor_signal, *sorted(pd.Timestamp(f.stem) for f in (cache / "members").glob("*.parquet"))]
    formation = {}
    rosters = {}
    for signal in signals:
        base = days[days.get_loc(signal) - LOOKBACK]
        if base > anchor_signal:
            continue
        snapshot = roster_dates[roster_dates.searchsorted(base, side="right") - 1]
        if snapshot not in rosters:
            rosters[snapshot] = market.check_snapshot(pd.read_parquet(member_files[snapshot]), snapshot).code.tolist()
        formation[signal] = (base, snapshot)
    names = sorted(set().union(*rosters.values()))
    raw = stock_history(dev, {2025: names}, "raw", ["date", "code", "amount", "turn", "peTTM", "tradestatus"], "3")
    adjusted = stock_history(dev, {2025: names}, "adjusted", ["date", "code", "open", "high", "low", "close"], "1")
    prices = adjusted[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    good = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1) &
            prices.high.ge(prices.max(axis=1)) & prices.low.le(prices.min(axis=1)))
    close = adjusted.assign(close=prices.close.where(good)).pivot(index="date", columns="code", values="close").reindex(history)
    vol = close.pct_change(fill_method=None).rolling(60).std()
    momentum = close.shift(20) / close.shift(120) - 1
    for field in ["amount", "turn", "peTTM"]:
        raw[field] = pd.to_numeric(raw[field], errors="coerce")
    active = raw.tradestatus.eq("1")
    mv = (raw.amount / (raw.turn / 100)).where(active & raw.amount.gt(0) & raw.turn.gt(0) &
                                                raw.turn.le(100) & np.isfinite(raw.amount) & np.isfinite(raw.turn))
    raw = raw.assign(mv=mv, active=active)
    def wide(field):
        return raw.pivot(index="date", columns="code", values=field).reindex(history)
    cap = wide("mv").rolling(20, min_periods=10).median()
    pe, active = wide("peTTM"), wide("active").eq(True)
    rows = []
    anchor_returns = {}
    for signal, (base, snap) in formation.items():
        universe = rosters[snap]
        available = active.loc[base, universe] & close.loc[base, universe].notna()
        group = pd.DataFrame({"code": universe, "signal_date": base, "signal_bar_usable": available.to_numpy(),
                              "float_mv": cap.loc[base, universe].to_numpy(), "VOL_60": vol.loc[base, universe].to_numpy(),
                              "peTTM": pe.loc[base, universe].to_numpy()})
        group["value"] = neutralized_valuation(group, "peTTM")
        group["trend"] = momentum.loc[base, universe].where(available).to_numpy()
        for style in ("value", "trend"):
            ordered = group.dropna(subset=[style]).sort_values([style, "code"], ascending=[False, True])
            picks = ordered.code.head(len(ordered) // 3).tolist()
            if signal == anchor_signal:
                anchor_returns[style] = (close.ffill().loc[signal, picks] / close.loc[base, picks] - 1).mean()
            if signal >= FIRST:
                rows.extend((signal, base, snap, style, code) for code in picks if code not in collected)
    ref = pd.read_csv(root / "research/v3_1/results/step1/style_state_audit.csv", parse_dates=["signal_date"])
    prior = ref.loc[ref.signal_date.eq(anchor_signal)].iloc[0]
    for style, value in anchor_returns.items():
        np.testing.assert_allclose(value, prior[f"{style}_60d_return"], rtol=0, atol=1e-10)
    return pd.DataFrame(rows, columns=["signal_date", "formation_date", "roster_snapshot_date", "style", "code"])


def collect(root, plan_only):
    plan = carry_plan(root)
    cache = root / "data/raw/v4_1/final_2026"
    out = root / "research/v4_1/final_2026/audit"
    market.save(plan, out / "style_carry_required.csv")
    limits = plan.groupby("code").signal_date.max()
    print(f"Required: {len(limits)} former members, {plan.signal_date.nunique()} affected signal weeks", flush=True)
    print(limits.to_string(), flush=True)
    if plan_only:
        return
    folder = cache / "style_carry"
    folder.mkdir(parents=True, exist_ok=True)
    calendar = trading_days(cache / "calendar.parquet")
    session = market.BaoSession()
    parts = []
    try:
        for code, end in limits.items():
            path = folder / f"{code}.parquet"
            dates = calendar[(calendar >= FIRST) & (calendar <= end)]
            fields = ["date", "code", "open", "high", "low", "close", "adjustflag", "tradestatus"]
            def check(frame):
                if not (frame.code.eq(code).all() and frame.adjustflag.eq("1").all() and
                        pd.DatetimeIndex(frame.date).equals(dates) and frame.tradestatus.isin(["0", "1"]).all()):
                    raise ValueError(f"{code}: incomplete style carry partition through {end.date()}")
                prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
                valid = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1) &
                         prices.high.ge(prices.max(axis=1)) & prices.low.le(prices.min(axis=1)))
                if (frame.tradestatus.eq("1") & ~valid).any():
                    raise ValueError(f"{code}: invalid traded OHLC")
            if path.is_file():
                check(pd.read_parquet(path))
            else:
                frame = market.daily(session, code, dates[0], end, fields, "1", empty=False)
                check(frame)
                market.save(frame, path)
            parts.append({"code": code, "through": str(end.date()), "sessions": len(dates),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"{code} ready | {len(dates)} sessions", flush=True)
    finally:
        session.close()
    save_json({"configuration_sha256": seal(root)["configuration_sha256"], "partitions": parts},
              folder / "complete.json")
    print(f"Style carry ready: {len(parts)}/{len(limits)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--plan-only", action="store_true", help="Offline coverage check")
    args = parser.parse_args()
    collect(args.project_root.resolve(), args.plan_only)

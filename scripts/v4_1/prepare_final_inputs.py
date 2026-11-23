"""Build registered h5 features, realized labels and carry-in benchmarks offline."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile

from collect_final_2026 import ROOT, ANCHOR, FIRST, LAST, market, seal
from collect_final_style_carry import carry_plan, trading_days
import build_v3_factors_and_labels as prices
import build_v3_1_factors_and_labels as features
from prepare_v3_1_base import proxy_weights, benchmark_nav
from v3_1_style_state import build_style_state, STYLE

DEST = ROOT / "research/v4_1/final_2026"
WORK = DEST / "work"
OLD = ROOT / "research/v3_1/results"
KEY = features.KEY


def build():
    cfg = seal(ROOT)["configuration_sha256"]
    dev = ROOT / "data/raw/v3/development"
    cache = ROOT / "data/raw/v4_1/final_2026"
    carry = cache / "style_carry"
    manifest = json.loads((carry / "complete.json").read_text())
    plan = carry_plan(ROOT).groupby("code").signal_date.max()
    if manifest["configuration_sha256"] != cfg or {p["code"] for p in manifest["partitions"]} != set(plan.index):
        raise ValueError("Style carry registration mismatch")
    for p in manifest["partitions"]:
        data = carry / (p["code"] + ".parquet")
        if hashlib.sha256(data.read_bytes()).hexdigest() != p["sha256"] or pd.Timestamp(p["through"]) != plan[p["code"]]:
            raise ValueError(f"Invalid carry mark series {data}")
    def roster(path):
        return pd.concat([market.check_snapshot(pd.read_parquet(f), pd.Timestamp(f.stem)) for f in sorted(path)])
    prior = roster(dev.glob("members/2025-*.parquet"))
    recent = roster((cache / "members").glob("*.parquet"))
    last_week = prior.signal_date.max()
    rosters = pd.concat([prior, recent], ignore_index=True)
    members = rosters.loc[rosters.signal_date.ge(last_week)].sort_values(KEY)
    stamps = pd.DatetimeIndex(members.signal_date.unique())
    days = trading_days(dev / "calendar.parquet").union(trading_days(cache / "calendar.parquet"))
    days = days[(days.year >= 2024) & (days <= LAST)]
    daily_codes = set(json.loads((cache / "collection_complete.json").read_text())["stock_codes"])
    universe = daily_codes | set(prior.loc[prior.signal_date.ge("2025-09-01"), "code"])
    def path(year, basis, code):
        fname = Path(basis) / str(year) / (code + ".parquet")
        if year < 2026 and (dev / fname).is_file():
            return dev / fname
        return cache / fname
    tasks = {y: sorted(c for c in universe if path(y, "raw", c).is_file()) for y in (2024, 2025, 2026)}
    if set(tasks[2026]) != daily_codes:
        raise ValueError("Missing 2026 partition")
    def history(source, groups, basis, cols, flag):
        chunks = []
        for y, names in groups.items():
            files = [path(y, basis, c) for c in names]
            if y == 2026 and basis == "adjusted":
                files.extend(carry / (c + ".parquet") for c in plan.index)
            for f in files:
                a = pd.read_parquet(f, columns=cols + ["adjustflag"])
                chunks.append(a.loc[a.adjustflag.eq(flag), cols])
        return pd.concat(chunks, ignore_index=True)
    class IndexFiles:
        def __truediv__(self, item):
            if item == "price_index":
                return self
            year = int(Path(item).stem)
            return (dev if year < 2026 else cache) / "price_index" / item
    with patch.object(prices, "stock_history", history):
        panel, index = prices.panel_and_index(IndexFiles(), days, tasks)
        style = build_style_state(None, days, rosters, tasks, stamps, features.neutralized_valuation)
    w = proxy_weights(panel, members, days)
    extra = ["psTTM", "pcfNcfTTM"]
    vals = []
    for (year, code), group in members.groupby([members.signal_date.dt.year, "code"]):
        f = path(year, "raw", code)
        if not set(extra).issubset(ParquetFile(f).schema_arrow.names):
            part = Path("valuations") / str(year) / (code + ".parquet")
            f = ROOT / "data/raw/v3_1/development" / part
            if not f.is_file():
                f = cache / part
        a = pd.read_parquet(f, columns=["date", "code", "adjustflag", *extra])
        vals.append(a.loc[a.adjustflag.eq("3") & a.date.isin(group.signal_date)])
    vals = pd.concat(vals, ignore_index=True).rename(columns={"date":"signal_date"})
    vals[extra] = vals[extra].apply(pd.to_numeric, errors="coerce")
    frame, states = features.signals(panel, index, members, w.merge(vals[KEY+extra], on=KEY, how="left", validate="one_to_one"), days)
    states = states.merge(style[["signal_date", STYLE]], on="signal_date", validate="one_to_one")
    if not np.isfinite(states.drop(columns="signal_date")).all().all():
        raise ValueError("Missing holdout state")
    next_day = pd.Series(days, index=days).shift(-1)
    schedule = pd.DataFrame({"signal_date": stamps, "execution_date": stamps.map(next_day)})
    official = pd.concat([pd.read_parquet(OLD/"step0/H00300_development.parquet"),pd.read_parquet(cache/"H00300.parquet")])
    official = official.drop_duplicates("date").set_index("date").close
    with patch.object(features, "DEVELOPMENT_LAST", LAST), patch.object(features, "HORIZONS", (5,)):
        labels = features.build_labels(frame, panel, official, days, schedule)
    config = json.loads((OLD/"step1/v3_1_selected_factors.json").read_text())
    def extend(name, new, keys, check):
        old = pd.read_parquet(OLD / "step1" / name)
        if name == "labels.parquet":
            old = old.loc[old.horizon.eq(5)]
        if check:
            old_anchor = old.loc[old.signal_date.eq(last_week)].set_index(keys).sort_index()
            new_anchor = new.loc[new.signal_date.eq(last_week)].set_index(keys).sort_index()
            pd.testing.assert_index_equal(old_anchor.index, new_anchor.index)
            np.testing.assert_allclose(old_anchor[check], new_anchor[check], atol=1e-8, rtol=1e-8, equal_nan=True)
        return pd.concat([old.loc[old.signal_date.lt(last_week)], new], ignore_index=True)
    for name, new, keys, check in [
        ("factor_panel.parquet", frame, KEY, config["selected"] + ["VOL_20","BETA_120","IDIO_VAR_60"]),
        ("states.parquet", states, ["signal_date"], config["states"]),
        ("labels.parquet", labels, KEY, [])]:
        market.save(extend(name, new, keys, check), WORK / "step1" / name)
    shutil.copyfile(OLD/"step1/v3_1_selected_factors.json", WORK/"step1/v3_1_selected_factors.json")
    market.save(w, WORK/"weights.parquet")
    market.save(style, DEST/"audit/style_state.csv")
    cols = ["date","code","raw_open","raw_close","preclose","tradestatus","isST"]
    market.save(panel.loc[panel.date.ge(FIRST) & panel.raw_close.notna(),cols], WORK/"execution.parquet")
    adj = panel.loc[panel.adj_ohlc_ok.eq(True),["date","code","adj_open","adj_close"]].rename(columns={"adj_open":"open","adj_close":"close"})
    bench = benchmark_nav(w, adj, schedule.dropna(), days).set_index("date")
    cols = ["float_mv_nav","equal_weight_nav"]
    prior_nav = pd.read_parquet(OLD/"step0/benchmark_nav.parquet").set_index("date")
    bench[cols] *= prior_nav.loc[ANCHOR,cols].astype(float) / bench.loc[ANCHOR,cols].astype(float)
    overlap = bench.index[bench.index <= ANCHOR]
    np.testing.assert_allclose(bench.loc[overlap,cols], prior_nav.loc[overlap,cols], rtol=1e-8)
    bench["h00300"] = official.reindex(bench.index)
    market.save(bench.loc[ANCHOR:].reset_index(), WORK/"benchmarks.parquet")
    print(f"Boundary verified: {last_week.date()}, {len(stamps)-1} new signals, {len(bench.loc[FIRST:])} sessions")


if __name__ == "__main__":
    build()

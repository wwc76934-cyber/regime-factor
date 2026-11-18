"""Build point-in-time tier features, mature labels and descriptive sleeve returns."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


FACTORS = ["RET_5", "VOL_60", "CORR_PV_10"]
KEY = ["signal_date", "code"]
GROUP = ["signal_date", "vol_tier"]
TIERS = ["low_vol", "mid_vol", "high_vol"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def make_tiers(candidates, labels, schedule):
    require(not candidates.duplicated(KEY).any() and not labels.duplicated(KEY).any(), "Duplicate stock/week")
    usable = candidates["signal_bar_usable"].eq(True) & np.isfinite(candidates[FACTORS]).all(axis=1)
    tiered = candidates.loc[usable, KEY + FACTORS].sort_values(["signal_date", "VOL_60", "code"]).copy()
    weekly = tiered.groupby("signal_date")
    tiered["q_vol"] = (weekly.cumcount() + 1) / weekly["code"].transform("size")
    tiered["vol_tier"] = pd.cut(tiered["q_vol"], [0, 1/3, 2/3, 1], labels=TIERS).astype("str")
    require(tiered.groupby(GROUP).size().min() >= 90, "Unexpected tier coverage")
    for factor in FACTORS:
        ranks = (-tiered[factor]).groupby([tiered[key] for key in GROUP]).rank(method="average")
        size = tiered.groupby(GROUP)[factor].transform("size")
        tiered[f"tier_pct_{factor}"] = (ranks - .5) / size - .5
    tiered = tiered.merge(schedule[["signal_date", "execution_date"]], on="signal_date", validate="many_to_one")
    require(tiered["execution_date"].gt(tiered["signal_date"]).all(), "Invalid execution dates")
    tiered = tiered.merge(labels[KEY + ["exit_date", "forward_return"]], on=KEY, how="left", validate="one_to_one")
    valid = tiered["forward_return"].where(np.isfinite(tiered["forward_return"]))
    require(tiered.loc[valid.notna(), "exit_date"].gt(tiered.loc[valid.notna(), "execution_date"]).all(),
            "Label exit must follow the execution open")
    by_tier = valid.groupby([tiered[key] for key in GROUP])
    tiered["tier_target_y"] = (by_tier.rank(method="average") - .5) / by_tier.transform("count") - .5
    for column in ["tier_target_y", *[f"tier_pct_{factor}" for factor in FACTORS]]:
        require(tiered.groupby(GROUP)[column].mean().dropna().abs().lt(1e-12).all(), "Tier rank centering failed")
    return tiered


def sleeve_nav(tiered, panel, benchmark, first, end):
    dates = benchmark.index[benchmark.index <= end]
    anchor = dates[dates < first][-1]
    dates = dates[dates >= anchor]
    benchmark = benchmark.loc[dates] / benchmark.loc[anchor]
    require(dates.is_unique and benchmark.gt(0).all() and dates[-1] == end, "Bad benchmark calendar")
    close = panel.pivot(index="date", columns="code", values="adj_close").reindex(dates)
    close = close.where(close.gt(0) & np.isfinite(close))
    found = close.ffill()
    rows = []
    for tier in TIERS:
        choices = {date: list(week["code"]) for date, week in
                   tiered.loc[tiered["vol_tier"].eq(tier)].groupby("execution_date")}
        base, positions, excluded, stale = 1.0, None, 0, 0
        for date in dates:
            if positions is not None:
                marks = found.loc[date, positions.index]
                stale = int(close.loc[date, positions.index].isna().sum())
                base = float(positions @ marks)
            if date in choices:
                current = close.loc[date, choices[date]].dropna()
                excluded = len(choices[date]) - len(current)
                require(len(current) >= 85, "Too few prices for the descriptive sleeve")
                positions = base / (len(current) * current)
            else:
                excluded = 0
            rows.append((date, tier, base, benchmark.loc[date], excluded, stale))
    return pd.DataFrame(rows, columns=["date", "vol_tier", "nav", "benchmark_nav", "unpriced_at_rebalance", "stale_marks"])


def summarize(nav):
    results = []
    for tier, group in nav.groupby("vol_tier"):
        daily = group.set_index("date")[["nav", "benchmark_nav"]].pct_change().dropna()
        for year, period in daily.groupby(daily.index.year):
            weekly = (1 + period).groupby(period.index.to_period("W-SUN")).prod() - 1
            variance = weekly["benchmark_nav"].var()
            results.append({"year": year, "vol_tier": tier,
                            "tier_return": (1 + period["nav"]).prod() - 1,
                            "benchmark_return": (1 + period["benchmark_nav"]).prod() - 1,
                            "realized_beta": weekly["nav"].cov(weekly["benchmark_nav"]) / variance,
                            "beta_weeks": len(weekly)})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root / "research/v2_regime_tiered/results"
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    from v1_holdout_inputs import build_inputs

    processed = root / "data/processed"
    factors = processed / "factors"
    candidates = pd.read_parquet(factors / "csi300_weekly_candidates.parquet",
                                 columns=KEY + ["signal_bar_usable", *FACTORS])
    labels = pd.read_parquet(factors / "csi300_weekly_forward_labels.parquet")
    schedule = pd.read_parquet(processed / "universe/csi300_weekly_schedule.parquet")
    dev = make_tiers(candidates, labels, schedule)
    require(dev["signal_date"].dt.year.le(2024).all(), "Unexpected development dates")
    panel = pd.read_parquet(processed / "csi300_daily_panel.parquet", columns=["date", "code", "adj_close"])
    index = pd.read_parquet(root / "data/raw/index/csi300_H00300_development.parquet")
    bench = index.set_index("date")["close"].sort_index()
    dev_nav = sleeve_nav(dev, panel, bench, pd.Timestamp("2021-01-18"), pd.Timestamp("2024-12-31"))
    cache = root / "data/raw/holdout_2025"
    days, schedule, members, panel, _, states, labels, _ = build_inputs(cache)
    from build_candidate_factors import build_candidate_panel

    candidates, _ = build_candidate_panel(panel, members[KEY], days)
    post = make_tiers(candidates, labels, schedule)
    bench = pd.read_parquet(cache / "H00300.parquet").set_index("date")["close"].sort_index()
    post_nav = sleeve_nav(post, panel, bench, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    for name, frame in [("development", dev), ("review_2025", post)]:
        frame.to_parquet(output / f"tiered_{name}.parquet", index=False)
    states.reset_index().to_parquet(output / "states_2025.parquet", index=False)
    nav = pd.concat([dev_nav.assign(sample="development"), post_nav.assign(sample="retrospective_2025")])
    nav.to_parquet(output / "tier_reference_daily_nav.parquet", index=False)
    annual = pd.concat([summarize(dev_nav), summarize(post_nav)]).sort_values(["year", "vol_tier"])
    annual.to_csv(output / "tier_annual_returns_beta.csv", index=False, encoding="utf-8-sig")
    print(annual.round(4).to_string(index=False))
    print("Rows:", len(dev), len(post), "| Output:", output)


if __name__ == "__main__":
    main()

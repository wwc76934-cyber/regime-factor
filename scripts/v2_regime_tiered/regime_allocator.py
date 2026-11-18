"""Assign 30 slots using frozen state rules and actual holdings."""

from math import floor

import numpy as np
import pandas as pd

from tiered_data import TIERS, require


QUOTAS = {"defensive": (18, 8, 4), "neutral": (10, 10, 10), "offensive": (5, 10, 15)}
STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d"]
BEAR_BREADTH = .4217
BULL_BREADTH = .50
VOL_REPORT_THRESHOLD = .010875


def allocation_table(states, sleeve_nav):
    frame = states[["signal_date", *STATES]].sort_values("signal_date").copy()
    require(frame["signal_date"].is_unique and np.isfinite(frame[STATES]).all().all(), "Invalid market states")
    trend, vol, breadth = (frame[state] for state in STATES)
    defensive = trend.le(0) & breadth.lt(BEAR_BREADTH)
    offensive = trend.gt(0) & breadth.ge(BULL_BREADTH)
    frame["regime"] = np.select([defensive, offensive], ["defensive", "offensive"], default="neutral")
    frame["risk_on_score"] = trend.gt(0).astype(int) + breadth.ge(BULL_BREADTH).astype(int) + vol.lt(
        VOL_REPORT_THRESHOLD).astype(int)
    nav_groups = {}
    for sample, rows in sleeve_nav.groupby("sample"):
        series = rows.pivot(index="date", columns="vol_tier", values="nav").sort_index()
        nav_groups[sample] = (series["high_vol"].pct_change(20) - series["low_vol"].pct_change(20))
    frame["trailing_high_minus_low_20d"] = np.nan
    for sample, series in nav_groups.items():
        if sample == "development":
            mask = frame["signal_date"].dt.year.le(2024)
        else:
            mask = frame["signal_date"].dt.year.eq(2025)
        frame.loc[mask, "trailing_high_minus_low_20d"] = series.reindex(
            frame.loc[mask, "signal_date"]).to_numpy()
    for index, tier in enumerate(TIERS):
        frame[f"slots_{tier}"] = frame["regime"].map(lambda regime: QUOTAS[regime][index])
    require(frame[[f"slots_{tier}" for tier in TIERS]].sum(axis=1).eq(30).all(), "Quota does not sum to 30")
    return frame


def select_targets(scores, allocations, dynamic, first_live):
    states = allocations.set_index("signal_date")["regime"]
    schedules = {}
    audit = []
    for day, week in scores.groupby("execution_date", sort=True):
        signal = week["signal_date"].iat[0]
        require(week["signal_date"].nunique() == 1 and week["code"].is_unique and signal < day, "Invalid signal schedule")
        if dynamic and day >= first_live:
            regime = states.at[signal]
        else:
            regime = "neutral"
        slots = QUOTAS[regime]
        score_col = "score_ema05" if day >= first_live else "score"
        tiers = {key: group.sort_values([score_col, "code"], ascending=[False, True])["code"].tolist()
                 for key, group in week.groupby("vol_tier")}
        require(set(tiers) == set(TIERS) and all(len(tiers[key]) >= floor(1.5 * count)
                for key, count in zip(TIERS, slots, strict=True)), "Insufficient stocks for tier quota")
        schedules[day] = (regime, tiers)

    def pick(day, positions, record):
        regime, names_by_tier = schedules[day]
        selected = []
        for tier, count in zip(TIERS, QUOTAS[regime], strict=True):
            ranked = names_by_tier[tier]
            cutoff = floor(1.5 * count)
            retain = [code for code in ranked[:cutoff] if positions.get(code, 0) > 0][:count]
            picks = list(dict.fromkeys(retain + ranked))[:count]
            selected.extend(picks)
            if record:
                lookup = {code: rank for rank, code in enumerate(ranked, 1)}
                audit.extend((day, regime, tier, count, cutoff, code, lookup[code], code in retain) for code in picks)
        require(len(selected) == len(set(selected)) == 30, "Expected thirty distinct target positions")
        return {code: rank for rank, code in enumerate(selected, 1)}

    schedule = {day: pick(day, {}, False) for day in schedules}
    return schedule, lambda day, _original, positions: pick(day, positions, True), audit

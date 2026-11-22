"""Build and screen point-in-time, multi-horizon V3.1 development features."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v3_1_base import DEVELOPMENT_LAST, initialize
from v3_1_style_state import STYLE, build_style_state


KEY = ["signal_date", "code"]
VALUE = {"EP_TTM": "peTTM", "BP_MRQ": "pbMRQ", "SP_TTM": "psTTM", "CFP_TTM": "pcfNcfTTM"}
GROUPS = {
    "fast": ["RET_5", "CORR_PV_10", "TURN_20", "TURN_RATIO_5_20", "IDIO_VOL_20", "AMIHUD_20"],
    "trend": ["MOM_60_SKIP10", "HIGH_52W_PROX", "SHARPE_60", "BIAS_20", "MOM_120_SKIP20"],
    "value": list(VALUE),
    "risk_reference": ["BETA_120", "VOL_60"],
}
HORIZONS = (5, 10, 20)
NATURAL = {"fast": 5, "trend": 20, "value": 20, "risk_reference": 5}
STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d", "tier_vol_spread_20d"]
SCREEN_END = pd.Timestamp("2018-12-31")
VALIDATION_END = pd.Timestamp("2022-12-31")
QUOTAS = {"fast": 2, "trend": 2, "value": 2}


def rolling_risk(stock_r, market_r, length, minimum):
    right = stock_r.notna().mul(market_r, axis=0).where(stock_r.notna())
    left = stock_r.where(market_r.notna(), axis=0)
    market_var = right.rolling(length, min_periods=minimum).var().where(lambda x: x.gt(0))
    covariance = left.rolling(length, min_periods=minimum).cov(right)
    beta = covariance / market_var
    residual_variance = left.rolling(length, min_periods=minimum).var() - covariance.pow(2) / market_var
    return beta, residual_variance.clip(lower=0)


def neutralized_valuation(frame, column):
    """Residualize positive inverse valuation ratios on size and volatility at each signal."""
    result = pd.Series(index=frame.index, dtype=float)
    for _, week in frame.groupby("signal_date", sort=False):
        value = (1 / week[column].where(week[column].gt(0))).replace([np.inf, -np.inf], np.nan)
        cap = np.log(week.float_mv.where(week.float_mv.gt(0)))
        regressors = pd.DataFrame({"size": cap, "vol": week.VOL_60})
        eligible = week.signal_bar_usable & value.notna() & regressors.notna().all(axis=1)
        if eligible.sum() < 50:
            continue
        x = regressors.loc[eligible]
        scale = x.std().replace(0, 1)
        design = np.c_[np.ones(len(x)), ((x - x.mean()) / scale).to_numpy()]
        y = value.loc[eligible]
        y = y.clip(y.quantile(.01), y.quantile(.99)).to_numpy()
        result.loc[x.index] = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return result


def signals(panel, index, members, inputs, days):
    from build_candidate_factors import rolling_corr

    def wide(name):
        return panel.pivot(index="date", columns="code", values=name).reindex(days)

    good = wide("adj_ohlc_ok").eq(True)
    close = wide("adj_close").where(good)
    high = wide("adj_high").where(good)
    observed = wide("signal_bar_usable").eq(True)
    market = index.close.pct_change(fill_method=None)
    stock = close.pct_change(fill_method=None)
    vol20 = stock.rolling(20).std()
    vol60 = stock.rolling(60).std()
    beta, _ = rolling_risk(stock, market, 120, 100)
    _, idio20 = rolling_risk(stock, market, 20, 16)
    _, idio60 = rolling_risk(stock, market, 60, 50)
    volume = wide("volume").where(lambda x: x.ge(0) & np.isfinite(x))
    amount = wide("amount").where(lambda x: x.gt(0) & np.isfinite(x))
    turn = wide("turn").where(observed).where(lambda x: x.ge(0) & x.le(100))
    turn20 = turn.rolling(20).mean()
    frame = members[KEY].merge(inputs, on=KEY, validate="one_to_one", how="left")
    stamps = pd.DatetimeIndex(frame.signal_date.unique()).sort_values()
    target = pd.MultiIndex.from_frame(frame[KEY])

    def take(values):
        weekly = values.reindex(stamps).rename_axis(index="signal_date", columns="code").stack()
        return weekly.reindex(target).to_numpy()

    def pct(period):
        return close.pct_change(periods=period, fill_method=None)

    estimates = {
        "signal_bar_usable": observed,
        "RET_5": pct(5), "RET_20": pct(20),
        "CORR_PV_10": rolling_corr(close, np.log1p(volume), 10),
        "TURN_20": turn20,
        "TURN_RATIO_5_20": turn.rolling(5).mean() / turn20.where(turn20.gt(0)),
        "IDIO_VOL_20": np.sqrt(idio20), "IDIO_VAR_60": idio60,
        "AMIHUD_20": (stock.abs() / amount).rolling(20).mean(),
        "MOM_60_SKIP10": close.shift(10) / close.shift(60) - 1,
        "MOM_120_SKIP20": close.shift(20) / close.shift(120) - 1,
        "HIGH_52W_PROX": close / high.rolling(240).max(),
        "SHARPE_60": stock.rolling(60).mean() / vol60.where(vol60.gt(0)),
        "BIAS_20": close / close.rolling(20).mean() - 1,
        "VOL_20": vol20, "VOL_60": vol60,
        "BETA_120": beta.clip(.2, 2.2).where(np.isfinite(beta)),
        "peTTM": wide("peTTM"), "pbMRQ": wide("pbMRQ"),
    }
    for name, values in estimates.items():
        frame[name] = take(values)
    frame["signal_bar_usable"] = frame.signal_bar_usable == True
    for name, col in VALUE.items():
        frame[name] = neutralized_valuation(frame, col)
    eligible = frame.loc[frame.signal_bar_usable & frame.VOL_60.notna() & frame.RET_20.notna()]
    eligible = eligible.sort_values(["signal_date", "VOL_60", "code"])
    groups = eligible.groupby("signal_date")
    index_rank = (groups.cumcount() + .5) / groups.code.transform("size")
    low = eligible.loc[index_rank.lt(1/3)].groupby("signal_date").RET_20.mean()
    high_vol = eligible.loc[index_rank.ge(2/3)].groupby("signal_date").RET_20.mean()
    breadth = eligible.RET_20.gt(0).groupby(eligible.signal_date).mean()
    states = pd.DataFrame({
        STATES[0]: index.close.pct_change(periods=20, fill_method=None),
        STATES[1]: market.rolling(20).std(),
        STATES[2]: breadth, STATES[3]: high_vol - low,
    }).reindex(stamps).rename_axis("signal_date").reset_index()
    return frame.drop(columns=["peTTM", "pbMRQ", *[VALUE[f] for f in ("SP_TTM", "CFP_TTM")]]), states


def build_labels(frame, panel, index, calendar, schedule):
    schedule = schedule[["signal_date", "execution_date"]].copy()
    close = panel.loc[panel.signal_bar_usable, ["date", "code", "adj_close"]]
    entry = schedule.rename(columns={"execution_date": "entry_date"})
    from_entry = frame[KEY + ["signal_bar_usable", "BETA_120", "VOL_20"]].merge(entry, on="signal_date", validate="many_to_one")
    from_entry = from_entry.merge(close.rename(columns={"date": "entry_date", "adj_close": "entry_close"}),
                                  on=["entry_date", "code"], validate="many_to_one", how="left")
    results = []
    for h in HORIZONS:
        mapper = pd.Series(calendar, index=calendar).shift(-h)
        work = from_entry.assign(exit_date=from_entry.entry_date.map(mapper))
        work = work.merge(close.rename(columns={"date": "exit_date", "adj_close": "exit_close"}),
                          on=["exit_date", "code"], validate="many_to_one", how="left")
        stock = (work.exit_close / work.entry_close - 1).where(work.signal_bar_usable)
        benchmark = work.exit_date.map(index) / work.entry_date.map(index) - 1
        raw = ((stock - work.BETA_120 * benchmark) / work.VOL_20.clip(lower=.01)).replace([np.inf, -np.inf], np.nan)
        group = raw.groupby(work.signal_date)
        mean, sd = group.transform("mean"), group.transform("std")
        trimmed = raw.clip(mean - 3 * sd, mean + 3 * sd)
        cross = trimmed.groupby(work.signal_date)
        response = (trimmed - cross.transform("mean")) / cross.transform("std").where(lambda x: x.gt(0))
        results.append(work[KEY + ["entry_date", "exit_date"]].assign(
            horizon=h, stock_return=stock, index_return=benchmark, risk_adjusted_target=response))
    result = pd.concat(results, ignore_index=True)
    ok = result.loc[result.exit_date.notna()]
    if not (ok.exit_date.le(DEVELOPMENT_LAST) & ok.exit_date.gt(ok.entry_date) &
            ok.entry_date.gt(ok.signal_date)).all():
        raise ValueError("Invalid label calendar")
    return result


def split_weeks(day, end):
    if day <= SCREEN_END and end <= SCREEN_END:
        return "screen"
    if SCREEN_END < day <= VALIDATION_END and end <= VALIDATION_END:
        return "validation"
    return None


def evaluate_factors(frame, labels):
    rows = []
    candidates = [(group, name) for group, names in GROUPS.items() for name in names]
    joined = labels.merge(frame[KEY + ["signal_bar_usable", *[name for _, name in candidates]]],
                          on=KEY, validate="many_to_one")
    for (h, day), sample in joined.groupby(["horizon", "signal_date"], sort=True):
        end = sample.exit_date.iat[0]
        split = split_weeks(day, end)
        if split is None or pd.isna(end) or sample.index_return.isna().all():
            continue
        benchmark = sample.index_return.iat[0]
        eligible = sample.loc[sample.signal_bar_usable]
        base = eligible.stock_return.mean()
        if eligible.stock_return.notna().sum() < 100:
            continue
        for group, name in candidates:
            observed = eligible.loc[eligible[name].notna(), ["code", name, "stock_return", "risk_adjusted_target"]]
            if len(observed) < 50 or observed[name].nunique() < 5:
                continue
            pairs = observed.dropna(subset=["stock_return"])
            risk_pairs = observed.dropna(subset=["risk_adjusted_target"])
            ic = pairs[name].rank().corr(pairs.stock_return.rank()) if len(pairs) > 1 else np.nan
            risk_ic = risk_pairs[name].rank().corr(risk_pairs.risk_adjusted_target.rank()) if len(risk_pairs) > 1 else np.nan
            for direction in [1, -1]:
                ordered = observed.sort_values([name, "code"], ascending=[direction == 1, True])
                top = observed.sort_values([name, "code"], ascending=[direction == -1, True]).head(30).stock_return
                if top.count() < 24:
                    continue
                buckets = np.floor(np.arange(len(ordered)) * 5 / len(ordered)).astype(int)
                quintiles = ordered.groupby(buckets).stock_return.mean().reindex(range(5))
                rows.append({"signal_date": day, "horizon": h, "split": split, "group": group,
                             "factor": name, "direction": direction, "eligible": len(eligible), "factor_n": len(observed),
                             "top30_realized_n": int(top.count()), "rank_ic": direction * ic, "risk_target_rank_ic": direction * risk_ic,
                             "top30_return": top.mean(), "top30_excess_equal": top.mean() - base,
                             "top30_excess_H00300": top.mean() - benchmark,
                             **{f"q{i+1}": quintiles.iloc[i] for i in range(5)}})
    return pd.DataFrame(rows)


def summarize(weekly):
    stats = weekly.groupby(["group", "factor", "direction", "horizon", "split"], as_index=False).agg(
        weeks=("signal_date", "size"), average_coverage=("factor_n", "mean"), top30_realized_n=("top30_realized_n", "mean"),
        rank_ic=("rank_ic", "mean"), risk_target_rank_ic=("risk_target_rank_ic", "mean"),
        top30_return=("top30_return", "mean"), top30_excess_equal=("top30_excess_equal", "mean"),
        top30_excess_H00300=("top30_excess_H00300", "mean"), **{f"q{i}": (f"q{i}", "mean") for i in range(1, 6)})
    quintiles = [f"q{i}" for i in range(1, 6)]
    stats["quintile_monotonicity"] = stats[quintiles].rank(axis=1).apply(
        lambda row: row.corr(pd.Series(range(1, 6), index=quintiles)), axis=1)
    return stats


def correlations(frame):
    cols = [name for names in GROUPS.values() for name in names]
    source = frame.loc[frame.signal_bar_usable & frame.signal_date.le(VALIDATION_END)]
    weeks = [g[cols].corr(method="spearman", min_periods=50).stack() for _, g in source.groupby("signal_date")]
    return pd.concat(weeks, axis=1).median(axis=1).rename("median_spearman")


def coverage_by_split(frame):
    frame = frame.loc[frame.signal_bar_usable & frame.signal_date.le(VALIDATION_END)].copy()
    fields = [name for names in GROUPS.values() for name in names]
    observed = frame[fields].notna().groupby(frame.signal_date).mean()
    period = pd.Series(np.where(observed.index <= SCREEN_END, "screen", "validation"), index=observed.index)
    return observed.groupby(period).mean().T


def choose(stats, corr, coverage):
    natural = stats.loc[stats.apply(lambda row: row.horizon == NATURAL[row.group], axis=1)].copy()
    rows = []
    for group, names in GROUPS.items():
        for name in names:
            entries = natural.loc[natural.factor.eq(name)]
            initial = entries.loc[entries.split.eq("screen")]
            if group == "trend":
                initial = initial.loc[initial.direction.eq(1)]
            columns = ["top30_excess_equal", "top30_excess_H00300"]
            row = {"factor": name, "group": group, "natural_horizon": NATURAL[group],
                   "screen_effective_coverage": coverage.at[name, "screen"],
                   "validation_effective_coverage": coverage.at[name, "validation"]}
            covered = min(row["screen_effective_coverage"], row["validation_effective_coverage"]) >= .85
            if initial.empty:
                rows.append({**row, "direction": 0, "passed": False, "reason": "insufficient_initial_coverage"})
                continue
            leader = initial.sort_values(["top30_excess_equal", "direction"], ascending=False).iloc[0]
            matching = entries.loc[entries.direction.eq(leader.direction)].set_index("split")
            output = {}
            for split in ["screen", "validation"]:
                cols = ["weeks", "average_coverage", "top30_realized_n", "rank_ic", "risk_target_rank_ic",
                        "top30_return", *columns, "quintile_monotonicity"]
                for col in cols:
                    output[f"{split}_{col}"] = matching.at[split, col] if split in matching.index else np.nan
            screens = matching.reindex(["screen", "validation"])
            enough = bool(screens.weeks.gt(0).all())
            excess = screens.top30_excess_equal
            combined = float(np.average(excess, weights=screens.weeks)) if enough else np.nan
            stable_excess = bool(excess.gt(0).all())
            stable_ic = bool(screens.risk_target_rank_ic.ge(.015).all())
            monotonic = (screens.at["validation", "quintile_monotonicity"] >= .6 - 1e-12 or
                         (screens.quintile_monotonicity.notna().all() and
                          screens.quintile_monotonicity.mean() >= .4 - 1e-12))
            value_excess = combined > 0 and excess.max() >= .002 and excess.at["screen"] > 0
            checks = {
                "risk_reference": group != "risk_reference",
                "missing_split_observations": enough,
                "coverage_below_85_percent": covered,
            }
            if group == "value":
                checks["value_excess_below_gate"] = value_excess
            else:
                checks.update(stable_equal_excess_failed=stable_excess, stable_risk_ic_failed=stable_ic,
                              stable_monotonicity_failed=monotonic)
            failures = [key for key, passed in checks.items() if not passed]
            rows.append({**row, **output, "direction": int(leader.direction), "combined_top30_excess_equal": combined,
                         "passed": not failures, "reason": failures[0] if failures else "eligible",
                         "failures": "|".join(failures)})
    audit = pd.DataFrame(rows)
    qualified = audit.loc[audit.passed].set_index("factor")

    selected = []
    counts = {}
    for group, quota in QUOTAS.items():
        rows = qualified.loc[qualified.group.eq(group)].sort_values("combined_top30_excess_equal", ascending=False)
        counts[group] = len(rows)
        retained = 0
        for name in rows.index:
            if retained >= quota:
                break
            if all(abs(corr.get((name, other), np.inf)) < .70 for other in selected):
                selected.append(name)
                retained += 1
    balanced = len(selected) == sum(QUOTAS.values())
    audit["selected"] = audit.factor.isin(selected) if balanced else False
    pairs = [abs(corr.at[a, b]) for a, b in combinations(selected, 2)]
    decision = {
        "status": "go" if balanced else "no_go",
        "selected": selected if balanced else [],
        "directions": {name: int(qualified.at[name, "direction"]) for name in selected} if balanced else {},
        "quota": QUOTAS,
        "qualified_counts": counts,
        "provisional_only": selected if not balanced else [],
        "horizons": list(HORIZONS),
        "states": [*STATES, STYLE],
        "selection_end": str(VALIDATION_END.date()),
        "max_abs_median_spearman": max(pairs, default=None),
    }
    return audit, decision


def value_trend_correlations(weekly, audit):
    directions = audit[["factor", "direction"]]
    active = weekly.loc[weekly.horizon.eq(20) & weekly.group.isin(["value", "trend"])]
    active = active.merge(directions, on=["factor", "direction"], validate="many_to_one")
    rows = []
    for split in ["all", "screen", "validation"]:
        data = active if split == "all" else active.loc[active.split.eq(split)]
        matrix = data.pivot(index="signal_date", columns="factor", values="top30_excess_equal")
        for v in GROUPS["value"]:
            for t in GROUPS["trend"]:
                pairs = matrix.reindex(columns=[v, t]).dropna()
                rho = pairs[v].corr(pairs[t]) if len(pairs) > 1 else np.nan
                rows.append((split, v, t, len(pairs), rho))
    return pd.DataFrame(rows, columns=["split", "value_factor", "trend_factor", "overlapping_weeks", "pearson_excess_return"])


def run(root, step0, out, reuse_panel=None):
    v3, source, days, schedule, members = initialize(root)
    from build_v3_factors_and_labels import panel_and_index

    status = json.loads((step0 / "step0_status.json").read_text(encoding="utf-8"))
    v3.need(not status["data_collection_required"], "Complete the offline Step 0 audit before factor evaluation")
    index_total = pd.read_parquet(step0 / "H00300_development.parquet").set_index("date").close
    v3.need(index_total.index.equals(days[days.year >= 2015]), "H00300 must cover all 2015–2025 sessions")
    w = pd.read_parquet(step0 / "benchmark_weights.parquet")[KEY + ["float_mv", "weight_float_mv", "weight_equal"]]
    vals = pd.read_parquet(step0 / "valuation_panel.parquet")[KEY + ["psTTM", "pcfNcfTTM"]]
    inputs = w.merge(vals, on=KEY, validate="one_to_one")
    tasks = v3.ranges(schedule, members)
    if reuse_panel is None:
        panel, index = panel_and_index(source, days, tasks)
        frame, states = signals(panel, index, members, inputs, days)
        print(f"Built {len(frame):,} point-in-time feature rows", flush=True)
        labels = build_labels(frame, panel, index_total, days, schedule)
        coverage = labels.groupby("horizon", as_index=False).agg(
            member_rows=("code", "size"), raw_labeled=("stock_return", "count"),
            risk_adjusted_labeled=("risk_adjusted_target", "count"), last_exit_date=("exit_date", "max"))
        for name, data in [("factor_panel.parquet", frame), ("labels.parquet", labels), ("label_coverage.csv", coverage)]:
            v3.save(data, out / name)
        frame = frame.loc[frame.signal_date.le(VALIDATION_END)]
        weekly = evaluate_factors(frame, labels.loc[labels.exit_date.le(VALIDATION_END)])
        del panel, labels
    else:
        frame = pd.read_parquet(reuse_panel / "factor_panel.parquet", filters=[("signal_date", "<=", VALIDATION_END)])
        states = pd.read_parquet(reuse_panel / "states.parquet").drop(columns=STYLE, errors="ignore")
        weekly = pd.read_parquet(reuse_panel / "weekly_factor_metrics.parquet",
                                 filters=[("split", "in", ["screen", "validation"]),
                                          ("signal_date", "<=", VALIDATION_END)])
    v3.need(weekly.signal_date.le(VALIDATION_END).all(), "Screening includes out-of-window outcomes")
    stats = summarize(weekly)
    corr = correlations(frame)
    audit, decision = choose(stats, corr, coverage_by_split(frame))
    print(f"Factor gate before style-state check: {decision['status']} {decision['selected']}", flush=True)
    style = build_style_state(source, days, members, tasks, schedule.signal_date, neutralized_valuation)
    states = states.merge(style[["signal_date", STYLE]], on="signal_date", validate="one_to_one")
    miss = states[STYLE].isna()
    early_missing = float(miss.loc[states.signal_date.le(VALIDATION_END)].mean())
    decision["style_missing_rate"] = float(miss.mean())
    decision["style_missing_rate_2015_2022"] = early_missing
    if miss.mean() >= .05 or early_missing >= .05:
        decision["status"] = "no_go"
        decision["provisional_only"] = decision["selected"] or decision["provisional_only"]
        decision.update(selected=[], directions={})
        audit["selected"] = False
    if decision["status"] == "go":
        decision["provisional_7f"] = [*decision["selected"], "TURN_20"]
        decision["provisional_7f_directions"] = {**decision["directions"], "TURN_20": -1}
    corr = corr.rename_axis(index=["factor", "other"]).reset_index()
    corr["abs_median_spearman"] = corr.median_spearman.abs()
    for name, df in [("states.parquet", states), ("style_state_audit.csv", style),
                     ("weekly_factor_metrics.parquet", weekly), ("v3_1_factor_audit.csv", audit),
                     ("factor_horizon_summary.csv", stats),
                     ("value_trend_excess_correlations.csv", value_trend_correlations(weekly, audit)),
                     ("factor_correlations.csv", corr)]:
        v3.save(df, out / name)
    for file in ["features.json", "v3_1_selected_factors.json"]:
        (out / file).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(audit[["factor", "group", "direction", "reason", "selected"]].to_string(index=False))
    print(json.dumps(decision))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--step0", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--reuse-panel", type=Path, help="Reuse existing factor, state and weekly metric artifacts")
    args = p.parse_args()
    results = args.project_root / "research/v3_1/results"
    run(args.project_root.resolve(), args.step0 or results / "step0", args.output or results / "step1", args.reuse_panel)

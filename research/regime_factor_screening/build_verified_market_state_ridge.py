import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_all_factor_regimes import FACTOR_DIR, KEYS, SPLITS, STATES, STATES_FILE, icir, require, write_table


WALK = Path("data/processed/portfolio/walk_forward")
MIN_WEEKS = 52
WINDOW_WEEKS = 104
STATE_Z_LIMIT = 3.0
STATE_PENALTIES = [0.5, 1.0, 2.0]
LEGACY = ["equal_weight", "rolling_ridge", "state_ridge"]


def normalize_states(history, current, orthogonalize):
    history, current = history.copy(), current.copy()
    slope = np.nan
    if orthogonalize:
        trend = history[:, 0]
        centered = trend - trend.mean()
        denominator = centered @ centered
        slope = (centered @ (history[:, 2] - history[:, 2].mean())) / denominator if denominator else 0.0
        intercept = history[:, 2].mean() - slope * trend.mean()
        history[:, 2] -= intercept + slope * trend
        current[2] -= intercept + slope * current[0]
    center, scale = history.mean(axis=0), history.std(axis=0)
    safe = np.where(scale > 0, scale, 1)
    return (
        np.clip(np.where(scale > 0, (history - center) / safe, 0), -STATE_Z_LIMIT, STATE_Z_LIMIT),
        np.clip(np.where(scale > 0, (current - center) / safe, 0), -STATE_Z_LIMIT, STATE_Z_LIMIT),
        slope,
    )


def weekly_ic(scores, labels):
    matched = scores.merge(labels.loc[labels["evaluation_ready"] & labels["evaluation_split"].isin(SPLITS),
                                      [*KEYS, "evaluation_split", "forward_return"]], on=KEYS,
                           how="inner", validate="many_to_one")
    grouped = matched.groupby(["strategy", "signal_date"], sort=False)
    rows = []
    for (strategy, signal), week in grouped:
        require(week["evaluation_split"].nunique() == 1 and len(week) >= 50, "Invalid weekly IC sample")
        rows.append({"strategy": strategy, "signal_date": signal,
                     "evaluation_split": week["evaluation_split"].iat[0],
                     "fit_status": week["fit_status"].iat[0], "observations": len(week),
                     "rank_ic": week["score"].rank().corr(week["forward_return"].rank())})
    return pd.DataFrame(rows)


def rank_scores(scores, schedule):
    scores = scores.sort_values(["strategy", "signal_date", "score", "code"],
                                ascending=[True, True, False, True]).copy()
    scores["rank"] = scores.groupby(["strategy", "signal_date"]).cumcount() + 1
    return scores.merge(schedule, on="signal_date", validate="many_to_one")


def run(root, output, breadth_mode):
    spec_file = output / "configs/verified_model_spec.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    names = [item["factor"] for item in spec["selected_factors"]]
    directions = {item["factor"]: item["direction"] for item in spec["selected_factors"]}
    active = [(item["factor"], item["state_name"]) for item in spec["active_interactions"]]
    dimension = len(names)
    raw_features = [f"pct_{factor}" for factor in names]
    candidates = pd.read_parquet(root / FACTOR_DIR / "csi300_weekly_candidates.parquet", columns=KEYS + names)
    fixed = pd.read_parquet(root / FACTOR_DIR / "csi300_weekly_fixed_scores.parquet",
                            columns=KEYS + ["eligible", "score_without_bias"])
    eligible = fixed.loc[fixed["eligible"], KEYS + ["score_without_bias"]].merge(
        candidates, on=KEYS, how="left", validate="one_to_one")
    ranks = (eligible[names] * pd.Series(directions)).where(lambda values: np.isfinite(values))
    percentiles = ranks.groupby(eligible["signal_date"]).rank(pct=True)
    missing = percentiles.isna().sum()
    for factor, field in zip(names, raw_features, strict=True):
        eligible[field] = percentiles[factor].fillna(0.5)
    require(len(eligible) == fixed["eligible"].sum(), "Changed eligible universe")
    labels = pd.read_parquet(root / FACTOR_DIR / "csi300_weekly_forward_labels.parquet", columns=KEYS + [
        "exit_date", "forward_return", "evaluation_ready", "evaluation_split"])
    labels = labels.loc[labels["signal_date"].dt.year.between(2020, 2024)].copy()
    evaluation_days = labels.loc[labels["evaluation_split"].eq("screening"), "signal_date"].unique()
    states = pd.read_parquet(root / STATES_FILE, columns=["signal_date", *STATES]).set_index("signal_date")
    screening_corr = states.loc[evaluation_days, STATES].corr()
    write_table(screening_corr.rename_axis("state").reset_index(), output / "reports/screening_state_correlations.csv")
    correlation = screening_corr.loc[STATES[0], STATES[2]]
    residualize = breadth_mode == "residual" or (breadth_mode == "auto" and abs(correlation) > 0.80)
    raw_schedule = pd.read_parquet(root / "data/processed/universe/csi300_weekly_schedule.parquet",
                                  columns=["signal_date", "execution_date"])
    schedule = raw_schedule.loc[raw_schedule["execution_date"].dt.year.between(2020, 2024)]
    fit = pd.read_parquet(root / WALK / "walk_forward_weight_schedule.parquet",
                          columns=["signal_date", "fit_status"])
    links = pd.read_parquet(root / WALK / "walk_forward_training_weeks.parquet")
    require(fit["signal_date"].is_unique and set(schedule["signal_date"]) == set(fit["signal_date"]),
            "Existing fit calendar must be the development calendar")
    require(set(links["signal_date"]) == set(fit.loc[fit["fit_status"].eq("fitted"), "signal_date"])
            and not links.duplicated(["signal_date", "training_signal_date"]).any()
            and links["training_exit_date"].le(links["signal_date"]).all()
            and links["training_signal_date"].lt(links["signal_date"]).all(),
            "Invalid or look-ahead rolling schedule")
    require(np.isfinite(states.loc[fit["signal_date"], STATES]).all().all() and states.index.is_unique,
            "Invalid market state data")
    training = eligible.merge(labels, on=KEYS, validate="one_to_one")
    training = training.loc[training["exit_date"].notna() & np.isfinite(training["forward_return"])]
    moments = {}
    for day, week in training.groupby("signal_date"):
        xs = week[raw_features].to_numpy()
        ys = week["forward_return"].rank(pct=True).to_numpy()
        xc, yc = xs - xs.mean(axis=0), ys - ys.mean()
        require(week["exit_date"].nunique() == 1, f"Multiple label ends on {day}")
        moments[day] = (xc.T @ xc / len(xs), xc.T @ yc / len(xs), len(xs), week["exit_date"].iat[0])

    factor_positions = np.array([*range(dimension), *[names.index(name) for name, _ in active]], dtype=int)
    state_positions = np.array([0] * dimension + [STATES.index(state) + 1 for _, state in active], dtype=int)
    models = {"sparse_state_ridge": (0.01, 0.01, True),
              "verified_main_ridge_001": (0.01, 0.0, False),
              "verified_main_ridge_005": (0.05, 0.0, False)}
    models.update({f"diff_{penalty:g}": (0.05, penalty, True) for penalty in STATE_PENALTIES})
    score_frames, weight_rows = [], []
    stock_weeks = dict(tuple(eligible.groupby("signal_date")))
    for signal, history in links.groupby("signal_date"):
        history = history.sort_values("training_signal_date")
        past = history["training_signal_date"]
        require(MIN_WEEKS <= len(history) <= WINDOW_WEEKS, "Unexpected training window size")
        train_z, now_z, slope = normalize_states(states.loc[past, STATES].to_numpy(),
                                                states.loc[signal, STATES].to_numpy(), residualize)
        scales = np.c_[np.ones(len(history)), train_z][:, state_positions]
        current_scales = np.r_[1.0, now_z][state_positions]
        gram, rhs = [], []
        for row in history.itertuples(index=False):
            moment, response, count, exit_date = moments[row.training_signal_date]
            require(count == row.training_stocks and exit_date == row.training_exit_date,
                    f"Training links differ from control on {row.training_signal_date}")
            gram.append(moment[np.ix_(factor_positions, factor_positions)])
            rhs.append(response[factor_positions])
        gram = (np.asarray(gram) * scales[:, :, None] * scales[:, None, :]).mean(axis=0)
        rhs = (np.asarray(rhs) * scales).mean(axis=0)
        week = stock_weeks[signal]
        feature_values = week[raw_features].to_numpy() - 0.5
        for name, (base_penalty, state_penalty, interactions) in models.items():
            length = len(factor_positions) if interactions else dimension
            penalty = np.r_[np.full(dimension, base_penalty), np.full(length - dimension, state_penalty)]
            coef = np.linalg.solve(gram[:length, :length] + np.diag(penalty), rhs[:length])
            effective = coef[:dimension].copy()
            if interactions:
                np.add.at(effective, factor_positions[dimension:], coef[dimension:] * current_scales[dimension:])
            score_frames.append(week[KEYS].assign(strategy=name, fit_status="fitted",
                                                  score=np.sum(feature_values * effective, axis=1)))
            for pos, factor in enumerate(names):
                weight_rows.append({"strategy": name, "signal_date": signal, "factor": factor,
                                    "base_weight": coef[pos], "effective_weight": effective[pos],
                                    "state_contribution": effective[pos] - coef[pos], "breadth_on_trend_slope": slope,
                                    **{f"z_{state}": z for state, z in zip(STATES, now_z, strict=True)}})
            for pos, (factor, state) in enumerate(active, start=dimension):
                if interactions:
                    weight_rows[-dimension + names.index(factor)][f"coef_x_{state}"] = coef[pos]

    warmup_days = fit.loc[fit["fit_status"].eq("equal_weight_warmup"), "signal_date"]
    warmup = eligible.loc[eligible["signal_date"].isin(warmup_days), KEYS + ["score_without_bias"]]
    for name in models:
        score_frames.append(warmup.rename(columns={"score_without_bias": "score"})
                            .assign(strategy=name, fit_status="equal_weight_warmup"))
    new_scores = pd.concat(score_frames, ignore_index=True)
    ic_grid = weekly_ic(new_scores.loc[new_scores["strategy"].str.startswith("diff_")], labels)
    tuning = (ic_grid.loc[ic_grid["fit_status"].eq("fitted") & ic_grid["evaluation_split"].eq("validation")]
              .groupby("strategy")["rank_ic"].agg(validation_rank_ic="mean", ic_weeks="count").reset_index())
    tuning["state_penalty"] = tuning["strategy"].str.removeprefix("diff_").astype(float)
    tuning = tuning.sort_values(["validation_rank_ic", "state_penalty"], ascending=[False, False])
    require(len(tuning) == len(STATE_PENALTIES) and tuning["ic_weeks"].nunique() == 1
            and tuning["validation_rank_ic"].notna().all(), "Invalid penalty validation")
    winner, penalty = tuning[["strategy", "state_penalty"]].iloc[0]
    tuning["chosen"] = tuning["strategy"].eq(winner)
    write_table(tuning, output / "reports/state_penalty_validation.csv")
    write_table(ic_grid, output / "reports/state_penalty_weekly_rank_ic.parquet")
    new_scores = new_scores.loc[~new_scores["strategy"].str.startswith("diff_") | new_scores["strategy"].eq(winner)]
    new_scores = new_scores.replace({"strategy": {winner: "diff_reg_state_ridge"}})
    weights = pd.DataFrame(weight_rows)
    weights = weights.loc[~weights["strategy"].str.startswith("diff_") | weights["strategy"].eq(winner)]
    weights = weights.replace({"strategy": {winner: "diff_reg_state_ridge"}})
    write_table(weights, output / "data/verified_ridge_weights.parquet")
    write_table(weights, output / "reports/verified_ridge_weights.csv")

    legacy = pd.read_parquet(root / WALK / "walk_forward_full_rankings.parquet")
    require(set(legacy["strategy"]) == set(LEGACY[1:]), "Invalid legacy ranking strategies")
    common = eligible.merge(fit, on="signal_date", validate="many_to_one")
    equal = rank_scores(common[KEYS + ["fit_status", "score_without_bias"]]
                        .rename(columns={"score_without_bias": "score"}).assign(strategy="equal_weight"), schedule)
    ranked = pd.concat([equal, legacy, rank_scores(new_scores, schedule)], ignore_index=True)
    require(not ranked.duplicated(["strategy", *KEYS]).any() and np.isfinite(ranked["score"]).all()
            and ranked["execution_date"].gt(ranked["signal_date"]).all(), "Invalid final score panel")
    per_week = ranked.groupby(["signal_date", "strategy"])["code"].size().unstack()
    require(per_week.eq(per_week["equal_weight"], axis=0).all().all(), "Changed stock counts among models")
    config = tomllib.loads((root / "configs/baseline.toml").read_text(encoding="utf-8-sig"))
    top_n = config["portfolio"]["top_n"]
    require(top_n == 30 and per_week.ge(top_n).all().all(), "Expected Top 30 and adequate stocks")
    targets = ranked.loc[ranked["rank"].le(top_n),
                         ["strategy", *KEYS, "execution_date", "rank", "score"]].assign(target_weight=1 / top_n)
    original = pd.read_parquet(root / WALK / "walk_forward_three_variant_targets.parquet")
    compare = ["strategy", *KEYS, "execution_date", "rank", "score"]
    pd.testing.assert_frame_equal(targets.loc[targets["strategy"].isin(LEGACY), compare]
                                  .sort_values(compare[:3]).reset_index(drop=True),
                                  original[compare].sort_values(compare[:3]).reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    write_table(ranked, output / "data/verified_full_rankings.parquet")
    write_table(targets, output / "data/verified_targets.parquet")
    daily_ic = weekly_ic(ranked, labels)
    write_table(daily_ic, output / "reports/model_weekly_rank_ic.parquet")
    fitted_ic = daily_ic.loc[daily_ic["fit_status"].eq("fitted")]
    summary = fitted_ic.groupby(["strategy", "evaluation_split"])["rank_ic"].agg(
        rank_ic="mean", rank_icir=icir, ic_weeks="count").reset_index()
    write_table(summary, output / "reports/model_rank_ic_by_split.csv")
    metadata = {"breadth_mode_requested": breadth_mode, "breadth_residualized": bool(residualize),
                "screening_trend_breadth_pearson": float(correlation), "lambda_base_diff": 0.05,
                "lambda_state_validation_winner": float(penalty), "state_penalty_grid": STATE_PENALTIES,
                "percentile_missing_policy": "neutral 0.5 on the unchanged legacy eligible universe",
                "missing_signal_rows_by_factor": {key: int(value) for key, value in missing.items()},
                "min_weeks": MIN_WEEKS, "window_weeks": WINDOW_WEEKS, "state_z_limit": STATE_Z_LIMIT}
    (output / "configs/verified_fit_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nStep 3 | fit metadata:", metadata, flush=True)
    print(tuning.to_string(index=False), flush=True)
    print(summary.round(5).to_string(index=False), flush=True)

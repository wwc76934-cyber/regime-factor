import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = ["signal_date", "code"]
FACTORS = ["RET_5", "VOL_60", "CORR_PV_10"]
FEATURES = [f"pct_{factor}" for factor in FACTORS]
STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d"]
WALK = Path("data/processed/portfolio/walk_forward")
FACTORS_DIR = Path("data/processed/factors")
FIRST = pd.Timestamp("2020-01-01")
LAST = pd.Timestamp("2024-12-31")
LEGACY = ["equal_weight", "rolling_ridge", "state_ridge"]
FULL = list(range(12))
CORE = [0, 1, 2, 6, 10]
RIDGE = {
    "state_ridge_replication": (FULL, 0.01, 0.01),
    "three_main_ridge_005": (list(range(3)), 0.05, 0.0),
    "three_diff_reg_050": (FULL, 0.05, 0.5),
    "three_diff_reg_100": (FULL, 0.05, 1.0),
    "three_core_interaction": (CORE, 0.01, 0.01),
}
TREES = ["lightgbm_3f", "lightgbm_6f"]
TREE_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 300,
    "max_bin": 63,
    "lambda_l2": 1.0,
    "num_threads": 2,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "seed": 20261118,
}
TREE_ROUNDS = 64


def require(test, message):
    if not test:
        raise ValueError(message)


def read_development(path, columns=None, date_field="signal_date"):
    filters = [(date_field, ">=", FIRST), (date_field, "<=", LAST)]
    return pq.read_table(path, columns=columns, filters=filters).to_pandas()


def write_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_parquet(path, index=False)


def icir(values):
    std = values.std(ddof=1)
    return values.mean() / std if std > 0 else np.nan


def state_z(past, now):
    center, scale = past.mean(axis=0), past.std(axis=0)
    safe = np.where(scale > 0, scale, 1)
    scaled = [np.clip(np.where(scale > 0, (data - center) / safe, 0), -3, 3)
              for data in (past, now)]
    return tuple(scaled)


def weekly_ic(ranked, labels):
    usable = labels.loc[labels["evaluation_ready"] & labels["evaluation_split"].isin(["screening", "validation"])]
    matched = ranked.merge(usable[KEYS + ["evaluation_split", "forward_return"]], on=KEYS,
                           how="inner", validate="many_to_one")
    rows = []
    for (name, signal), week in matched.groupby(["strategy", "signal_date"], sort=False):
        require(week["evaluation_split"].nunique() == 1 and len(week) >= 50, "Invalid IC sample")
        rows.append({"strategy": name, "signal_date": signal, "evaluation_split": week["evaluation_split"].iat[0],
                     "fit_status": week["fit_status"].iat[0], "observations": len(week),
                     "rank_ic": week["score"].rank().corr(week["forward_return"].rank())})
    return pd.DataFrame(rows)


def rank_rows(data, schedule):
    result = data.sort_values(["strategy", "signal_date", "score", "code"],
                              ascending=[True, True, False, True]).copy()
    result["rank"] = result.groupby(["strategy", "signal_date"]).cumcount() + 1
    return result.merge(schedule, on="signal_date", validate="many_to_one")


def run(root, output):
    import lightgbm as lgb
    import scipy

    fixed = read_development(root / FACTORS_DIR / "csi300_weekly_fixed_scores.parquet",
                             KEYS + ["eligible", "score_without_bias", *FEATURES])
    eligible = fixed.loc[fixed["eligible"].fillna(False)].copy()
    require(not eligible.duplicated(KEYS).any() and np.isfinite(eligible[FEATURES]).all().all(),
            "Invalid eligible stocks")
    labels = read_development(root / FACTORS_DIR / "csi300_weekly_forward_labels.parquet",
                              KEYS + ["exit_date", "forward_return", "evaluation_ready", "evaluation_split"])
    joined = eligible.merge(labels, on=KEYS, validate="one_to_one")
    require(len(joined) == len(eligible), "Missing label rows")
    training = joined.loc[joined["exit_date"].between(FIRST, LAST) & np.isfinite(joined["forward_return"])]
    states = read_development(root / "data/processed/market/csi300_weekly_market_states.parquet",
                              ["signal_date", *STATES]).set_index("signal_date")
    fit = read_development(root / WALK / "walk_forward_weight_schedule.parquet",
                           ["signal_date", "fit_status"])
    links = read_development(root / WALK / "walk_forward_training_weeks.parquet")
    calendar = read_development(root / "data/processed/universe/csi300_weekly_schedule.parquet",
                                ["signal_date", "execution_date"])
    require(fit["signal_date"].is_unique and states.index.is_unique and np.isfinite(states).all().all()
            and set(fit["signal_date"]) == set(calendar["signal_date"])
            and calendar["execution_date"].between(FIRST, LAST).all(), "Invalid development calendar")
    require(not links.duplicated(["signal_date", "training_signal_date"]).any()
            and set(links["signal_date"]) == set(fit.loc[fit["fit_status"].eq("fitted"), "signal_date"])
            and links["training_signal_date"].lt(links["signal_date"]).all()
            and links["training_exit_date"].between(FIRST, LAST).all()
            and links["training_exit_date"].le(links["signal_date"]).all(), "Invalid rolling schedule")

    weeks = dict(tuple(eligible.groupby("signal_date")))
    moments = {}
    for day, week in training.groupby("signal_date"):
        xs = week[FEATURES].to_numpy(dtype=float)
        ys = week["forward_return"].rank(pct=True).to_numpy()
        xc, yc = xs - xs.mean(axis=0), ys - ys.mean()
        require(week["exit_date"].nunique() == 1, "Multiple exit dates")
        moments[day] = (xc.T @ xc / len(xs), xc.T @ yc / len(xs), xs - 0.5, ys, week["exit_date"].iat[0])

    score_frames, weight_rows, tree_rows = [], [], []
    started = perf_counter()
    tree_seconds = 0.0
    for count, (signal, history) in enumerate(links.groupby("signal_date", sort=True), start=1):
        history = history.sort_values("training_signal_date")
        require(52 <= len(history) <= 104, "Invalid window length")
        train_z, now_z = state_z(states.loc[history["training_signal_date"], STATES].to_numpy(dtype=float),
                                 states.loc[signal, STATES].to_numpy(dtype=float))
        normal, response = np.zeros((12, 12)), np.zeros(12)
        tx, ty, sample_weight = [], [], []
        for z, link in zip(train_z, history.itertuples(index=False), strict=True):
            gram, rhs, xs, ys, exit_date = moments[link.training_signal_date]
            require(len(xs) == link.training_stocks and exit_date == link.training_exit_date,
                    "Training rows or dates differ from original model")
            state = np.r_[1.0, z]
            normal += np.kron(np.outer(state, state), gram)
            response += np.kron(state, rhs)
            tx.append(np.c_[xs, np.broadcast_to(z, (len(xs), 3))])
            ty.append(ys)
            sample_weight.append(np.full(len(xs), 1 / len(xs)))
        normal /= len(history)
        response /= len(history)
        week = weeks[signal]
        xs = week[FEATURES].to_numpy(dtype=float) - 0.5
        current = np.r_[1.0, now_z]
        for name, (active, alpha_base, alpha_state) in RIDGE.items():
            penalty = np.r_[np.full(3, alpha_base), np.full(len(active) - 3, alpha_state)]
            coefficients = np.zeros(12)
            coefficients[active] = np.linalg.solve(normal[np.ix_(active, active)] + np.diag(penalty), response[active])
            coefficient = coefficients.reshape(4, 3)
            effective = current @ coefficient
            score_frames.append(week[KEYS].assign(strategy=name, fit_status="fitted",
                                                  score=np.sum(xs * effective, axis=1)))
            for pos, factor in enumerate(FACTORS):
                weight_rows.append({"strategy": name, "signal_date": signal, "factor": factor,
                                    "base_weight": coefficient[0, pos], "effective_weight": effective[pos],
                                    "state_contribution": effective[pos] - coefficient[0, pos],
                                    **{f"coef_x_{state}": val for state, val in zip(STATES, coefficient[1:, pos])},
                                    **{f"z_{state}": val for state, val in zip(STATES, now_z)}})
        tx, ty = np.concatenate(tx), np.concatenate(ty)
        weights = np.concatenate(sample_weight) * (len(ty) / len(history))
        prediction_x = np.c_[xs, np.broadcast_to(now_z, (len(xs), 3))]
        tree_start = perf_counter()
        for name, width in zip(TREES, [3, 6], strict=True):
            fields = [*FEATURES, *STATES][:width]
            dataset = lgb.Dataset(tx[:, :width].copy(), label=ty, weight=weights, feature_name=fields)
            tree = lgb.train(TREE_PARAMS, dataset, num_boost_round=TREE_ROUNDS)
            prediction = tree.predict(prediction_x[:, :width].copy(), num_threads=TREE_PARAMS["num_threads"])
            score_frames.append(week[KEYS].assign(strategy=name, fit_status="fitted", score=prediction))
            tree_rows.append({"strategy": name, "signal_date": signal, "training_weeks": len(history),
                              "training_rows": len(ty), "trees": tree.num_trees(),
                              **dict(zip(fields, tree.feature_importance(importance_type="split"), strict=True))})
        tree_seconds += perf_counter() - tree_start
        if count % 50 == 0:
            print(f"Fitted {count} weeks; LightGBM CPU wall time so far: {tree_seconds:.1f}s", flush=True)

    legacy = read_development(root / WALK / "walk_forward_full_rankings.parquet")
    require(set(legacy["strategy"]) == set(LEGACY[1:]), "Unexpected legacy rankings")
    old = legacy.loc[legacy["strategy"].eq("state_ridge") & legacy["fit_status"].eq("fitted")]
    new = pd.concat(score_frames, ignore_index=True)
    replica = new.loc[new["strategy"].eq("state_ridge_replication")]
    compare = old[KEYS + ["score"]].merge(replica[KEYS + ["score"]], on=KEYS, validate="one_to_one")
    require(len(compare) == len(old) == len(replica), "Replication score keys differ")
    np.testing.assert_allclose(compare["score_x"], compare["score_y"], rtol=0, atol=1e-12)

    warmup = eligible.loc[eligible["signal_date"].isin(
        fit.loc[fit["fit_status"].eq("equal_weight_warmup"), "signal_date"]), KEYS + ["score_without_bias"]]
    additions = [new.loc[~new["strategy"].eq("state_ridge_replication")]]
    for name in [*RIDGE.keys() - {"state_ridge_replication"}, *TREES]:
        additions.append(warmup.rename(columns={"score_without_bias": "score"})
                         .assign(strategy=name, fit_status="equal_weight_warmup"))
    equal = eligible[KEYS + ["score_without_bias"]].rename(columns={"score_without_bias": "score"})
    equal = equal.merge(fit, on="signal_date", validate="many_to_one").assign(strategy="equal_weight")
    ranked = pd.concat([legacy, rank_rows(pd.concat([equal, *additions], ignore_index=True), calendar)],
                       ignore_index=True)
    require(not ranked.duplicated(["strategy", *KEYS]).any() and np.isfinite(ranked["score"]).all()
            and ranked["execution_date"].gt(ranked["signal_date"]).all(), "Invalid model rankings")
    counts = ranked.groupby(["signal_date", "strategy"]).size().unstack()
    require(counts.eq(counts["equal_weight"], axis=0).all().all(), "Scoring universes differ")
    targets = ranked.loc[ranked["rank"].le(30)].assign(target_weight=1 / 30)
    original = read_development(root / WALK / "walk_forward_three_variant_targets.parquet")
    keys = ["strategy", *KEYS, "execution_date", "rank", "score"]
    pd.testing.assert_frame_equal(targets.loc[targets["strategy"].isin(LEGACY), keys]
                                  .sort_values(keys[:3]).reset_index(drop=True),
                                  original[keys].sort_values(keys[:3]).reset_index(drop=True),
                                  check_dtype=False, check_exact=True)

    ic = weekly_ic(ranked, labels)
    prediction = (ic.loc[ic["fit_status"].eq("fitted")]
                  .groupby(["strategy", "evaluation_split"])["rank_ic"]
                  .agg(rank_ic="mean", rank_icir=icir, ic_weeks="count").reset_index())
    write_table(ranked, output / "data/full_rankings.parquet")
    write_table(targets, output / "data/targets.parquet")
    write_table(pd.DataFrame(weight_rows), output / "reports/ridge_weekly_weights.parquet")
    write_table(pd.DataFrame(tree_rows), output / "reports/lightgbm_weekly_fit.csv")
    write_table(ic, output / "reports/model_weekly_rank_ic.parquet")
    write_table(prediction, output / "reports/model_rank_ic_by_split.csv")
    config = {"factors": FEATURES, "states": STATES, "state_transform": "training window z score, clip ±3; no residualization",
              "ridge_columns": ["base", *STATES], "ridge_models": RIDGE, "tree_models": TREES,
              "tree_parameters": TREE_PARAMS, "tree_rounds": TREE_ROUNDS, "lightgbm": lgb.__version__,
              "scipy": scipy.__version__, "fitted_weeks": len(links["signal_date"].unique()),
              "training_links": len(links), "tree_training_and_prediction_seconds": tree_seconds,
              "build_seconds": perf_counter() - started}
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(prediction.round(5).to_string(index=False), flush=True)

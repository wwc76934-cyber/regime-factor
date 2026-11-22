"""Rolling development-only state Ridge and Huber LightGBM with mature labels."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prepare_v3_1_base import DEVELOPMENT_LAST, initialize


PREDICT_FIRST = pd.Timestamp("2017-01-01")
WINDOW = 260
MIN_WEEKS = 52
REFIT = 4
TREE_ROUNDS = 60
KEY = ["signal_date", "code"]
COMMON = ["RET_5", "VOL_60", "CORR_PV_10"]
RIDGE, TREE, BASELINE = "v3_1_state_ridge", "v3_1_lgb_huber", "v3_0_state_ridge"
TREE_OPTIONS = {"objective":"huber", "alpha":.9, "learning_rate":.05, "max_depth":4,
    "num_leaves":12, "min_data_in_leaf":200, "colsample_bytree":.7, "max_bin":63, "lambda_l2":1.,
    "force_col_wise":True, "deterministic":True, "verbosity":-1, "seed":731}


@dataclass
class Week:
    day: pd.Timestamp
    state: np.ndarray
    x: np.ndarray
    output: pd.DataFrame
    exit: pd.Timestamp
    train_x: np.ndarray
    train_y: np.ndarray
    xx: np.ndarray
    xy: np.ndarray


def transform_inputs(frame, factors, directions):
    aligned = frame[factors].mul(pd.Series(directions)[factors], axis=1)
    return (aligned.rank(pct=True).fillna(.5) - .5).to_numpy(dtype=np.float32)


def weeks_from(frame, labels, states, factors, directions):
    joined = frame.merge(labels[KEY + ["entry_date", "exit_date", "risk_adjusted_target"]], on=KEY, validate="one_to_one")
    state_map = states.set_index("signal_date")
    blocks = []
    for day, week in joined.groupby("signal_date", sort=True):
        z = state_map.loc[day].to_numpy(dtype=float)
        if not np.isfinite(z).all() or week.empty:
            continue
        x = transform_inputs(week, factors, directions)
        valid = week.risk_adjusted_target.notna().to_numpy()
        mat, target = x[valid], week.risk_adjusted_target.to_numpy(dtype=np.float32)[valid]
        center = mat - mat.mean(axis=0) if len(mat) else mat
        output = week[KEY + ["entry_date", "BETA_120"]].reset_index(drop=True)
        n = max(1, len(mat))
        blocks.append(Week(day, z, x, output, week.exit_date.iat[0], mat, target,
                           center.T @ center / n, center.T @ target / n))
    return blocks


def standardized(train, current):
    states = np.stack([w.state for w in train])
    center = states.mean(axis=0)
    scale = states.std(axis=0)
    scale[scale == 0] = 1
    z = np.clip((states - center) / scale, -3, 3)
    now = np.clip((current.state - center) / scale, -3, 3)
    return z, now, center, scale


def tree_features(x, state):
    # Feed explicit cross-products so even a shallow tree can model state interactions.
    z = np.asarray(state, dtype=np.float32)
    interaction = (z[None, :, None] * x[:, None, :]).reshape(len(x), -1)
    return np.concatenate([x, np.broadcast_to(z, (len(x), len(z))), interaction], axis=1)


def produce(weeks, workers):
    predictions, audit, weights = [], [], []
    tree = None
    tree_source = None
    count = 0
    features = weeks[0].x.shape[1]
    state_dim = len(weeks[0].state)
    penalty = np.diag(np.r_[np.full(features,.05),np.full(features * state_dim,.5)])
    options = {**TREE_OPTIONS, "num_threads":workers}
    for offset, current in enumerate(weeks):
        if not PREDICT_FIRST <= current.day <= DEVELOPMENT_LAST:
            continue
        train = [w for w in weeks[:offset] if pd.notna(w.exit) and w.exit <= current.day
                 and len(w.train_y) >= 50][-WINDOW:]
        if len(train) < MIN_WEEKS:
            continue
        z, now, center, scale = standardized(train, current)
        blocks = np.c_[np.ones(len(z)), z]
        gram = np.einsum("na,nb,nij->aibj", blocks, blocks, np.stack([w.xx for w in train])).reshape(penalty.shape)
        rhs = np.einsum("na,ni->ai", blocks, np.stack([w.xy for w in train])).ravel()
        coef = np.linalg.solve(gram / len(train) + penalty, rhs / len(train)).reshape(state_dim + 1, features)
        actual_weights = np.r_[1., now] @ coef
        predictions.append(current.output.assign(strategy=RIDGE, score=current.x @ actual_weights))
        weights.append((current.day, *actual_weights))
        if count % REFIT == 0 or tree is None:
            xt = np.concatenate([tree_features(w.train_x, q) for w, q in zip(train, z, strict=True)])
            yt = np.concatenate([w.train_y for w in train])
            sample_weight = np.concatenate([np.full(len(w.train_y), len(yt)/(len(train)*len(w.train_y)),
                                                    dtype=np.float32) for w in train])
            tree = lgb.train(options, lgb.Dataset(xt, label=yt, weight=sample_weight), num_boost_round=TREE_ROUNDS)
            tree_source = current.day
            tree_center, tree_scale = center, scale
            tree_first = train[0].day
            tree_exit = max(w.exit for w in train)
        tree_z = np.clip((current.state - tree_center) / tree_scale, -3, 3)
        score = tree.predict(tree_features(current.x, tree_z), num_threads=workers)
        predictions.append(current.output.assign(strategy=TREE, score=score))
        audit.append((current.day, train[0].day, max(w.exit for w in train), len(train),
                      tree_source, tree_first, tree_exit))
        count += 1
        if count % 50 == 0:
            print(f"Predicted {count} weeks through {current.day:%Y-%m-%d}", flush=True)
    logs = pd.DataFrame(audit, columns=["signal_date","ridge_oldest_training_signal","ridge_latest_training_exit",
        "ridge_training_weeks","tree_last_fit_signal","tree_oldest_training_signal","tree_latest_training_exit"])
    if logs.empty or not (logs.ridge_latest_training_exit.le(logs.signal_date) &
        logs.tree_latest_training_exit.le(logs.tree_last_fit_signal) & logs.tree_last_fit_signal.le(logs.signal_date)).all():
        raise ValueError("Unmatured labels or invalid time windows")
    return pd.concat(predictions, ignore_index=True), logs, weights


def include_baseline(scores, path):
    legacy = pd.read_parquet(path, columns=KEY+["entry_date","BETA_120","strategy","score"],
                             filters=[("strategy","==","v3_state_ridge")])
    legacy["strategy"] = BASELINE
    common = scores[KEY].drop_duplicates().merge(legacy[KEY], on=KEY, validate="one_to_one")
    return pd.concat([scores.merge(common, on=KEY, validate="many_to_one"), legacy.merge(common, on=KEY, validate="one_to_one")],
                     ignore_index=True)


def evaluate(scores, labels, horizon):
    rows = []
    view = scores.merge(labels[KEY+["stock_return","index_return","risk_adjusted_target","exit_date"]],
                        on=KEY, validate="many_to_one")
    for (name, day), week in view.groupby(["strategy","signal_date"], sort=True):
        test = week.dropna(subset=["risk_adjusted_target"])
        if len(test) < 50 or pd.isna(week.exit_date.iat[0]):
            continue
        top = week.sort_values(["score","code"],ascending=[False,True]).head(30)
        if top.stock_return.count() < 24:
            continue
        best = test.sort_values(["risk_adjusted_target","code"],ascending=[False,True]).head(30)
        top_return = top.stock_return.mean()
        index = top.index_return.iat[0]
        rank = test.score.rank()
        ic = rank.corr(test.risk_adjusted_target.rank()) if rank.nunique() > 1 else np.nan
        raw_ic = rank.corr(test.stock_return.rank()) if rank.nunique() > 1 else np.nan
        rows.append((name, horizon, day, ic, raw_ic, len(set(top.code) & set(best.code))/30,
                     top_return, index, top_return-index, top.BETA_120.mean(), len(test), top.stock_return.count()))
    monthly = pd.DataFrame(rows, columns=["strategy","horizon","signal_date","rank_ic","raw_return_rank_ic","top30_hit",
        "top30_return","index_return","top30_excess_H00300","top30_signal_beta","labelled_universe","top30_realized_n"])
    tables = []
    for name,h in monthly.groupby("strategy"):
        for period,sample in [("all",h),("2017-2022",h.loc[h.signal_date.dt.year.le(2022)]),
                              ("2023-2025",h.loc[h.signal_date.dt.year.ge(2023)])]:
            ic = sample.rank_ic.mean()
            sd = sample.rank_ic.std()
            beta = sample.top30_return.cov(sample.index_return) / sample.index_return.var()
            tables.append({"strategy":name,"horizon":horizon,"period":period,"weeks":len(sample),"rank_ic":ic,
                "icir":ic/sd if sd>0 else np.nan,"raw_return_rank_ic":sample.raw_return_rank_ic.mean(),
                "top30_hit":sample.top30_hit.mean(),"top30_excess_H00300":sample.top30_excess_H00300.mean(),
                "top30_realized_beta":beta,"top30_signal_beta":sample.top30_signal_beta.mean()})
    return monthly, pd.DataFrame(tables)


def run(root, step1, inputs, out, horizon, workers):
    v3, _, _, _, _ = initialize(root)
    spec = json.loads((step1 / "v3_1_selected_factors.json").read_text(encoding="utf-8"))
    v3.need(spec["status"]=="go" and len(spec["selected"])==6 and spec["style_missing_rate"]<.05,
            "V3.1 factor and style gate has not passed")
    factors = spec["selected"]
    frame = pd.read_parquet(inputs/"factor_panel.parquet",
        columns=KEY+["signal_bar_usable","BETA_120","VOL_20",*dict.fromkeys([*factors,*COMMON])])
    frame = frame.loc[frame.signal_bar_usable & frame[["BETA_120","VOL_20",*COMMON]].notna().all(axis=1)].sort_values(KEY)
    labels = pd.read_parquet(inputs/"labels.parquet",filters=[("horizon","==",horizon)])
    states = pd.read_parquet(step1/"states.parquet")[["signal_date",*spec["states"]]]
    v3.need(frame.signal_date.le(DEVELOPMENT_LAST).all() and labels.exit_date.dropna().le(DEVELOPMENT_LAST).all(),
            "V3.1 development cannot contain 2026")
    weeks = weeks_from(frame, labels, states, factors, spec["directions"])
    scores, windows, weights = produce(weeks, workers)
    scores = include_baseline(scores, root/"research/v3/results/step3/predictions.parquet")
    monthly, metrics = evaluate(scores, labels, horizon)
    for name,data in [("predictions.parquet",scores),("training_windows.csv",windows),("weekly_signal_metrics.parquet",monthly),
                      ("model_signal_audit.csv",metrics),("ridge_effective_weights.csv",pd.DataFrame(weights,columns=["signal_date",*factors]))]:
        v3.save(data,out/name)
    (out/"model_spec.json").write_text(json.dumps({"horizon":horizon,"ridge_l2":[.05,.5],
        "tree":TREE_OPTIONS,"tree_rounds":TREE_ROUNDS,"tree_refit_weeks":REFIT,"min_weeks":MIN_WEEKS,"window":WINDOW},
        indent=2),encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--step1",type=Path)
    p.add_argument("--inputs",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--horizon",type=int,choices=[5,10,20],default=10)
    p.add_argument("--workers",type=int,default=2)
    a = p.parse_args()
    study=a.project_root/"research/v3_1/results"
    run(a.project_root.resolve(),a.step1 or study/"step1",a.inputs or study/"step1",
        a.output or study/"step2"/f"h{a.horizon}",a.horizon,a.workers)

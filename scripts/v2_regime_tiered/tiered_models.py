"""Fit within each volatility sleeve and attach causal score EMA variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tiered_data import FACTORS, TIERS, require


FEATURES = [f"tier_pct_{factor}" for factor in FACTORS]
STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d"]
MODELS = ["tiered_pure_alpha_equal", "tiered_pure_alpha_ridge", "tiered_three_factor_ridge", "tiered_lightgbm_6f"]
LABEL = "tier_target_y"
TREE = dict(objective="regression", learning_rate=.05, num_leaves=7, max_depth=3,
            min_data_in_leaf=300, max_bin=63, lambda_l2=1., num_threads=2,
            deterministic=True, force_col_wise=True, verbosity=-1, seed=20261118)


def smooth(frame):
    pieces = []
    for _, model in frame.groupby("strategy"):
        previous = pd.Series(dtype=float)
        for _, week in model.groupby("signal_date", sort=True):
            week = week.copy()
            raw = week.set_index("code")["score"]
            previous = .5 * raw + .5 * previous.reindex(raw.index).fillna(raw)
            week["score_ema05"] = week["code"].map(previous)
            pieces.append(week)
    return pd.concat(pieces, ignore_index=True)


def state_z(states, history, signal):
    past = states.loc[history, STATES].to_numpy(dtype=float)
    center, scale = past.mean(axis=0), past.std(axis=0)
    def normalize(values):
        return np.clip(np.divide(values - center, scale, out=np.zeros_like(values), where=scale > 0), -3, 3)
    return normalize(past), normalize(states.loc[signal, STATES].to_numpy(dtype=float))


def fit(root, source, output, include_review=False):
    import lightgbm as lgb

    require(lgb.__version__ == "4.6.0", "Expected frozen LightGBM 4.6.0")
    sample = pd.read_parquet(source / "tiered_development.parquet")
    states = pd.read_parquet(root / "data/processed/market/csi300_weekly_market_states.parquet").set_index("signal_date")
    if include_review:
        sample = pd.concat([sample, pd.read_parquet(source / "tiered_review_2025.parquet")], ignore_index=True)
        states = pd.concat([states[STATES], pd.read_parquet(source / "states_2025.parquet").set_index("signal_date")[STATES]])
    sample = sample.sort_values(["signal_date", "code"])
    require(not sample.duplicated(["signal_date","code"]).any() and states.index.is_unique, "Duplicate inputs")
    groups = dict(tuple(sample.groupby(["signal_date", "vol_tier"])))
    dates = sorted(sample["signal_date"].unique())
    valid = sample.loc[sample[LABEL].notna() & sample["exit_date"].notna()]
    moments = {}
    for (day, tier), rows in valid.groupby(["signal_date","vol_tier"]):
        if len(rows) < 50:
            continue
        require(rows["exit_date"].nunique() == 1, "Inconsistent label maturity")
        xs = rows[FEATURES].to_numpy()
        ys = rows[LABEL].to_numpy()
        xc = xs - xs.mean(axis=0)
        moments[(day,tier)] = (rows["exit_date"].iat[0], xs, ys, xc.T @ xc / len(xs), xc.T @ ys / len(xs))
    scores, audit, coefficients = [], [], []
    ridge_names = MODELS[1:3]
    for index, signal in enumerate(dates, 1):
        for tier in TIERS:
            week = groups[(signal,tier)]
            features = week[FEATURES].to_numpy()
            equal = features[:, [0,2]].mean(axis=1)
            history = [day for day in dates if day < signal and (day,tier) in moments
                       and moments[(day,tier)][0] <= signal][-104:]
            ready = len(history) >= 52
            base = week[["signal_date","execution_date","code","vol_tier"]].assign(fit_status="fitted" if ready else "warmup")
            scores.append(base.assign(strategy=MODELS[0], score=equal))
            if not ready:
                scores.extend(base.assign(strategy=name, score=equal) for name in MODELS[1:])
                continue
            past, now = state_z(states, history, signal)
            normal, rhs = np.zeros((12,12)), np.zeros(12)
            tree_x, tree_y, tree_w = [], [], []
            for day, z in zip(history,past,strict=True):
                exit_date, xs, ys, gram, response = moments[(day,tier)]
                state = np.r_[1.,z]
                normal += np.kron(np.outer(state,state),gram)
                rhs += np.kron(state,response)
                tree_x.append(np.c_[xs,np.broadcast_to(z,(len(xs),3))])
                tree_y.append(ys)
                tree_w.append(np.full(len(xs),1/len(xs)))
                audit.append((signal,tier,day,exit_date,len(ys)))
            normal /= len(history)
            rhs /= len(history)
            for name, chosen in zip(ridge_names,[[0,2],[0,1,2]],strict=True):
                used = [block*3+factor for block in range(4) for factor in chosen]
                penalty = np.tile([.05,.5,.5,.5],(len(chosen),1)).T.ravel()
                beta = np.linalg.solve(normal[np.ix_(used,used)]+np.diag(penalty),rhs[used]).reshape(4,-1)
                effective = np.r_[1.,now] @ beta
                scores.append(base.assign(strategy=name, score=np.sum(features[:,chosen]*effective,axis=1)))
                for factor, value in zip(chosen,effective,strict=True):
                    coefficients.append((signal,tier,name,FACTORS[factor],value))
            tree_x, tree_y = np.concatenate(tree_x), np.concatenate(tree_y)
            weights = np.concatenate(tree_w) * len(tree_y) / len(history)
            data = lgb.Dataset(tree_x,label=tree_y,weight=weights,feature_name=FEATURES+STATES)
            model = lgb.train(TREE,data,num_boost_round=64)
            pred = model.predict(np.c_[features,np.broadcast_to(now,(len(features),3))],num_threads=2)
            scores.append(base.assign(strategy=MODELS[3],score=pred))
        if index % 50 == 0:
            print(f"Scored {index}/{len(dates)} signal weeks",flush=True)
    scored = smooth(pd.concat(scores,ignore_index=True))
    require(np.isfinite(scored[["score","score_ema05"]]).all().all(),"Nonfinite scores")
    links = pd.DataFrame(audit,columns=["signal_date","vol_tier","training_signal_date","training_exit_date","rows"])
    require(links.training_exit_date.le(links.signal_date).all() and links.training_signal_date.lt(links.signal_date).all(),
            "Training leakage")
    eligible = sample[["signal_date","code","vol_tier",LABEL]]
    quality = scored.loc[scored.fit_status.eq("fitted")].merge(eligible,on=["signal_date","code","vol_tier"],
                                                              validate="many_to_one").dropna(subset=[LABEL])
    metrics = []
    for (name,day,tier), week in quality.groupby(["strategy","signal_date","vol_tier"]):
        target = week[LABEL].rank()
        for field, gamma in [("score",0.),("score_ema05",.5)]:
            rank = week[field].rank()
            ic = rank.corr(target) if rank.nunique() > 1 and target.nunique() > 1 else np.nan
            metrics.append((name,day,tier,gamma,len(week),ic))
    weekly_ic = pd.DataFrame(metrics,columns=["strategy","signal_date","vol_tier","gamma","stocks","rank_ic"])
    weekly_ic["sample"] = np.where(weekly_ic.signal_date.dt.year.le(2024),"development","retrospective_2025")
    output.mkdir(parents=True,exist_ok=True)
    scored.to_parquet(output/"tiered_weekly_scores.parquet",index=False)
    links.to_parquet(output/"tiered_training_audit.parquet",index=False)
    weekly_ic.to_parquet(output/"tiered_weekly_ic.parquet",index=False)
    weekly_ic.groupby(["sample","strategy","vol_tier","gamma"])["rank_ic"].agg(["mean","count"]).reset_index().to_csv(
        output/"tiered_ic_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(coefficients,columns=["signal_date","vol_tier","strategy","factor","effective_weight"]).to_parquet(
        output/"tiered_ridge_weights.parquet",index=False)
    (output/"model_spec.json").write_text(json.dumps({"models":MODELS,"window":[52,104],"ridge":[.05,.5],
        "tree":TREE,"rounds":64,"gamma":[0,.5],"include_review":include_review},indent=2),encoding="utf-8")
    print("Scored rows:",len(scored),"| Training links:",len(links),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--tier-data-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--include-review",action="store_true",help="Also score the already revealed 2025 sample")
    options = parser.parse_args()
    root = options.project_root.resolve()
    default = root/"research/v2_regime_tiered/results"
    fit(root,options.tier_data_dir or default,options.output_dir or default/"stage2",options.include_review)

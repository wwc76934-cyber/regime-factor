"""Audit oriented factor contributions and replay Ridge with nonnegative effective weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_v3_1_portfolio import FUSION, scores_by_variant, zscore
from prepare_v3_1_base import initialize
from train_v3_1_models import COMMON, KEY, RIDGE, TREE, transform_inputs


def recalculate(root, audit, constrained):
    v3,_,_,_,_=initialize(root)
    result = root/"research/v3_1/results"
    step1,step2 = result/"step1",result/"step2"
    cfg = json.loads((step1/"v3_1_selected_factors.json").read_text(encoding="utf-8"))
    factors = cfg["selected"]
    features = pd.read_parquet(step1/"factor_panel.parquet",
        columns=KEY+["signal_bar_usable","BETA_120","VOL_20",*dict.fromkeys([*factors,*COMMON])])
    features = features.loc[features.signal_bar_usable & features[["BETA_120","VOL_20",*COMMON]].notna().all(axis=1)].sort_values(KEY)
    ranks = []
    for day,group in features.groupby("signal_date"):
        rank = pd.DataFrame(transform_inputs(group,factors,cfg["directions"]),columns=factors)
        ranks.append(pd.concat([group[KEY].reset_index(drop=True),rank],axis=1))
    design = pd.concat(ranks,ignore_index=True)
    annual,monthly,weekly = [],[],[]
    for h in (5,10,20):
        src = step2/f"h{h}"
        scores = pd.read_parquet(src/"predictions.parquet")
        beta = pd.read_csv(src/"ridge_effective_weights.csv",parse_dates=["signal_date"]).set_index("signal_date")
        ridge = scores.loc[scores.strategy.eq(RIDGE),KEY+["score"]].merge(design,on=KEY,validate="one_to_one")
        original = beta.reindex(ridge.signal_date)[factors].to_numpy()
        observed = ridge[factors].to_numpy()
        if not np.allclose((observed*original).sum(axis=1),ridge.score,atol=1e-11,rtol=1e-10):
            raise ValueError("Rolling Ridge forecasts did not reproduce from their exported weights")
        native = beta.reset_index().assign(horizon=h,model=RIDGE,method="exact_effective_coefficient")
        weekly.append(native)
        labels = pd.read_parquet(step1/"labels.parquet",filters=[("horizon","==",h)])
        fusion,_ = scores_by_variant(scores,labels,FUSION,.3)
        proxy = [scores.loc[scores.strategy.eq(TREE),KEY+["score"]].assign(model=TREE),
                 fusion[KEY+["score"]].assign(model="fusion_50_50_ema03")]
        if h in (5,10):
            revised = scores.copy()
            replay = ridge[KEY].assign(score_with_floor=(observed*np.maximum(original,0)).sum(axis=1))
            revised = revised.merge(replay,on=KEY,how="left",validate="many_to_one")
            fixed = revised.strategy.eq(RIDGE)
            if revised.loc[fixed,"score_with_floor"].isna().any():
                raise ValueError("A constrained forecast was not matched")
            revised.loc[fixed,"score"] = revised.loc[fixed,"score_with_floor"]
            revised = revised.drop(columns="score_with_floor")
            v3.save(revised,constrained/f"h{h}"/"predictions.parquet")
            weekly.append(native.assign(model="v3_1_state_ridge_nonnegative",**{f:native[f].clip(lower=0) for f in factors}))
            adjusted_fusion,_ = scores_by_variant(revised,labels,FUSION,.3)
            proxy.append(adjusted_fusion[KEY+["score"]].assign(model="fusion_50_50_ema03_nonnegative_ridge"))
        coefs = []
        for (model,day),group in pd.concat(proxy,ignore_index=True).merge(design,on=KEY,validate="many_to_one").groupby(["model","signal_date"]):
            x = group[factors].to_numpy(dtype=float,copy=True)
            x -= x.mean(axis=0)
            y = zscore(group.score).to_numpy()
            estimated = np.linalg.lstsq(x,y,rcond=None)[0]
            coefs.append((day,h,model,"linear_projection_of_normalized_scores",*estimated))
        weekly.append(pd.DataFrame(coefs,columns=["signal_date","horizon","model","method",*factors]))
    w = pd.concat(weekly,ignore_index=True)
    m = w.melt(id_vars=["signal_date","horizon","model","method"],value_vars=factors,var_name="factor",value_name="weight")
    m["year"] = m.signal_date.dt.year
    m["month"] = m.signal_date.dt.strftime("%Y-%m")
    m["negative"] = m.weight.lt(0)
    keys = ["model","horizon","method","factor"]
    summary = dict(sequences=("weight","size"),mean_net_weight=("weight","mean"),negative_share=("negative","mean"),
                   min_net_weight=("weight","min"),max_net_weight=("weight","max"))
    for name,table in [("annual_effective_weights.csv",m.groupby(["year",*keys],as_index=False).agg(**summary)),
                       ("monthly_2024_effective_weights.csv",m.loc[m.year.eq(2024)].groupby(["month",*keys],as_index=False).agg(**summary)),
                       ("weekly_effective_weights.parquet",m)]:
        v3.save(table,audit/name)
    print(m.loc[m.year.eq(2024)&m.model.eq(RIDGE)&m.factor.isin(["EP_TTM","SP_TTM"])].groupby(["horizon","factor"])
          .agg(mean=("weight","mean"),negative_share=("negative","mean")).to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path)
    parser.add_argument("--constrained",type=Path)
    args=parser.parse_args()
    base=args.project_root/"research/v3_1/results/step4"
    recalculate(args.project_root.resolve(),args.output or base,args.constrained or base/"nonnegative_forecasts")

"""Expanding, mature-label development replay for the V4.1 state router."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from v4_1_regime_moe_model import RegimeMoE, ic_moments, objective

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/v3_1"))
from train_v3_1_models import evaluate, KEY, RIDGE, TREE

FACTORS=["RET_5","CORR_PV_10","MOM_120_SKIP20","HIGH_52W_PROX","EP_TTM","SP_TTM"]
EXPERTS=["fast","trend","value"]
SEEDS=[42,1024,2026,777,999]
MOE="v4_1_regime_moe"
END=pd.Timestamp("2025-12-31")


def standardize(values,winsor):
    frame=values.where(np.isfinite(values))
    if winsor:
        mu,sd=frame.mean(),frame.std(ddof=0)
        frame=frame.clip(mu-3*sd,mu+3*sd,axis=1)
    return ((frame-frame.mean())/frame.std(ddof=0).replace(0,np.nan)).fillna(0.)


def prepare(source):
    factor=pd.read_parquet(source/"step1/factor_panel.parquet",columns=KEY+FACTORS+["signal_bar_usable","BETA_120"])
    labels=pd.read_parquet(source/"step1/labels.parquet",filters=[("horizon","==",5)])
    config=json.loads((source/"step1/v3_1_selected_factors.json").read_text(encoding="utf-8"))
    state=pd.read_parquet(source/"step1/states.parquet").set_index("signal_date")[config["states"]].sort_index()
    if max(factor.signal_date.max(),labels.signal_date.max(),labels.exit_date.max(),state.index.max())>END:
        raise ValueError("Development cutoff exceeded")
    if set(FACTORS)!=set(config["selected"]) or not state.index.is_unique or factor.duplicated(KEY).any():
        raise ValueError("Factor configuration or input keys inconsistent")
    joint=factor.sort_values(KEY).merge(labels[KEY+["risk_adjusted_target","entry_date","exit_date"]],
                                       on=KEY,validate="one_to_one",how="left",indicator=True)
    if not joint._merge.eq("both").all() or len(joint)!=len(labels):
        raise ValueError("Stock/label keys do not match")
    weeks=pd.DatetimeIndex(joint.signal_date.unique())
    if not weeks.equals(state.index) or not joint.groupby("signal_date").size().eq(300).all():
        raise ValueError("Incomplete point-in-time universe")
    vectors=[]
    for t,g in joint.groupby("signal_date",sort=True):
        aligned=standardize(g[FACTORS].mul(pd.Series(config["directions"]),axis=1),True)
        group=pd.DataFrame({e:aligned[FACTORS[2*j:2*j+2]].mean(axis=1) for j,e in enumerate(EXPERTS)})
        vectors.append(standardize(group,False).to_numpy("float32"))
    e=torch.from_numpy(np.stack(vectors))
    y=torch.from_numpy(joint.risk_adjusted_target.to_numpy("float32").reshape(-1,300))
    z=state.to_numpy("float32")
    seq=np.full((len(weeks),12,5),np.nan,dtype=np.float32)
    for i in range(11,len(weeks)):
        seq[i]=z[i-11:i+1]
    by_week=joint.groupby("signal_date")
    if not by_week.exit_date.nunique().le(1).all():
        raise ValueError("Mixed exits")
    matured=joint.loc[joint.risk_adjusted_target.notna()]
    if not (matured.signal_date.lt(matured.entry_date)&matured.entry_date.lt(matured.exit_date)).all():
        raise ValueError("Invalid holding windows")
    return joint,labels,weeks,e,y,z,torch.from_numpy(seq),by_week.exit_date.first().to_numpy()


def plot_gate_history(gates,source,out):
    idx=pd.read_parquet(source/"step0/H00300_development.parquet").set_index("date").close.reindex(gates.index)
    if idx.isna().any():
        raise ValueError("Missing benchmark close at a forecast date")
    fig,(ax,market)=plt.subplots(2,1,figsize=(11,6),sharex=True,height_ratios=(2,1),layout="constrained")
    ax.stackplot(gates.index,*gates.to_numpy().T,labels=EXPERTS,colors=["#438598","#d19445","#626c9f"])
    ax.set(ylim=(0,1),ylabel="Expert weight",title="Weekly market routing | Five-seed average")
    ax.legend(ncol=3,loc="upper left")
    market.plot(idx.index,idx/idx.iloc[0]*100,color="#354254")
    market.set(ylabel="H00300 (start = 100)",xlabel="Signal date")
    market.grid(axis="y",alpha=.25)
    fig.savefig(out/"gate_weights_history.png",dpi=160)
    plt.close(fig)


def fit(root):
    torch.set_num_threads(2)
    source=root/"research/v3_1/results"
    out=root/"research/v4_1/results/step1"
    out.mkdir(parents=True,exist_ok=True)
    joint,labels,weeks,e,y,z,seq,exits=prepare(source)
    gram,xy,yy=ic_moments(e,y)
    good=seq.isfinite().all(dim=(1,2)).numpy()
    valid=good & (y.isfinite().sum(1).numpy()>=50) & (yy.numpy()>0)
    forecast=np.flatnonzero(good & (weeks.year>=2017))
    if len(forecast)!=460:
        raise ValueError("Unexpected development forecast calendar")
    logs,weights,perseed=[],[],[]
    for offset in range(0,len(forecast),12):
        ids=forecast[offset:offset+12]
        i=ids[0]
        train=np.flatnonzero(valid & (np.arange(len(weeks))<i) & (exits<=weeks[i].to_datetime64()))
        if len(train)<52 or (logs and len(train)<logs[-1]["training_weeks"]):
            raise ValueError("Invalid expanding window")
        center=z[train].mean(0)
        scale=z[train].std(0)
        norm=((seq-torch.from_numpy(center))/torch.from_numpy(np.where(scale>0,scale,1.))).clamp(-3,3)
        adjacent=torch.from_numpy(np.diff(train)==1)
        draws=[]
        for seed in SEEDS:
            torch.manual_seed(seed)
            net=RegimeMoE()
            opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.02)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)
            for _ in range(25):
                opt.zero_grad(set_to_none=True)
                w=net(norm[train])
                loss=objective(w,gram[train],xy[train],yy[train],adjacent)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                scheduler.step()
            net.eval()
            with torch.inference_mode():
                w=net(norm[ids]).numpy()
            draws.append(w)
            perseed.append(pd.DataFrame(w,columns=EXPERTS).assign(signal_date=weeks[ids],refit=weeks[i],seed=seed))
            logs.append(dict(refit=weeks[i],seed=seed,epochs=25,training_weeks=len(train),first_train=weeks[train[0]],
                             last_train=weeks[train[-1]],latest_training_exit=exits[train].max(),
                             first_scored=weeks[ids[0]],last_scored=weeks[ids[-1]],last_training_loss=float(loss.detach())))
        weights.extend(np.mean(draws,axis=0))
    w=np.stack(weights)
    if not (np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-6)):
        raise ValueError("Invalid routing distribution")
    scores=np.einsum("tne,te->tn",e.numpy()[forecast],w)
    selected=joint.loc[joint.signal_date.isin(weeks[forecast])].copy()
    selected["score"]=scores.ravel()
    selected["strategy"]=MOE
    pred=selected.loc[selected.signal_bar_usable,KEY+["entry_date","BETA_120","score","strategy"]]
    pred.to_parquet(out/"predictions.parquet",index=False)
    gates=pd.DataFrame(w,columns=EXPERTS,index=weeks[forecast]).rename_axis("signal_date")
    gates.to_csv(out/"gate_weights.csv")
    pd.concat(perseed).to_csv(out/"gate_weights_by_seed.csv",index=False)
    pd.DataFrame(logs).to_csv(out/"training_windows.csv",index=False)
    legacy=pd.read_parquet(source/"step4/nonnegative_forecasts/h5/predictions.parquet",
                           filters=[("strategy","==",RIDGE)])
    pair=pred.merge(legacy[KEY],on=KEY,validate="one_to_one")
    if len(pair)!=len(legacy):
        raise ValueError("V3.1 reference pool not fully covered")
    base=legacy.assign(strategy="v3_1_nonnegative_ridge")
    monthly,audit=evaluate(pd.concat([pair,base],ignore_index=True),labels,5)
    _,own=evaluate(pred,labels,5)
    pd.concat([audit.assign(universe="paired"),own.assign(universe="all_usable")]).to_csv(out/"v4_1_signal_audit.csv",index=False)
    monthly.to_parquet(out/"paired_weekly_metrics.parquet",index=False)
    correlations=np.stack([pd.DataFrame(x,columns=EXPERTS).corr(method="spearman").to_numpy() for x in e.numpy()[forecast]])
    pd.DataFrame(np.nanmean(correlations,axis=0),index=EXPERTS,columns=EXPERTS).to_csv(out/"expert_rank_correlation.csv")
    plot_gate_history(gates,source,out)
    print(audit[["strategy","period","rank_ic","icir","top30_hit","top30_excess_H00300"]].to_string(index=False))
    print(len(logs),"seed fits;",sum(p.numel() for p in net.parameters()),"parameters")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    fit(p.parse_args().project_root.resolve())

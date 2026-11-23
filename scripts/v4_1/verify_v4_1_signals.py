"""Reconstruct stored routing decisions and verify the masked IC objective and gradient."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_v4_1_regime_moe import ROOT, SEEDS, EXPERTS, FACTORS, KEY, prepare
from v4_1_regime_moe_model import RegimeMoE, ic_moments, objective


def verify(root):
    torch.set_num_threads(2)
    out=root/"research/v4_1/results/step1"
    joint,_,weeks,e,y,_,seq,exits=prepare(root/"research/v3_1/results")
    logs=pd.read_csv(out/"training_windows.csv",parse_dates=["refit","latest_training_exit","first_train","last_train"])
    draw=pd.read_csv(out/"gate_weights_by_seed.csv",parse_dates=["signal_date","refit"])
    w=pd.read_csv(out/"gate_weights.csv",parse_dates=["signal_date"]).set_index("signal_date")[EXPERTS]
    param_count=sum(p.numel() for p in RegimeMoE().parameters())
    assert param_count<1000 and len(w)==460 and len(logs)==39*len(SEEDS)
    np.testing.assert_allclose(draw.groupby("signal_date")[EXPERTS].mean(),w,atol=1e-7)
    np.testing.assert_allclose(w.sum(axis=1),1,atol=2e-7)
    assert (w>=0).all().all() and logs.epochs.eq(25).all() and logs.latest_training_exit.le(logs.refit).all()
    valid=seq.isfinite().all(dim=(1,2)).numpy() & (y.isfinite().sum(1).numpy()>=50) & (y.nan_to_num().var(1).numpy()>0)
    for day,g in logs.groupby("refit"):
        ids=np.flatnonzero(valid & (weeks<day) & (exits<=day.to_datetime64()))
        assert len(g)==5 and set(g.seed)==set(SEEDS) and g.training_weeks.eq(len(ids)).all()
        assert g.first_train.eq(weeks[ids[0]]).all() and g.last_train.eq(weeks[ids[-1]]).all()
    idx=weeks.get_indexer(w.index)
    p=joint.loc[joint.signal_date.isin(w.index),KEY+["signal_bar_usable"]].copy()
    p["score"]=np.einsum("tne,te->tn",e.numpy()[idx],w.to_numpy()).ravel()
    p=p[p.signal_bar_usable].set_index(KEY).score.sort_index()
    saved=pd.read_parquet(out/"predictions.parquet").set_index(KEY).score.sort_index()
    pd.testing.assert_index_equal(p.index,saved.index)
    np.testing.assert_allclose(p,saved,atol=1e-6)
    # Independent direct-stock Pearson calculation on multiple incomplete cross-sections.
    sample=np.flatnonzero(valid)[:15]
    experts,target=e[sample].double(),y[sample].double()
    weights=torch.linspace(.15,.65,45,dtype=torch.float64).reshape(-1,3).softmax(1).requires_grad_()
    adj=torch.tensor(np.diff(sample)==1)
    fast=objective(weights,*ic_moments(experts,target),adj)
    rho=[]
    for a,b,c in zip(experts,target,weights,strict=True):
        mask=b.isfinite()
        rho.append(torch.corrcoef(torch.stack([a[mask]@c,b[mask]]))[0,1])
    r=torch.stack(rho)
    direct=-r.mean()/(r.std(unbiased=False)+.05)+.15*weights.diff(dim=0).abs().sum(1)[adj].mean()+.05*(weights-1/3).square().sum(1).mean()
    torch.testing.assert_close(fast,direct)
    torch.testing.assert_close(torch.autograd.grad(fast,weights)[0],torch.autograd.grad(direct,weights)[0])
    params=dict(factors=FACTORS,model_parameters=param_count,temperature=1.5,seeds=SEEDS,epochs=25,
                refits=logs.refit.nunique(),prediction_weeks=len(w),seed_fits=len(logs),matched_prediction_rows=len(saved),
                reproduced_scores=True,exact_masked_icir_loss_and_gradient=True,
                min_train=int(logs.training_weeks.min()),max_train=int(logs.training_weeks.max()),
                first_training_signal=str(logs.first_train.min().date()),max_prediction_date=str(w.index.max().date()),
                torch_version=str(torch.__version__))
    (out/"verification.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    print(json.dumps(params))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    verify(p.parse_args().project_root.resolve())

"""Continue the frozen tree, Ridge and MoE training calendars from their last development block."""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from prepare_final_inputs import ROOT, WORK, OLD, DEST, ANCHOR, LAST, KEY, market
import train_v3_1_models as linear
import train_v4_1_regime_moe as deep


def traditional():
    spec = json.loads((WORK/"step1/v3_1_selected_factors.json").read_text())
    f = pd.read_parquet(WORK/"step1/factor_panel.parquet").sort_values(KEY)
    f = f.loc[f.signal_bar_usable & f[["BETA_120","VOL_20",*linear.COMMON]].notna().all(axis=1)]
    l = pd.read_parquet(WORK/"step1/labels.parquet")
    z = pd.read_parquet(WORK/"step1/states.parquet")[["signal_date",*spec["states"]]]
    weeks = linear.weeks_from(f, l, z, spec["selected"], spec["directions"])
    old = pd.read_parquet(OLD/"step4/nonnegative_forecasts/h5/predictions.parquet")
    ref = old.loc[old.strategy.isin([linear.RIDGE,linear.TREE])]
    calendar = pd.DatetimeIndex(sorted(ref.signal_date.unique()))
    first = calendar[(len(calendar)-1)//linear.REFIT*linear.REFIT]
    with patch.object(linear,"PREDICT_FIRST",first), patch.object(linear,"DEVELOPMENT_LAST",LAST):
        pred, windows, coef = linear.produce(weeks,2)
    scores = []
    blocks = {w.day:w for w in weeks}
    for day,*weights in coef:
        w = blocks[day]
        scores.append(w.output.assign(strategy=linear.RIDGE,score=w.x @ np.maximum(weights,0)))
    scores = pd.concat([*scores, pred.loc[pred.strategy.eq(linear.TREE)]],ignore_index=True)
    keys = [*KEY, "strategy"]
    prior = ref.loc[ref.signal_date.ge(first)].set_index(keys).sort_index()
    recovered = scores.set_index(keys).reindex(prior.index)
    np.testing.assert_allclose(prior.score, recovered.score, rtol=1e-7, atol=1e-7)
    market.save(windows, DEST/"audit/traditional_refits.csv")
    return scores.loc[scores.signal_date.ge(ANCHOR)].copy()


def neural():
    torch.set_num_threads(2)
    with patch.object(deep,"END",LAST):
        joint,_,weeks,e,y,z,seq,exits = deep.prepare(WORK)
    gram,xy,yy = deep.ic_moments(e,y)
    good = seq.isfinite().all(dim=(1,2)).numpy()
    valid = good & (y.isfinite().sum(1).numpy()>=50) & (yy.numpy()>0)
    forecast = np.flatnonzero(good & (weeks.year>=2017))
    prior = pd.read_csv(ROOT/"research/v4_1/results/step1/gate_weights.csv",parse_dates=["signal_date"]).set_index("signal_date")
    if not prior.index.equals(weeks[forecast[:len(prior)]]):
        raise ValueError("MoE forecast calendar drift")
    start = (len(prior)-1)//12*12
    route,logs = [],[]
    for offset in range(start,len(forecast),12):
        ids = forecast[offset:offset+12]
        i = ids[0]
        train = np.flatnonzero(valid & (np.arange(len(weeks))<i) & (exits<=weeks[i].to_datetime64()))
        if len(train)<52:
            raise ValueError("Insufficient mature weeks")
        center,scale = z[train].mean(0),z[train].std(0)
        norm = ((seq-torch.from_numpy(center))/torch.from_numpy(np.where(scale>0,scale,1.))).clamp(-3,3)
        adjacent = torch.from_numpy(np.diff(train)==1)
        draws = []
        for seed in deep.SEEDS:
            torch.manual_seed(seed)
            net = deep.RegimeMoE()
            opt = torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.02)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)
            for _ in range(25):
                opt.zero_grad(set_to_none=True)
                loss = deep.objective(net(norm[train]),gram[train],xy[train],yy[train],adjacent)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                scheduler.step()
            net.eval()
            with torch.inference_mode():
                draws.append(net(norm[ids]).numpy())
            logs.append((weeks[i],seed,len(train),weeks[train[0]],exits[train].max(),weeks[ids[-1]]))
        route.extend(np.mean(draws,axis=0))
    w = np.stack(route)
    ids = forecast[start:]
    gates = pd.DataFrame(w,index=weeks[ids],columns=deep.EXPERTS)
    np.testing.assert_allclose(prior.iloc[start:],gates.loc[prior.index[start:]],atol=1e-6)
    if not (np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1)):
        raise ValueError("Invalid gate")
    result = joint.loc[joint.signal_date.isin(weeks[ids]),KEY].copy()
    result["score"] = np.einsum("tne,te->tn",e.numpy()[ids],w).ravel()
    market.save(gates.rename_axis("signal_date").reset_index(),DEST/"audit/gate_weights.csv")
    market.save(pd.DataFrame(logs,columns=["refit","seed","training_weeks","first_signal","latest_exit","last_signal"]),
                DEST/"audit/moe_refits.csv")
    return result.loc[result.signal_date.ge(ANCHOR)]


def main():
    t = traditional()
    d = neural().set_index(KEY)
    market.save(t,WORK/"baseline_predictions.parquet")
    primary = t.copy()
    mask = primary.strategy.eq(linear.RIDGE)
    primary.loc[mask,"score"] = d.score.reindex(pd.MultiIndex.from_frame(primary.loc[mask,KEY])).to_numpy()
    if primary.score.isna().any():
        raise ValueError("Missing paired score")
    market.save(primary,WORK/"primary_predictions.parquet")
    print("Continued forecasts:", t.signal_date.nunique(), "weeks; development forecast overlap reproduced")


if __name__=="__main__":
    main()

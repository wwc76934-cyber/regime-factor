"""Optional full-epoch NumPy gate, with purged time validation, for the 10-session label."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v3_1_models import (COMMON, KEY, MIN_WEEKS, PREDICT_FIRST, REFIT, WINDOW, evaluate,
                              include_baseline, initialize, standardized, weeks_from)


NAME = "v3_1_mgg_numpy"
HORIZON = 10
VALIDATION_WEEKS = 12
MAX_EPOCHS = 5
PATIENCE = 2
BATCH_WEEKS = 16
MSE_WEIGHT = .15


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Gate:
    def __init__(self, states, features):
        self.rng = np.random.default_rng(731)
        sizes = [(states, 16), (16, features), (features, 16), (16, 1)]
        self.p = [self.rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in sizes]
        self.p += [np.zeros(b, dtype=np.float32) for _, b in sizes]
        self.m = [np.zeros_like(p) for p in self.p]
        self.v = [np.zeros_like(p) for p in self.p]
        self.step = 0

    def forward(self, x, state):
        w1,w2,w3,w4,b1,b2,b3,b4 = self.p
        pre = state @ w1 + b1
        h = np.maximum(0, pre)
        gg = h @ w2 + b2
        e = np.exp(gg - gg.max(axis=1, keepdims=True))
        gates = e / e.sum(axis=1, keepdims=True)
        contribution = x * (x.shape[-1] * gates[:, None, :])
        logit = contribution @ w3 + b3
        act = sigmoid(logit)
        hidden = logit * act
        score = (hidden @ w4 + b4)[...,0]
        return score, (pre,h,gates,contribution,logit,act,hidden)

    def batch(self, x, state, y, mask, top_weights, optimize):
        p, arrays = self.forward(x,state)
        n = mask.sum(axis=1,keepdims=True)
        pc = np.where(mask, p - (p*mask).sum(axis=1,keepdims=True)/n, 0)
        yc = np.where(mask, y - (y*mask).sum(axis=1,keepdims=True)/n, 0)
        pn = np.sqrt((pc*pc).sum(axis=1,keepdims=True) + 1e-7)
        yn = np.sqrt((yc*yc).sum(axis=1,keepdims=True) + 1e-7)
        corr = (pc*yc).sum(axis=1,keepdims=True)/(pn*yn)
        delta = p-y
        w_norm = top_weights / top_weights.sum(axis=1,keepdims=True)
        loss = (1-corr.ravel()+MSE_WEIGHT*(w_norm*delta**2).sum(axis=1)).mean()
        if not optimize:
            return float(loss)
        q = (-yc/(pn*yn)+corr*pc/(pn**2)+2*MSE_WEIGHT*w_norm*delta)/len(y)
        pre,h,gates,scaled,logit,act,hidden = arrays
        w1,w2,w3,w4,*_ = self.p
        q3 = q[...,None]*w4.ravel()*(act+logit*act*(1-act))
        qg = ((q3@w3.T)*x).sum(axis=1)*x.shape[-1]
        qlog = gates*(qg-(qg*gates).sum(axis=1,keepdims=True))
        qp = (qlog@w2.T)*(pre>0)
        grads = [state.T@qp,h.T@qlog,scaled.reshape(-1,scaled.shape[-1]).T@q3.reshape(-1,q3.shape[-1]),
                 hidden.reshape(-1,hidden.shape[-1]).T@q.reshape(-1,1),
                 qp.sum(axis=0),qlog.sum(axis=0),q3.sum(axis=(0,1)),np.array([q.sum()],dtype=np.float32)]
        self.step += 1
        for i, grad in enumerate(grads):
            self.m[i] = .9*self.m[i]+.1*grad
            self.v[i] = .999*self.v[i]+.001*grad**2
            self.p[i] -= .001*(.01*self.p[i]+(self.m[i]/(1-.9**self.step)) /
                               (np.sqrt(self.v[i]/(1-.999**self.step))+1e-8))
        return float(loss)

    def checkpoint(self):
        return tuple([a.copy() for a in group] for group in (self.p,self.m,self.v)), self.step

    def restore(self, checkpoint):
        (self.p,self.m,self.v),self.step = checkpoint


def pack(items, z):
    max_items = max(len(w.train_y) for w in items)
    x = np.zeros((len(items),max_items,items[0].train_x.shape[1]),dtype=np.float32)
    y = np.zeros((len(items),max_items),dtype=np.float32)
    mask = np.zeros_like(y,dtype=bool)
    weights = np.zeros_like(y)
    for i,w in enumerate(items):
        n = len(w.train_y)
        x[i,:n] = w.train_x
        y[i,:n] = w.train_y
        mask[i,:n] = True
        weights[i,:n] = 1
        weights[i,np.argsort(-w.train_y,kind="stable")[:30]] = 2
    return x,z.astype(np.float32),y,mask,weights


def produce(weeks, factors):
    predictions, logs, gating = [], [], []
    n = 0
    for offset,current in enumerate(weeks):
        if current.day < PREDICT_FIRST:
            continue
        history = [w for w in weeks[:offset] if pd.notna(w.exit) and w.exit <= current.day
                   and len(w.train_y) >= 50][-WINDOW:]
        if len(history) < MIN_WEEKS:
            continue
        if n % REFIT == 0:
            # The previous refit's validation can overlap this validation block.
            # Initialize anew so the checkpoint has not used any of these labels.
            model = Gate(len(current.state), len(factors))
            val = history[-VALIDATION_WEEKS:]
            train = [w for w in history[:-VALIDATION_WEEKS] if w.exit <= val[0].day]
            if not train:
                raise ValueError("Missing purged chronological training set")
            if not max(w.exit for w in train) <= val[0].day <= max(w.exit for w in val) <= current.day:
                raise ValueError("Gate chronological validation is not causal")
            z,_,center,scale = standardized(train,current)
            xt = pack(train,z)
            xv = pack(val,np.clip((np.stack([w.state for w in val])-center)/scale,-3,3))
            best_loss = np.inf
            stalls = 0
            first_log = len(logs)
            for epoch in range(1,MAX_EPOCHS+1):
                order = model.rng.permutation(len(train))
                losses = []
                for i in range(0,len(order),BATCH_WEEKS):
                    ids = order[i:i+BATCH_WEEKS]
                    loss = model.batch(*(a[ids] for a in xt),optimize=True)
                    losses.append((len(ids),loss))
                training_loss = sum(size*loss for size,loss in losses)/len(train)
                validation_loss = model.batch(*xv,optimize=False)
                logs.append([current.day,epoch,len(train),len(val),max(w.exit for w in train),val[0].day,
                             max(w.exit for w in val),training_loss,validation_loss,False])
                if validation_loss < best_loss-1e-6:
                    best_loss,best_model,best_epoch = validation_loss,model.checkpoint(),epoch
                    stalls = 0
                else:
                    stalls += 1
                if stalls >= PATIENCE:
                    break
            logs[first_log+best_epoch-1][-1] = True
            model.restore(best_model)
        now = np.clip((current.state-center)/scale,-3,3).astype(np.float32)[None,:]
        y,parts = model.forward(current.x[None,:,:],now)
        predictions.append(current.output.assign(strategy=NAME,score=y[0]))
        gating.append((current.day,*parts[2][0]))
        n += 1
        if n % 50 == 0:
            print(f"Gate predicted {n} weeks through {current.day:%Y-%m-%d}",flush=True)
    ledger = pd.DataFrame(logs,columns=["signal_date","epoch","train_weeks","validation_weeks","last_train_exit",
        "validation_first_signal","last_validation_exit","training_loss","validation_loss","selected_epoch"])
    return pd.concat(predictions,ignore_index=True),ledger,pd.DataFrame(gating,columns=["signal_date",*factors])


def run(root, step1, inputs, out):
    v3,_,_,_,_ = initialize(root)
    spec = json.loads((step1/"v3_1_selected_factors.json").read_text(encoding="utf-8"))
    v3.need(spec["status"]=="go","V3.1 factor gate has not passed")
    factors = spec["selected"]
    frame = pd.read_parquet(inputs/"factor_panel.parquet",
        columns=KEY+["signal_bar_usable","BETA_120","VOL_20",*dict.fromkeys([*factors,*COMMON])])
    frame = frame.loc[frame.signal_bar_usable & frame[["BETA_120","VOL_20",*COMMON]].notna().all(axis=1)].sort_values(KEY)
    labels = pd.read_parquet(inputs/"labels.parquet",filters=[("horizon","==",HORIZON)])
    states = pd.read_parquet(step1/"states.parquet")[["signal_date",*spec["states"]]]
    v3.need(frame.signal_date.dt.year.le(2025).all() and labels.exit_date.dropna().dt.year.le(2025).all(),
            "Development cannot contain 2026")
    weeks = weeks_from(frame,labels,states,factors,spec["directions"])
    scores,ledger,weights = produce(weeks,factors)
    scores = include_baseline(scores,root/"research/v3/results/step3/predictions.parquet")
    scores = scores.loc[scores.strategy.eq(NAME)]
    weekly,metrics = evaluate(scores,labels,HORIZON)
    for name,data in [("predictions.parquet",scores),("training_curve.csv",ledger),("gate_weights.csv",weights),
                      ("weekly_signal_metrics.parquet",weekly),("model_signal_audit.csv",metrics)]:
        v3.save(data,out/name)
    (out/"model_spec.json").write_text(json.dumps({"horizon":HORIZON,"backend":"numpy","epoch_limit":MAX_EPOCHS,
        "validation_weeks":VALIDATION_WEEKS,"patience":PATIENCE,"batch_weeks":BATCH_WEEKS,
        "refit_weeks":REFIT,"mse_weight":MSE_WEIGHT,"optimizer":"AdamW","initialization":"cold_each_refit"},
        indent=2),encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--step1", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    study = a.project_root/"research/v3_1/results"
    run(a.project_root.resolve(),a.step1 or study/"step1",a.inputs or study/"step1",
        a.output or study/"step2"/"gate_h10")

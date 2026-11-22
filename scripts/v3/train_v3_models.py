"""Fit three causal rankers; the gate uses a CPU NumPy implementation of ListNet/AdamW."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build_v3_factors_and_labels import CORE, STATES, KEY
from fetch_and_audit_v3_data import need, save


TREE=dict(objective="lambdarank",metric="ndcg",ndcg_at=[30],lambdarank_truncation_level=33,
          learning_rate=.05,max_depth=3,num_leaves=7,min_data_in_leaf=300,lambda_l2=1.,
          force_col_wise=True,deterministic=True,num_threads=2,verbosity=-1,seed=731)
MODELS=["v1_rolling_104","core_long_raw","core_long_alpha","v3_state_ridge","v3_lgb_lambdarank","v3_master_gating_net","expanded_long_raw","v3_state_ridge_similarity"]


def softmax(a, axis=-1):
    exp=np.exp(a-a.max(axis=axis,keepdims=True))
    return exp/exp.sum(axis=axis,keepdims=True)


class Gate:
    def __init__(self,k,backend="numpy"):
        rng=np.random.default_rng(731)
        self.rng=rng
        self.p=[rng.normal(0,(2/a)**.5,size=(a,b)) for a,b in [(4,16),(16,k),(k,32),(32,1)]]
        self.p += [np.zeros(value.shape[1]) for value in self.p]
        self.m=[np.zeros_like(x) for x in self.p]
        self.v=[np.zeros_like(x) for x in self.p]
        self.n=0
        self.backend=backend
        if backend=="torch":
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(731)
            self.torch=torch
            self.params=[torch.nn.Parameter(torch.tensor(a,dtype=torch.float32)) for a in self.p]
            self.optimizer=torch.optim.AdamW(self.params,lr=.001,weight_decay=.01)

    def predict(self,x,z,train=False):
        w1,w2,w3,w4,b1,b2,b3,b4=self.p
        u=z@w1+b1
        h=np.maximum(u,0)
        gate=softmax(h@w2+b2)
        scaled=x*(x.shape[-1]*gate[:,None,:])
        h3=scaled@w3+b3
        sigmoid=1/(1+np.exp(-np.clip(h3,-40,40)))
        d=(self.rng.random(h3.shape)>=.2)/.8 if train else 1.
        rep=h3*sigmoid*d
        score=(rep@w4+b4)[...,0]
        return score,(u,h,gate,scaled,h3,sigmoid,d,rep)

    def step(self,x,z,y,mask):
        if self.backend=="torch":
            t=self.torch
            w1,w2,w3,w4,b1,b2,b3,b4=self.params
            xs,zs,ys=(t.as_tensor(v,dtype=t.float32) for v in (x,z,y))
            eligible=t.as_tensor(mask,dtype=t.bool)
            g=(t.relu(zs@w1+b1)@w2+b2).softmax(-1)*x.shape[-1]
            h=t.nn.functional.dropout(t.nn.functional.silu((xs*g[:,None,:])@w3+b3),p=.2,training=True)
            score=(h@w4+b4)[...,0].masked_fill(~eligible,-1e9)
            dist=(5*ys).masked_fill(~eligible,-1e9).softmax(-1)
            loss=-(dist*score.log_softmax(-1)).sum(-1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.p=[a.detach().numpy().astype(float) for a in self.params]
            return
        score,aux=self.predict(x,z,True)
        u,h,g,scaled,h3,sigmoid,d,rep=aux
        q=(softmax(np.where(mask,score,-1e9))-softmax(np.where(mask,5*y,-1e9)))/len(y)
        w1,w2,w3,w4,*_=self.p
        q3=q[...,None]*w4.ravel()*d*(sigmoid+h3*sigmoid*(1-sigmoid))
        qg=(np.sum((q3@w3.T)*x,axis=1))*x.shape[-1]
        qlog=g*(qg-np.sum(qg*g,axis=1,keepdims=True))
        q1=(qlog@w2.T)*(u>0)
        grads=[z.T@q1,h.T@qlog,np.einsum("bnk,bnh->kh",scaled,q3),
               np.einsum("bnh,bn->h",rep,q)[:,None],q1.sum(0),qlog.sum(0),q3.sum((0,1)),np.array([q.sum()])]
        self.n+=1
        for i,grad in enumerate(grads):
            self.m[i]=.9*self.m[i]+.1*grad
            self.v[i]=.999*self.v[i]+.001*grad**2
            self.p[i]-=.001*(.01*self.p[i]+(self.m[i]/(1-.9**self.n))/(np.sqrt(self.v[i]/(1-.999**self.n))+1e-8))


def transform(data,names,directions):
    signed=data[names].mul(pd.Series(directions).reindex(names),axis=1)
    ranks=signed.groupby(data.signal_date).rank(pct=True)
    return (ranks.fillna(.5)-.5).to_numpy()


def fit(data,states,spec,eval_first,eval_end,backend="numpy"):
    import lightgbm as lgb

    eligible=data.loc[data.signal_bar_usable&data[CORE].notna().all(axis=1)&data.BETA_120.notna()].copy()
    eligible=eligible.sort_values(KEY)
    eligible["_row"]=np.arange(len(eligible))
    features=spec["selected"]
    xs=transform(eligible,features,spec["directions"])
    xc=transform(eligible,CORE,{x:-1 for x in CORE})
    states=states.set_index("signal_date")[STATES]
    weeks={day:w for day,w in eligible.groupby("signal_date")}
    moments={}
    for day,w in weeks.items():
        valid=w.loc[w.alpha_return.notna()&w.forward_return.notna()]
        if len(valid)<50 or not np.isfinite(states.loc[day]).all():
            continue
        ar=valid.alpha_return.rank(pct=True).to_numpy()-.5
        rr=valid.forward_return.rank(pct=True).to_numpy()-.5
        ids=valid["_row"].to_numpy()
        moments[day]=(valid.exit_date.iat[0],ids,ar,rr)
    mats={}
    for day,(_,ids,ar,rr) in moments.items():
        ss,cc=xs[ids],xc[ids]
        ss=ss-ss.mean(0);cc=cc-cc.mean(0)
        mats[day]=(ss.T@ss/len(ids),ss.T@ar/len(ids),cc.T@cc/len(ids),cc.T@ar/len(ids),cc.T@rr/len(ids),ss.T@rr/len(ids))
    gate=Gate(len(features),backend)
    outputs,info,links=[],[],[]
    fitted=0
    for day,w in weeks.items():
        if not eval_first<=w.entry_date.iat[0]<=eval_end:
            continue
        dates=[past for past,value in moments.items() if past<day and value[0]<=day][-260:]
        if len(dates)<52 or not np.isfinite(states.loc[day]).all():
            continue
        fitted+=1
        history=states.loc[dates].to_numpy()
        center,scale=history.mean(0),history.std(0)
        scale=np.where(scale>0,scale,1.)
        z=np.clip((history-center)/scale,-3,3)
        now=np.clip((states.loc[day].to_numpy()-center)/scale,-3,3)
        ix=w["_row"].to_numpy()
        for model,source,target,window,idx in [
            (MODELS[0],xc,4,104,2),(MODELS[1],xc,4,260,2),(MODELS[2],xc,3,260,2)]:
            used=dates[-window:]
            matrix=np.mean([mats[p][idx] for p in used],axis=0)+.01*np.eye(3)
            rhs=np.mean([mats[p][target] for p in used],axis=0)
            outputs.append(w[KEY+["entry_date","BETA_120"]].assign(strategy=model,score=source[ix]@np.linalg.solve(matrix,rhs)))
        k=len(features)
        penalty=np.diag(np.r_[np.full(k,.05),np.full(k*4,.5)])
        for model, weights, label_moment in [
            (MODELS[6],np.ones(len(dates)),5),
            (MODELS[7],np.exp(-np.square(z-now).mean(axis=1)/2),1),
        ]:
            weights=weights/weights.sum()
            gram=sum(weight*np.kron(np.outer(np.r_[1.,state],np.r_[1.,state]),mats[p][0])
                     for p,state,weight in zip(dates,z,weights,strict=True))
            target=sum(weight*np.kron(np.r_[1.,state],mats[p][label_moment])
                       for p,state,weight in zip(dates,z,weights,strict=True))
            coefficients=np.linalg.solve(gram+penalty,target).reshape(5,k)
            outputs.append(w[KEY+["entry_date","BETA_120"]].assign(strategy=model,score=xs[ix]@(np.r_[1.,now]@coefficients)))
        normal=np.mean([np.kron(np.outer(np.r_[1.,state],np.r_[1.,state]),mats[p][0])
                        for p,state in zip(dates,z,strict=True)],axis=0)
        rhs=np.mean([np.kron(np.r_[1.,state],mats[p][1]) for p,state in zip(dates,z,strict=True)],axis=0)
        beta=np.linalg.solve(normal+np.diag(np.r_[np.full(k,.05),np.full(k*4,.5)]),rhs).reshape(5,k)
        outputs.append(w[KEY+["entry_date","BETA_120"]].assign(strategy=MODELS[3],score=xs[ix]@(np.r_[1.,now]@beta)))
        xt,yt,groups=[],[],[]
        nmax=max(len(moments[p][1]) for p in dates)
        gx=np.zeros((len(dates),nmax,k));gy=np.zeros((len(dates),nmax));mask=np.zeros_like(gy,dtype=bool)
        for j,(p,state) in enumerate(zip(dates,z,strict=True)):
            exit_day,ids,ar,_=moments[p]
            n=len(ids)
            xt.append(np.c_[xs[ids],np.broadcast_to(state,(n,4))])
            yt.append(np.clip(np.ceil((ar+.5)*5).astype(int)-1,0,4))
            groups.append(n)
            gx[j,:n]=xs[ids];gy[j,:n]=ar;mask[j,:n]=True
            links.append((day,p,exit_day,n))
        if fitted % 4 == 1:
            tree_center,tree_scale=center,scale
            booster=lgb.train(TREE,lgb.Dataset(np.concatenate(xt),label=np.concatenate(yt),group=groups),num_boost_round=80)
        score=booster.predict(np.c_[xs[ix],np.broadcast_to(np.clip((states.loc[day].to_numpy()-tree_center)/tree_scale,-3,3),(len(ix),4))],num_threads=2)
        outputs.append(w[KEY+["entry_date","BETA_120"]].assign(strategy=MODELS[4],score=score))
        # Mini-batches of whole weeks preserve the cross-sectional ListNet objective.
        for _ in range(30 if fitted==1 else 8):
            choice=gate.rng.choice(len(dates),size=min(8,len(dates)),replace=False)
            gate.step(gx[choice],z[choice],gy[choice],mask[choice])
        score,parts=gate.predict(xs[ix][None,:,:],now[None,:])
        outputs.append(w[KEY+["entry_date","BETA_120"]].assign(strategy=MODELS[5],score=score[0]))
        info.append({"signal_date":day,**dict(zip(features,parts[2][0],strict=True))})
        if fitted%50==0:
            print(f"{fitted} signal weeks scored through {day.date()}",flush=True)
    need(outputs,"No predictions")
    links=pd.DataFrame(links,columns=["signal_date","training_signal_date","training_exit_date","stocks"])
    need(links.training_signal_date.lt(links.signal_date).all()&links.training_exit_date.le(links.signal_date).all(),"Leakage")
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(info),links


def describe(scores,data):
    results=[]
    evals=scores.merge(data[KEY+["alpha_return","forward_return"]],on=KEY,validate="many_to_one")
    for (model,day),g in evals.groupby(["strategy","signal_date"]):
        for label in ["alpha_return","forward_return"]:
            p=g.dropna(subset=[label])
            if len(p)<50:continue
            top=p.sort_values(["score","code"],ascending=[False,True]).head(30)
            realized=p.sort_values([label,"code"],ascending=[False,True]).head(30)
            ranks=p.score.rank()
            ic=ranks.corr(p[label].rank()) if ranks.nunique()>1 else np.nan
            results.append((model,day,label,ic,len(set(top.code)&set(realized.code))/30))
    ic=pd.DataFrame(results,columns=["strategy","signal_date","label","rank_ic","top30_hit"])
    out=ic.groupby(["strategy","label"]).agg(weeks=("rank_ic","count"),rank_ic=("rank_ic","mean"),
          sd=("rank_ic","std"),top30_hit=("top30_hit","mean")).reset_index()
    out["icir"]=out.rank_ic/out.sd
    return ic,out


def run(source,out,backend):
    spec=json.loads((source/"features.json").read_text())
    data=pd.read_parquet(source/"factor_panel.parquet")
    states=pd.read_parquet(source/"states.parquet")
    need(data.signal_date.dt.year.le(2025).all(),"Development must not see 2026")
    scores,gates,links=fit(data,states,spec,pd.Timestamp("2017-01-01"),pd.Timestamp("2025-12-31"),backend)
    ic,table=describe(scores,data)
    for name,frame in [("predictions.parquet",scores),("gates.csv",gates),("training_links.parquet",links),
                       ("weekly_ic.parquet",ic),("model_metrics.csv",table)]:
        save(frame,out/name)
    (out/"model_spec.json").write_text(json.dumps({"backend":backend,"tree":TREE,"tree_rounds":80,"tree_refit_weeks":4,
        "training_window":260,"min_weeks":52,"gate_steps_first":30,"gate_steps_later":8,"gate_batch_weeks":8},indent=2))
    print(table.to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--source",type=Path)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--gate-backend",choices=["numpy","torch"],default="numpy")
    a=p.parse_args()
    base=a.project_root/"research/v3/results"
    run(a.source or base/"step2",a.output_dir or base/"step3",a.gate_backend)

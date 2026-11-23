"""Time a complete three-seed first refit before committing to the 2017–2025 replay."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch

from v4_tcsa_mgg_model import TCSAMGG, ic_loss

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42,1024,2026]


class Dataset:
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as d:
            self.x = torch.from_numpy(d["x_seq"])
            mask = d["valid_mask"]
            # Eligibility is observable on the signal date. Labels only mask the supervised loss.
            self.stock = torch.from_numpy(mask[:,:,-1,2] & mask.any(axis=(2,3)))
            self.y = torch.from_numpy(d["y"])
            self.state = torch.from_numpy(d["z_state"])
            self.signal,self.exit = d["signal_date"],d["exit_date"]
        self.state_ok = self.state.isfinite().all(dim=1).numpy()
        self.usable = ((self.stock & self.y.isfinite()).sum(dim=1)>=50).numpy() & self.state_ok
        self.refits = self.forecast_indices()[::12]

    def forecast_indices(self):
        return np.array([i for i,day in enumerate(self.signal) if day>=np.datetime64("2017-01-01")
                         and self.state_ok[i] and self.stock[i].any() and len(self.history(i))>=52])

    def history(self,i):
        return np.flatnonzero(self.usable & (self.exit<=self.signal[i]) & (self.signal<self.signal[i]))[-260:]

    def split(self,i):
        history=self.history(i)
        train,val = history[:-18],history[-16:]
        if len(history)<52 or self.exit[train].max()>=self.signal[val].min() or self.exit[val].max()>self.signal[i]:
            raise ValueError("Overlapping fit, validation or prediction windows")
        return train,val


def rank_ic(pred,y,mask):
    values=[]
    for a,b,ok in zip(pred,y,mask,strict=True):
        x,z=rankdata(a[ok]),rankdata(b[ok])
        if len(x)>=50 and x.std()>0 and z.std()>0:
            values.append(float(np.corrcoef(x,z)[0,1]))
    if not values:
        raise ValueError("No usable validation weeks")
    return float(np.mean(values))


def fit(data,i,seed,dim,batch,attention=True,gating=True):
    start=time.perf_counter()
    torch.manual_seed(seed)
    train,val=data.split(i)
    st=data.state[train]
    center,scale=st.mean(0),st.std(0,unbiased=False).clamp_min(1e-6)
    z=((data.state-center)/scale).clamp(-3,3)
    model=TCSAMGG(dim,attention,gating)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=35)
    rng=np.random.default_rng(seed)
    best=-np.inf
    for epoch in range(1,36):
        model.train()
        shuffled=rng.permutation(train)
        for start_idx in range(0,len(train),batch):
            idx=shuffled[start_idx:start_idx+batch]
            optimizer.zero_grad(set_to_none=True)
            scores,_=model(data.x[idx],z[idx],data.stock[idx])
            loss=ic_loss(scores,data.y[idx],data.stock[idx] & data.y[idx].isfinite())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            scores,_=model(data.x[val],z[val],data.stock[val])
        ic=rank_ic(scores.numpy(),data.y[val].numpy(),(data.stock[val]&data.y[val].isfinite()).numpy())
        if ic>best:
            best,best_epoch,saved=ic,epoch,deepcopy(model.state_dict())
        if epoch-best_epoch>=6:
            break
    model.load_state_dict(saved)
    record=dict(seed=seed,epochs=epoch,best_epoch=best_epoch,validation_ic=best,
                seconds=time.perf_counter()-start)
    return model,center,scale,record


def benchmark(data,dim,batch):
    i=data.refits[0]
    train,val=data.split(i)
    records=[]
    for seed in SEEDS:
        model,_,_,record=fit(data,i,seed,dim,batch)
        records.append(record)
        print(f"dim={dim} batch={batch} seed={seed} epochs={record['epochs']} seconds={record['seconds']:.1f}",flush=True)
    seconds=sum(r["seconds"] for r in records)
    # Scale by train batches; early stopping in later quarters can differ.
    relative_work=sum(math.ceil(len(data.split(j)[0])/batch) for j in data.refits)/math.ceil(len(train)/batch)
    return dict(dim=dim,batch_size=batch,parameters=sum(p.numel() for p in model.parameters()),
                refit_signal=str(data.signal[i]),training_weeks=len(train),purged_weeks=2,
                validation_weeks=len(val),last_training_exit=str(data.exit[train].max()),
                first_validation_signal=str(data.signal[val].min()),last_validation_exit=str(data.exit[val].max()),
                seeds=records,measured_three_seed_seconds=seconds,refits=len(data.refits),
                projected_minutes_same_quarter=seconds*len(data.refits)/60,
                projected_minutes_scaled_train_batches=seconds*relative_work/60)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    out=args.project_root/"research/v4"
    dest=out/"results/step1"
    data=Dataset(out/"cache/weekly_sequences.npz")
    report=json.loads((dest/"dataset.json").read_text(encoding="utf-8"))
    with (out/"cache/weekly_sequences.npz").open("rb") as cache:
        report["cache_sha256"] = hashlib.file_digest(cache, "sha256").hexdigest()
    report.update(torch_version=str(torch.__version__),device="cpu",threads=args.threads,seeds=SEEDS,
                  forecast_weeks=len(data.forecast_indices()),benchmarks=[])
    for dim,batch in [(24,16),(16,32)]:
        run=benchmark(data,dim,batch)
        report["benchmarks"].append(run)
        report["within_15_minute_budget"]=run["projected_minutes_scaled_train_batches"]<=15
        if report["within_15_minute_budget"] or dim==16:
            break
    report["selected_dim"],report["selected_batch_size"]=dim,batch
    (dest/"dataset_and_benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["benchmarks"]),flush=True)


if __name__=="__main__":
    main()

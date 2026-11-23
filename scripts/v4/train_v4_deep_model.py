"""Walk forward on the locked V4 dataset. Each completed refit is independently resumable."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_v4_cpu import Dataset, ROOT, SEEDS, fit

FULL = "v4_deep_tcsa_mgg"
ABLATIONS = ["without_attention", "without_gating"]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def predict(data, idx, model, center, scale):
    model.eval()
    active = data.stock[idx]
    with torch.inference_mode():
        scores, gates = model(data.x[idx], ((data.state[idx]-center)/scale).clamp(-3,3), active)
        n = active.sum(1,keepdim=True)
        centered = scores-(scores*active).sum(1,keepdim=True)/n
        sd = ((centered.square()*active).sum(1,keepdim=True)/n).sqrt()
        if (sd<=0).any():
            raise ValueError("Degenerate weekly score")
        scores = (centered/sd).masked_fill(~active,torch.nan)
    if not torch.isfinite(scores[active]).all() or not torch.isfinite(gates).all():
        raise ValueError("Invalid forecast")
    return scores.numpy(),gates.numpy()


def quarter(data, idx, name, dim, batch, file):
    train,val = data.split(idx[0])
    scores,gates,records = [],[],[]
    for seed in SEEDS:
        model,center,scale,record = fit(data,idx[0],seed,dim,batch,
                                        attention=name!="without_attention",gating=name!="without_gating")
        s,g = predict(data,idx,model,center,scale)
        scores.append(s)
        gates.append(g)
        records.append(dict(**record,strategy=name,refit=str(data.signal[idx[0]]),train_weeks=len(train),
                            train_start=str(data.signal[train[0]]),last_train_exit=str(data.exit[train].max()),
                            first_validation_signal=str(data.signal[val[0]]),last_validation_exit=str(data.exit[val].max())))
    tmp = file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,idx=idx,score=np.mean(scores,axis=0),gate=np.stack(gates),logs=json.dumps(records))
    tmp.replace(file)


def run(root,threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    study=root/"research/v4"
    cache=study/"cache/weekly_sequences.npz"
    spec=json.loads((study/"results/step1/dataset_and_benchmark.json").read_text(encoding="utf-8"))
    if digest(cache)!=spec["cache_sha256"]:
        raise ValueError("Dataset differs from Step 1")
    data=Dataset(cache)
    with np.load(cache,allow_pickle=False) as d:
        codes=d["codes"]
    dates=data.forecast_indices()
    if data.signal[dates[-1]]>np.datetime64("2025-12-31") or len(dates)!=spec["forecast_weeks"]:
        raise ValueError("Unexpected evaluation period")
    out=study/"results/step2"
    out.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).parent
    signature=dict(cache=spec["cache_sha256"],seeds=SEEDS,dim=spec["selected_dim"],batch=spec["selected_batch_size"],
                   cpu_threads=threads,torch_version=str(torch.__version__),ablation_refit="first",
                   sources={n:digest(src/n) for n in ["train_v4_deep_model.py","benchmark_v4_cpu.py","v4_tcsa_mgg_model.py"]})
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    parts=study/"cache"/"step2"/fingerprint
    parts.mkdir(parents=True,exist_ok=True)
    logs,predictions,gate_rows=[],[],[]
    for k,start in enumerate(range(0,len(dates),12)):
        idx=dates[start:start+12]
        for name in ([FULL,*ABLATIONS] if k==0 else [FULL]):
            file=parts/f"{data.signal[idx[0]]}_{name}.npz"
            if not file.exists():
                quarter(data,idx,name,signature["dim"],signature["batch"],file)
            with np.load(file,allow_pickle=False) as d:
                np.testing.assert_array_equal(idx,d["idx"])
                scores,gates=d["score"],d["gate"]
                logs.extend(json.loads(d["logs"].item()))
            sel,stock=np.nonzero(data.stock[idx].numpy())
            predictions.append(pd.DataFrame(dict(signal_date=data.signal[idx[sel]],code=codes[idx[sel],stock],
                                strategy=name,score=scores[sel,stock],refit=data.signal[idx[0]])))
            if name==FULL:
                for seed,gate in zip(SEEDS,gates,strict=True):
                    gate_rows.append(pd.DataFrame(gate,columns=[f"gate_{i:02d}" for i in range(gate.shape[1])])
                                     .assign(seed=seed,refit=data.signal[idx[0]],signal_date=data.signal[idx]))
        print(f"Refit {k+1}/{len(data.refits)} complete through {data.signal[idx[-1]]}",flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    if pred.duplicated(["strategy","signal_date","code"]).any() or not np.isfinite(pred.score).all():
        raise ValueError("Invalid merged predictions")
    pred.to_parquet(out/"predictions.parquet",index=False)
    pd.concat(gate_rows,ignore_index=True).to_csv(out/"weekly_gates_by_seed.csv",index=False)
    pd.DataFrame(logs).to_csv(out/"refit_audit.csv",index=False)
    (out/"training_spec.json").write_text(json.dumps(signature,indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    run(a.project_root.resolve(),a.threads)

"""Verify the cached point-in-time universe, label alignment and sampled raw-bar endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from build_v4_sequence_dataset import ROOT, END


def verify(root):
    out=root/"research/v4"
    raw=root/"data/raw/v3/development"
    labels=pd.read_parquet(root/"research/v3_1/results/step1/labels.parquet",filters=[("horizon","==",5)])
    labels=labels.set_index(["signal_date","code"]).sort_index()
    with np.load(out/"cache/weekly_sequences.npz",allow_pickle=False) as d:
        weeks,codes=d["signal_date"],d["codes"]
        x,valid=d["x_seq"],d["valid_mask"]
        y,entry,exit_dates=d["y"],d["entry_date"],d["exit_date"]
        calendar=d["sequence_dates"]
    if not np.isfinite(x).all() or (x[~valid]!=0).any() or np.max(weeks)>END.to_datetime64():
        raise ValueError("Invalid cached features")
    keys=pd.MultiIndex.from_arrays([pd.to_datetime(np.repeat(weeks,300)),codes.ravel()],names=labels.index.names)
    if not keys.is_unique or len(keys)!=len(labels) or not keys.isin(labels.index).all():
        raise ValueError("Unexpected point-in-time stock/date keys")
    aligned=labels.reindex(keys)
    np.testing.assert_equal(y.ravel(),aligned.risk_adjusted_target.to_numpy(dtype=np.float32))
    for recorded,col in [(entry,"entry_date"),(exit_dates,"exit_date")]:
        np.testing.assert_array_equal(np.repeat(recorded,300),aligned[col].to_numpy("datetime64[D]"))
    if not ((np.diff(calendar,axis=1)>np.timedelta64(0,"D")).all() and (calendar[:,-1]==weeks).all()):
        raise ValueError("Invalid feature chronology")
    rng=np.random.default_rng(42)
    checks=[]
    for i in sorted(rng.choice(np.flatnonzero(np.isfinite(y).any(1)),10,replace=False)):
        stamp=pd.Timestamp(weeks[i])
        names=codes[i]
        member=pd.read_parquet(raw/"members"/f"{stamp.date()}.parquet").code.sort_values()
        np.testing.assert_array_equal(names,member.to_numpy())
        paths=[str(raw/"raw"/str(stamp.year)/f"{code}.parquet") for code in names]
        date_field=ds.dataset(paths[0],format="parquet").schema.field("date")
        key=str(stamp.date()) if str(date_field.type) in ("string","large_string") else stamp.to_pydatetime()
        endpoint=ds.dataset(paths,format="parquet").to_table(
            columns=["code","open","close","tradestatus","high","low"],
            filter=(ds.field("date")==key)&(ds.field("adjustflag")=="3")).to_pandas().set_index("code").reindex(names)
        q=endpoint.apply(pd.to_numeric,errors="coerce")
        observed=(q.tradestatus==1)&(q.low>0)&(q.high>=np.maximum(q.open,q.close))&(q.low<=np.minimum(q.open,q.close))
        a=(q.close/q.open-1).where(observed)
        cropped=a.clip(a.mean()-3*a.std(ddof=0),a.mean()+3*a.std(ddof=0))
        expected=((cropped-cropped.mean())/cropped.std(ddof=0)).fillna(0)
        np.testing.assert_array_equal(valid[i,:,-1,2],observed.to_numpy())
        np.testing.assert_allclose(x[i,:,-1,2],expected,atol=2e-6,rtol=1e-6)
        checks.append(str(stamp.date()))
    result=dict(weekly_keys_checked=len(keys),signal_weeks=len(weeks),labels_match_v31_h5=True,
                all_missing_features_zero=True,all_windows_end_on_signal=True,
                independent_raw_intraday_and_universe_checks=checks)
    (out/"results/step1/verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    verify(p.parse_args().project_root.resolve())

"""Build the point-in-time, development-only TCSA-MGG input tensor."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3"))
from fetch_and_audit_v3_data import load_universe, ranges

CHANNELS = ["ret_1d","rel_ret_1d","intraday_ret","hl_range","close_pos","log_vol_ratio","turn_z"]
END = pd.Timestamp("2025-12-31")


def daily(cache, schedule, members, days, index):
    codes = pd.Index(sorted(members.code.unique()))
    fields = ["open","high","low","close","preclose","volume","turn","tradestatus"]
    a = {c:np.full((len(days),len(codes)),np.nan) for c in [*fields,"adj"]}
    for year, subset in ranges(schedule,members).items():
        for basis,cols,flag in [("raw",fields,"3"),("adjusted",["close"],"1")]:
            paths = [str(cache/basis/str(year)/f"{c}.parquet") for c in sorted(subset)]
            tab = ds.dataset(paths,format="parquet").to_table(columns=["date","code",*cols],
                                                               filter=ds.field("adjustflag")==flag).to_pandas()
            dates = pd.to_datetime(tab.date)
            rows,stocks = days.get_indexer(dates),codes.get_indexer(tab.code)
            if (rows<0).any() or (stocks<0).any() or dates.dt.year.ne(year).any() or tab.duplicated(["date","code"]).any():
                raise ValueError("Invalid stock/date partition")
            for field in cols:
                a[field if basis=="raw" else "adj"][rows,stocks] = pd.to_numeric(tab[field],errors="coerce")
    o,h,l,c,p,v,turn,status,adj = a.values()
    ok = (status==1)&(l>0)&(h>=np.maximum(o,c))&(l<=np.minimum(o,c))
    prev = np.vstack([np.full((1,len(codes)),np.nan),adj[:-1]])
    market = index.set_index("date").close.reindex(days).pct_change(fill_method=None).to_numpy()
    mean_vol = pd.DataFrame(np.where(v>=0,v,np.nan)).rolling(20,min_periods=20).mean().to_numpy()
    with np.errstate(divide="ignore",invalid="ignore"):
        ret = np.clip(np.where((prev>0)&(adj>0),adj/prev-1,np.nan),-.2,.2)
        result = np.stack([ret,ret-market[:,None],c/o-1,np.where(p>0,(h-l)/p,np.nan),
                           (c-l)/(h-l+1e-8)-.5,np.where(v>=0,np.log((v+1)/(mean_vol+1)),np.nan),
                           np.where(turn>=0,turn,np.nan)],axis=-1)
    result[~ok] = np.nan
    return result.astype("float32"),codes


def standardize(x):
    # Turnover first becomes its own clipped cross-sectional z-score.
    # Then all seven input channels share the same winsorization and z-score step.
    x = x.copy()
    turn = x[..., -1]
    observed = np.isfinite(turn)
    count = np.maximum(observed.sum(0), 1)
    mean = np.where(observed, turn, 0.).sum(0, dtype=np.float64) / count
    scale = np.sqrt(np.where(observed, (turn-mean)**2, 0.).sum(0)/count)
    x[..., -1] = np.clip((turn-mean)/np.where(scale>1e-10, scale, 1.), -3., 3.)
    mask = np.isfinite(x)
    x = np.where(mask,x,0.).astype(np.float64)
    n = np.maximum(mask.sum(0),1)
    mu = x.sum(0)/n
    sd = np.sqrt(np.where(mask,(x-mu)**2,0.).sum(0)/n)
    clipped = np.clip(x,mu-3*sd,mu+3*sd)
    mu = np.where(mask,clipped,0.).sum(0)/n
    sd = np.sqrt(np.where(mask,(clipped-mu)**2,0.).sum(0)/n)
    return np.where(mask,(clipped-mu)/np.where(sd>1e-10,sd,1.),0.).astype(np.float32), mask


def build(root):
    raw = root/"data/raw/v3/development"
    source = root/"research/v3_1/results"
    out = root/"research/v4"
    days,schedule,members = load_universe(raw)
    index = pd.read_parquet(source/"step0/H00300_development.parquet")
    spec = json.loads((source/"step1/v3_1_selected_factors.json").read_text(encoding="utf-8"))
    states = pd.read_parquet(source/"step1/states.parquet").set_index("signal_date")[spec["states"]]
    labels = pd.read_parquet(source/"step1/labels.parquet",filters=[("horizon","==",5)])
    if days.max()!=END or max(index.date.max(),states.index.max(),labels.signal_date.max(),labels.exit_date.max())>END:
        raise ValueError("Invalid development dates")
    panel,codes = daily(raw,schedule,members,days,index)
    labels = labels.set_index(["signal_date","code"])
    if not labels.index.is_unique:
        raise ValueError("Duplicate development target keys")
    weeks = pd.DatetimeIndex(schedule.signal_date)
    groups = members.groupby("signal_date")
    data = {k:[] for k in ["x_seq","valid_mask","codes","y","entry_date","exit_date","sequence_dates"]}
    execution = schedule.set_index("signal_date").execution_date
    for t in weeks:
        names = groups.get_group(t).code.sort_values()
        if names.nunique()!=300 or len(names)!=300:
            raise ValueError(f"Invalid universe {t}")
        end = days.get_loc(t)
        if end<19:
            raise ValueError("Insufficient calendar warm-up")
        sl = slice(end-19,end+1)
        x,mask = standardize(panel[sl][:,codes.get_indexer(names)].transpose(1,0,2))
        targets = labels.loc[t].reindex(names)
        y = targets.risk_adjusted_target.to_numpy(dtype=np.float32)
        entry,exit_day = targets.entry_date.dropna().unique(),targets.exit_date.dropna().unique()
        if len(exit_day)>1 or len(entry)>1:
            raise ValueError("Mixed holding windows")
        if np.isfinite(y).any() and (len(entry)!=1 or len(exit_day)!=1 or
                not t < entry[0] <= exit_day[0] <= END or entry[0]!=execution.loc[t]):
            raise ValueError("Invalid h5 label window")
        values = [x,mask,names.to_numpy(dtype="U9"),y,
                  np.datetime64(entry[0],"D") if len(entry) else np.datetime64("NaT","D"),
                  np.datetime64(exit_day[0],"D") if len(exit_day) else np.datetime64("NaT","D"),
                  days[sl].to_numpy("datetime64[D]")]
        for key,value in zip(data,values,strict=True):
            data[key].append(value)
    data = {k:np.stack(v) for k,v in data.items()}
    data.update(signal_date=weeks.to_numpy("datetime64[D]"),z_state=states.reindex(weeks).to_numpy(np.float32),
                channels=np.array(CHANNELS),states=np.array(spec["states"]))
    ends = data["sequence_dates"][:,-1]
    valid = np.isfinite(data["y"]).any(1)
    if not (np.array_equal(ends,data["signal_date"]) and np.all(data["exit_date"][valid]>ends[valid])):
        raise ValueError("Chronology audit failed")
    checks = []
    for i in sorted(np.random.default_rng(42).choice(np.flatnonzero(valid),10,replace=False)):
        t = weeks[i]
        past = days[days<=t][-20:]
        test,mask = standardize(panel[days.get_indexer(past)][:,codes.get_indexer(data["codes"][i])].transpose(1,0,2))
        np.testing.assert_array_equal(test,data["x_seq"][i])
        np.testing.assert_array_equal(mask,data["valid_mask"][i])
        checks.append(dict(signal=str(t.date()),last_input=str(past[-1].date()),label_exit=str(data["exit_date"][i])))
    cache = out/"cache"
    cache.mkdir(parents=True,exist_ok=True)
    tmp = cache/"weekly_sequences.tmp.npz"
    np.savez_compressed(tmp,**data)
    tmp.replace(cache/"weekly_sequences.npz")
    audit = dict(shape=list(data["x_seq"].shape),channels=CHANNELS,first_signal=str(weeks[0].date()),
                 last_signal=str(weeks[-1].date()),missing_by_channel=dict(zip(CHANNELS,
                 (1-data["valid_mask"].mean(axis=(0,1,2))).tolist())),missing_labels=float((~np.isfinite(data["y"])).mean()),
                 complete_state_weeks=int(np.isfinite(data["z_state"]).all(1).sum()),all_weeks_chronology=True,
                 random_date_checks=checks,index_first=str(index.date.min().date()),states=spec["states"])
    dest = out/"results/step1"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"dataset.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    build(p.parse_args().project_root.resolve())

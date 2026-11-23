"""Offline frozen-cache and event reconciliation, without portfolio performance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from collect_final_2026 import ROOT, FIRST, LAST, ANCHOR, seal, market
from normalize_corporate_actions import normalize_event, EVENT_COLUMNS


def audit(root):
    cfg=seal(root)
    for name,expected in cfg["configuration"]["development_artifact_sha256"].items():
        with (root/name).open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=expected:
                raise ValueError("Changed development artifact: "+name)
    cache=root/"data/raw/v4_1/final_2026"
    out=root/"research/v4_1/final_2026/audit"
    out.mkdir(parents=True,exist_ok=True)
    ready=json.loads((cache/"collection_complete.json").read_text())
    assert ready["configuration_sha256"]==cfg["configuration_sha256"]
    codes=ready["stock_codes"]
    cal=pd.read_parquet(cache/"calendar.parquet")
    sessions=pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_trading_day.eq("1"),"calendar_date"]))
    idx=pd.read_parquet(cache/"H00300.parquet")
    assert np.array_equal(idx.date,sessions) and idx.code.eq("H00300").all() and idx.close.gt(0).all()
    snapshots=pd.concat([market.check_snapshot(pd.read_parquet(f),pd.Timestamp(f.stem)) for f in sorted((cache/"members").glob("*.parquet"))])
    expected=sessions.to_series().groupby(sessions.to_period("W-SUN")).tail(1)
    assert np.array_equal(snapshots.signal_date.unique(),expected) and set(snapshots.code)<=set(codes)
    events,review,checks,flags=[],[],[],[]
    for code in codes:
        listing=pd.read_parquet(cache/"listings"/f"{code}.parquet").iloc[0]
        ipo=pd.to_datetime(listing.ipoDate,errors="coerce"); end=pd.to_datetime(listing.outDate,errors="coerce")
        exp=sessions[(sessions>=max(FIRST,ipo) if pd.notna(ipo) else sessions>=FIRST)&(sessions<=end if pd.notna(end) else True)]
        values={}
        for basis,flag in [("raw","3"),("adjusted","1")]:
            f=pd.read_parquet(cache/basis/"2026"/f"{code}.parquet").sort_values("date")
            assert f.code.eq(code).all() and f.date.is_unique and f.date.between(FIRST,LAST).all() and f.date.isin(sessions).all()
            wrong=f[~f.adjustflag.eq(flag)]
            values[basis]=f[f.adjustflag.eq(flag)].set_index("date")
            if len(wrong):
                for d in wrong.date: checks.append((code,d,basis+"_flag_quarantined"))
            for d in exp.difference(pd.DatetimeIndex(f.date)): checks.append((code,d,basis+"_missing_session"))
        f=values["raw"]
        assert f.tradestatus.isin(["0","1"]).all()
        for d in f.index[f.tradestatus.eq("0")]: checks.append((code,d,"suspended"))
        if pd.notna(end): review.append((code,end,"delisting","Verify exit settlement if held"))
        prevpath=root/"data/raw/v3/development/raw/2025"/f"{code}.parquet"
        if not prevpath.exists(): prevpath=cache/"raw/2025"/f"{code}.parquet"
        prev=pd.read_parquet(prevpath)
        close=pd.concat([prev.loc[prev.date.eq(ANCHOR)].set_index("date").close,f.close])
        gap=pd.to_numeric(f.preclose,errors="coerce")-pd.to_numeric(close,errors="coerce").reindex(sessions).shift().reindex(f.index)
        for d,g in gap[f.tradestatus.eq("1")&gap.abs().gt(.005)].items():
            flags.append((code,d,float(g)))
        actions=pd.read_parquet(cache/"actions/2026"/f"{code}.parquet")
        if len(actions):
            assert actions.code.eq(code).all()
            for d,group in actions.groupby(pd.to_datetime(actions.dividOperateDate,errors="coerce"),dropna=False):
                if pd.isna(d): raise ValueError("Undated action "+code)
                if d>LAST:continue
                try: events.append(normalize_event(code,d,group))
                except ValueError as error:review.append((code,d,"terms",str(error)))
    df=pd.DataFrame(events,columns=EVENT_COLUMNS)
    known=set(zip(df.code,df.ex_date))|{(c,d) for c,d,_,_ in review}
    review.extend((c,d,"reference_gap",str(v)) for c,d,v in flags if (c,d) not in known)
    issues=pd.DataFrame(review,columns=["code","ex_date","category","reason"])
    price=pd.DataFrame(checks,columns=["code","date","condition"])
    for n,t in [("normalized_actions.parquet",df),("action_review.csv",issues),("quote_audit.csv",price),
                ("reference_gaps.csv",pd.DataFrame(flags,columns=["code","date","gap"]))]:
        market.save(t,out/n)
    print("Stocks, sessions, normalized events, event reviews:",len(codes),len(sessions)-1,len(df),len(issues))
    print(issues.to_string(index=False))
    print(price.condition.value_counts().to_string())
    return df,issues


if __name__=="__main__":
    audit(ROOT)

"""Acquire the first post-seal weekly extension. Store it separately from the signed holdout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_final_2026 import EXTRA, LAST, ROOT, market, read_source, save_json, seal, validate_history
from collect_final_style_carry import trading_days

SIGNAL = pd.Timestamp("2026-11-20")
OHLC = ["open","high","low","close"]


def cache_week(root, mirror=None, check_only=False):
    report = root/"research/v4_1/final_2026"
    reg = seal(root)
    signed = json.loads((report/"holdout_seal.json").read_text(encoding="utf-8"))
    if not signed["all_ledgers_verified"] or reg["configuration_sha256"]!=signed["configuration_sha256"]:
        raise ValueError("Holdout release is required")
    for f,h in signed["sha256"].items():
        if hashlib.sha256((root/f).read_bytes()).hexdigest()!=h:
            raise ValueError("Changed holdout artifact: "+f)
    old = root/"data/raw/v4_1/final_2026"
    legacy = root/"data/raw/forward/2026-11-20"
    out = root/"data/raw/v4_1/paper_week_20261120"
    cal = trading_days(legacy/"calendar.parquet")
    days = cal[(cal>=LAST)&(cal<=SIGNAL)]
    if len(days)!=6 or cal[cal>SIGNAL].min()!=pd.Timestamp("2026-11-23"):
        raise ValueError("Unexpected week or execution calendar")
    members = market.check_snapshot(pd.read_parquet(legacy/"members/2026-11-20.parquet"),SIGNAL)
    floor = cal[cal.get_loc(SIGNAL)-60]
    formation_file = max(p for p in (old/"members").glob("*.parquet") if pd.Timestamp(p.stem)<=floor)
    source = pd.read_parquet(formation_file)
    carried = pd.read_csv(report/"holdings_at_cutoff.csv")
    codes = sorted(set(members.code)|set(source.code)|set(carried.code)|set(pd.read_parquet(old/"members/2026-11-13.parquet").code))
    former = {}
    for c in codes:
        listing = old/"listings"/(c+".parquet")
        if not listing.exists():
            raise ValueError("New member requires historical warm-up: "+c)
        end = pd.to_datetime(pd.read_parquet(listing).outDate.iloc[0],errors="coerce")
        if pd.notna(end) and end<=LAST:
            former[c] = str(end.date())
    codes = [c for c in codes if c not in former]
    print(f"{len(days)-1} sessions, {len(codes)} live symbols, {len(former)} retired, last signal {SIGNAL.date()}",flush=True)
    if check_only:
        return
    out.mkdir(parents=True,exist_ok=True)
    market.save(members,out/"members.parquet")
    market.save(pd.read_parquet(legacy/"calendar.parquet").loc[lambda x:pd.to_datetime(x.calendar_date)>=LAST],out/"calendar.parquet")
    session = market.BaoSession()

    def anchor_check(frame,original,fields,code):
        for table in (frame,original):
            if table.date.duplicated().any() or not table.code.eq(code).all():
                raise ValueError("Duplicate date or unexpected code: "+code)
        a,b = [t.loc[t.date.eq(LAST),fields].apply(pd.to_numeric,errors="coerce") for t in (frame,original)]
        return len(a)==len(b)==1 and np.allclose(a,b,equal_nan=True,atol=1e-8,rtol=1e-10)

    try:
        if not (out/"H00300.parquet").exists():
            frame,raw,url,kind = read_source(argparse.Namespace(mirror_csv=mirror,mirror=False),days)
            index = validate_history(frame,days)
            prev = pd.read_parquet(old/"H00300.parquet")
            np.testing.assert_allclose(index.close.iat[0],prev.close.iat[-1],atol=.005,rtol=0)
            market.save(index,out/"H00300.parquet")
            save_json({"source":url,"kind":kind,"sha256":hashlib.sha256(raw).hexdigest()},out/"H00300.json")
        if not (out/"price_index.parquet").exists():
            frame = market.daily(session,"sh.000300",LAST,SIGNAL,market.INDEX,"3",False)
            prev = pd.read_parquet(old/"price_index/2026.parquet")
            if not pd.DatetimeIndex(frame.date).equals(days) or not anchor_check(frame,prev,OHLC,"sh.000300"):
                raise ValueError("Price index discontinuity")
            market.save(frame,out/"price_index.parquet")
        for i,c in enumerate(codes,1):
            for basis,flag in [("raw","3"),("adjusted","1")]:
                dest = out/basis/(c+".parquet")
                if dest.exists():
                    continue
                fields = list(market.RAW)+EXTRA if basis=="raw" else ["date","code",*OHLC,"adjustflag"]
                prev = pd.read_parquet(old/basis/"2026"/(c+".parquet"))
                v1 = legacy/basis/(c+".parquet")
                local = pd.read_parquet(v1) if v1.exists() else pd.DataFrame()
                if not local.empty:
                    local["date"] = pd.to_datetime(local.date)
                    local = local.loc[local.date.isin(days)].sort_values("date")
                reusable = (not local.empty and pd.DatetimeIndex(local.date).equals(days) and
                            local.adjustflag.eq(flag).all() and anchor_check(local,prev,OHLC,c))
                if reusable:
                    missing = [x for x in fields if x not in local]
                    if missing:
                        tail = market.daily(session,c,LAST,SIGNAL,["date","code","adjustflag",*missing],flag,False)
                        frame = local.merge(tail,on=["date","code","adjustflag"],validate="one_to_one")
                    else:
                        frame = local
                else:
                    frame = market.daily(session,c,LAST,SIGNAL,fields,flag,False)
                if not (pd.DatetimeIndex(frame.date).equals(days) and frame.adjustflag.eq(flag).all()
                        and anchor_check(frame,prev,OHLC,c)):
                    raise ValueError("Price/adjustment anchor mismatch: "+c+" "+basis)
                market.save(frame[fields],dest)
            dest = out/"actions"/(c+".parquet")
            if not dest.exists():
                data = session.query("query_dividend_data",c+" actions",code=c,year="2026",yearType="operate",empty=True)
                if len(data):
                    ex = pd.to_datetime(data.dividOperateDate,errors="coerce")
                    rec = pd.to_datetime(data.dividRegistDate,errors="coerce")
                    keep = (ex.gt(LAST)&ex.le(SIGNAL)) | (rec.gt(LAST)&rec.le(SIGNAL))
                    if not data.code.eq(c).all() or (ex.isna()&rec.isna()).any():
                        raise ValueError("Unidentified dividend for "+c)
                    data = data.loc[keep]
                market.save(data,dest)
            if i%25==0:
                print(f"{i}/{len(codes)}",flush=True)
    finally:
        session.close()
    digest = {p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*.parquet"))}
    save_json({"signal_date":str(SIGNAL.date()),"holdout_configuration":reg["configuration_sha256"],
               "code_count":len(codes),"retired":former,"sha256":digest},out/"complete.json")
    print("Paper week ready:",SIGNAL.date(),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--h00300-csv",type=Path)
    p.add_argument("--check-only",action="store_true")
    a = p.parse_args()
    cache_week(a.project_root.resolve(),a.h00300_csv,a.check_only)

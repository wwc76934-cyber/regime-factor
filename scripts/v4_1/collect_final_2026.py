"""User-run acquisition for the registered blind window only. Never fits or scores a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from pyarrow.parquet import ParquetFile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"scripts/v3"))
import fetch_and_audit_v3_data as market
from fetch_csi300_total_return import read_source, validate_history

FIRST=pd.Timestamp("2026-01-01")
LAST=pd.Timestamp("2026-11-13")
ANCHOR=FIRST-pd.Timedelta(days=1)
FREEZE="353f9891ec7b40cea39a8deb8525325ec9013f69ac3d1cfeeb0db7a6f99e6286"
EXTRA=["psTTM","pcfNcfTTM"]


def seal(root):
    path=root/"research/v4_1/freeze_registration_2026_final.json"
    if hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=FREEZE:
        raise ValueError("Wrong freeze registration")
    receipt=json.loads(path.read_text(encoding="utf-8"))
    cfg=receipt["configuration"]
    b=json.dumps(cfg,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    if hashlib.sha256(b).hexdigest()!=receipt["configuration_sha256"]:
        raise ValueError("Configuration digest mismatch")
    for name,version in cfg["source_sha256"].items():
        source=(root/name).read_bytes()
        if (hashlib.sha256(source).hexdigest()!=version["local_bytes_sha256"] and
            hashlib.sha256(source.replace(b"\r\n",b"\n")).hexdigest()!=version["git_blob_sha256"]):
            raise ValueError("Frozen source changed: "+name)
    return receipt


def save_json(value,path):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


def part(path,fetch):
    if not path.exists():
        market.save(fetch(),path)
    return pd.read_parquet(path)


def collect(root,mirror):
    receipt=seal(root)
    cache=root/"data/raw/v4_1/final_2026"
    dev=root/"data/raw/v3/development"
    cache.mkdir(parents=True,exist_ok=True)
    owner=cache/"registration.json"
    token=dict(configuration_sha256=receipt["configuration_sha256"],registration_file_sha256=FREEZE,
               freeze_commit="d4a05d9baa182c6d4b03d2a2087ff5c16bb0c24e",through=str(LAST.date()))
    if owner.exists() and json.loads(owner.read_text(encoding="utf-8"))!=token:
        raise ValueError("Cache belongs to a different registration")
    save_json(token,owner)
    session=market.BaoSession()
    def dates(a):
        return str(a.date())
    try:
        # Include December 31 so a shortened year-end week is handled by the joint calendar.
        cal=part(cache/"calendar.parquet",lambda:session.query("query_trade_dates","calendar",
                 start_date=dates(ANCHOR),end_date=dates(LAST)))
        civ=pd.DatetimeIndex(pd.to_datetime(cal.calendar_date))
        if not civ.equals(pd.date_range(ANCHOR,LAST)) or not cal.is_trading_day.astype(str).isin(["0","1"]).all():
            raise ValueError("Incomplete calendar")
        days=civ[cal.is_trading_day.astype(str).eq("1")]
        if days[0]!=ANCHOR or days[-1]!=LAST:
            raise ValueError("Unexpected first or last trading day")
        grouped=pd.Series(days,index=days).groupby(days.to_period("W-SUN")).tail(1)
        signals=pd.DatetimeIndex(grouped)
        portfolio_codes={h["code"] for a in receipt["configuration"]["accounts"] for h in a["holdings"]}
        prior=dev/"members/2025-12-26.parquet"
        codes=portfolio_codes|set(pd.read_parquet(prior).code)
        for t in signals:
            snapshot=part(cache/"members"/f"{dates(t)}.parquet",lambda:market.check_snapshot(
                session.query("query_hs300_stocks",dates(t),date=dates(t)),t))
            codes.update(market.check_snapshot(snapshot,t).code)
        print(f"{len(signals)} PIT snapshots, {len(codes)} distinct symbols (including carry-in)",flush=True)
        idx=part(cache/"price_index/2026.parquet",lambda:market.daily(session,"sh.000300",FIRST,LAST,market.INDEX,"3",False))
        if not pd.DatetimeIndex(idx.date).equals(days[days>=FIRST]):
            raise ValueError("Price-index date coverage mismatch")
        official=cache/"H00300.parquet"
        if not official.exists() or not official.with_suffix(".json").exists():
            chunks,provenance=[],[]
            for year in sorted(set(days.year)):
                d=days[days.year==year]
                frame,raw,source,kind=read_source(argparse.Namespace(mirror_csv=mirror,mirror=False),d)
                chunks.append(validate_history(frame,d))
                provenance.append(dict(source=source,kind=kind,sha256=hashlib.sha256(raw).hexdigest()))
            market.save(pd.concat(chunks,ignore_index=True),official)
            save_json(provenance,official.with_suffix(".json"))
        hindex=pd.read_parquet(official)
        if not pd.DatetimeIndex(hindex.date).equals(days) or not hindex.close.gt(0).all() or not hindex.code.eq("H00300").all():
            raise ValueError("H00300 incomplete")
        frozen_index=pd.read_parquet(root/"research/v3_1/results/step0/H00300_development.parquet")
        anchor_close=frozen_index.loc[frozen_index.date.eq(ANCHOR),"close"].item()
        if abs(float(hindex.close.iloc[0])-float(anchor_close))>.005:
            raise ValueError("H00300 anchor disagrees with development")
        for i,code in enumerate(sorted(codes),1):
            for year in (2024,2025,2026):
                a=pd.Timestamp(year,1,1)
                b=LAST if year==2026 else pd.Timestamp(year,12,31)
                for basis,flag in [("raw","3"),("adjusted","1")]:
                    f=Path(basis)/str(year)/f"{code}.parquet"
                    is_dev=year<2026 and (dev/f).exists()
                    loc=dev/f if is_dev else cache/f
                    fields=list(market.RAW)+(EXTRA if basis=="raw" else [])
                    if not is_dev:
                        part(loc,lambda:market.daily(session,code,a,b,fields,flag))
                    if year==2025 and basis=="raw" and is_dev and not set(EXTRA).issubset(ParquetFile(loc).schema_arrow.names):
                        v=Path("valuations")/str(year)/f"{code}.parquet"
                        if not (root/"data/raw/v3_1/development"/v).exists():
                            part(cache/v,lambda:market.daily(session,code,a,b,["date","code","adjustflag",*EXTRA],"3"))
            def actions():
                f=session.query("query_dividend_data",f"actions {code}",code=code,year="2026",yearType="operate",empty=True)
                if f.empty:return f
                if not f.code.eq(code).all():raise ValueError("Dividend symbol mismatch")
                # Keep only terms relevant by record or ex date. No later performance/quotes are requested.
                ex=pd.to_datetime(f.dividOperateDate,errors="coerce")
                reg=pd.to_datetime(f.dividRegistDate,errors="coerce")
                if (ex.isna()&reg.isna()).any():
                    raise ValueError(f"{code}: event missing both dates; official terms required")
                return f.loc[(ex<=LAST)|(reg<=LAST)].reset_index(drop=True)
            part(cache/"actions/2026"/f"{code}.parquet",actions)
            def basic():
                f=session.query("query_stock_basic",f"listing {code}",code=code)
                f=f.loc[f.code.eq(code),["code","ipoDate","outDate","type"]].copy()
                if len(f)!=1:raise ValueError(f"Missing listing record for {code}")
                f.loc[pd.to_datetime(f.outDate,errors="coerce").gt(LAST),"outDate"]=""
                return f
            part(cache/"listings"/f"{code}.parquet",basic)
            if i%25==0 or i==len(codes):
                print(f"{i}/{len(codes)} symbols cached",flush=True)
        save_json(dict(**token,stocks=len(codes),stock_codes=sorted(codes),signal_snapshots=len(signals),
                       h00300_points=len(days),holdout_trading_days=int((days>=FIRST).sum()),
                       cached_pending_offline_audit=True),cache/"collection_complete.json")
        print("Cache ready for offline audit:",cache/"collection_complete.json",flush=True)
    finally:
        session.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--h00300-csv",type=Path,help="Optional H00300 local mirror with date,csi300 columns")
    parser.add_argument("--check-freeze-only",action="store_true",help="Verify the local receipt and source hashes; no network")
    a=parser.parse_args()
    if a.check_freeze_only:
        print("Freeze verified:",seal(a.project_root.resolve())["configuration_sha256"])
    else:
        collect(a.project_root.resolve(),a.h00300_csv)

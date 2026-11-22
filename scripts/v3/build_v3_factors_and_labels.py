"""Develop factors using only matured 2015–2025 labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from fetch_and_audit_v3_data import END, FIRST, need, load_universe, ranges, save

FACTORS = ["RET_5","MOM_20_SKIP5","RET_60","INTRA_RET_10","VOL_60","DOWN_VOL_20","BETA_120",
           "CORR_PV_10","TURN_20","TURN_RATIO_5_20","AMIHUD_20","EP_TTM","BP_MRQ","VOL_20","BIAS_20"]
STATES = ["market_return_20d","market_volatility_20d","market_breadth_20d","tier_vol_spread_20d"]
KEY = ["signal_date","code"]
CORE = ["RET_5","VOL_60","CORR_PV_10"]
HOLDOUT_END = pd.Timestamp("2026-11-13")


def stock_history(cache, tasks, basis, columns, flag):
    chunks = []
    for year, codes in tasks.items():
        frames = []
        for code in codes:
            part = pd.read_parquet(cache / basis / str(year) / f"{code}.parquet", columns=columns+["adjustflag"])
            frames.append(part.loc[part.adjustflag.eq(flag),columns])
        chunks.append(pd.concat(frames,ignore_index=True))
        print(f"Read {basis} {year}",flush=True)
    return pd.concat(chunks,ignore_index=True)


def prices(table, prefix):
    fields=["open","high","low","close"]
    table[fields]=table[fields].apply(pd.to_numeric,errors="coerce")
    values=table[fields]
    ok=np.isfinite(values).all(axis=1)&values.gt(0).all(axis=1)&values.high.ge(values.max(axis=1))&values.low.le(values.min(axis=1))
    return table.assign(**{f"{prefix}_ohlc_ok":ok}).rename(columns={x:f"{prefix}_{x}" for x in fields})


def panel_and_index(cache, days, tasks):
    rawcols=["date","code","open","high","low","close","volume","amount","turn","peTTM","pbMRQ","tradestatus","isST","preclose"]
    adjcols=["date","code","open","high","low","close"]
    raw=prices(stock_history(cache,tasks,"raw",rawcols,"3"),"raw")
    adj=prices(stock_history(cache,tasks,"adjusted",adjcols,"1"),"adj")
    panel=raw.merge(adj,on=["date","code"],how="outer",validate="one_to_one")
    nums=["volume","amount","turn","peTTM","pbMRQ","preclose"]
    panel[nums]=panel[nums].apply(pd.to_numeric,errors="coerce")
    panel["signal_bar_usable"]=panel.raw_ohlc_ok.eq(True)&panel.adj_ohlc_ok.eq(True)&panel.tradestatus.eq("1")
    ix=pd.concat([pd.read_parquet(cache/"price_index"/f"{y}.parquet") for y in tasks]).set_index("date").reindex(days)
    need(ix.code.eq("sh.000300").all(),"Incomplete price index")
    ix[["open","close"]]=ix[["open","close"]].apply(pd.to_numeric)
    need(np.isfinite(ix[["open","close"]]).all().all() and ix[["open","close"]].gt(0).all().all(),"Invalid price index")
    return panel,ix


def make_data(panel, ix, members, schedule, days, cutoff):
    from build_candidate_factors import calculate_factors
    from build_forward_labels import build_holding_windows, join_open_price

    mats=calculate_factors(panel,days)
    def wide(field):
        return panel.pivot(index="date",columns="code",values=field).reindex(days)
    close=wide("adj_close").where(wide("adj_ohlc_ok").eq(True))
    returns=close.pct_change(fill_method=None)
    market=ix.close.pct_change(fill_method=None)
    # Match market variance to the very same valid stock/day pairs.
    market_pairs=returns.notna().mul(market,axis=0).where(returns.notna())
    b=returns.rolling(120,min_periods=100).cov(market_pairs) / market_pairs.rolling(120,min_periods=100).var()
    mats["BETA_120"]=b.where(np.isfinite(b)).clip(.2,2.2)
    valid_turn=wide("turn").where(lambda x:x.ge(0))
    mats["TURN_20"]=valid_turn.rolling(20).mean()
    mats["TURN_RATIO_5_20"]=valid_turn.rolling(5).mean()/mats["TURN_20"].where(mats["TURN_20"].gt(0))
    for source,target in [("peTTM","EP_TTM"),("pbMRQ","BP_MRQ")]:
        value=wide(source)
        mats[target]=1/value.where(value.gt(0))
    frame=members[KEY].merge(panel[["date","code","signal_bar_usable"]].rename(columns={"date":"signal_date"}),
                            on=KEY,how="left",validate="one_to_one").set_index(KEY)
    signal=pd.DatetimeIndex(schedule.signal_date)
    for name in [*FACTORS,"RET_20"]:
        frame[name]=mats[name].reindex(signal).rename_axis(index="signal_date",columns="code").stack().reindex(frame.index)
    frame=frame.reset_index()
    frame["signal_bar_usable"]=frame.signal_bar_usable.eq(True)
    price=frame.loc[frame.signal_bar_usable&frame.VOL_60.notna(),KEY+["VOL_60","RET_20"]].copy()
    price=price.sort_values(["signal_date","VOL_60","code"])
    frac=(price.groupby("signal_date").cumcount()+.5)/price.groupby("signal_date").code.transform("size")
    low=price[frac.lt(1/3)].groupby("signal_date").RET_20.mean()
    high=price[frac.ge(2/3)].groupby("signal_date").RET_20.mean()
    breadth=frame.RET_20.gt(0).where(frame.RET_20.notna()).groupby(frame.signal_date).mean()
    states=pd.DataFrame({STATES[0]:ix.close.pct_change(20),STATES[1]:market.rolling(20).std(),
                         STATES[2]:breadth,STATES[3]:high-low}).reindex(signal).reset_index()
    windows=build_holding_windows(schedule)
    frame=frame.merge(windows,on="signal_date",validate="many_to_one")
    for side in ["entry","exit"]:
        frame=join_open_price(frame,panel[["date","code","adj_open","tradestatus","adj_ohlc_ok"]],side)
    ok=frame.signal_bar_usable&frame.exit_date.le(cutoff)
    for side in ["entry","exit"]:
        ok &= frame[f"{side}_status"].eq("1")&frame[f"{side}_price_ok"].eq(True)&frame[f"{side}_open_adj"].gt(0)
    frame["forward_return"]=(frame.exit_open_adj/frame.entry_open_adj-1).where(ok)
    entry=frame.entry_date.map(ix.open)
    exit_price=frame.exit_date.map(ix.open)
    frame["index_forward_return"]=(exit_price/entry-1).where(frame.exit_date.le(cutoff))
    frame["alpha_return"]=frame.forward_return-frame.BETA_120*frame.index_forward_return
    official=frame[KEY+["signal_bar_usable","entry_date","exit_date",*FACTORS,"forward_return","index_forward_return","alpha_return"]]
    return official,states


def weekly_metrics(data):
    rows=[]
    for day, w in data.groupby("signal_date"):
        if w.exit_date.dropna().empty:
            continue
        end=w.exit_date.dropna().iat[0]
        split=("screen" if day.year<=2018 and end.year<=2018 else
               "validation" if 2019<=day.year<=2022 and end.year<=2022 else
               "later_dev" if day.year>=2023 else None)
        if split is None:
            continue
        sample=w.loc[w.signal_bar_usable&w.alpha_return.notna()&w.forward_return.notna()]
        for factor in FACTORS:
            q=sample[[factor,"forward_return","alpha_return"]].dropna()
            if len(q)<50 or q[factor].nunique()<2:
                continue
            rank=q[factor].rank()
            percentile=rank/len(q)
            stats={}
            for target in ["forward_return","alpha_return"]:
                stats[f"ic_{target}"]=rank.corr(q[target].rank())
                stats[f"spread_{target}"]=q.loc[percentile.gt(.8),target].mean()-q.loc[percentile.le(.2),target].mean()
            rows.append(dict(signal_date=day,split=split,factor=factor,n=len(q),**stats))
    return pd.DataFrame(rows)


def select(data, weekly):
    stats=weekly.groupby(["factor","split"])[["ic_forward_return","ic_alpha_return","spread_alpha_return","n"]].mean().unstack()
    stats.columns=["_".join(pair) for pair in stats.columns]
    stats=stats.reindex(FACTORS)
    c=stats["ic_alpha_return_screen"]; v=stats["ic_alpha_return_validation"]
    raw=stats["ic_forward_return_screen"]
    stats["direction"]=np.sign(c)
    stats["abs_screen"]=c.abs()
    stats["passes"]=c.abs().ge(.015)&v.abs().ge(.01)&c.mul(v).gt(0)&raw.mul(c).gt(0)&(
        stats["spread_alpha_return_validation"].mul(c).gt(0))
    corrs=[]
    for _, week in data.loc[data.signal_date.dt.year.le(2018)&data.signal_bar_usable].groupby("signal_date"):
        corrs.append(week[FACTORS].corr(method="spearman",min_periods=50).abs().stack())
    median=pd.concat(corrs,axis=1).median(axis=1)
    selected=[]
    stats["decision"]="failed_thresholds"
    for name in stats.sort_values("abs_screen",ascending=False).index:
        if not stats.at[name,"passes"]:
            continue
        conflict=[other for other in selected if median.get((name,other),np.nan)>=.75]
        if conflict:
            stats.at[name,"decision"]="redundant_with_"+conflict[0]
        elif len(selected)>=8:
            stats.at[name,"decision"]="ranked_below_cap"
        else:
            selected.append(name)
            stats.at[name,"decision"]="selected"
    need(len(selected)>=2,"Fewer than two candidates qualified; do not relax thresholds on holdout")
    return stats.rename_axis("factor").reset_index(), selected, median


def run(root,output):
    sys.path.insert(0,str(root/"scripts"))
    cache=root/"data/raw/v3/development"
    days,schedule,members=load_universe(cache)
    panel,ix=panel_and_index(cache,days,ranges(schedule,members))
    data,states=make_data(panel,ix,members,schedule,days,END)
    weekly=weekly_metrics(data)
    audit,selected,corr=select(data,weekly)
    save(data,output/"factor_panel.parquet")
    save(states,output/"states.parquet")
    save(panel.loc[panel.date.ge(FIRST),["date","code","raw_open","raw_close","preclose","tradestatus","isST"]],
         output/"execution_panel.parquet")
    save(weekly,output/"weekly_factor_audit.parquet")
    save(audit,output/"v3_factor_audit.csv")
    save(corr.rename("median_abs_spearman").rename_axis(index=["factor","other"]).reset_index(),output/"redundancy.csv")
    chosen=dict(zip(audit.factor,audit.direction))
    (output/"features.json").write_text(json.dumps({"selected":selected,"directions":{x:int(chosen[x]) for x in selected},
        "v1_directions":{x:-1 for x in CORE},"selection":"screen 2015–2018, validate 2019–2022, same rows for raw and adjusted IC",
        "beta":"120 sessions, >=100 matched returns, clip [0.2,2.2]", "states":STATES},indent=2),encoding="utf-8")
    print(audit[["factor","ic_alpha_return_screen","ic_alpha_return_validation","decision"]].to_string(index=False))
    print(data[["forward_return","alpha_return"]].count().to_dict())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    run(a.project_root.resolve(),a.output_dir or a.project_root/"research/v3/results/step2")

"""Extend the signed features by one completed signal week in a separate directory."""
import hashlib
import json
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_final_inputs as hist
import forecast_final_2026 as forecast
from collect_paper_week_20261120 import SIGNAL
from prepare_final_inputs import ROOT, WORK, LAST, KEY, market, features, prices, proxy_weights, trading_days

CACHE = ROOT/"data/raw/v4_1/paper_week_20261120"
BASE = ROOT/"data/raw/v4_1/final_2026"
OUT = ROOT/"research/v4_1/paper_week_20261120"
LOCAL = OUT/"work"


def verify():
    signed = json.loads((hist.DEST/"holdout_seal.json").read_text())
    receipt = json.loads((CACHE/"complete.json").read_text())
    if (not signed["all_ledgers_verified"] or receipt["signal_date"]!=str(SIGNAL.date()) or
        receipt["holdout_configuration"]!=signed["configuration_sha256"] or
        signed["configuration_sha256"]!=hist.seal(ROOT)["configuration_sha256"]):
        raise ValueError("Registration mismatch")
    for root, manifest in [(ROOT,signed),(CACHE,receipt)]:
        for f,h in manifest["sha256"].items():
            if hashlib.sha256((root/f).read_bytes()).hexdigest()!=h:
                raise ValueError("Changed input: "+f)


def build():
    verify()
    dev = ROOT/"data/raw/v3/development"
    days = trading_days(dev/"calendar.parquet").union(trading_days(BASE/"calendar.parquet")).union(trading_days(CACHE/"calendar.parquet"))
    days = days[(days.year>=2025)&(days<=SIGNAL)]
    members = pd.concat([pd.read_parquet(p) for p in sorted((BASE/"members").glob("*.parquet"))]+[pd.read_parquet(CACHE/"members.parquet")],ignore_index=True)
    two = members.loc[members.signal_date.isin([LAST,SIGNAL])].sort_values(KEY)
    names = sorted(set(members.code))
    source = {}
    for basis,flag in [("raw","3"),("adjusted","1")]:
        chunks = []
        for c in names:
            old = dev/basis/"2025"/(c+".parquet")
            if not old.exists():
                old = BASE/basis/"2025"/(c+".parquet")
            for f in (old,BASE/basis/"2026"/(c+".parquet"),CACHE/basis/(c+".parquet")):
                if not f.exists():
                    if CACHE in f.parents:
                        continue
                    raise FileNotFoundError(f)
                a = pd.read_parquet(f)
                if CACHE in f.parents:
                    a = a.loc[a.date.gt(LAST)]
                chunks.append(a.loc[a.adjustflag.eq(flag)])
        source[basis] = pd.concat(chunks,ignore_index=True)
    def history(_, groups, basis, cols, flag):
        return source[basis][cols].copy()
    ix = pd.read_parquet(BASE/"price_index/2026.parquet")
    tail_ix = pd.read_parquet(CACHE/"price_index.parquet")
    for year, table in [(2025,pd.read_parquet(dev/"price_index/2025.parquet")),
                        (2026,pd.concat([ix,tail_ix.loc[tail_ix.date.gt(LAST)]]))]:
        market.save(table,LOCAL/"price_index"/f"{year}.parquet")
    stamps = pd.DatetimeIndex([LAST,SIGNAL])
    with patch.object(prices,"stock_history",history):
        panel,index = prices.panel_and_index(LOCAL,days,{2025:names,2026:names})
        style = hist.build_style_state(None,days,members,{},stamps,features.neutralized_valuation)
    weights = proxy_weights(panel,two,days)
    extra = ["psTTM","pcfNcfTTM"]
    val = source["raw"].loc[lambda x:x.date.isin(stamps),["date","code",*extra]].rename(columns={"date":"signal_date"})
    val[extra] = val[extra].apply(pd.to_numeric,errors="coerce")
    frame,z = features.signals(panel,index,two,weights.merge(val,on=KEY,how="left",validate="one_to_one"),days)
    z = z.merge(style[["signal_date",hist.STYLE]],on="signal_date",validate="one_to_one")
    spec = json.loads((WORK/"step1/v3_1_selected_factors.json").read_text())
    def append_checked(name,new,keys,cols):
        old = pd.read_parquet(WORK/"step1"/name)
        def anchor(t):
            return t[t.signal_date.eq(LAST)].set_index(keys)[cols].sort_index()
        pd.testing.assert_frame_equal(anchor(old),anchor(new),check_dtype=False,rtol=1e-8,atol=1e-8)
        full = pd.concat([old,new[new.signal_date.eq(SIGNAL)]],ignore_index=True)
        market.save(full,LOCAL/"step1"/name)
        return full
    full = append_checked("factor_panel.parquet",frame,KEY,[*spec["selected"],"BETA_120","VOL_20","IDIO_VAR_60"])
    append_checked("states.parquet",z,["signal_date"],spec["states"])
    if not np.isfinite(z[spec["states"]]).all().all():
        raise ValueError("Missing weekly state")
    old_labels = pd.read_parquet(WORK/"step1/labels.parquet")
    recent = full.loc[full.signal_date.ge("2026-10-30")]
    dates = pd.DatetimeIndex(sorted(recent.signal_date.unique()))
    schedule = pd.DataFrame({"signal_date":dates,"execution_date":dates.map(pd.Series(days,index=days).shift(-1))})
    benchmarks = pd.concat([pd.read_parquet(BASE/"H00300.parquet"),pd.read_parquet(CACHE/"H00300.parquet")]).drop_duplicates("date").set_index("date").close
    with patch.object(features,"DEVELOPMENT_LAST",SIGNAL),patch.object(features,"HORIZONS",(5,)):
        labels = features.build_labels(recent,panel,benchmarks,days,schedule)
    known = old_labels.loc[old_labels.signal_date.ge(dates[0])&old_labels.exit_date.notna()].set_index(KEY)
    pd.testing.assert_series_equal(known.risk_adjusted_target,labels.set_index(KEY).loc[known.index,"risk_adjusted_target"],check_names=False,atol=1e-8,rtol=1e-8)
    market.save(pd.concat([old_labels.loc[old_labels.signal_date.lt(dates[0])],labels]),LOCAL/"step1/labels.parquet")
    shutil.copyfile(WORK/"step1/v3_1_selected_factors.json",LOCAL/"step1/v3_1_selected_factors.json")
    market.save(weights.loc[weights.signal_date.eq(SIGNAL)],LOCAL/"weights.parquet")
    market.save(z.loc[z.signal_date.eq(SIGNAL)],OUT/"market_state.csv")
    cols = ["date","code","raw_open","raw_close","preclose","tradestatus","isST"]
    market.save(panel.loc[panel.date.gt(LAST)&panel.raw_close.notna(),cols],LOCAL/"execution.parquet")
    with patch.multiple(forecast,WORK=LOCAL,DEST=LOCAL,LAST=SIGNAL):
        forecast.main()
    for role in ["baseline","primary"]:
        fname = role+"_predictions.parquet"
        old = pd.read_parquet(WORK/fname).set_index(KEY+["strategy"]).sort_index()
        all_scores = pd.read_parquet(LOCAL/fname)
        prior = all_scores.loc[all_scores.signal_date.le(LAST)].set_index(KEY+["strategy"]).sort_index()
        pd.testing.assert_series_equal(old.score,prior.score,rtol=1e-6,atol=1e-6)
        # Preserve the sealed historical score bytes in the EMA calculation.
        market.save(pd.concat([old.reset_index(),all_scores.loc[all_scores.signal_date.eq(SIGNAL)]]),LOCAL/fname)
    print("Weekly features ready; historical scores and last sealed cross-section matched.")


if __name__=="__main__":
    build()

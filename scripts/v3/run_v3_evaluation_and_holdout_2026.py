"""Register V3, prepare an isolated user-run 2026 cache, and run the frozen evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import fetch_and_audit_v3_data as market
from build_v3_factors_and_labels import KEY, STATES, make_data, panel_and_index
from optimize_v3_portfolio import BUFFER, CAP, CONC, FEES, TURN, beta_band, make_rules, run as backtest
from train_v3_models import describe, fit
from v3_execution import stock_rules

FIRST = pd.Timestamp("2026-01-01")
CUTOFF = pd.Timestamp("2026-11-13")
HERE = Path(__file__).resolve().parent
V3_FILES = ["fetch_and_audit_v3_data.py","build_v3_factors_and_labels.py","train_v3_models.py",
            "v3_execution.py","optimize_v3_portfolio.py","run_v3_evaluation_and_holdout_2026.py"]
DEPENDENCIES = ["apply_verified_actions.py","normalize_corporate_actions.py","build_candidate_factors.py",
                "build_forward_labels.py","fetch_csi300_total_return.py","holdout_2025/verified_actions_2025.csv"]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def json_text(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n"


def json_digest(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def environment(backend):
    packages=["numpy","pandas","pyarrow","scipy","lightgbm"]
    if backend=="torch":
        packages.append("torch")
    return {"python":list(sys.version_info[:3]),**{x:importlib.metadata.version(x) for x in packages}}


def path_for(root, name):
    return HERE/name[3:] if name.startswith("v3/") else root/"scripts"/name


def register(root, results, primary, out):
    market.need(not out.exists(),f"Freeze registration already exists: {out}")
    source,models,execution=[results/f"step{step}" for step in [2,3,4]]
    portfolio=pd.read_csv(execution/"portfolio_metrics.csv")
    market.need(portfolio.unresolved_exposures.eq(0).all(),"Resolve development position exposures before freezing")
    market.need(primary in set(portfolio.loc[portfolio.fee_bp.eq(10),"strategy"]),
                "Primary forward strategy must exist at 10 bp in development")
    configuration={
        "last_development_date":"2025-12-31","holdout_start":str(FIRST.date()),"holdout_end":str(CUTOFF.date()),
        "factors":json.loads((source/"features.json").read_text(encoding="utf-8")),
        "models":json.loads((models/"model_spec.json").read_text(encoding="utf-8")),
        "portfolio":{"max_weight":CAP,"turnover_penalty":TURN,"concentration_penalty":CONC,"buffer_rank":BUFFER,
                     "fee_bp":list(FEES),"initial_cash":1_000_000,
                     "beta_bands":{"offensive":[.95,1.10],"neutral":[.85,.95],"defensive":[.65,.80]}},
        "variants":sorted(portfolio.strategy.tolist()),"primary_forward":primary,
        "source_sha256":{name:sha(path_for(root,name)) for name in [*map(lambda x:"v3/"+x,V3_FILES),*DEPENDENCIES]},
        "development_sha256":{name:sha(results/name) for name in ["step2/factor_panel.parquet","step2/states.parquet",
                     "step2/features.json","step3/predictions.parquet","step3/model_spec.json","step3/model_metrics.csv","step4/portfolio_metrics.csv",
                     "step4/unresolved_exposure.csv"]},
        "previous_exposure":"Earlier V1 forward data exists locally. V3 freeze uses only 2015–2025 inputs.",
    }
    configuration["runtime"]=environment(configuration["models"]["backend"])
    receipt={"registered_utc":datetime.now(timezone.utc).isoformat(),
             "configuration_sha256":json_digest(configuration),"configuration":configuration}
    with out.open("x",encoding="utf-8") as handle:
        handle.write(json_text(receipt))
    print(f"Frozen SHA256: {receipt['configuration_sha256']}")


def verify(root, results, path):
    receipt=json.loads(path.read_text(encoding="utf-8"))
    cfg=receipt["configuration"]
    market.need(json_digest(cfg)==receipt["configuration_sha256"],"Configuration digest mismatch")
    market.need(cfg["runtime"]==environment(cfg["models"]["backend"]),"Runtime changed after freeze")
    for files,base in [(cfg["source_sha256"],lambda name:path_for(root,name)),
                       (cfg["development_sha256"],lambda name:results/name)]:
        for name,expected in files.items():
            market.need(sha(base(name))==expected,f"Hash mismatch: {name}")
    return receipt


def holdout_dates():
    market.WARM=pd.Timestamp("2025-05-01")
    market.FIRST=pd.Timestamp("2025-12-01")
    market.END=CUTOFF


def download(cache, mirror):
    """The user runs this command. Resume from any existing partition."""
    holdout_dates()
    market.collect(cache,2026,"all")
    benchmark=cache/"H00300.parquet"
    if not benchmark.exists() or mirror:
        market.collect_benchmark(cache,mirror,pd.Timestamp("2025-12-31"))
    forthcoming=cache/"forward_schedule.json"
    snapshot=cache/"members"/f"{CUTOFF:%Y-%m-%d}.parquet"
    if not forthcoming.exists() or not snapshot.exists():
        session=market.BaoSession()
        try:
            snapshot_date=str(CUTOFF.date())
            frame=session.query("query_hs300_stocks","latest signal snapshot",date=snapshot_date)
            market.save(market.check_snapshot(frame,CUTOFF),snapshot)
            cal=session.query("query_trade_dates","next session",
                              start_date=str((CUTOFF+pd.Timedelta(days=1)).date()),
                              end_date=str((CUTOFF+pd.Timedelta(days=10)).date()))
            dates=pd.to_datetime(cal.loc[cal.is_trading_day.eq("1"),"calendar_date"])
            market.need(not dates.empty,"Cannot determine next execution session")
            forthcoming.write_text(json_text({"signal_date":snapshot_date,"execution_date":str(dates.min().date())}),encoding="utf-8")
        finally:
            session.close()
    _,schedule,members=market.load_universe(cache)
    tasks=market.ranges(schedule,members)
    new_codes=set(pd.read_parquet(snapshot).code)-set(tasks[2026])
    for year in [2025,2026]:
        for code in sorted(new_codes):
            for basis,flag in [("raw","3"),("adjusted","1")]:
                path=cache/basis/str(year)/f"{code}.parquet"
                if not path.exists():
                    session=market.BaoSession()
                    try:
                        frame=market.daily(session,code,*market.year_bounds(year),market.RAW,flag)
                        market.save(frame,path)
                    finally:
                        session.close()
    print(f"Download complete: {cache}")


def prepare(root, cache, out, receipt):
    holdout_dates()
    market.audit(root,cache,out/"audit")
    days,schedule,members=market.load_universe(cache)
    forward=json.loads((cache/"forward_schedule.json").read_text(encoding="utf-8"))
    next_day=pd.Timestamp(forward["execution_date"])
    market.need(pd.Timestamp(forward["signal_date"])==CUTOFF and next_day>CUTOFF,"Invalid latest signal dates")
    additional=market.check_snapshot(pd.read_parquet(cache/"members"/f"{CUTOFF:%Y-%m-%d}.parquet"),CUTOFF)
    tasks=market.ranges(schedule,members)
    # Check new cutoff-day entries explicitly. These are never filled with next day's data.
    missing=set(additional.code)-set(tasks[2026])
    for code in missing:
        for year in [2025,2026]:
            for basis in ["raw","adjusted"]:
                market.need((cache/basis/str(year)/f"{code}.parquet").exists(),
                            f"Latest snapshot requires {basis}/{year}/{code}.parquet")
    if missing:
        tasks={y:sorted(set(codes)|missing) for y,codes in tasks.items()}
    panel,index=panel_and_index(cache,days,tasks)
    schedule=pd.concat([schedule,pd.DataFrame({"signal_date":[CUTOFF],"execution_date":[next_day]})],ignore_index=True)
    data,states=make_data(panel,index,pd.concat([members,additional]),schedule,days,CUTOFF)
    market.need(data.loc[data.signal_date.eq(CUTOFF),"forward_return"].isna().all(),"Latest signal leaked a future return")
    market.save(data,out/"factor_panel.parquet")
    market.save(states,out/"states.parquet")
    market.save(panel.loc[panel.date.ge(FIRST),["date","code","raw_open","raw_close","preclose","tradestatus","isST"]],
                out/"execution_panel.parquet")
    (out/"prepared.json").write_text(json_text({"configuration_sha256":receipt["configuration_sha256"],
         "latest_execution_date":str(next_day.date())}),encoding="utf-8")


def combined_data(results, work):
    dev=pd.read_parquet(results/"step2/factor_panel.parquet").set_index(KEY)
    observed=pd.read_parquet(work/"factor_panel.parquet").set_index(KEY)
    cols=["exit_date","forward_return","index_forward_return","alpha_return"]
    common=dev.index.intersection(observed.index)
    overlaps=dev.loc[common,"entry_date"].eq(observed.loc[common,"entry_date"])
    backfill=dev.loc[common,"exit_date"].isna()&overlaps
    true_keys=common[backfill.to_numpy()]
    dev.loc[true_keys,cols]=observed.loc[true_keys,cols]
    combined=pd.concat([dev,observed.loc[~observed.index.isin(dev.index)]],verify_integrity=True).reset_index()
    dev_states=pd.read_parquet(results/"step2/states.parquet")
    new_states=pd.read_parquet(work/"states.parquet")
    states=pd.concat([dev_states,new_states.loc[~new_states.signal_date.isin(dev_states.signal_date)]],ignore_index=True)
    return combined,states


def make_forward(work, artifacts, scores, gates, factors, states, cfg, next_day, ledger_clear):
    model,method,_=cfg["primary_forward"].split("__")
    ranks=scores.loc[scores.strategy.eq(model)&scores.signal_date.eq(CUTOFF)].copy()
    market.need(len(ranks)>=280,"Latest eligible universe has fewer than 280 stocks")
    ledger=artifacts/"ledgers"/cfg["primary_forward"]
    holdings=pd.read_parquet(ledger/"holdings.parquet")
    holdings=holdings.loc[holdings.date.eq(CUTOFF)].set_index("code")
    nav=pd.read_parquet(ledger/"nav.parquet").set_index("date")
    positions=holdings.shares.astype(int).to_dict()
    marks=pd.read_parquet(work/"execution_panel.parquet")
    marks=marks.loc[marks.date.eq(CUTOFF)].set_index("code")
    equity=nav.at[CUTOFF,"equity"]
    details=[]
    targets,selector,optimizer=make_rules(ranks,states,cfg["primary_forward"],method,details)
    if optimizer:
        locked={i:(code,CUTOFF,CUTOFF,int(row.locked_shares)) for i,(code,row) in enumerate(holdings.iterrows())
                if row.locked_shares}
        weights=optimizer(next_day,positions,locked,lambda code,_:marks.at[code,"raw_close"]
                          if code in marks.index and pd.notna(marks.at[code,"raw_close"]) else holdings.at[code,"mark"],equity)
    else:
        chosen=selector(next_day,targets[next_day],positions)
        weights={code:1/len(chosen) for code in chosen}
    actual=factors.loc[factors.signal_date.eq(CUTOFF)&factors.code.isin(ranks.code)].set_index("code")
    alpha_factors=cfg["factors"]["selected"]
    signs=pd.Series(cfg["factors"]["directions"])
    percentile=actual[alpha_factors].mul(signs).rank(pct=True).fillna(.5)
    state=states.set_index("signal_date").loc[CUTOFF]
    regime,lo,hi=beta_band(state)
    gate=gates.loc[gates.signal_date.eq(CUTOFF)]
    market.save(gate,artifacts/"forward_gate_weights.csv")
    market.save(pd.DataFrame([{"signal_date":CUTOFF,"proposed_execution_date":next_day,
        "primary":cfg["primary_forward"],"regime":regime,"beta_lower":lo,"beta_upper":hi,
        "ledger_quality":"verified_events" if ledger_clear else "provisional_action_review",
        **state[STATES].to_dict()}]),artifacts/"forward_market_state.csv")
    ranks=ranks.sort_values(["score","code"],ascending=[False,True]).set_index("code")
    ranks["rank"]=np.arange(1,len(ranks)+1)
    rows=[]
    orders=[]
    for code in dict.fromkeys([*weights,*positions]):
        prior=positions.get(code,0)
        weight=weights.get(code,0)
        locked=int(holdings.at[code,"locked_shares"]) if code in holdings.index else 0
        reference=marks.at[code,"raw_close"] if code in marks.index else np.nan
        desired=0 if not weight else prior
        if pd.notna(reference) and reference>0 and weight:
            desired=max(0,int(equity*weight/reference)-locked)
        delta=desired-prior
        if code in marks.index:
            lot,step,_=stock_rules(code,next_day,marks.loc[code])
            if delta>0 or (delta<0 and desired>0):
                sign=1 if delta>0 else -1
                size=(abs(delta)//step)*step
                delta=sign*size if size>=lot else 0
        orders.append({"code":code,"prior_shares":prior,"locked_shares":locked,"reference_close":reference,
                       "target_weight":weight,"indicative_order_shares":delta,"status":"indicative_unfilled"})
        if weight:
            p=percentile.loc[code].to_dict()
            strongest=min(p,key=lambda k:-p[k])
            rows.append({"code":code,"rank":ranks.at[code,"rank"],"model_score":ranks.at[code,"score"],
                         "beta":actual.at[code,"BETA_120"],"weight":weight,"strongest_signed_factor":strongest,
                         **{f"{key}_pct":value for key,value in p.items()}})
    market.save(pd.DataFrame(rows),artifacts/"forward_targets.csv")
    market.save(pd.DataFrame(orders),artifacts/"forward_indicative_orders.csv")
    market.save(ranks.head(30).reset_index(),artifacts/"forward_score_top30.csv")


def evaluate(root, results, cache, work, out, receipt):
    market.need(not out.exists(),f"Holdout ledger exists: {out}. Preserve this run and its registration.")
    market.need((work/"prepared.json").exists(),"Prepare and audit the isolated cache first")
    metadata=json.loads((work/"prepared.json").read_text(encoding="utf-8"))
    market.need(metadata["configuration_sha256"]==receipt["configuration_sha256"],"Prepared data and freeze disagree")
    cfg=receipt["configuration"]
    next_day=pd.Timestamp(metadata["latest_execution_date"])
    data,states=combined_data(results,work)
    fitted,gates,links=fit(data,states,cfg["factors"],pd.Timestamp("2017-01-01"),next_day,cfg["models"]["backend"])
    # Historical replay must reproduce the frozen predictions on every available development key.
    original=pd.read_parquet(results/"step3/predictions.parquet")
    compare=original.merge(fitted,on=KEY+["strategy","entry_date"],suffixes=("_dev","_replayed"),validate="one_to_one")
    market.need(len(compare)==len(original) and np.allclose(compare.score_dev,compare.score_replayed,rtol=1e-10,atol=1e-11),
                "Frozen development predictions do not replay")
    scores=fitted.loc[fitted.entry_date.ge(FIRST)].copy()
    market.save(scores,work/"predictions.parquet")
    market.save(links.loc[links.signal_date.dt.year.eq(2026)],work/"training_links.parquet")
    market.save(gates.loc[gates.signal_date.dt.year.eq(2026)],work/"gates_2026.csv")
    market.need(len(set(scores.loc[scores.entry_date.lt(next_day),"entry_date"]))>0,"Missing holdout score calendar")
    out.mkdir(parents=True)
    (out/"RUN_STARTED.json").write_text(json_text({"freeze_sha256":receipt["configuration_sha256"],
            "started_utc":datetime.now(timezone.utc).isoformat()}),encoding="utf-8")
    table=backtest(root,work,work,work/"audit",out,cache/"H00300.parquet",FIRST,CUTOFF)
    market.need(set(table.strategy)==set(cfg["variants"]),"Holdout variants differ from freeze")
    _,rank_metrics=describe(scores,data)
    market.save(rank_metrics,out/"model_metrics.csv")
    clear=table.set_index("strategy").at[cfg["primary_forward"],"unresolved_exposures"]==0
    make_forward(work,out,scores,gates,data,states,cfg,next_day,clear)
    (out/"RUN_FINISHED.json").write_text(json_text({"freeze_sha256":receipt["configuration_sha256"],
         "corporate_action_exposure_rows":int(table.unresolved_exposures.sum()),
         "forward_execution_observed":False}),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["freeze","download","prepare","evaluate"])
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--results",type=Path)
    p.add_argument("--registration",type=Path)
    p.add_argument("--cache",type=Path)
    p.add_argument("--work",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--benchmark-csv",type=Path)
    p.add_argument("--primary")
    a=p.parse_args()
    root=a.project_root.resolve()
    sys.path.insert(0,str(root/"scripts"))
    results=a.results or root/"research/v3/results"
    path=a.registration or root/"research/v3/freeze_registration_2026.json"
    cache=a.cache or root/"data/raw/v3/holdout_2026"
    work=a.work or root/"data/processed/v3/holdout_2026"
    if a.stage=="freeze":
        market.need(a.primary is not None,"Supply the chosen development forward strategy")
        register(root,results,a.primary,path)
        return
    receipt=verify(root,results,path)
    if a.stage=="download":
        download(cache,a.benchmark_csv)
    elif a.stage=="prepare":
        prepare(root,cache,work,receipt)
    else:
        evaluate(root,results,cache,work,a.output or results/"holdout_2026",receipt)


if __name__=="__main__":
    main()

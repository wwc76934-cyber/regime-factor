"""Compare buffered equal sizing with long-only beta-constrained targets."""
import argparse
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fetch_and_audit_v3_data import need, save
from v3_execution import simulate


CAP = .045
TURN = .16
CONC = 2.5
BUFFER = 45
FEES = (10, 20)
NAV_COLUMNS = ["strategy","date","cash","stock_value","pending_value","dividends_due","rights_prepaid",
               "equity","fee","gross_traded","opening_equity","stale_count"]
HOLDING_COLUMNS = ["strategy","date","code","shares","locked_shares","mark","value","stale"]
ORDER_COLUMNS = ["strategy","date","code","old","desired","requested","filled","new","price","gross","fee","cash","status"]
ACTION_COLUMNS = ["strategy","date","code","ex_date","stage","amount","shares","fraction"]
EXPOSURE_COLUMNS = ["strategy","date","code","shares","locked_shares","category","reason"]


def beta_band(state):
    trend, breadth = state.market_return_20d, state.market_breadth_20d
    if trend > 0 and breadth >= .5:
        return "offensive", .95, 1.10
    if trend <= 0 and breadth < .45:
        return "defensive", .65, .80
    return "neutral", .85, .95


def solve(score, beta, previous, lower, upper, turn=TURN, conc=CONC):
    """Solve the full-universe L1 quadratic exactly via the two scalar Lagrange multipliers."""
    need(len(score)*CAP >= 1 and conc > 0, "Insufficient candidates or invalid concentration penalty")
    ordered = np.sort(beta)
    count = int(1/CAP)
    remainder = 1-count*CAP
    min_beta = CAP*ordered[:count].sum()+remainder*ordered[count]
    max_beta = CAP*ordered[-count:].sum()+remainder*ordered[-count-1]
    feasible = lower <= max_beta and upper >= min_beta
    if not feasible:
        # Return a flagged boundary portfolio when the full invested mandate cannot reach the band.
        order = np.argsort(beta if lower <= max_beta else -beta, kind="stable")
        w = np.zeros_like(score)
        w[order[:count]] = CAP
        w[order[count]] = remainder
        return w, "unattainable_band"

    def at_multiplier(multiplier):
        adjusted = score - multiplier*beta
        kink = np.clip(previous,0,CAP)
        shift = adjusted-2*conc*kink

        def at_budget(budget):
            delta=shift-budget
            weight=kink+(np.maximum(delta-turn,0)+np.minimum(delta+turn,0))/(2*conc)
            return np.clip(weight,0,CAP)

        left=adjusted.min()-2*conc*CAP-turn-1
        right=adjusted.max()+turn+1
        budget=brentq(lambda value: at_budget(value).sum()-1,left,right,xtol=1e-10)
        return at_budget(budget)

    w=at_multiplier(0.)
    current=float(w@beta)
    target = np.clip(current,max(lower,min_beta),min(upper,max_beta))
    if abs(current-target)>1e-9:
        def gap(multiplier):
            return at_multiplier(multiplier)@beta-target
        bound=1.
        sign=1 if current>target else -1
        while sign*gap(sign*bound)>1e-9 and bound<2**20:
            bound*=2
        endpoint=sign*bound
        multiplier=endpoint if sign*gap(endpoint)>=0 else brentq(gap,*sorted([0.,endpoint]),xtol=1e-10)
        w=at_multiplier(multiplier)
    need(abs(w.sum()-1)<1e-7 and lower-1e-7<=w@beta<=upper+1e-7, "Optimization constraint tolerance exceeded")
    return w, "ok"


def load_actions(root, audit):
    from apply_verified_actions import OVERRIDES
    from normalize_corporate_actions import EVENT_COLUMNS

    events=pd.read_parquet(audit/"normalized_actions.parquet")
    review=pd.read_csv(audit/"action_review.csv",parse_dates=["ex_date"])
    # Prior project audit, CNNC 2024-082. Only applied to the matching review key.
    cnnp={"code":"sh.601985","record_date":"2024-11-29","ex_date":"2024-12-02","pay_date":"2024-12-02",
          "cash_per_share":"0.02","source_url":"https://static.cninfo.com.cn/finalpage/2024-11-25/1221831318.PDF"}
    verified=pd.DataFrame([*OVERRIDES,cnnp])
    extra=root/"scripts/holdout_2025/verified_actions_2025.csv"
    if extra.exists():
        verified=pd.concat([verified,pd.read_csv(extra,dtype=str)],ignore_index=True)
    for col in ["record_date","ex_date","pay_date"]:
        verified[col]=pd.to_datetime(verified[col])
    key=["code","ex_date"]
    need(not verified.duplicated(key).any(),"Duplicate verified amendments")
    verified=verified.merge(review[key],on=key,how="inner")
    need(verified.record_date.lt(verified.ex_date).all() and verified.pay_date.ge(verified.ex_date).all(),
         "Invalid verified dates")
    if not verified.empty:
        verified=verified.assign(stock_listing_date=pd.NaT,bonus_shares_per_share="0",
                                 reserve_shares_per_share="0",event_type="cash_only",source_rows=1)
        events=pd.concat([events,verified[EVENT_COLUMNS]],ignore_index=True)
    review=review.merge(verified[key],on=key,how="left",indicator=True).query('_merge == "left_only"')
    need(not events.duplicated(key).any(),"Duplicate execution events")
    return events, review.drop(columns="_merge")


def make_rules(rankings, state_table, name, method, details, overrides=None):
    by_day={day:g.sort_values(["score","code"],ascending=[False,True]).reset_index(drop=True)
            for day,g in rankings.groupby("entry_date")}
    start={day:{code:i+1 for i,code in enumerate(g.code.head(30))} for day,g in by_day.items()}
    state_by_signal=state_table.set_index("signal_date")

    def buffered(day, original, positions):
        g=by_day[day]
        top=g.code.head(BUFFER).tolist()
        keep=[code for code in top if positions.get(code,0)>0][:30]
        chosen=list(dict.fromkeys(keep+list(original)))[:30]
        return {code:top.index(code)+1 for code in chosen}

    def optimizer(day, positions, pending, mark, equity):
        g=by_day[day]
        state=state_by_signal.loc[g.signal_date.iat[0]]
        regime,lo,hi=beta_band(state)
        locked={}
        for code, _, _, quantity in pending.values():
            locked[code]=locked.get(code,0)+quantity
        previous=np.array([float(mark(code,"raw_open")*(positions.get(code,0)+locked.get(code,0))/equity)
                           if positions.get(code,0) or locked.get(code,0) else 0 for code in g.code])
        beta=g.BETA_120.to_numpy()
        score=g.score.rank(pct=True).to_numpy()-.5
        w,status=solve(score,beta,previous,lo,hi,**(overrides or {}))
        for i in np.flatnonzero(w>1e-9):
            details.append((name,day,g.signal_date.iat[0],g.code.iat[i],g.score.iat[i],float(w[i]),
                            float(previous[i]),float(beta[i]),regime,lo,hi,float(w@beta),status))
        return {g.code.iat[i]:float(w[i]) for i in np.flatnonzero(w>1e-9)}

    return start, buffered if method=="buffer" else None, optimizer if method=="optimized" else None


def execution_objects(frame):
    frame=frame.sort_values("date")
    return {day:{row.code:row for row in rows.itertuples(index=False)} for day,rows in frame.groupby("date")}


def coverage_metrics(nav, benchmark, anchor):
    v=nav.set_index("date").sort_index()
    b=benchmark.reindex(v.index)
    need(b.notna().all() and anchor>0,"Benchmark incomplete or missing anchor")
    daily=v.equity.div(v.equity.shift(fill_value=1_000_000))-1
    index_r=b.div(b.shift(fill_value=anchor))-1
    relative=(v.equity/1_000_000)/(b/anchor)
    excess=daily-index_r
    nav_path=pd.Series(np.r_[1.,v.equity.to_numpy()/1_000_000])
    excess_path=pd.Series(np.r_[1.,relative.to_numpy()])
    fills=v.opening_equity.notna()
    return dict(days=len(v),net_return=v.equity.iat[-1]/1_000_000-1,benchmark_return=b.iat[-1]/anchor-1,
         annualized_excess=relative.iat[-1]**(252/len(v))-1,
         information_ratio=excess.mean()/excess.std()*np.sqrt(252) if excess.std()>0 else np.nan,
         max_drawdown=(nav_path/nav_path.cummax()-1).min(),excess_max_drawdown=(excess_path/excess_path.cummax()-1).min(),
         realized_beta=daily.cov(index_r)/index_r.var(),two_way_turnover=(v.loc[fills,"gross_traded"]/v.loc[fills,"opening_equity"]).mean(),
         fees=v.fee.sum())


def run(root,source,models,audit,out,benchmark_path,start,end,model_names=None,methods=("buffer","optimized")):
    sys.path.insert(0,str(root/"scripts"))
    scores=pd.read_parquet(models/"predictions.parquet")
    states=pd.read_parquet(source/"states.parquet")
    frame=pd.read_parquet(source/"execution_panel.parquet")
    bench=pd.read_parquet(benchmark_path).set_index("date").close.astype(float).sort_index()
    dates=bench.loc[start:end].index
    anchor=bench.loc[bench.index<dates[0]].iat[-1]
    all_models=sorted(scores.strategy.unique()) if model_names is None else model_names
    events,review=load_actions(root,audit)
    scores=scores.loc[scores.entry_date.isin(dates)]
    scoring_dates=set(scores.entry_date)
    need(len(dates)>0 and scoring_dates,"Missing prices or predictions")
    need(set(dates).issubset(set(frame.date)),"Missing trading sessions in execution panel")
    bars=execution_objects(frame.loc[frame.date.isin(dates)])
    details,metrics,navs,issues,annual,beta_audits=[],[],[],[],[],[]
    out.mkdir(parents=True,exist_ok=True)
    for model in all_models:
        ranks=scores.loc[scores.strategy.eq(model)]
        need(set(ranks.entry_date)==scoring_dates and not ranks.duplicated(["entry_date","code"]).any(),
             f"Incomparable forecasts: {model}")
        chosen_methods=methods if model.startswith("v3_") and model!="v3_state_ridge_similarity" else ("buffer",)
        for method in chosen_methods:
            for fee in FEES:
                name=f"{model}__{method}__{fee}bp"
                targets,select,weight=make_rules(ranks,states,name,method,details)
                results=simulate(name,list(dates),bars,targets,events,review,Decimal("1000000"),
                                 Decimal(fee)/Decimal(10000),select,weight)
                nav,orders,holdings,actions,exposure=(pd.DataFrame(values,columns=columns)
                   for values,columns in zip(results,[NAV_COLUMNS,ORDER_COLUMNS,HOLDING_COLUMNS,ACTION_COLUMNS,EXPOSURE_COLUMNS],strict=True))
                for file,data in [("nav",nav),("orders",orders),("holdings",holdings),("actions",actions)]:
                    save(data,out/"ledgers"/name/f"{file}.parquet")
                lookup=ranks[["entry_date","code","BETA_120"]].rename(columns={"entry_date":"date"})
                actual=holdings.loc[holdings.date.isin(scoring_dates)].merge(lookup,on=["date","code"],how="left")
                actual["weighted_beta"]=actual.value*actual.BETA_120
                actual["known_beta_value"]=actual.value.where(actual.BETA_120.notna(),0)
                actual=actual.groupby("date")[["weighted_beta","known_beta_value"]].sum()
                actual=actual.join(nav.set_index("date")[["equity","stock_value","pending_value"]])
                actual["estimated_beta"]=actual.weighted_beta/actual.equity
                actual["unknown_beta_weight"]=(actual.stock_value+actual.pending_value-actual.known_beta_value)/actual.equity
                weeks=ranks[["entry_date","signal_date"]].drop_duplicates().merge(states,on="signal_date",validate="many_to_one")
                regimes=pd.DataFrame([beta_band(row) for row in weeks.itertuples()],columns=["regime","beta_lower","beta_upper"])
                actual=actual.join(regimes.set_index(pd.DatetimeIndex(weeks.entry_date)),how="left")
                beta_audits.append(actual.reset_index().assign(strategy=name))
                save(exposure,out/"ledgers"/name/"unresolved_exposure.csv")
                issues.append(exposure)
                navs.append(nav)
                row={"strategy":name,"model":model,"method":method,"fee_bp":fee,
                     "unresolved_exposures":len(exposure),"stale_marks":int(nav.stale_count.sum()),
                     **coverage_metrics(nav,bench,anchor)}
                metrics.append(row)
                prior_equity=1_000_000.
                for year,sample in nav.groupby(nav.date.dt.year):
                    bench_before=bench.loc[bench.index<sample.date.iat[0]].iat[-1]
                    end_equity=sample.equity.iat[-1]
                    annual.append((name,year,end_equity/prior_equity-1,bench.loc[sample.date.iat[-1]]/bench_before-1))
                    prior_equity=end_equity
                print(f"{name}: {row['net_return']:+.2%}, turnover {row['two_way_turnover']:.2%}, unverified exposures {len(exposure)}",flush=True)
    outputs={"portfolio_metrics.csv":pd.DataFrame(metrics),"daily_nav.parquet":pd.concat(navs),
        "unresolved_exposure.csv":pd.concat(issues),
        "executed_beta.csv":pd.concat(beta_audits),
        "annual_returns.csv":pd.DataFrame(annual,columns=["strategy","year","net_return","benchmark_return"]),
        "optimized_targets.parquet":pd.DataFrame(details,columns=["strategy","date","signal_date","code","score","weight",
                          "previous_weight","beta","regime","beta_lower","beta_upper","target_beta","solver_status"])}
    for path,value in outputs.items():
        save(value,out/path)
    return outputs["portfolio_metrics.csv"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--source",type=Path)
    p.add_argument("--models",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    results=a.project_root/"research/v3/results"
    run(a.project_root,a.source or results/"step2",a.models or results/"step3",a.audit or results/"step1",
        a.output_dir or results/"step4",a.project_root/"data/raw/v3/development/H00300.parquet","2017-01-01","2025-12-31")

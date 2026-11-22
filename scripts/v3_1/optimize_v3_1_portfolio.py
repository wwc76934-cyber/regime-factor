"""Replay V3.1 development portfolios against H00300, FloatMV and CSI300_EW."""
import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v3_1_base import initialize
from v3_1_active_weights import ACTIVE_LIMIT, RISK_AVERSION, SINGLE_LIMIT, lot_targets, solve_active


RIDGE = "v3_1_state_ridge"
HUBER = "v3_1_lgb_huber"
FUSION = "fusion_50_50"
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2025-12-31")
KEY = ["signal_date", "code"]
INITIAL = Decimal("1000000")
BENCHMARKS = {"H00300": "h00300", "FloatMV": "float_mv_nav", "CSI300_EW": "equal_weight_nav"}


@dataclass(frozen=True)
class Variant:
    model: str
    horizon: int
    fee_bps: int
    construction: str = "dynamic"
    eta: float = .96
    ema: float = 1.

    @property
    def name(self):
        return f"{self.model}_h{self.horizon}_{self.construction}_eta{self.eta:g}_ema{self.ema:g}_{self.fee_bps}bp"


def variants(block):
    for fee in (10,20):
        if block=="core":
            for horizon in (10,20):
                for model in (RIDGE,HUBER,FUSION):
                    yield Variant(model,horizon,fee)
        elif block=="frequency":
            for smoothing in (.3,.5):
                yield Variant(FUSION,5,fee,ema=smoothing)
        elif block=="turnover_sensitivity":
            for horizon in (10,20):
                yield Variant(FUSION,horizon,fee,ema=.3)
        else:
            for horizon in (10,20):
                yield Variant(FUSION,horizon,fee,eta=1.)
                yield Variant(FUSION,horizon,fee,construction="static")
                yield Variant(FUSION,horizon,fee,construction="unanchored",eta=1.)


def zscore(values):
    scale = values.std(ddof=0)
    return (values-values.mean())/scale if scale>0 else values*0


def scores_by_variant(predictions, labels, model, ema):
    p = predictions.loc[predictions.strategy.isin([RIDGE,HUBER])]
    wide = p.pivot(index=KEY,columns="strategy",values="score").sort_index()
    if wide.isna().any().any():
        raise ValueError("Ridge and Huber stock keys do not match")
    normalized = wide.groupby(level="signal_date")[[RIDGE,HUBER]].transform(zscore)
    raw = normalized.mean(axis=1) if model==FUSION else normalized[model]
    result = raw.rename("score").reset_index()
    memory = pd.Series(dtype=float)
    if ema<1:
        for _,group in result.groupby("signal_date"):
            by_stock = group.set_index("code").score
            previous = memory.reindex(by_stock.index).fillna(by_stock)
            memory = ema*by_stock+(1-ema)*previous
            result.loc[group.index,"score"] = memory.to_numpy()
    result["score_z"] = result.groupby("signal_date").score.transform(zscore).clip(-3,3)
    test = result.merge(labels[KEY+["exit_date","risk_adjusted_target"]],on=KEY,validate="one_to_one")
    by_day = {}
    for day,g in test.groupby("signal_date"):
        pair = g.dropna(subset=["risk_adjusted_target"])
        if len(pair)>=50 and pair.score.nunique()>1:
            by_day[day] = (g.exit_date.iat[0],pair.score.rank().corr(pair.risk_adjusted_target.rank()))
    ic = {}
    for day in result.signal_date.unique():
        mature = [s for stamp,(exit,s) in by_day.items() if stamp<day and pd.notna(exit) and exit<=day and np.isfinite(s)][-52:]
        ic[day] = (float(np.clip(np.mean(mature),.02,.08)) if mature else .02,len(mature))
    return result,ic


def regime_bands(states):
    if states.market_return_20d>0 and states.market_breadth_20d>=.50:
        return "offensive",.98,1.08
    if states.market_return_20d<=0 and states.market_breadth_20d<.45:
        return "defensive",.82,.92
    return "neutral",.92,1.00


def portfolio_schedule(days, scoring_dates, first, spacing):
    days = days[days>=first][::spacing]
    shift = scoring_dates.searchsorted(days,side="left")-1
    if (shift<0).any():
        raise ValueError("Portfolio schedule requires an earlier signal")
    return pd.Series(scoring_dates[shift],index=days)


def prepare_market(factors, anchor):
    f = factors[KEY+["BETA_120","VOL_20","IDIO_VAR_60"]].merge(
        anchor[KEY+["weight_float_mv"]],on=KEY,how="inner",validate="one_to_one")
    return {day:g.set_index("code").sort_index() for day,g in f.groupby("signal_date")}


def make_planner(variant, schedule, signals, calibration, market, states, quotes, first_day, plan_log, signal_log):
    from optimize_v3_portfolio import solve as legacy
    from v3_execution import money, stock_rules

    weekly = {day:g.set_index("code") for day,g in signals.groupby("signal_date")}
    fee = variant.fee_bps/10000
    name = variant.name
    horizons = variant.horizon

    def choose(day, positions, pending, mark, equity):
        sig = schedule[day]
        scores = weekly[sig]
        base = market[sig].join(scores[["score_z"]])
        st = states.loc[sig]
        regime, nominal_lower, nominal_upper = regime_bands(st)
        lo,hi = (.95,1.05) if variant.construction=="static" else (nominal_lower,nominal_upper)
        pending_by_code = {}
        for code, _, _, quantity in pending.values():
            pending_by_code[code] = pending_by_code.get(code,0)+quantity
        old = {code: float(mark(code,"raw_open"))*(positions.get(code,0)+pending_by_code.get(code,0))/float(equity)
               for code in set(positions)|set(pending_by_code)}
        fitted_ic,n_ic = calibration[sig]
        anchor = base.weight_float_mv.to_numpy(dtype=float)
        if not np.isclose(anchor.sum(),1) or (anchor<0).any():
            raise ValueError("Point-in-time FloatMV weights do not sum to one")
        estimate = base.BETA_120.fillna(1.).to_numpy()
        prior = pd.Series(old,dtype=float).reindex(base.index,fill_value=0).to_numpy()
        score = base.score_z.fillna(0).to_numpy()
        if variant.construction=="unanchored":
            order = base.dropna(subset=["score_z"]).sort_values("score_z",ascending=False,kind="stable").index[:30]
            index = base.index.get_indexer(order)
            selected, status = legacy(pd.Series(score[index]).rank(pct=True).to_numpy()-.5,
                                      estimate[index],prior[index],lo,hi)
            target = np.zeros(len(base))
            target[index] = selected
            cap = np.where(target>0,.045,0)
            attainable_lo = attainable_hi = np.nan
            infeasible = status!="ok"
        else:
            horizon_sigma = np.maximum(base.VOL_20.fillna(base.VOL_20.median()).to_numpy(),.01)*np.sqrt(horizons)
            idio = base.IDIO_VAR_60.fillna(base.IDIO_VAR_60.median()).clip(lower=.005**2).to_numpy()*horizons
            target,attainable_lo,attainable_hi,infeasible = solve_active(
                anchor, fitted_ic*horizon_sigma*score, estimate, idio,
                float(st.market_volatility_20d**2*horizons), prior, fee, lo/variant.eta,hi/variant.eta)
            cap = np.minimum(SINGLE_LIMIT,anchor+ACTIVE_LIMIT)
        target_data = {}
        for i, code in enumerate(base.index):
            if target[i] > 1e-10 or code in old:
                if code in old:
                    opening_mark = float(mark(code,"raw_open"))
                else:
                    opening = quotes.get(day,{}).get(code)
                    opening_mark = float(opening.raw_open) if opening is not None else np.nan
                target_data[code] = {"weight":float(target[i]),"cap":float(cap[i]),"mark":opening_mark}
            if target[i]>1e-10:
                plan_log.append((day,sig,code,float(anchor[i]),float(target[i]),float(prior[i]),float(estimate[i])))
        for code in set(old)-set(target_data):
            target_data[code] = {"weight":0.,"cap":0.,"mark":float(mark(code,"raw_open"))}
        planned = lot_targets(target_data,positions,pending_by_code,quotes.get(day,{}),day,first_day,
                              equity,fee,stock_rules,money)
        # A non-empty placeholder keeps the reused engine defined if no order can be sized.
        if not planned:
            planned = {base.index[0]:0.}
        signal_log.append((day,sig,regime,lo,hi,nominal_lower,nominal_upper,float(target@estimate),
                           attainable_lo,attainable_hi,infeasible,fitted_ic,n_ic,float(anchor@estimate),
                           float(anchor[base.BETA_120.isna()].sum())))
        return dict(sorted(planned.items(),key=lambda kv:(-kv[1],kv[0])))

    return choose


def compute_actual_beta(nav, holdings, panel, schedule, plan_stats, beta_history):
    trade_days = schedule.index
    actual = holdings.loc[holdings.date.isin(trade_days)].merge(
        panel.loc[panel.date.isin(trade_days),["date","code","raw_open","prior_mark"]],
        on=["date","code"],how="left",validate="one_to_one")
    actual["signal_date"] = actual.date.map(schedule)
    actual = actual.merge(beta_history,on=["signal_date","code"],how="left",validate="many_to_one")
    quote = actual.raw_open.where(actual.raw_open.gt(0),actual.prior_mark).fillna(actual.mark)
    actual["stock_mark_value"] = (actual.shares+actual.locked_shares)*quote
    actual["beta_mass"] = actual.stock_mark_value*actual.BETA_120
    actual["unknown_mark_value"] = actual.stock_mark_value.where(actual.BETA_120.isna(),0)
    exposure = actual.groupby("date")[["beta_mass","unknown_mark_value","stock_mark_value"]].sum()
    opens = nav.set_index("date").loc[trade_days]
    after_fee = opens.opening_equity-opens.fee
    frame = plan_stats.set_index("date").join(exposure,how="left").fillna(
        {"beta_mass":0,"unknown_mark_value":0,"stock_mark_value":0})
    frame["posttrade_beta_lower"] = (frame.beta_mass+.2*frame.unknown_mark_value)/after_fee
    frame["posttrade_beta_upper"] = (frame.beta_mass+2.2*frame.unknown_mark_value)/after_fee
    frame["posttrade_beta"] = (frame.beta_mass+frame.unknown_mark_value)/after_fee
    frame["unknown_beta_equity_weight"] = frame.unknown_mark_value/after_fee
    frame["posttrade_stock_weight"] = frame.stock_mark_value/after_fee
    frame["inside_band"] = (frame.posttrade_beta_lower.ge(frame.policy_lo-1e-10) &
                            frame.posttrade_beta_upper.le(frame.policy_hi+1e-10))
    frame["inside_official_regime_band"] = (frame.posttrade_beta_lower.ge(frame.regime_lo-1e-10) &
                                            frame.posttrade_beta_upper.le(frame.regime_hi+1e-10))
    return frame.reset_index()


def benchmark_metrics(nav, benchmarks):
    d = nav.set_index("date")
    portfolio = d.equity.div(d.equity.shift(fill_value=float(INITIAL)))-1
    rows,annual = [],[]
    for name,col in BENCHMARKS.items():
        b = benchmarks[col]
        index = b.loc[d.index].div(b.shift().loc[d.index])-1
        if index.isna().any():
            raise ValueError("Incomplete benchmark daily returns")
        for year, subset in [("all", portfolio), *portfolio.groupby(portfolio.index.year)]:
            active = subset-index.loc[subset.index]
            port = float((1+subset).prod()-1)
            anchor = float((1+index.loc[subset.index]).prod()-1)
            relative = pd.Series(np.r_[1.,((1+subset)/(1+index.loc[subset.index])).cumprod().to_numpy()])
            data = dict(strategy=nav.strategy.iat[0],benchmark=name,year=year,net_return=port,benchmark_return=anchor,
                        cumulative_excess_pp=port-anchor,geometric_excess=relative.iat[-1]-1,
                        annualized_excess=relative.iat[-1]**(252/len(subset))-1,
                        information_ratio=float(np.sqrt(252)*active.mean()/active.std()) if active.std()>0 else np.nan,
                        max_excess_drawdown=float((1-relative/relative.cummax()).max()),daily_win_rate=float(active.gt(0).mean()))
            (rows if year=="all" else annual).append(data)
    return pd.DataFrame(rows),pd.DataFrame(annual)


def run(root, out, block):
    v3, _, days, _, _ = initialize(root)
    from optimize_v3_portfolio import (ACTION_COLUMNS, EXPOSURE_COLUMNS, HOLDING_COLUMNS, NAV_COLUMNS,
        ORDER_COLUMNS, execution_objects, load_actions)
    from v3_execution import simulate

    directory = root/"research/v3_1/results"
    step1,step2 = directory/"step1",directory/"step2"
    if json.loads((step2/"step2_status.json").read_text(encoding="utf-8"))["status"]!="go":
        raise ValueError("Step 2 signal gate has not passed")
    panel = pd.read_parquet(root/"research/v3/results/step2/execution_panel.parquet").sort_values(["code","date"])
    if panel.date.max()!=END:
        raise ValueError("Development execution must end in 2025")
    valid = panel.raw_close.where(panel.raw_close.gt(0))
    panel["prior_mark"] = valid.groupby(panel.code).transform(lambda x:x.ffill().shift())
    days = days[(days>=START)&(days<=END)]
    quote_objects = execution_objects(panel.loc[panel.date.isin(days)])
    events,review = load_actions(root,root/"research/v3/results/step1")
    states = pd.read_parquet(step1/"states.parquet").set_index("signal_date")
    factors = pd.read_parquet(step1/"factor_panel.parquet")
    labels = pd.read_parquet(step1/"labels.parquet")
    anchor = pd.read_parquet(directory/"step0/benchmark_weights.parquet")
    benchmarks = pd.read_parquet(directory/"step0/benchmark_nav.parquet").set_index("date")
    market = prepare_market(factors,anchor)
    beta_history = factors.pivot(index="signal_date",columns="code",values="BETA_120").ffill().stack()
    beta_history = beta_history.rename("BETA_120").reset_index()
    predictions = {h:pd.read_parquet(step2/f"h{h}"/"predictions.parquet") for h in (5,10,20)}
    first = max(p.entry_date.min() for p in predictions.values())
    sets = {}
    result,tri,by_year,beta_table,all_issues = [],[],[],[],[]
    for v in variants(block):
        key = (v.model,v.horizon,v.ema)
        if key not in sets:
            sets[key] = scores_by_variant(predictions[v.horizon],labels.loc[labels.horizon.eq(v.horizon)],v.model,v.ema)
        score,calib = sets[key]
        schedule = portfolio_schedule(days,pd.DatetimeIndex(sorted(score.signal_date.unique())),first,v.horizon)
        details,signals = [],[]
        weights = make_planner(v,schedule,score,calib,market,states,quote_objects,days[0],details,signals)
        targets = {day:{"_weight_placeholder":1} for day in schedule.index}
        raw = simulate(v.name,list(days),quote_objects,targets,events,review,INITIAL,Decimal(v.fee_bps)/10000,
                       weight_targets=weights)
        nav,orders,holdings,actions,issues = [
            pd.DataFrame(rows,columns=cols) for rows,cols in zip(raw,[NAV_COLUMNS,ORDER_COLUMNS,HOLDING_COLUMNS,
                                                                  ACTION_COLUMNS,EXPOSURE_COLUMNS],strict=True)]
        stats = pd.DataFrame(signals,columns=["date","signal_date","regime","policy_lo","policy_hi","regime_lo",
            "regime_hi","target_beta","feasible_beta_lo","feasible_beta_hi","unreachable_target","calibration_ic",
            "calibration_weeks","float_mv_beta","anchor_unknown_beta_weight"])
        beta = compute_actual_beta(nav,holdings,panel,schedule,stats,beta_history).assign(strategy=v.name)
        triplet,annual = benchmark_metrics(nav,benchmarks)
        v3.save(triplet,out/"runs"/v.name/"benchmark_metrics.csv")
        v3.save(annual,out/"runs"/v.name/"annual_benchmarks.csv")
        tri.append(triplet)
        by_year.append(annual)
        for path,frame in [("nav",nav),("orders",orders),("holdings",holdings),("actions",actions),
                           ("targets",pd.DataFrame(details,columns=["date","signal_date","code","anchor_weight",
                            "target_weight","previous_weight","beta"]))]:
            v3.save(frame,out/"runs"/v.name/f"{path}.parquet")
        v3.save(beta,out/"runs"/v.name/"beta_rebalances.csv")
        v3.save(issues,out/"runs"/v.name/"unresolved_exposure.csv")
        all_issues.append(issues)
        groups = beta.groupby("regime").agg(decisions=("inside_band","size"),beta_hit_rate=("inside_band","mean"),
           regime_beta_hit_rate=("inside_official_regime_band","mean"),mean_target_beta=("target_beta","mean"),
           mean_posttrade_beta=("posttrade_beta","mean"),mean_stock_weight=("posttrade_stock_weight","mean"),
           unreachable=("unreachable_target","sum"),mean_unknown_beta_weight=("unknown_beta_equity_weight","mean"))
        groups = groups.reset_index().assign(strategy=v.name)
        beta_table.append(groups)
        ref = triplet.set_index("benchmark").loc["H00300"]
        progress = nav.gross_traded.div(nav.opening_equity).sum()*5/len(nav)
        offensive = groups.set_index("regime").loc["offensive","regime_beta_hit_rate"]
        row={"strategy":v.name,"model":v.model,"horizon":v.horizon,"fee_bps":v.fee_bps,
             "construction":v.construction,"eta":v.eta,"ema":v.ema,"rebalance_count":len(schedule),
             "weekly_equivalent_two_way_turnover":progress,"offensive_beta_hit":offensive,
             "unresolved_exposures":len(issues),"unreachable_target_days":int(beta.unreachable_target.sum()),
             "stale_marks":int(nav.stale_count.sum()),"fees":nav.fee.sum(),"net_return":ref.net_return,
             "H00300_excess":ref.cumulative_excess_pp,"H00300_ir":ref.information_ratio,
             "FloatMV_excess":triplet.set_index("benchmark").loc["FloatMV","cumulative_excess_pp"],
             "CSI300_EW_excess":triplet.set_index("benchmark").loc["CSI300_EW","cumulative_excess_pp"]}
        result.append(row)
        print(f"{v.name}: net {ref.net_return:+.2%}, H00300 excess {ref.cumulative_excess_pp:+.2%}, "
              f"weekly turnover {progress:.2%}, offensive beta hit {offensive:.2%}, issues {len(issues)}",flush=True)
    for name,table in [("model_portfolio_comparison_v3_1.csv",pd.DataFrame(result)),
                       ("benchmark_comparison_v3_1.csv",pd.concat(tri,ignore_index=True)),
                       ("annual_benchmarks_v3_1.csv",pd.concat(by_year,ignore_index=True)),
                       ("beta_by_regime_v3_1.csv",pd.concat(beta_table,ignore_index=True)),
                       ("unresolved_exposure.csv",pd.concat(all_issues,ignore_index=True))]:
        v3.save(table,out/name)
    (out/"experiment_spec.json").write_text(json.dumps({"risk_aversion":RISK_AVERSION,"initial_cash":str(INITIAL),
        "period":["2017-01-01","2025-12-31"],"turnover":"sum of daily gross traded / opening NAV, scaled by 5/sessions",
        "funding_buffer_eta":.96,"active_limit":ACTIVE_LIMIT,"absolute_limit":SINGLE_LIMIT,
        "execution":"every h sessions at open, latest earlier weekly signal", "issue_free":not any(len(x) for x in all_issues)},
        indent=2),encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--block",choices=["core","frequency","controls","turnover_sensitivity"],required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    run(args.project_root.resolve(),args.output or args.project_root/"research/v3_1/results/step3"/args.block,args.block)

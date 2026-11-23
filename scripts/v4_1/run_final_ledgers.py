"""Replay the four registered ledgers, verifying the 2025 anchor before releasing any return."""
import json
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

from prepare_final_inputs import ROOT, WORK, DEST, OLD, FIRST, LAST, ANCHOR, market, seal, trading_days
import optimize_v3_1_portfolio as engine
import v3_1_execution as execution
from optimize_v3_portfolio import (execution_objects, load_actions, NAV_COLUMNS, ORDER_COLUMNS,
                                  HOLDING_COLUMNS, ACTION_COLUMNS, EXPOSURE_COLUMNS)

CASH = [
    ("sh.600930", "2026-02-10", "2026-02-11", "0.03", "https://static.cninfo.com.cn/finalpage/2026-02-04/1224964356.PDF"),
    ("sh.603288", "2026-02-05", "2026-02-06", "0.30", "https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0130/2026013001658_c.pdf"),
    ("sz.300760", "2026-05-27", "2026-05-28", "1.56", "https://static.cninfo.com.cn/finalpage/2026-05-20/1225320390.PDF"),
    ("sz.301308", "2026-06-01", "2026-06-02", "0.9907442", "https://static.cninfo.com.cn/finalpage/2026-05-26/1225331662.PDF"),
    ("sz.000538", "2026-09-23", "2026-09-24", "1.038", "https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-09-17/60be8b82-bfcd-4d39-a079-a05d3e50b7b7.PDF"),
]
SWAP = execution.Exchange("sh.601059","sh.601995",*map(pd.Timestamp,["2026-10-16","2026-10-19","2026-10-23","2026-10-29"]),Decimal(".5210"))


def verified_events():
    events,prior_review = execution.verified_cash_event(*load_actions(ROOT,ROOT/"research/v3/results/step1"))
    verified = pd.read_parquet(DEST/"audit/normalized_actions.parquet")
    queue = pd.read_csv(DEST/"audit/action_review.csv",parse_dates=["ex_date"])
    extra = pd.DataFrame(CASH,columns=["code","record_date","ex_date","cash_per_share","source"])
    for c in ["record_date","ex_date"]:
        extra[c] = pd.to_datetime(extra[c])
    extra = extra.assign(pay_date=extra.ex_date, stock_listing_date=pd.NaT, bonus_shares_per_share="0",
                         reserve_shares_per_share="0",event_type="cash_only",source_rows=1)
    keys = set(zip(extra.code,extra.ex_date)) | {(SWAP.old,SWAP.retire)}
    if keys != set(zip(queue.code,queue.ex_date)):
        raise ValueError("Additional corporate actions require evidence")
    market.save(extra,DEST/"audit/verified_dividends.csv")
    market.save(pd.DataFrame([{**vars(SWAP),"source":"https://static.cninfo.com.cn/finalpage/2026-10-27/1225627349.PDF",
        "terms":"https://static.cninfo.com.cn/finalpage/2026-10-16/1225606336.PDF"}]),DEST/"audit/verified_exchange.csv")
    events = pd.concat([events.loc[events.ex_date.lt(FIRST)],verified,extra[verified.columns]],ignore_index=True)
    if events.duplicated(["code","ex_date"]).any():
        raise ValueError("Duplicate entitlement")
    return events, prior_review


def signal_series(account,labels,cfg):
    primary = account["role"]=="primary"
    old_file = (ROOT/"research/v4_1/results/step3/inputs/v4_1_moe_huber" if primary else OLD/"step4/nonnegative_forecasts")/"h5/predictions.parquet"
    pred = pd.concat([pd.read_parquet(old_file),pd.read_parquet(WORK/("primary_predictions.parquet" if primary else "baseline_predictions.parquet"))])
    signals,cal = engine.scores_by_variant(pred,labels,engine.FUSION,.3)
    anchor = cfg["continuity"]["EMA_anchors"][account["model"]]
    t = pd.Timestamp(anchor["signal_date"])
    saved = pd.Series(anchor["score_memory_by_code"]).sort_index()
    got = signals.loc[signals.signal_date.eq(t)].set_index("code").score.sort_index()
    if not saved.index.equals(got.index):
        raise ValueError("EMA constituent memory mismatch")
    np.testing.assert_allclose(saved,got,atol=1e-9)
    np.testing.assert_allclose(cal[t], [anchor["calibration_ic"],anchor["calibration_weeks"]],atol=1e-10)
    return signals,cal


def check(account, nav, orders, holdings, actions, issues, events, quotes):
    balances = account["end_balances"]
    n = nav.set_index("date")
    np.testing.assert_allclose(n.loc[ANCHOR,list(balances)].astype(float),list(balances.values()),atol=.004,rtol=0)
    old = pd.DataFrame(account["holdings"]).set_index("code").sort_index()
    h = holdings.loc[holdings.date.eq(ANCHOR)].set_index("code").sort_index()
    if not h.index.equals(old.index):
        raise ValueError("Carry-in stock codes mismatch")
    np.testing.assert_allclose(h[old.columns],old,rtol=0,atol=.004)
    prior_nav = pd.read_parquet(ROOT/account["development_run"]/"nav.parquet").set_index("date")
    np.testing.assert_allclose(n.loc[prior_nav.index,"equity"],prior_nav.equity,rtol=0,atol=.004)
    n = n.loc[FIRST:]
    trades = orders.loc[orders.date.ge(FIRST)]
    np.testing.assert_allclose(trades.groupby("date").fee.sum().reindex(n.index,fill_value=0),n.fee,atol=.004,rtol=0)
    np.testing.assert_allclose(trades.gross.abs().groupby(trades.date).sum().reindex(n.index,fill_value=0),n.gross_traded,atol=.004,rtol=0)
    for row in trades.loc[trades.filled.ne(0)].itertuples():
        quote = quotes[row.date][row.code]
        if str(quote.tradestatus)!="1" or quote.raw_open!=row.price:
            raise ValueError("Fill on a suspension or invalid opening mark")
    np.testing.assert_allclose(n[["cash","stock_value","pending_value","dividends_due","rights_prepaid"]].sum(axis=1),n.equity,atol=.004,rtol=0)
    daily_paid = actions.loc[actions.stage.eq("cash_paid")].groupby("date").amount.sum().reindex(n.index,fill_value=0)
    signed = trades.groupby("date").gross.sum().reindex(n.index,fill_value=0)
    np.testing.assert_allclose(nav.set_index("date").cash.diff().reindex(n.index), daily_paid-signed-n.fee,rtol=0,atol=.004)
    reg = holdings.set_index(["date","code"])
    observed = actions.loc[actions.date.ge(FIRST)]
    for e in events.loc[events.ex_date.between(FIRST,LAST)].itertuples():
        key = (e.record_date,e.code)
        owned = int(reg.loc[key,["shares","locked_shares"]].sum()) if key in reg.index else 0
        amt = execution.money(Decimal(e.cash_per_share)*owned)
        sub = observed.loc[observed.code.eq(e.code)&observed.ex_date.eq(e.ex_date)]
        for stage,date in [("cash_receivable",e.ex_date),("cash_paid",e.pay_date)]:
            if pd.isna(date) or date>LAST:
                continue
            hit = sub.loc[sub.stage.eq(stage)]
            if len(hit)!=int(amt>0) or (amt and (hit.date.iat[0]!=date or Decimal(str(hit.amount.iat[0]))!=amt)):
                raise ValueError(f"Cash entitlement mismatch: {e.code} {date}")
        ratio = Decimal(e.bonus_shares_per_share)+Decimal(e.reserve_shares_per_share)
        if ratio and owned:
            award = int(ratio*owned)
            hit = sub.loc[sub.stage.eq("stock_entitlement")]
            if len(hit)!=1 or hit.shares.iat[0]!=award or hit.date.iat[0]!=e.ex_date:
                raise ValueError("Entitled shares mismatch")
            if award and e.stock_listing_date<=LAST:
                listed = sub.loc[sub.stage.eq("shares_listed")]
                if len(listed)!=1 or listed.date.iat[0]!=e.stock_listing_date or listed.shares.iat[0]!=award:
                    raise ValueError("Listing mismatch")
    if len(issues) or n.index[-1]!=LAST or n.stale_count.iat[-1] or len(n)!=208:
        raise ValueError("Unresolved exposure, stale terminal position or missing session")
    return {"model":account["model"],"fee_bps":account["fee_bps"],"sessions":len(n),
            "fills":int(trades.filled.ne(0).sum()),"action_entries":len(observed),"unresolved":len(issues),
            "terminal_stale":int(n.stale_count.iat[-1]),"carry_in_verified":True,"all_ledgers_verified":True}


def main():
    cfg = seal(ROOT)["configuration"]
    events,review = verified_events()
    old_panel = pd.read_parquet(ROOT/"research/v3/results/step2/execution_panel.parquet")
    panel = pd.concat([old_panel,pd.read_parquet(WORK/"execution.parquet")]).sort_values(["code","date"])
    days = trading_days(ROOT/"data/raw/v3/development/calendar.parquet").union(trading_days(ROOT/"data/raw/v4_1/final_2026/calendar.parquet"))
    days = days[(days>=engine.START)&(days<=LAST)]
    quotes = execution_objects(panel.loc[panel.date.isin(days)])
    f = pd.read_parquet(WORK/"step1/factor_panel.parquet")
    w = pd.read_parquet(WORK/"weights.parquet")
    prior = pd.read_parquet(OLD/"step0/benchmark_weights.parquet")
    base = engine.prepare_market(f,pd.concat([prior,w.loc[w.signal_date.ge(ANCHOR)]],ignore_index=True))
    states = pd.read_parquet(WORK/"step1/states.parquet").set_index("signal_date")
    labels = pd.read_parquet(WORK/"step1/labels.parquet")
    summary,series = [],{}
    for acc in cfg["accounts"]:
        model,fee = acc["model"],acc["fee_bps"]
        print("Replaying",model,fee,flush=True)
        if model not in series:
            series[model] = signal_series(acc,labels,cfg)
        signals,cal = series[model]
        history = pd.read_csv(ROOT/acc["development_run"]/"beta_rebalances.csv",parse_dates=["date","signal_date"])
        schedule = engine.portfolio_schedule(days,pd.DatetimeIndex(sorted(signals.signal_date.unique())),history.date.iat[0],5)
        matches = schedule.loc[:ANCHOR]
        if not np.array_equal(matches.index,history.date) or not np.array_equal(matches,history.signal_date):
            raise ValueError("Rebalance phase differs")
        details,stats = [],[]
        variant = engine.Variant(engine.FUSION,5,fee,acc["policy"],.96,.3,20,15,.02)
        planner = engine.make_planner(variant,schedule,signals,cal,base,states,quotes,days[0],details,stats)
        with patch.object(execution,"EXCHANGES",(*execution.EXCHANGES,SWAP)):
            rows = execution.simulate(model,list(days),quotes,{d:{"_":1} for d in schedule.index},events,review,
                                      engine.INITIAL,Decimal(fee)/10000,weight_targets=planner)
        tables = [pd.DataFrame(r,columns=cols) for r,cols in zip(rows,[NAV_COLUMNS,ORDER_COLUMNS,HOLDING_COLUMNS,ACTION_COLUMNS,EXPOSURE_COLUMNS],strict=True)]
        result = check(acc,*tables,events,quotes)
        out = WORK/model/f"{fee}bp"
        for name,table in zip(["nav","orders","holdings","actions","issues"],tables,strict=True):
            market.save(table.loc[table.date.ge(ANCHOR)],out/(name+".parquet"))
        targets = pd.DataFrame(details,columns=["date","signal_date","code","anchor","target","prior","beta"])
        market.save(targets.loc[targets.date.ge(FIRST)],out/"targets.parquet")
        summary.append(result)
        print("Ledger verified",model,fee,flush=True)
    if len(summary)!=4:
        raise ValueError("Missing ledger")
    market.save(pd.DataFrame(summary),DEST/"audit/ledger_verification.csv")


if __name__=="__main__":
    main()

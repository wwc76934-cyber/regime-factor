"""Reconcile the first post-holdout week and size indicative orders using Friday's information."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_final_ledgers as ledger
from normalize_corporate_actions import normalize_event
from collect_final_2026 import save_json
from prepare_paper_week import CACHE, OUT, LOCAL, BASE, ROOT, WORK, LAST, SIGNAL, KEY, market, verify, trading_days

E = ledger.engine
EX = ledger.execution
NEXT = pd.Timestamp("2026-11-23")
SHEETS = ["nav","orders","holdings","actions","issues"]
COLUMNS = [ledger.NAV_COLUMNS,ledger.ORDER_COLUMNS,ledger.HOLDING_COLUMNS,ledger.ACTION_COLUMNS,ledger.EXPOSURE_COLUMNS]


def event_book():
    historical,review = EX.verified_cash_event(*ledger.load_actions(ROOT,ROOT/"research/v3/results/step1"))
    prior = pd.read_parquet(ledger.DEST/"audit/normalized_actions.parquet")
    extra = pd.read_csv(ledger.DEST/"audit/verified_dividends.csv",dtype={"cash_per_share":str,"bonus_shares_per_share":str,
                        "reserve_shares_per_share":str},parse_dates=["record_date","pay_date","stock_listing_date","ex_date"])
    tail = []
    for f in sorted((CACHE/"actions").glob("*.parquet")):
        table = pd.read_parquet(f)
        if len(table):
            for d,g in table.groupby("dividOperateDate",dropna=False):
                if pd.to_datetime(g.dividPlanDate).gt(SIGNAL).any():
                    raise ValueError("Unpublished terms")
                tail.append(normalize_event(f.stem,pd.Timestamp(d),g))
    tail = pd.DataFrame(tail)
    events = pd.concat([historical.loc[historical.ex_date.lt(ledger.FIRST)],prior,extra[prior.columns],tail],ignore_index=True)
    if events.duplicated(["code","ex_date"]).any():
        raise ValueError("Duplicate event")
    known = set(zip(tail.code,tail.ex_date))
    for f in (CACHE/"raw").glob("*.parquet"):
        q = pd.read_parquet(f).sort_values("date")
        gap = pd.to_numeric(q.preclose,errors="coerce")-pd.to_numeric(q.close,errors="coerce").shift()
        if any((f.stem,d) not in known for d in q.date[gap.abs().gt(.005)&q.tradestatus.eq("1")]):
            raise ValueError("Unexplained price reference: "+f.stem)
    market.save(tail,OUT/"company_actions.csv")
    return events,review,tail


def scenario_quotes(panel,tail):
    # Uses only published corporate terms and Friday closing marks for indicative sizing.
    q = panel.loc[panel.date.eq(SIGNAL)].copy()
    q["date"] = NEXT
    q["raw_open"] = q.raw_close
    ex = tail.loc[tail.ex_date.eq(NEXT)]
    if set(ex.code)!={"sz.301308"} or Decimal(ex.cash_per_share.item())!=Decimal("2.18"):
        raise ValueError("Review the known ex-date terms before scenario pricing")
    mask = q.code.eq("sz.301308")
    # Issuer announcement 2026-129, published Nov 16: differential distribution.
    mark = EX.money(Decimal(str(q.loc[mask,"raw_close"].item()))-Decimal("2.1683782"))
    q.loc[mask,"raw_open"] = float(mark)
    q["preclose"] = q.raw_open
    q["raw_close"] = q.raw_open
    return q


def validate_week(account, tables, tail, quotes):
    nav, orders, held, actions, issues = tables
    ref = WORK/account["model"]/f"{account['fee_bps']}bp"
    for name,frame,keys in [("nav",nav,["date"]),("holdings",held,["date","code"]),("orders",orders,["date","code"])]:
        old = pd.read_parquet(ref/(name+".parquet")).sort_values(keys).reset_index(drop=True)
        overlap = frame.loc[frame.date.between(ledger.ANCHOR,LAST)].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(old,overlap,check_dtype=False,atol=.00001,rtol=0)
    n = nav.set_index("date").loc[LAST:SIGNAL]
    o = orders.loc[orders.date.gt(LAST)&orders.date.le(SIGNAL)]
    if len(n)!=6 or len(issues) or n.stale_count.iloc[-1] or o.date.nunique()!=1 or o.date.min()!=pd.Timestamp("2026-11-16"):
        raise ValueError("Ledger coverage, rebalance or stale mark failure")
    if any(quotes[r.date][r.code].tradestatus!="1" or quotes[r.date][r.code].raw_open!=r.price
           for r in o[o.filled.ne(0)].itertuples()):
        raise ValueError("Invalid observed fill")
    received = actions.loc[actions.stage.eq("cash_paid")].groupby("date").amount.sum()
    flow = received.reindex(n.index,fill_value=0)-o.groupby("date").gross.sum().reindex(n.index,fill_value=0)-n.fee
    np.testing.assert_allclose(n.cash.diff().iloc[1:],flow.iloc[1:],atol=.004,rtol=0)
    np.testing.assert_allclose(n[["cash","stock_value","pending_value","dividends_due","rights_prepaid"]].sum(axis=1),n.equity,atol=.004,rtol=0)
    for e in tail.itertuples():
        reg = held.loc[held.date.eq(e.record_date)&held.code.eq(e.code)]
        owned = int(reg.shares.sum()+reg.locked_shares.sum())
        amt = EX.money(Decimal(e.cash_per_share)*owned)
        actual = actions.loc[actions.code.eq(e.code)&actions.ex_date.eq(e.ex_date)&actions.stage.eq("cash_paid")]
        if len(actual)!=int(bool(amt)) or (amt and (Decimal(str(actual.amount.item()))!=amt or actual.date.item()!=e.pay_date)):
            raise ValueError("Dividend mismatch: "+e.code)


def build():
    verify()
    cfg = ledger.seal(ROOT)["configuration"]
    historic = pd.read_parquet(ROOT/"research/v3/results/step2/execution_panel.parquet")
    panel = pd.concat([historic,pd.read_parquet(WORK/"execution.parquet"),pd.read_parquet(LOCAL/"execution.parquet")],ignore_index=True)
    events,review,tail = event_book()
    panel = pd.concat([panel,scenario_quotes(panel,tail)],ignore_index=True)
    days = trading_days(ROOT/"data/raw/v3/development/calendar.parquet").union(trading_days(BASE/"calendar.parquet")).union(trading_days(CACHE/"calendar.parquet"))
    days = days[(days>=E.START)&(days<=NEXT)]
    if days[-2]!=SIGNAL or days[-1]!=NEXT:
        raise ValueError("Unexpected next execution")
    quotes = ledger.execution_objects(panel.loc[panel.date.isin(days)])
    factors = pd.read_parquet(LOCAL/"step1/factor_panel.parquet")
    states = pd.read_parquet(LOCAL/"step1/states.parquet").set_index("signal_date")
    labels = pd.read_parquet(LOCAL/"step1/labels.parquet")
    weights = pd.concat([pd.read_parquet(ledger.OLD/"step0/benchmark_weights.parquet"),
                         pd.read_parquet(WORK/"weights.parquet").loc[lambda x:x.signal_date.ge(ledger.ANCHOR)],
                         pd.read_parquet(LOCAL/"weights.parquet")])
    base = E.prepare_market(factors,weights)
    names = pd.read_parquet(CACHE/"members.parquet").set_index("code").code_name
    factor_names = json.loads((LOCAL/"step1/v3_1_selected_factors.json").read_text())["directions"]
    oriented = factors.loc[factors.signal_date.eq(SIGNAL)&factors.signal_bar_usable].set_index("code")
    percentiles = oriented[list(factor_names)].mul(pd.Series(factor_names),axis=1).rank(pct=True).add_prefix("pct_")
    outputs = {}
    summaries = []
    forecast_cache = {}
    def save(name,table):
        outputs.setdefault(name,[]).append(table)
    for a in cfg["accounts"]:
        model,fee = a["model"],a["fee_bps"]
        print("Continuing", model,fee,flush=True)
        if model not in forecast_cache:
            with patch.object(ledger,"WORK",LOCAL):
                forecast_cache[model] = ledger.signal_series(a,labels,cfg)
        scores,cal = forecast_cache[model]
        first = pd.read_csv(ROOT/a["development_run"]/"beta_rebalances.csv",nrows=1,parse_dates=["date"]).date.item()
        schedule = E.portfolio_schedule(days,pd.DatetimeIndex(sorted(scores.signal_date.unique())),first,5)
        if schedule.get(NEXT)!=SIGNAL:
            raise ValueError("Execution phase drift")
        detail,stats = [],[]
        variant = E.Variant(E.FUSION,5,fee,a["policy"],.96,.3,20,15,.02)
        plan = E.make_planner(variant,schedule,scores,cal,base,states,quotes,days[0],detail,stats)
        with patch.object(EX,"EXCHANGES",(*EX.EXCHANGES,ledger.SWAP)):
            rows = EX.simulate(model,list(days),quotes,{d:{"_":1} for d in schedule.index},events,review,
                               E.INITIAL,Decimal(fee)/10000,weight_targets=plan)
        tables = [pd.DataFrame(raw,columns=cols) for raw,cols in zip(rows,COLUMNS,strict=True)]
        validate_week(a,tables,tail,quotes)
        nav,orders,holdings,actions,_ = tables
        for name,table in zip(SHEETS[:4],tables[:4],strict=True):
            save("week_"+name,table.loc[table.date.gt(LAST)&table.date.le(SIGNAL)].assign(fee_bps=fee))
        n = nav.loc[nav.date.eq(SIGNAL)].iloc[0]
        h = holdings.loc[holdings.date.eq(SIGNAL)].set_index("code")
        d = pd.DataFrame(detail,columns=["execution_date","signal_date","code","anchor_weight","target_weight","prior_weight","beta"])
        target = d.loc[d.execution_date.eq(NEXT)].set_index("code")
        size = orders.loc[orders.date.eq(NEXT)].set_index("code")
        pred = scores.loc[scores.signal_date.eq(SIGNAL)].set_index("code").score_z
        all_codes = h.index.union(target.index).union(size.index)
        t = pd.DataFrame(index=all_codes).rename_axis("code")
        t["stock_name"] = names
        t["signal_date"],t["planned_execution"],t["model"],t["fee_bps"] = SIGNAL,NEXT,model,fee
        t["prior_shares"] = h.shares.reindex(t.index,fill_value=0)
        t["prior_locked_shares"] = h.locked_shares.reindex(t.index,fill_value=0)
        t["prior_close_weight"] = h.value.reindex(t.index,fill_value=0)/n.equity
        t["anchor_weight"] = base[SIGNAL].weight_float_mv.reindex(t.index,fill_value=0)
        t["continuous_target_weight"] = target.target_weight.reindex(t.index,fill_value=0)
        t["active_weight"] = t.continuous_target_weight-t.anchor_weight
        t["ema_score_z"] = pred
        t["ema_rank"] = pred.rank(ascending=False,method="min")
        t = t.join(percentiles)
        t["indicative_price"] = [quotes[NEXT][c].raw_open for c in t.index]
        t["indicative_change"] = size.filled.reindex(t.index,fill_value=0).astype(int)
        t["indicative_shares"] = t.prior_shares+t.indicative_change
        t["indicative_fee"] = size.fee.reindex(t.index,fill_value=0)
        t["order_side"] = np.select([t.indicative_change.gt(0),t.indicative_change.lt(0)],["买入","卖出"],default="持有")
        if not np.isclose(t.continuous_target_weight.sum(),1) or (t.indicative_shares<0).any():
            raise ValueError("Unbalanced target portfolio")
        ph = holdings.loc[holdings.date.eq(NEXT)].set_index("code")
        np.testing.assert_array_equal(t.indicative_shares,ph.shares.reindex(t.index,fill_value=0))
        decision = stats[-1]
        t["beta_policy"],t["market_regime"] = a["policy"],decision[2]
        def explain(row):
            observed = row[percentiles.columns].dropna().astype(float)
            if observed.empty:
                factor = "当周无可用的定向因子百分位"
            else:
                leader = observed.idxmax()
                factor = f"{leader[4:]} 定向百分位 {observed[leader]:.1%}"
            score = f"组合评分排名 {int(row.ema_rank)}" if pd.notna(row.ema_rank) else "无当周模型排名"
            return (f"{factor}，{score}。流通市值锚点 {row.anchor_weight:.2%}，"
                    f"优化目标 {row.continuous_target_weight:.2%}，原持仓 {row.prior_close_weight:.2%}。")
        t["position_context"] = t.apply(explain,axis=1)
        save("target_positions",t.reset_index().sort_values("continuous_target_weight",ascending=False))
        save("indicative_orders",t[t.indicative_change.ne(0)].reset_index().sort_values("indicative_change"))
        summaries.append(dict(model=model,fee_bps=fee,signal_date=SIGNAL,planned_execution=NEXT,regime=decision[2],
                              beta_low=decision[3],beta_high=decision[4],prior_equity=n.equity,prior_cash=n.cash,
                              prior_positions=len(h),continuous_names=int(t.continuous_target_weight.gt(0).sum()),
                              indicative_positions=int(t.indicative_shares.gt(0).sum()),
                              buys=int(t.indicative_change.gt(0).sum()),sells=int(t.indicative_change.lt(0).sum())))
    if len(summaries)!=4:
        raise ValueError("Expected four accounts")
    for name,blocks in outputs.items():
        market.save(pd.concat(blocks,ignore_index=True),OUT/(name+".csv"))
    market.save(pd.DataFrame(summaries),OUT/"account_summary.csv")
    gates = pd.read_csv(LOCAL/"audit/gate_weights.csv",parse_dates=["signal_date"])
    market.save(gates.loc[gates.signal_date.eq(SIGNAL)],OUT/"gate_weights.csv")
    verify()
    from hashlib import sha256
    save_json({"generated_utc":datetime.now(timezone.utc).isoformat(),"signal_as_of":str(SIGNAL.date()),
               "indicative_execution_date":str(NEXT.date()),"last_observed_quote":str(SIGNAL.date()),
               "holdout_seal_sha256":sha256((ledger.DEST/"holdout_seal.json").read_bytes()).hexdigest(),
               "input_receipt_sha256":sha256((CACHE/"complete.json").read_bytes()).hexdigest(),
               "csv_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))}},
              OUT/"paper_week_receipt.json")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=="__main__":
    build()

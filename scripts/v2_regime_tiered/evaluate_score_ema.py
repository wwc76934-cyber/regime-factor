"""Compare score EMA under fixed 10/10/10 slots and unchanged execution."""

import argparse
from decimal import Decimal
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

from tiered_data import TIERS, require


def static_selector(ranking):
    schedules = {}
    for day, week in ranking.groupby("execution_date"):
        tiers = {}
        for tier, stocks in week.groupby("vol_tier"):
            tiers[tier] = stocks.sort_values(["used_score","code"],ascending=[False,True])["code"].tolist()
        require(set(tiers)==set(TIERS) and all(len(names)>=15 for names in tiers.values()),"Insufficient tier coverage")
        schedules[day] = tiers
    def choose(day, original, positions):
        selected = []
        for tier in TIERS:
            names = schedules[day][tier]
            kept = [code for code in names[:15] if positions.get(code,0)>0][:10]
            slots = list(dict.fromkeys(kept+names[:10]))[:10]
            selected.extend(slots)
        require(len(set(selected))==30,"Expected thirty target positions")
        return {code:rank for rank,code in enumerate(selected,1)}
    schedule = {day:choose(day,None,{}) for day in schedules}
    return schedule,choose


def verify_actions(events, review):
    code, date = "sh.601985", pd.Timestamp("2024-12-02")
    exists = events["code"].eq(code) & events["ex_date"].eq(date)
    if exists.any():
        require(exists.sum()==1 and Decimal(events.loc[exists,"cash_per_share"].iat[0])==Decimal(".02"),
                "Conflicting verified China National Nuclear Power dividend")
    else:
        event = dict(code=code,ex_date=date,record_date=pd.Timestamp("2024-11-29"),pay_date=date,
                     stock_listing_date=pd.NaT,cash_per_share=".02",bonus_shares_per_share="0",
                     reserve_shares_per_share="0",event_type="cash_only")
        events = pd.concat([events,pd.DataFrame([event])],ignore_index=True)
    return events,review.loc[~(review["code"].eq(code)&review["ex_date"].eq(date))]


def evaluate(root, source, output):
    sys.path.insert(0,str(root/"scripts"))
    from run_holdings_buffer_backtest import simulate

    config = tomllib.loads((root/"configs/baseline.toml").read_text(encoding="utf-8-sig"))
    start,end = config["research"]["bars_start"],config["research"]["development_end"]
    initial = Decimal(str(config["portfolio"]["initial_cash"]))
    calendar = pd.read_parquet(root/f"data/raw/calendar/baostock_{start}_{end}.parquet")
    days = pd.DatetimeIndex(calendar.loc[calendar["is_trading_day"].eq("1"),"calendar_date"]).sort_values()
    scored = pd.read_parquet(source/"tiered_weekly_scores.parquet")
    require(scored["execution_date"].dt.year.le(2024).all(),"Stage 2 evaluation is development-only")
    days = days[(days>=scored.execution_date.min()) & (days<=pd.Timestamp(end))]
    panel = pd.read_parquet(root/"data/processed/csi300_daily_panel.parquet",columns=[
        "date","code","raw_open","raw_close","preclose","tradestatus","isST"])
    bars = {date:{row.code:row for row in group.itertuples(index=False)} for date,group in
            panel.loc[panel["date"].isin(days) & panel["code"].isin(scored["code"])].groupby("date")}
    events = pd.read_parquet(root/"data/processed/corporate_actions/csi300_backtest_actions.parquet")
    review = pd.read_parquet(root/"reports/portfolio/corporate_actions_remaining_review.parquet")
    events,review=verify_actions(events,review)
    nav_columns="strategy date cash stock pending receivables prepaid equity fee gross opening stale".split()
    first=pd.Timestamp("2021-01-18")
    anchor=days[days<first][-1]
    results, ledgers, issues = [],[],[]
    for name,week in scored.groupby("strategy"):
        for field,gamma in [("score",0.),("score_ema05",.5)]:
            ranking=week.copy()
            warm=ranking["execution_date"].lt(first)
            ranking["used_score"]=ranking[field].where(~warm,ranking["score"])
            schedule,selector=static_selector(ranking)
            for fee in [10,20]:
                nav,_,_,_,unresolved=simulate(name,days,bars,schedule,events,review,initial,
                    fee_rate=Decimal(fee)/Decimal(10000),select_targets=selector)
                table=pd.DataFrame(nav,columns=nav_columns).set_index("date").loc[anchor:].copy()
                table["nav"]=table["equity"]/table.at[anchor,"equity"]
                trades=table.loc[table.index>=first]
                turnover=(trades["gross"]/trades["opening"]).dropna()
                require(len(turnover)==202 and len(table)==960,"Changed evaluation window")
                summary={"strategy":name,"gamma":gamma,"fee_bps":fee,"rebalance_weeks":len(turnover),
                    "total_return":table["nav"].iat[-1]-1,"two_way_turnover":turnover.mean(),
                    "fees_paid":trades["fee"].sum(),"max_drawdown":-(table["nav"]/table["nav"].cummax()-1).min(),
                    "unresolved_exposures":len(unresolved),"stale_mark_days":int(trades["stale"].gt(0).sum())}
                results.append(summary)
                ledgers.append(table.reset_index().assign(gamma=gamma,fee_bps=fee))
                issues.extend((*item,gamma,fee) for item in unresolved)
        print("Evaluated:",name,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(issues,columns=["strategy","date","code","opening_shares","pending_shares","category","reason",
                               "gamma","fee_bps"]).to_csv(output/"ema_unresolved_exposures.csv",index=False,encoding="utf-8-sig")
    summary=pd.DataFrame(results)
    summary.to_csv(output/"ema_static_10_10_10_summary.csv",index=False,encoding="utf-8-sig")
    pd.concat(ledgers,ignore_index=True).to_parquet(output/"ema_static_daily_nav.parquet",index=False)
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path.cwd())
    parser.add_argument("--scores-dir",type=Path)
    opts=parser.parse_args()
    root=opts.project_root.resolve()
    source=opts.scores_dir or root/"research/v2_regime_tiered/results/stage2"
    evaluate(root,source,source)

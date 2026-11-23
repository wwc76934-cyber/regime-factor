"""Reconcile the ten accounts, link daily attribution, and publish the development review."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/v3_1"))
import report_v3_1_development as attribution
from prepare_v3_1_base import initialize
from v3_1_execution import EXCHANGES


def check_account(folder,reference=None):
    nav=pd.read_parquet(folder/"nav.parquet").set_index("date").sort_index()
    held=pd.read_parquet(folder/"holdings.parquet")
    actions=pd.read_parquet(folder/"actions.parquet")
    order=pd.read_parquet(folder/"orders.parquet")
    issues=pd.read_csv(folder/"unresolved_exposure.csv")
    parts=["cash","stock_value","pending_value","dividends_due","rights_prepaid"]
    residual=(nav[parts].sum(axis=1)-nav.equity).abs().max()
    trades=order.assign(gross=order.gross.abs()).groupby("date")[["fee","gross"]].sum().reindex(nav.index,fill_value=0)
    fee_error=(trades.fee-nav.fee).abs().max()
    gross_error=(trades.gross-nav.gross_traded).abs().max()
    stale=len(held.loc[held.date.eq(nav.index.max())&held.stale])
    base_error=None
    if reference is not None:
        base=pd.read_parquet(reference/"nav.parquet").set_index("date").sort_index()
        pd.testing.assert_index_equal(nav.index,base.index)
        base_error=(nav.equity-base.equity).abs().max()
    def shares(code,date,locked=False):
        x=held[held.code.eq(code)&held.date.eq(pd.Timestamp(date))]
        return int(x.shares.sum()+(x.locked_shares.sum() if locked else 0))
    def stage(code,date,ex,name,col):
        f=actions[actions.code.eq(code)&actions.date.eq(pd.Timestamp(date))&
                  actions.ex_date.eq(pd.Timestamp(ex))&actions.stage.eq(name)]
        return f[col].sum()
    moutai=shares("sh.600519","2022-12-26",True)*21.91
    terms=[abs(stage("sh.600519","2022-12-27","2022-12-27",s,"amount")-moutai)<.005
           for s in ["cash_receivable","cash_paid"]]
    events={"moutai_dividend":round(moutai,2)}
    for e in EXCHANGES:
        owned=shares(e.old,e.record)
        qty=float(e.ratio*owned)
        events[e.old+"_shares"]=owned
        for code,date,name,amount in [(e.old,e.record,"swap_entitlement",owned),(e.old,e.retire,"swap_old_retired",owned),
                                      (e.new,e.confirmation,"swap_confirmed",qty),(e.new,e.list_date,"swap_new_listed",qty)]:
            terms.append(stage(code,date,e.retire,name,"shares")==amount)
        terms.append(shares(e.old,e.retire)==0)
    verified=bool(nav.index.is_unique and nav.index.max()==pd.Timestamp("2025-12-31") and
                  np.isfinite(nav.equity).all() and max(residual,gross_error,fee_error)<.005 and
                  not len(issues) and not stale and all(terms) and (base_error is None or base_error<.005))
    return dict(ledger_verified=verified,unresolved=len(issues),terminal_stale=stale,
                max_nav_difference=residual,max_fee_difference=fee_error,max_gross_difference=gross_error,
                baseline_max_nav_difference=base_error,verified_event_terms=all(terms),**events)


def mdtable(f,cols,formats):
    lines=["| "+" | ".join(cols.values())+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for _,row in f.iterrows():
        lines.append("| "+" | ".join(format(row[k],formats.get(k,"")) for k in cols)+" |")
    return lines


def report(root):
    study=root/"research/v3_1/results"
    dest=root/"research/v4_1"
    out=dest/"results/step3"
    records=pd.read_csv(out/"accounts.csv")
    _,_,_,schedule,_=initialize(root)
    risk,proxy=attribution.held_context(study,schedule)
    bench=pd.read_parquet(study/"step0/benchmark_nav.parquet").set_index("date").rename(
        columns={"h00300":"H00300","float_mv_nav":"FloatMV","equal_weight_nav":"CSI300_EW"})[["H00300","FloatMV","CSI300_EW"]]
    ledgers,annual,summary=[],[],[]
    for r in records.itertuples():
        block=out/r.block
        path=block/"runs"/r.engine_strategy
        key=dict(model=r.model,policy=r.policy,fee_bps=r.fee_bps)
        old=attribution.frontier_run(study/"step4/nonnegative_portfolios",5,r.fee_bps) if r.model=="v3_1_champion" else None
        ledgers.append({**key,**check_account(path,old)})
        data=attribution.make_daily(path,bench,risk,proxy)
        annual.extend({**key,"year":int(year),**attribution.linked(g)} for year,g in data.groupby(data.index.year))
        full=attribution.linked(data)
        metric=pd.read_csv(block/"model_portfolio_comparison_v3_1.csv").set_index("strategy").loc[r.engine_strategy]
        tests={"h00300_excess":"H00300_excess","float_mv_excess":"FloatMV_excess",
               "equal_weight_excess":"CSI300_EW_excess","h00300_ir":"H00300_ir"}
        if not all(np.isclose(full[k],metric[v],atol=1e-10) for k,v in tests.items()):
            raise ValueError("Attribution does not reproduce account summary")
        beta=pd.read_csv(path/"beta_rebalances.csv")
        summary.append({**key,**full,"weekly_turnover":metric.weekly_equivalent_two_way_turnover,
                        "offensive_policy_hit":beta.loc[beta.regime.eq("offensive"),"inside_band"].mean(),
                        "offensive_reference_hit":metric.offensive_beta_hit})
    accounts=pd.DataFrame(summary)
    audits=pd.DataFrame(ledgers)
    years=pd.DataFrame(annual)
    audits.to_csv(out/"ledger_audit.csv",index=False)
    years.to_csv(out/"annual_attribution.csv",index=False)
    accounts.to_csv(out/"portfolio_comparison.csv",index=False)
    verified=bool(len(audits)==10 and audits.ledger_verified.all())
    signal=pd.read_csv(dest/"results/step1/v4_1_signal_audit.csv")
    signal=signal[signal.universe.eq("paired")]
    lines=["# V4.1 三专家门控开发报告","","## 研究设计","",
           "三路专家各取两项已定向因子的等权平均。先在当期全体 300 只成分股上按均值正负三个标准差缩尾、标准化，缺失填零，再对组分数标准化。"
           "五维状态使用截至当期的 12 周序列，标准化统计量只使用该次扩展训练窗的成熟信号周。759 个参数输出三个非负、和为一的风格系数。"
           "专家分数可以为负，非负路由约束的是风格系数。","",
           "39 次重训，每次 5 个种子、全批量 25 轮。周度可微 Pearson IC 的均值除以其总体标准差加 0.05，另含相邻周权重平滑及等权先验惩罚。"
           "六因子名单来自已使用 2015–2022 数据选因子的 V3.1，2017–2025 的成绩均属开发证据。","",
           "## 信号与门控","",
           *mdtable(signal,{"strategy":"信号","period":"区间","rank_ic":"Rank IC","icir":"ICIR","top30_hit":"Top 30 命中",
                            "top30_excess_H00300":"五日费前超额"},{"rank_ic":".4f","icir":".3f","top30_hit":".2%","top30_excess_H00300":"+.3%"}),
           "","![三专家周度权重](results/step1/gate_weights_history.png)","",
           "## 固定组合对照","",
           "按方案列出的五种组合 × 两档实际费率建立十本账。共同股票池沿用 V3.1，单模型分数在周截面标准化后融合，"
           "再执行 EMA 0.3、h5、20 bp 优化器换手惩罚、风险厌恶 15、主动限额 2%、资金缓冲 0.96。"
           "超额为同期累计收益差，IR 基于日度主动收益年化。进攻期命中率按各自执行政策的 Beta 区间计算。",""]
    cols={"model":"模型","policy":"Beta","fee_bps":"费率 bp","portfolio_return":"净收益","h00300_excess":"超 H00300",
          "float_mv_excess":"超 FloatMV","equal_weight_excess":"超 CSI300_EW","h00300_ir":"IR","weekly_turnover":"双边周换手","offensive_policy_hit":"进攻期命中"}
    formats={k:"+.2%" for k in ["portfolio_return","h00300_excess","float_mv_excess","equal_weight_excess"]}
    formats.update(h00300_ir=".3f",weekly_turnover=".2%",offensive_policy_hit=".2%")
    lines+=mdtable(accounts,cols,formats)
    lines+=["","## 2024 与 2025 年归因","",
            "表中四项逐日贡献已链接至年度相对 H00300 收益差，成本为负贡献。选股残差也包含未建模暴露。下表采用 10 bp；全部费率和九个年度见明细。",""]
    cols={"year":"年","model":"模型","policy":"Beta","h00300_excess":"净超额","beta_timing":"Beta 择时","float_mv_basis":"FloatMV 基准差",
          "stock_selection_residual":"选股残差","cost":"成本","portfolio_max_drawdown":"组合最大回撤"}
    lines+=mdtable(years[years.year.isin([2024,2025])&years.fee_bps.eq(10)].sort_values(["year","model","policy"]),cols,
                   {k:"+.2%" for k in list(cols)[3:]})
    best=accounts[(accounts.model!="v3_1_champion")&accounts.fee_bps.eq(10)].sort_values("h00300_ir",ascending=False).iloc[0]
    annual10=years[years.fee_bps.eq(10)].set_index(["model","policy","year"])
    baseline24=annual10.loc[("v3_1_champion","dynamic",2024)]
    static24=annual10.loc[("v4_1_moe_huber","static",2024)]
    yr25=annual10.xs(2025,level="year").h00300_excess
    lines+=["","## 开发封板与审阅","",
            f"10 bp 下，2024 年传统冠军相对 H00300 为 {baseline24.h00300_excess:+.2%}，静态融合为 {static24.h00300_excess:+.2%}。"
            f"其中选股残差分别贡献 {baseline24.stock_selection_residual*100:+.2f} 和 {static24.stock_selection_residual*100:+.2f} 个百分点。"
            f"2025 年静态融合、动态融合、传统冠军的净超额依次为 {yr25.at[('v4_1_moe_huber','static')]:+.2%}、"
            f"{yr25.at[('v4_1_moe_huber','dynamic')]:+.2%}、{yr25.at[('v3_1_champion','dynamic')]:+.2%}。","",
            f"按 10 bp 全期 H00300 IR 描述性排序，V4.1 首位是 {best.model} / {best.policy}，IR 为 {best.h00300_ir:.4f}。"
            f"十本账的综合核验 all_ledgers_verified = {str(verified).lower()}。"
            "逐日核对资产组成、成交总额、手续费、期末陈旧仓位、未核验敞口、登记日茅台分红及武钢、海通换股权益，另逐日对照两档 V3.1 历史净值。","",
            "可选的固定 20 轮无 CSA 时序模型未纳入本次实验。根据 V4 已有无注意力首窗的 3 种子用时，按固定 20 轮和后续窗口大小估算超出 5 分钟快速排查预算。"
            "三专家模型与四套 V4.1 执行方案已完成开发期对照。保留期为 2026-01-01 至 2026-11-13，双线冻结与一次性评估等待审阅。","",
            "[全期三基准](results/step3/portfolio_comparison.csv) · [逐年归因](results/step3/annual_attribution.csv) · [账本核验](results/step3/ledger_audit.csv) · [信号明细](results/step1/v4_1_signal_audit.csv)","",
            "## 本地复跑","",
            "在项目根目录使用现有 CPU 虚拟环境依次运行。","",
            "```powershell",*[f".\\.venv\\Scripts\\python.exe -B scripts/v4_1/{name}.py" for name in
                           ["train_v4_1_regime_moe","verify_v4_1_signals","run_v4_1_backtest","report_v4_1"]],"```"]
    (dest/"DEVELOPMENT_REPORT_V4_1.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status=dict(stage="development_complete_pending_review" if verified else "ledger_review_required",accounts=len(accounts),
                all_ledgers_verified=verified,highest_development_ir_v41=dict(model=best.model,policy=best.policy,ir=best.h00300_ir),
                holdout_opened=False)
    (out/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status))
    if not verified:
        raise ValueError("Unverified ledger; see ledger_audit.csv")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    report(p.parse_args().project_root.resolve())

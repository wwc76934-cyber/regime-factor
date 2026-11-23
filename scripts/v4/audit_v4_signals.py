"""Evaluate sealed weekly forecasts, without changing their sign, weights or admission rules."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
KEY=["signal_date","code"]
FULL="v4_deep_tcsa_mgg"
RIDGE="v3_1_state_ridge"
TREE="v3_1_lgb_huber"
AB=["without_attention","without_gating"]
PERIODS={"all":(2017,2025),"2017-2022":(2017,2022),"2023-2025":(2023,2025)}


def summarize(weekly, periods):
    records=[]
    for (name,universe),g in weekly.groupby(["strategy","universe"]):
        for period,(a,b) in periods.items():
            w=g[g.signal_date.dt.year.between(a,b)]
            if w.empty:
                continue
            ic=w.rank_ic.mean()
            std=w.rank_ic.std()
            records.append(dict(strategy=name,universe=universe,period=period,weeks=len(w),
                          rank_ic=ic,icir=ic/std if std>0 else np.nan,top30_hit=w.top30_hit.mean(),
                          top30_excess_H00300=w.top30_excess_H00300.mean(),mean_labeled=w.labeled.mean()))
    return pd.DataFrame(records)


def plot_signal_history(weekly,corr,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout="constrained")
    series=weekly.loc[weekly.strategy.eq(FULL)&weekly.universe.eq("v4_eligible")].set_index("signal_date").rank_ic.sort_index()
    axes[0].plot(series.rolling(26).mean(),color="#236D81",label="Deep model")
    axes[0].axhline(.04,color="#6B7280",ls="--",label="Full-period reference (0.04)")
    axes[0].set_ylabel("Rank IC")
    for name,label,color in [(RIDGE,"Nonnegative Ridge","#335F92"),(TREE,"Huber LightGBM","#BC612B")]:
        s=corr.loc[corr.model_a.eq(FULL)&corr.model_b.eq(name)].set_index("signal_date").rho.sort_index()
        axes[1].plot(s.rolling(26).mean(),color=color,label=label)
    axes[1].axhline(.65,color="#6B7280",ls="--",label="Full-period reference (0.65)")
    axes[1].set_ylabel("Score rank correlation")
    axes[0].set_title("Development signal history | 26-week trailing means",loc="left")
    for ax in axes:
        ax.axhline(0,color="#999999",lw=.6)
        ax.legend(frameon=False,ncol=2,fontsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(out/"signal_history.png",dpi=150)
    plt.close(fig)


def score_weeks(p,label,universe):
    merged=p.merge(label,on=KEY,validate="many_to_one",how="left",indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Missing stock label key")
    rows=[]
    for (name,t),w in merged.groupby(["strategy","signal_date"]):
        good=w.dropna(subset=["risk_adjusted_target"])
        if len(good)<50 or w.exit_date.isna().all():
            continue
        top=w.sort_values(["score","code"],ascending=[False,True]).head(30)
        if top.stock_return.count()<24:
            continue
        best=good.sort_values(["risk_adjusted_target","code"],ascending=[False,True]).head(30)
        ic=good.score.rank().corr(good.risk_adjusted_target.rank())
        rows.append(dict(strategy=name,universe=universe,signal_date=t,rank_ic=ic,labeled=len(good),
                         top30_hit=len(set(top.code)&set(best.code))/30,
                         top30_excess_H00300=top.stock_return.mean()-w.index_return.dropna().iloc[0]))
    return pd.DataFrame(rows)


def gate_tables(path,states,out,expected_gates):
    gates=pd.read_csv(path,parse_dates=["signal_date","refit"])
    cols=[c for c in gates if c.startswith("gate_")]
    actual=pd.MultiIndex.from_frame(gates[["signal_date","refit","seed"]])
    if not actual.is_unique or len(actual)!=len(expected_gates) or not expected_gates.isin(actual).all():
        raise ValueError("Incomplete gate weeks, refits or seeds")
    if not ((gates[cols]>0)&(gates[cols]<2)).all().all():
        raise ValueError("Invalid gating activation")
    gates=gates.merge(states,on="signal_date",validate="many_to_one",how="left")
    if gates.drop(columns=cols).isna().any().any():
        raise ValueError("Missing state in gating audit")
    trend=gates.market_return_20d
    breadth=gates.market_breadth_20d
    gates["market_regime"]=np.select([(trend>0)&(breadth>=.50),(trend<=0)&(breadth<.45)],
                                      ["offensive","defensive"],default="neutral")
    gates["style_regime"]=np.where(gates.value_minus_trend_60d>=0,"value","trend")
    # Channel numbers are local to each independently trained seed and refit.
    # Also report allocation amplitude, which is invariant to permutations of channels.
    gates["mean_scale"]=gates[cols].mean(axis=1)
    gates["mean_abs_shift_from_one"]=(gates[cols]-1).abs().mean(axis=1)
    groups=[]
    summary=[]
    for col in ["market_regime","style_regime"]:
        for (fit,seed,reg),sample in gates.groupby(["refit","seed",col]):
            groups.append(dict(refit=fit,seed=seed,axis=col,regime=reg,weeks=len(sample),
                               **sample[cols].mean().to_dict()))
        for reg,sample in gates.groupby(col):
            summary.append(dict(axis=col,regime=reg,weeks=sample.signal_date.nunique(),
                                mean_scale=sample.mean_scale.mean(),mean_abs_shift=sample.mean_abs_shift_from_one.mean()))
    by_fit=pd.DataFrame(groups)
    by_fit.to_csv(out/"gate_means_by_refit_seed_regime.csv",index=False)
    differences=[]
    for axis,one,two in [("market_regime","offensive","defensive"),("style_regime","value","trend")]:
        panel=by_fit.loc[by_fit.axis.eq(axis)].set_index(["refit","seed","regime"])[cols]
        # Subtract only the same learned channels from the same fitted network.
        delta=(panel.xs(one,level="regime")-panel.xs(two,level="regime")).dropna()
        differences.extend(dict(refit=fit,seed=seed,contrast=f"{one}_minus_{two}",
                                mean_signed_channel_change=row.mean(),mean_absolute_channel_change=row.abs().mean(),
                                **row.to_dict()) for (fit,seed),row in delta.iterrows())
    pd.DataFrame(differences).to_csv(out/"gate_within_refit_contrasts.csv",index=False)
    frame=pd.DataFrame(summary)
    frame.to_csv(out/"gate_regime_summary.csv",index=False)
    return frame


def audit(root):
    source=root/"research/v3_1/results"
    out=root/"research/v4/results/step2"
    p=pd.read_parquet(out/"predictions.parquet")
    logs=pd.read_csv(out/"refit_audit.csv",parse_dates=["refit","last_train_exit","first_validation_signal","last_validation_exit"])
    if (logs.duplicated(["strategy","seed","refit"]).any() or not logs.last_train_exit.lt(logs.first_validation_signal).all()
        or not logs.last_validation_exit.le(logs.refit).all() or not logs.best_epoch.le(logs.epochs).all()):
        raise ValueError("Invalid training or validation timeline")
    spec=json.loads((out/"training_spec.json").read_text(encoding="utf-8"))
    expected=json.loads((root/"research/v4/results/step1/dataset_and_benchmark.json").read_text(encoding="utf-8"))
    full=p.loc[p.strategy.eq(FULL)].copy()
    forecasts=sorted(full.signal_date.unique())
    if len(forecasts)!=expected["forecast_weeks"] or max(forecasts)>pd.Timestamp("2025-12-31"):
        raise ValueError("Incomplete full development coverage")
    refits=forecasts[::12]
    fit_calendar=pd.Series([refits[i//12] for i in range(len(forecasts))],index=forecasts)
    if not p.signal_date.map(fit_calendar).eq(p.refit).all():
        raise ValueError("Predictions outside the registered refit block")
    expected_refits=dict.fromkeys(AB,refits[:1])|{FULL:refits}
    for model,fit_days in expected_refits.items():
        observed=logs[logs.strategy.eq(model)]
        if set(observed.refit)!=set(fit_days) or set(observed.seed)!=set(spec["seeds"]) or len(observed)!=len(fit_days)*len(spec["seeds"]):
            raise ValueError("Incomplete refit/seed matrix")
    legacy=pd.read_parquet(source/"step4/nonnegative_forecasts/h5/predictions.parquet",
                           filters=[("strategy","in",[RIDGE,TREE])])
    labels=pd.read_parquet(source/"step1/labels.parquet",filters=[("horizon","==",5)])
    if labels.exit_date.dropna().max()>pd.Timestamp("2025-12-31"):
        raise ValueError("Invalid target maturity")
    trad=legacy.pivot(index=KEY,columns="strategy",values="score")
    wide=trad.join(full.set_index(KEY)[["score"]].rename(columns={"score":FULL}),how="inner").dropna()
    if wide.index.get_level_values("signal_date").nunique()!=len(forecasts):
        raise ValueError("Unpaired forecast dates")
    correlations=[]
    for t,w in wide.groupby(level="signal_date"):
        c=w.rank().corr()
        for a,b in [(FULL,RIDGE),(FULL,TREE),(RIDGE,TREE)]:
            correlations.append(dict(signal_date=t,model_a=a,model_b=b,n=len(w),rho=c.at[a,b]))
    corr=pd.DataFrame(correlations)
    if not np.isfinite(corr.rho).all():
        raise ValueError("Invalid rank correlation")
    corr.to_csv(out/"weekly_rank_correlations.csv",index=False)
    reports=[]
    for period,(a,b) in PERIODS.items():
        g=corr[corr.signal_date.dt.year.between(a,b)].copy()
        reports.append(g.assign(abs_rho=g.rho.abs()).groupby(["model_a","model_b"]).agg(weeks=("rho","size"),
                       mean_spearman=("rho","mean"),mean_abs_spearman=("abs_rho","mean"),mean_stocks=("n","mean"))
                       .reset_index().assign(period=period))
    orth=pd.concat(reports,ignore_index=True)
    orth.to_csv(out/"orthogonality.csv",index=False)
    models=[FULL,RIDGE,TREE]
    matrix=pd.DataFrame(np.eye(len(models)),index=models,columns=models)
    for pair in orth[orth.period.eq("all")].itertuples():
        matrix.at[pair.model_a,pair.model_b]=pair.mean_spearman
        matrix.at[pair.model_b,pair.model_a]=pair.mean_spearman
    matrix.to_csv(out/"orthogonality_matrix.csv",index_label="model")
    data=wide.rename_axis(columns="strategy").stack().rename("score").reset_index()
    weekly=pd.concat([score_weeks(full,labels,"v4_eligible"),score_weeks(data,labels,"paired")],ignore_index=True)
    summary=summarize(weekly,PERIODS)
    weekly.to_csv(out/"weekly_signal_audit.csv",index=False)
    summary.to_csv(out/"v4_signal_audit.csv",index=False)
    plot_signal_history(weekly,corr,out)
    # The prespecified static block covers the first twelve prediction dates only.
    first=p[p.signal_date.isin(forecasts[:12])]
    template=first.loc[first.strategy.eq(FULL),KEY].sort_values(KEY).reset_index(drop=True)
    for model in AB:
        pd.testing.assert_frame_equal(template,first.loc[first.strategy.eq(model),KEY].sort_values(KEY).reset_index(drop=True))
    ab=summarize(score_weeks(first,labels,"v4_eligible"),{"first_refit":(2017,2017)})
    ab.to_csv(out/"static_ablation.csv",index=False)
    expected_gates=pd.MultiIndex.from_tuples([(t,fit_calendar.at[t],seed) for t in forecasts for seed in spec["seeds"]])
    gates=gate_tables(out/"weekly_gates_by_seed.csv",pd.read_parquet(source/"step1/states.parquet"),out,expected_gates)
    m=summary.query("strategy==@FULL and universe=='v4_eligible' and period=='all'").iloc[0]
    cross=orth.query("model_a==@FULL and period=='all'").set_index("model_b")
    checks={"rank_ic_gte_0.04":bool(m.rank_ic>=.04),"top30_hit_gte_12pct":bool(m.top30_hit>=.12),
            "ridge_mean_rho_lt_0.65":bool(cross.loc[RIDGE,"mean_spearman"]<.65),
            "huber_mean_rho_lt_0.65":bool(cross.loc[TREE,"mean_spearman"]<.65)}
    result=dict(status="go" if all(checks.values()) else "no_go",checks=checks,development_end="2025-12-31",
                forecast_weeks=len(forecasts),validated_signal_weeks=int(m.weeks),full_refits=len(refits))
    (out/"decision.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    write_review(out,result,summary,cross,ab,gates)
    print(json.dumps(result))
    print(summary.to_string(index=False),flush=True)


def write_review(out,result,summary,cross,ab,gates):
    lines=["# V4 独立信号审计","",f"第二阶段判定为 **{result['status'].upper()}**。全期覆盖 {result['forecast_weeks']} 个预测周、"
           f"{result['full_refits']} 次重训，每次采用三个随机种子。下表按当周可评分的点时股票池计算。","",
           "| 区间 | 有效周数 | Rank IC | 周度 ICIR | Top 30 命中率 | Top 30 相对 H00300 平均五日收益差 |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in summary[summary.universe.eq("v4_eligible")].itertuples():
        lines.append(f"| {r.period} | {r.weeks} | {r.rank_ic:.4f} | {r.icir:.3f} | {r.top30_hit:.2%} | {r.top30_excess_H00300*10000:+.1f} bp |")
    criteria=[("全期 Rank IC ≥ 0.04","rank_ic_gte_0.04"),("Top 30 命中率 ≥ 12%","top30_hit_gte_12pct"),
              ("与非负 Ridge 的平均秩相关 < 0.65","ridge_mean_rho_lt_0.65"),
              ("与原 Huber 的平均秩相关 < 0.65","huber_mean_rho_lt_0.65")]
    lines.extend(["","| 准入标准 | 结果 |","|---|---|"])
    for caption,check in criteria:
        lines.append(f"| {caption} | {'通过' if result['checks'][check] else '未通过'} |")
    if result["status"]=="no_go":
        lines.extend(["","开发期独立信号门槛未全部通过，项目停在第二阶段。2026 留出期继续封存。"])
    lines.extend(["","命中率取预测前 30 名与风险调整收益前 30 名的交集占比。收益差按等权持股计算，尚未扣除费用。"
                  "周度 ICIR 取 Rank IC 均值除以样本标准差。","",
                  "每周在三个模型均有分数的股票上计算截面秩相关，再对各周等权平均。"
                  f"深度模型与非负 Ridge 的相关系数为 {cross.at[RIDGE,'mean_spearman']:.4f}，"
                  f"与原 Huber 的相关系数为 {cross.at[TREE,'mean_spearman']:.4f}。"
                  "[分段相关性](orthogonality.csv)和[共同股票池结果](v4_signal_audit.csv)可供复核。","",
                  "![26 周滚动 IC 与截面相关性](signal_history.png)","",
                  "## 首个重训窗口的静态消融","",
                  "| 模型 | 周数 | Rank IC | Top 30 命中率 |","|---|---:|---:|---:|"])
    for r in ab.itertuples():
        lines.append(f"| {r.strategy} | {r.weeks} | {r.rank_ic:.4f} | {r.top30_hit:.2%} |")
    block=ab.set_index("strategy")
    for model,label in [("without_attention","跨股注意力"),("without_gating","市场门控")]:
        metrics=["rank_ic","top30_hit"]
        delta=block.loc[FULL,metrics]-block.loc[model,metrics]
        lines.extend(["",f"首季度保留{label}对应的 Rank IC 差为 {delta.rank_ic:+.4f}，"
                         f"Top 30 命中率差为 {100*delta.top30_hit:+.2f} 个百分点。"])
    lines.extend(["","三种结构分别重训，使用相同的训练窗、验证窗、种子与预测股票池。去注意力版本整体移除跨股注意力及其残差归一化块。去门控版本把通道乘数固定为 1。这一组消融仅覆盖首个季度。","",
                  "## 市场状态与门控","",
                  "| 状态维度 | 状态 | 信号周 | 均值 | 与 1 的平均绝对距离 |","|---|---|---:|---:|---:|"])
    for r in gates.itertuples():
        lines.append(f"| {r.axis} | {r.regime} | {r.weeks} | {r.mean_scale:.4f} | {r.mean_abs_shift:.4f} |")
    lines.extend(["","市场状态沿用 V3.1 的进攻、中性、防御定义；价值与趋势按 60 日风格收益差的符号区分。"
                  "独立重训可以重新排列隐通道。逐通道解释应在同一次重训、同一个种子内比较。"
                  "[逐通道分状态记录](gate_means_by_refit_seed_regime.csv) · [同模型内状态差](gate_within_refit_contrasts.csv) · [逐项门槛](decision.json)"])
    (out/"REVIEW.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    audit(p.parse_args().project_root.resolve())

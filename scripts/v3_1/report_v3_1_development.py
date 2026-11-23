"""Produce annual and 2024 linked portfolio attribution for the V3.1 frontier."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v3_1_base import initialize


FRONTIER = [(h,fee) for h in (5,10) for fee in (10,20)]
COMPONENTS = ["beta_timing","float_mv_basis","stock_selection_residual","cost"]
EPSILON = 1e-10


def frontier_name(h, fee):
    return f"fusion_50_50_h{h}_dynamic_eta0.96_ema0.3_{fee}bp_tc20_risk15_active0.02"


def frontier_run(root, h, fee):
    return root/"optimizer_grid"/f"h{h}_{fee}bp"/"runs"/frontier_name(h,fee)


def held_context(study, calendar):
    factors = pd.read_parquet(study/"step1/factor_panel.parquet",
        columns=["signal_date","code","BETA_120","EP_TTM","SP_TTM"])
    factors[["EP_TTM","SP_TTM"]] = (factors.groupby("signal_date")[["EP_TTM","SP_TTM"]].rank(pct=True)-.5).fillna(0)
    pivot = factors.pivot(index="signal_date",columns="code",values="BETA_120").ffill()
    risk = pivot.stack().rename("BETA_120").reset_index().merge(factors.drop(columns="BETA_120"),
         on=["signal_date","code"],how="left",validate="one_to_one")
    anchor = pd.read_parquet(study/"step0/benchmark_weights.parquet",columns=["signal_date","code","weight_float_mv"])
    anchor = anchor.merge(factors,on=["signal_date","code"],how="left",validate="one_to_one")
    fields = ["BETA_120","EP_TTM","SP_TTM"]
    weights = anchor[fields].fillna({"BETA_120":1.,"EP_TTM":0.,"SP_TTM":0.}).mul(anchor.weight_float_mv,axis=0)
    reference = weights.groupby(anchor.signal_date).sum()
    proxy = calendar.set_index("execution_date").signal_date.map(reference["BETA_120"])
    ep = calendar.set_index("execution_date").signal_date.map(reference["EP_TTM"])
    sp = calendar.set_index("execution_date").signal_date.map(reference["SP_TTM"])
    proxy_context = pd.DataFrame({"float_beta":proxy,"float_EP":ep,"float_SP":sp})
    return risk,proxy_context


def make_daily(run, benchmarks, risk, proxy):
    nav = pd.read_parquet(run/"nav.parquet").set_index("date").sort_index()
    held = pd.read_parquet(run/"holdings.parquet",columns=["date","code","value"])
    index = benchmarks.pct_change().reindex(nav.index)
    if index.isna().any().any():
        raise ValueError("Missing daily benchmark marks")
    available = pd.DatetimeIndex(risk.signal_date.unique()).sort_values()
    idx = available.searchsorted(nav.index,side="right")-1
    if (idx<0).any():
        raise ValueError("Cannot identify stock beta known by the close")
    week = pd.Series(available[idx],index=nav.index)
    held["signal_date"] = held.date.map(week)
    held = held.merge(risk,on=["signal_date","code"],how="left",validate="many_to_one")
    x = held[["BETA_120","EP_TTM","SP_TTM"]].fillna({"BETA_120":1.,"EP_TTM":0.,"SP_TTM":0.}).mul(held.value,axis=0)
    weights = x.groupby(held.date).sum().reindex(nav.index,fill_value=0).div(nav.equity,axis=0)
    aligned = proxy.reindex(proxy.index.union(nav.index)).sort_index().ffill().reindex(nav.index)
    opening = nav.equity.shift(fill_value=1_000_000)
    r = nav.equity/opening-1
    one = pd.DataFrame(index,columns=["H00300","FloatMV","CSI300_EW"])
    one["portfolio_return"] = r
    one["portfolio_beta_prior"] = weights.BETA_120.shift(fill_value=0.)
    one["float_beta_prior"] = aligned.float_beta.shift(fill_value=aligned.float_beta.iat[0])
    one["EP_active_exposure_prior"] = (weights.EP_TTM-aligned.float_EP).shift()
    one["SP_active_exposure_prior"] = (weights.SP_TTM-aligned.float_SP).shift()
    one["beta_timing"] = (one.portfolio_beta_prior-one.float_beta_prior)*one.H00300
    one["beta_vs_one_diagnostic"] = (one.portfolio_beta_prior-1)*one.H00300
    one["float_mv_basis"] = one.FloatMV-one.H00300
    one["cost"] = -nav.fee/opening
    one["stock_selection_residual"] = r-one.H00300-one.beta_timing-one.float_mv_basis-one.cost
    if not np.allclose(one[COMPONENTS].sum(axis=1),one.portfolio_return-one.H00300,atol=EPSILON):
        raise ValueError("Daily attribution does not reconcile")
    return one


def linked(sample):
    p = 1+sample.portfolio_return
    b = 1+sample.H00300
    # Portfolio wealth before the day and benchmark wealth after it.
    multipliers = p.cumprod().shift(fill_value=1.) * b.iloc[::-1].cumprod().iloc[::-1].shift(-1,fill_value=1.)
    contributions = sample[COMPONENTS].mul(multipliers,axis=0).sum()
    totals = (1+sample[["portfolio_return","H00300","FloatMV","CSI300_EW"]]).prod()-1
    excess = totals.portfolio_return-totals.H00300
    if not np.isclose(contributions.sum(),excess,atol=EPSILON,rtol=0):
        raise ValueError("Linked attribution does not reconcile")
    active = sample.portfolio_return-sample.H00300
    relative = pd.Series(np.r_[1.,(p/b).cumprod().to_numpy()])
    portfolio = pd.Series(np.r_[1.,p.cumprod().to_numpy()])
    return {**contributions.to_dict(),"portfolio_return":totals.portfolio_return,"h00300_return":totals.H00300,
        "h00300_excess":excess,"float_mv_excess":totals.portfolio_return-totals.FloatMV,
        "equal_weight_excess":totals.portfolio_return-totals.CSI300_EW,
        "float_mv_minus_h00300":totals.FloatMV-totals.H00300,
        "h00300_ir":float(np.sqrt(252)*active.mean()/active.std()) if active.std()>0 else np.nan,
        "portfolio_max_drawdown":float((1-portfolio/portfolio.cummax()).max()),
        "max_excess_drawdown":float((1-relative/relative.cummax()).max()),
        "mean_portfolio_beta":sample.portfolio_beta_prior.mean(),"mean_float_beta":sample.float_beta_prior.mean(),
        "mean_EP_active_exposure":sample.EP_active_exposure_prior.mean(),"mean_SP_active_exposure":sample.SP_active_exposure_prior.mean(),
        "beta_vs_one_diagnostic":float((sample.beta_vs_one_diagnostic*multipliers).sum())}


def charts(out, before):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = before.loc[before.index.year==2024]
    monthly = pd.DataFrame([linked(g) for _,g in selected.groupby(selected.index.to_period("M"))],
                           index=pd.period_range("2024-01","2024-12",freq="M").astype(str))
    fig,ax=plt.subplots(figsize=(10.5,4.6),layout="constrained")
    (100*monthly[COMPONENTS]).rename(columns=dict(zip(COMPONENTS,["Beta timing","FloatMV basis","Selection residual","Fees"],strict=True))).plot.bar(
        stacked=True,ax=ax,color=["#287a9e","#dda54a","#546640","#9f3e35"])
    ax.axhline(0,color="#333333",lw=.8)
    ax.set(ylabel="Linked contribution (percentage points)",title="2024 monthly attribution: weekly fusion, 10 bp, original ledger",
           xlabel="")
    ax.legend(ncol=4,fontsize=8)
    fig.savefig(out/"attribution_2024.png",dpi=160)
    plt.close(fig)


def run(root, original, corrected, constrained, out):
    v3,_,_,schedule,_=initialize(root)
    study=root/"research/v3_1/results"
    data=pd.read_parquet(study/"step0/benchmark_nav.parquet").set_index("date").rename(
        columns={"h00300":"H00300","float_mv_nav":"FloatMV","equal_weight_nav":"CSI300_EW"})
    bench=data[["H00300","FloatMV","CSI300_EW"]]
    risk,proxy=held_context(study,schedule)
    annual,quarterly,monthly,windows,daily=[],[],[],[],[]
    sources = {"original":original}
    if corrected:
        sources["corrected"]=corrected
    if constrained:
        sources["nonnegative_ridge"]=constrained
    for version,folder in sources.items():
        for h,fee in FRONTIER:
            sample=make_daily(frontier_run(folder,h,fee),bench,risk,proxy)
            fields={"version":version,"horizon":h,"fee_bps":fee,"strategy":frontier_name(h,fee)}
            daily.append(sample.rename_axis("date").reset_index().assign(**fields))
            annual.extend({**fields,"period":str(year),**linked(group)} for year,group in sample.groupby(sample.index.year))
            y=sample.loc["2024"]
            monthly.extend({**fields,"period":str(period),**linked(group)} for period,group in y.groupby(y.index.to_period("M")))
            quarterly.extend({**fields,"period":str(period),**linked(group)} for period,group in y.groupby(y.index.to_period("Q")))
            for name,start,end in [("first_half","2024-01-01","2024-06-30"),
                                   ("before_924","2024-07-01","2024-09-23"),
                                   ("924_to_1008","2024-09-24","2024-10-08"),
                                   ("after_1008","2024-10-09","2024-12-31")]:
                windows.append({**fields,"period":name,**linked(y.loc[start:end])})
            if version=="original" and h==5 and fee==10:
                v3.save(pd.DataFrame(annual),out/"annual_attribution.csv")
                charts(out,sample)
    for name,table in [("annual_attribution.csv",pd.DataFrame(annual)),("quarterly_2024_attribution.csv",pd.DataFrame(quarterly)),
                       ("monthly_2024_attribution.csv",pd.DataFrame(monthly)),("windows_2024_attribution.csv",pd.DataFrame(windows)),
                       ("daily_attribution.parquet",pd.concat(daily,ignore_index=True))]:
        v3.save(table,out/name)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--corrected",type=Path)
    p.add_argument("--constrained",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    run(a.project_root.resolve(),a.original,a.corrected,a.constrained,a.output or a.project_root/"research/v3_1/results/step4")

"""Build the V3 development tables and NAV figure from saved local outputs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fetch_and_audit_v3_data import save

MAIN=["v1_rolling_104","core_long_raw","core_long_alpha","expanded_long_raw",
      "v3_state_ridge","v3_state_ridge_similarity","v3_lgb_lambdarank","v3_master_gating_net"]


def build(results, benchmark, report):
    models=pd.read_csv(results/"step3/model_metrics.csv")
    weekly=pd.read_parquet(results/"step3/weekly_ic.parquet")
    weekly["period"]=weekly.signal_date.dt.year.lt(2023).map({True:"2017–2022",False:"2023–2025"})
    sub=weekly.groupby(["strategy","label","period"],as_index=False).agg(
        weeks=("rank_ic","count"),rank_ic=("rank_ic","mean"),sd=("rank_ic","std"),top30_hit=("top30_hit","mean"))
    sub["icir"]=sub.rank_ic/sub.sd
    save(sub,report/"model_subperiods.csv")

    bands=pd.read_csv(results/"step4/executed_beta.csv",parse_dates=["date"])
    bands["inside_band"]=bands.estimated_beta.between(bands.beta_lower,bands.beta_upper)
    targets=pd.read_parquet(results/"step4/optimized_targets.parquet")
    grouped=targets.groupby(["strategy","date","regime"],as_index=False).agg(
        target_beta=("target_beta","first"),beta_lower=("beta_lower","first"),
        beta_upper=("beta_upper","first"),target_count=("code","size"),
        largest_weight=("weight","max"),solver_status=("solver_status","first"))
    grouped["inside_target_band"]=grouped.target_beta.between(grouped.beta_lower-1e-7,grouped.beta_upper+1e-7)
    actual=bands.groupby(["strategy","regime"],as_index=False).agg(
        weeks=("date","size"),estimated_beta=("estimated_beta","mean"),
        executed_inside_band=("inside_band","mean"),unknown_beta_weight=("unknown_beta_weight","mean"))
    summary=grouped.groupby(["strategy","regime"],as_index=False).agg(
        target_beta=("target_beta","mean"),target_inside_band=("inside_target_band","mean"),
        target_count=("target_count","mean"),largest_weight=("largest_weight","max"))
    save(actual.merge(summary,on=["strategy","regime"],how="left"),report/"beta_by_regime.csv")
    gates=pd.read_csv(results/"step3/gates.csv",parse_dates=["signal_date"])
    states=pd.read_parquet(results/"step2/states.parquet")
    from optimize_v3_portfolio import beta_band
    states["regime"]=[beta_band(row)[0] for row in states.itertuples()]
    gate_factors=gates.columns.drop("signal_date").tolist()
    g=gates.merge(states[["signal_date","regime"]],on="signal_date",validate="one_to_one")
    save(g.groupby("regime")[gate_factors].mean().reset_index(),report/"mean_gate_by_regime.csv")

    portfolio=pd.read_csv(results/"step4/portfolio_metrics.csv")
    yearly=pd.read_csv(results/"step4/annual_returns.csv")
    y2025=yearly.loc[yearly.year.eq(2025),["strategy","net_return","benchmark_return"]].rename(
        columns={"net_return":"net_return_2025","benchmark_return":"benchmark_return_2025"})
    joined=portfolio.merge(y2025,on="strategy",validate="one_to_one")
    save(joined,report/"model_portfolio_comparison.csv")

    nav=pd.read_parquet(results/"step4/daily_nav.parquet").pivot(index="date",columns="strategy",values="equity")
    bench=pd.read_parquet(benchmark).set_index("date").close.sort_index()
    norm=bench.loc[bench.index<nav.index[0]].iat[-1]
    names=["v1_rolling_104__buffer","v3_state_ridge__optimized","v3_lgb_lambdarank__buffer",
           "v3_lgb_lambdarank__optimized","v3_master_gating_net__optimized"]
    labels=["Core Ridge 104w + buffer","State Ridge + optimized","LambdaRank + buffer",
            "LambdaRank + optimized","Gating net + optimized"]
    fig,axes=plt.subplots(1,2,figsize=(13,4.3),sharey=True,constrained_layout=True)
    for ax,fee in zip(axes,[10,20],strict=True):
        ax.plot(nav.index,bench.reindex(nav.index)/norm,color="#222222",lw=1.1,ls="--",label="CSI 300 total return")
        for name,label in zip(names,labels,strict=True):
            ax.plot(nav.index,nav[f"{name}__{fee}bp"]/1_000_000,lw=1.1,label=label)
        ax.set_title(f"{fee} bp per traded side")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Growth of 1 unit")
    axes[0].legend(fontsize=8,loc="upper left")
    fig.savefig(report/"development_nav.png",dpi=170)
    plt.close(fig)
    print(joined[["strategy","net_return","net_return_2025","information_ratio","unresolved_exposures"]].to_string(index=False))
    print(models.loc[models.label.eq("alpha_return"),["strategy","rank_ic","icir","top30_hit"]].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--results",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    results=a.results or a.project_root/"research/v3/results"
    build(results,a.project_root/"data/raw/v3/development/H00300.parquet",a.output or a.project_root/"research/v3/results/development")

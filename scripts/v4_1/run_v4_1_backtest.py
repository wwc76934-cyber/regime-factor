"""Replay exactly the five listed strategies at two fees with the original V3.1 engine."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/v3_1"))
import optimize_v3_1_portfolio as engine

KEY=engine.KEY
MODELS=["v3_1_champion","v4_1_moe","v4_1_moe_huber"]


def prepare(root,output):
    study=root/"research/v3_1/results"
    if not json.loads((study/"step3/step3_status.json").read_text(encoding="utf-8"))["all_ledgers_verified"]:
        raise ValueError("V3.1 ledger verification required")
    baseline=study/"step4/nonnegative_forecasts"
    old=pd.read_parquet(baseline/"h5/predictions.parquet")
    moe=pd.read_parquet(root/"research/v4_1/results/step1/predictions.parquet")
    reference=old[old.strategy.isin([engine.RIDGE,engine.HUBER])].copy()
    ref_keys=reference.pivot(index=KEY,columns="strategy",values="score")
    deep=moe.set_index(KEY).score.reindex(ref_keys.index)
    if ref_keys.isna().any().any() or not np.isfinite(deep).all():
        raise ValueError("Incomplete common stock universe")
    # V3.1 already standardizes the two score streams and averages them before EMA.
    # Feeding the same MoE stream twice gives exactly the single-model portfolio.
    for model in MODELS[1:]:
        data=reference.copy()
        replace=data.strategy.eq(engine.RIDGE) if model==MODELS[2] else np.ones(len(data),dtype=bool)
        data.loc[replace,"score"]=deep.reindex(pd.MultiIndex.from_frame(data.loc[replace,KEY])).to_numpy()
        dest=output/"inputs"/model/"h5"
        dest.mkdir(parents=True,exist_ok=True)
        data.to_parquet(dest/"predictions.parquet",index=False)
    return baseline


def run(root):
    out=root/"research/v4_1/results/step3"
    baseline=prepare(root,out)
    manifest=[]
    for model in MODELS:
        if model==MODELS[0]:
            blocks=[]
            for fee in (10,20):
                folder=out/model/f"{fee}bp"
                engine.run(root,folder,"optimizer_grid",5,fee,predictions_dir=baseline,frontier_only=True)
                blocks.append(folder)
        else:
            folder=out/model
            engine.run(root,folder,"symmetric_beta",predictions_dir=out/"inputs"/model)
            blocks=[folder]
        for block in blocks:
            table=pd.read_csv(block/"model_portfolio_comparison_v3_1.csv")
            for row in table.itertuples():
                manifest.append(dict(model=model,policy=row.construction,fee_bps=row.fee_bps,engine_strategy=row.strategy,
                                     block=block.relative_to(out).as_posix()))
    manifest=pd.DataFrame(manifest)
    if len(manifest)!=10 or manifest.duplicated(["model","policy","fee_bps"]).any():
        raise ValueError("Unexpected strategy/fee matrix")
    manifest.to_csv(out/"accounts.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    run(p.parse_args().project_root.resolve())

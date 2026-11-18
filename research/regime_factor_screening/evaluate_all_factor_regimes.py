from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d"]
SPLITS = ["screening", "validation"]
FACTOR_DIR = Path("data/processed/factors")
REPORT_DIR = Path("reports/factor_evaluation")
STATES_FILE = Path("data/processed/market/csi300_weekly_market_states.parquet")
KEYS = ["signal_date", "code"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_table(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_parquet(path, index=False)


def registry(root):
    data = pd.read_parquet(root / FACTOR_DIR / "candidate_factor_registry.parquet")
    data = data.loc[data["status"].eq("candidate")].copy()
    require(len(data) == 24 and data["factor"].is_unique, "Expected 24 registered candidate factors")
    return data


def icir(values):
    sd = values.std(ddof=1)
    return values.mean() / sd if pd.notna(sd) and sd > 0 else np.nan


def run(root, output):
    factors = registry(root)["factor"]
    weekly = pd.read_parquet(root / REPORT_DIR / "single_factor_weekly.parquet")
    weekly = weekly.loc[weekly["factor"].isin(factors) & weekly["evaluation_split"].isin(SPLITS)]
    dates = weekly[["signal_date", "evaluation_split"]].drop_duplicates()
    require(dates["signal_date"].is_unique and set(dates["evaluation_split"]) == set(SPLITS)
            and dates["signal_date"].dt.year.between(2020, 2024).all(), "Invalid development dates or splits")
    require(not weekly.duplicated(["signal_date", "factor"]).any() and len(weekly) == len(dates) * len(factors),
            "Incomplete weekly factor grid")
    states = pd.read_parquet(root / STATES_FILE, columns=["signal_date", *STATES])
    joined = weekly.merge(states, on="signal_date", how="left", validate="many_to_one")
    screening = dates.loc[dates["evaluation_split"].eq("screening")].merge(
        states, on="signal_date", how="left", validate="one_to_one")
    require(np.isfinite(joined[STATES]).all().all(), "Missing or nonfinite market states")
    thresholds = screening[STATES].median()
    rows = []
    for (factor, split), week in joined.groupby(["factor", "evaluation_split"]):
        for state, threshold in thresholds.items():
            row = {"factor": factor, "state_name": state, "evaluation_split": split, "threshold": threshold}
            high = week[state].ge(threshold)
            for side, mask in [("high", high), ("low", ~high)]:
                values = week.loc[mask, "rank_ic"]
                spread = week.loc[mask, "q5_minus_q1"]
                row.update({f"weeks_{side}": mask.sum(), f"ic_weeks_{side}": values.count(),
                            f"mean_ic_{side}": values.mean(), f"icir_{side}": icir(values),
                            f"spread_weeks_{side}": spread.count(), f"mean_spread_bps_{side}": spread.mean() * 10000})
            row["delta_ic"] = row["mean_ic_high"] - row["mean_ic_low"]
            row["delta_spread_bps"] = row["mean_spread_bps_high"] - row["mean_spread_bps_low"]
            pair = week[["rank_ic", state]].dropna()
            row["rho_ic_state"] = pair["rank_ic"].rank().corr(pair[state].rank())
            rows.append(row)
    result = pd.DataFrame(rows)
    require(len(result) == 144, "Expected 24 factors × 3 states × 2 splits")
    write_table(result, output / "data/factor_regime_evaluation.parquet")
    write_table(result, output / "reports/factor_regime_evaluation.csv")
    ranking = result.pivot(index=["factor", "state_name"], columns="evaluation_split",
                           values=["delta_ic", "rho_ic_state", "delta_spread_bps"])
    ranking.columns = ["_".join(pair) for pair in ranking.columns]
    ranking = ranking.sort_values("delta_ic_screening", key=lambda values: -values.abs()).head(15).reset_index()
    write_table(ranking, output / "reports/top15_factor_regimes.csv")
    print("Step 1 | screening medians:", thresholds.round(6).to_dict(), flush=True)
    print(ranking.round(4).to_string(index=False), flush=True)
    return result

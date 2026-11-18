"""Reconstruct full weekly rankings and verify the original top-30 selections."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_walk_forward_controls import FEATURES, KEYS, WEIGHT_COLUMNS


MODELS = {"rolling_ridge": "walk_forward_weight_schedule.parquet",
          "state_ridge": "walk_forward_state_weights.parquet"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    processed = args.project_root / "data" / "processed"
    folder = processed / "portfolio" / "walk_forward"
    existing = pd.read_parquet(folder / "walk_forward_three_variant_targets.parquet")
    schedule = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet",
        columns=["signal_date", "execution_date"],
    )
    fixed = pd.read_parquet(
        processed / "factors" / "csi300_weekly_fixed_scores.parquet",
        columns=KEYS + ["eligible", "score_without_bias", *FEATURES],
    )
    eligible = fixed.loc[fixed["eligible"].fillna(False)].copy()
    if eligible.duplicated(KEYS).any():
        raise ValueError("Duplicated eligible stocks")
    frames = []
    for strategy, filename in MODELS.items():
        fit = pd.read_parquet(folder / filename, columns=["signal_date", "fit_status", *WEIGHT_COLUMNS])
        ranked = eligible.merge(fit, on="signal_date", how="left", validate="many_to_one")
        if not ranked["fit_status"].isin(["fitted", "equal_weight_warmup"]).all():
            raise ValueError(f"Missing or unexpected fit status for {strategy}")
        fitted = ranked["fit_status"].eq("fitted")
        ranked["score"] = ranked["score_without_bias"]
        x = ranked.loc[fitted, FEATURES].to_numpy(dtype=float) - 0.5
        coef = ranked.loc[fitted, WEIGHT_COLUMNS].to_numpy(dtype=float)
        ranked.loc[fitted, "score"] = np.sum(x * coef, axis=1)
        if not np.isfinite(ranked["score"]).all():
            raise ValueError(f"Invalid score for {strategy}")
        ranked = ranked.sort_values(["signal_date", "score", "code"], ascending=[True, False, True])
        ranked["rank"] = ranked.groupby("signal_date").cumcount() + 1
        ranked = ranked.merge(schedule, on="signal_date", how="left", validate="many_to_one")
        if not ranked["execution_date"].gt(ranked["signal_date"]).all():
            raise ValueError("Invalid execution date")
        original = existing.loc[existing["strategy"].eq(strategy), KEYS + ["execution_date", "rank", "score"]]
        top_n = int(original["rank"].max())
        actual = ranked.loc[ranked["rank"].le(top_n), original.columns]
        pd.testing.assert_frame_equal(
            original.sort_values(KEYS).reset_index(drop=True),
            actual.sort_values(KEYS).reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        ranked["strategy"] = strategy
        frames.append(ranked[["strategy", *KEYS, "execution_date", "fit_status", "rank", "score"]])
        print(f"{strategy}: {len(ranked):,} ranks; {len(original):,} original targets verified")

    output = args.output_dir or folder
    output.mkdir(parents=True, exist_ok=True)
    path = output / "walk_forward_full_rankings.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    print("Output:", path)


if __name__ == "__main__":
    main()

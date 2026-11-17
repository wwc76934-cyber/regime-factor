"""Build a market-state Ridge portfolio using the exact control training weeks."""

import argparse
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from build_walk_forward_controls import (
    ALPHA, FACTORS, FEATURES, KEYS, WEIGHT_COLUMNS, make_target_rows,
    weekly_statistics,
)


STATE_COLUMNS = [
    "market_return_20d", "market_volatility_20d", "market_breadth_20d",
]
STATE_Z_LIMIT = 3.0


def fit_state_weights(fit, links, training, states):
    if links.duplicated(["signal_date", "training_signal_date"]).any():
        raise ValueError("Duplicated training week in the rolling schedule")
    if not (
        links["training_exit_date"].le(links["signal_date"])
        & links["training_signal_date"].lt(links["signal_date"])
    ).all():
        raise ValueError("Found a training label unavailable on the signal day")

    moments = {}
    for day, week in training.groupby("signal_date", sort=True):
        exits = week["exit_date"].unique()
        if len(exits) != 1:
            raise ValueError(f"Unexpected number of exits for {day}")
        gram, rhs = weekly_statistics(week)
        moments[day] = (gram, rhs, len(week), pd.Timestamp(exits[0]))

    fitted_days = fit.loc[fit["fit_status"].eq("fitted"), "signal_date"]
    if set(fitted_days) != set(links["signal_date"]):
        raise ValueError("Control fitted weeks and training links differ")

    rows = []
    for signal, history in links.groupby("signal_date", sort=True):
        history = history.sort_values("training_signal_date")
        train_days = history["training_signal_date"].tolist()

        past_states = states.loc[train_days, STATE_COLUMNS].to_numpy(dtype=float)
        current = states.loc[signal, STATE_COLUMNS].to_numpy(dtype=float)
        # A week carries equal weight in the normalization as well as in the fit.
        center = past_states.mean(axis=0)
        scale = past_states.std(axis=0)
        usable_scale = scale > 0
        safe_scale = np.where(usable_scale, scale, 1)
        train_z = np.where(usable_scale, (past_states - center) / safe_scale, 0)
        current_z = np.where(usable_scale, (current - center) / safe_scale, 0)
        train_z = np.clip(train_z, -STATE_Z_LIMIT, STATE_Z_LIMIT)
        current_z = np.clip(current_z, -STATE_Z_LIMIT, STATE_Z_LIMIT)

        normal_matrix = np.zeros((12, 12))
        rhs_vector = np.zeros(12)
        for z, link in zip(train_z, history.itertuples(index=False), strict=True):
            gram, rhs, n, exit_date = moments[link.training_signal_date]
            if n != link.training_stocks or exit_date != link.training_exit_date:
                raise ValueError(f"Training rows no longer match the control for {link.training_signal_date}")
            state_vector = np.r_[1.0, z]
            # The stock features are x, z_trend*x, z_vol*x, z_breadth*x.
            normal_matrix += np.kron(np.outer(state_vector, state_vector), gram)
            rhs_vector += np.kron(state_vector, rhs)

        coefficient = np.linalg.solve(
            normal_matrix / len(history) + ALPHA * np.eye(12),
            rhs_vector / len(history),
        ).reshape(4, 3)
        effective = np.r_[1.0, current_z] @ coefficient
        row = {"signal_date": signal}
        row.update(dict(zip(WEIGHT_COLUMNS, effective, strict=True)))
        for index, (state, z) in enumerate(zip(STATE_COLUMNS, current_z, strict=True)):
            row[f"z_{state}"] = z
            for factor, value in zip(FACTORS, coefficient[index + 1], strict=True):
                row[f"interaction_{state}_{factor}"] = value
        for factor, value in zip(FACTORS, coefficient[0], strict=True):
            row[f"intercept_weight_{factor}"] = value
        rows.append(row)

    state_weights = pd.DataFrame(rows)
    base_metadata = fit.drop(columns=WEIGHT_COLUMNS)
    return base_metadata.merge(state_weights, on="signal_date", how="left", validate="one_to_one")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root
    processed = root / "data" / "processed"
    walk = processed / "portfolio" / "walk_forward"
    factor_dir = processed / "factors"
    config = tomllib.loads((root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig"))

    fit = pd.read_parquet(walk / "walk_forward_weight_schedule.parquet")
    if not np.allclose(fit["ridge_alpha"], ALPHA):
        raise ValueError("Ridge penalty differs from the control")
    links = pd.read_parquet(walk / "walk_forward_training_weeks.parquet")
    states = pd.read_parquet(
        processed / "market" / "csi300_weekly_market_states.parquet",
        columns=["signal_date", *STATE_COLUMNS],
    ).set_index("signal_date")
    if not states.index.is_unique or not np.isfinite(states.to_numpy(dtype=float)).all():
        raise ValueError("Missing or duplicate market features")

    scored = pd.read_parquet(
        factor_dir / "csi300_weekly_fixed_scores.parquet",
        columns=KEYS + ["eligible", "score_without_bias", *FEATURES],
    )
    eligible = scored.loc[scored["eligible"].fillna(False)].copy()
    labels = pd.read_parquet(
        factor_dir / "csi300_weekly_forward_labels.parquet",
        columns=KEYS + ["exit_date", "forward_return"],
    )
    training = eligible.merge(labels, on=KEYS, how="left", validate="one_to_one")
    training = training.loc[training["exit_date"].notna() & np.isfinite(training["forward_return"])]

    state_fit = fit_state_weights(fit, links, training, states)
    calendar = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet",
        columns=["signal_date", "execution_date"],
    )
    new_targets = make_target_rows(eligible, state_fit, calendar, config["portfolio"]["top_n"])
    controls = pd.read_parquet(walk / "walk_forward_control_targets.parquet")
    fixed = controls.loc[controls["strategy"].eq("equal_weight")]
    copied_fixed = new_targets.loc[new_targets["strategy"].eq("equal_weight")]
    functions = ["signal_date", "execution_date", "code", "rank", "target_weight"]
    pd.testing.assert_frame_equal(
        fixed.sort_values(KEYS)[functions].reset_index(drop=True),
        copied_fixed.sort_values(KEYS)[functions].reset_index(drop=True),
        check_dtype=False,
    )

    state = new_targets.loc[new_targets["strategy"].eq("rolling_ridge")].copy()
    state["strategy"] = "state_ridge"
    targets = pd.concat([controls, state], ignore_index=True)
    warmup = state_fit.loc[state_fit["fit_status"].ne("fitted"), "signal_date"]
    pd.testing.assert_frame_equal(
        fixed.loc[fixed["signal_date"].isin(warmup)].sort_values(KEYS)[functions].reset_index(drop=True),
        state.loc[state["signal_date"].isin(warmup)].sort_values(KEYS)[functions].reset_index(drop=True),
        check_dtype=False,
    )

    output = args.output_dir or walk
    output.mkdir(parents=True, exist_ok=True)
    state_fit.to_parquet(output / "walk_forward_state_weights.parquet", index=False)
    targets.to_parquet(output / "walk_forward_three_variant_targets.parquet", index=False)
    fitted = state_fit.loc[state_fit["fit_status"].eq("fitted")]
    rolling = controls.loc[
        controls["strategy"].eq("rolling_ridge") & controls["signal_date"].isin(fitted["signal_date"])
    ]
    overlap = state[KEYS].merge(rolling[KEYS], on=KEYS, validate="one_to_one")
    print(f"Fitted: {len(fitted)} | Warm-up: {len(warmup)} | Total target rows: {len(targets):,}")
    print(f"Mean overlap with rolling Ridge after warm-up: {len(overlap) / len(fitted):.2f}")
    print("First state-weighted signal:")
    print(fitted[["signal_date", "training_weeks", *WEIGHT_COLUMNS]].head(1).to_string(index=False))
    print("Output:", output)


if __name__ == "__main__":
    main()

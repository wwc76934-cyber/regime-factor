"""Build an auditable walk-forward schedule and two three-factor controls."""

from pathlib import Path
import argparse
import tomllib

import numpy as np
import pandas as pd


FACTORS = ["RET_5", "VOL_60", "CORR_PV_10"]
FEATURES = [f"pct_{factor}" for factor in FACTORS]
KEYS = ["signal_date", "code"]
MIN_WEEKS = 52
WINDOW_WEEKS = 104
MIN_STOCKS = 50
ALPHA = 0.01
WEIGHT_COLUMNS = [f"coef_{factor}" for factor in FACTORS]


def weekly_statistics(group):
    # Each week carries equal total weight regardless of its stock count.
    x = group[FEATURES].to_numpy(dtype=float)
    y = group["forward_return"].rank(method="average", pct=True).to_numpy()
    x = x - x.mean(axis=0)
    y = y - y.mean()
    return x.T @ x / len(group), x.T @ y / len(group)


def fit_schedule(score_days, training):
    moments = []
    for signal, group in training.groupby("signal_date", sort=True):
        if len(group) < MIN_STOCKS:
            continue
        exits = group["exit_date"].unique()
        if len(exits) != 1 or exits[0] <= signal:
            raise ValueError(f"Invalid label window for {signal}")
        gram, rhs = weekly_statistics(group)
        moments.append((signal, pd.Timestamp(exits[0]), gram, rhs, len(group)))

    rows = []
    links = []
    for day in score_days:
        # The label ends at an open. An exit on the signal day is known by its close.
        available = [item for item in moments if item[0] < day and item[1] <= day]
        history = available[-WINDOW_WEEKS:]
        fitted = len(history) >= MIN_WEEKS
        if fitted:
            gram = np.mean([item[2] for item in history], axis=0)
            rhs = np.mean([item[3] for item in history], axis=0)
            coefficients = np.linalg.solve(gram + ALPHA * np.eye(len(FACTORS)), rhs)
            links.extend(
                (day, train_day, exit_day, n)
                for train_day, exit_day, _, _, n in history
            )
        else:
            coefficients = np.full(len(FACTORS), np.nan)

        rows.append({
            "signal_date": day,
            "fit_status": "fitted" if fitted else "equal_weight_warmup",
            "available_weeks": len(available),
            "training_weeks": len(history) if fitted else 0,
            "first_training_signal": history[0][0] if fitted else pd.NaT,
            "latest_training_label_end": max(item[1] for item in history) if fitted else pd.NaT,
            "training_rows": sum(item[4] for item in history) if fitted else 0,
            "ridge_alpha": ALPHA,
            **dict(zip(WEIGHT_COLUMNS, coefficients)),
        })

    schedule = pd.DataFrame(rows)
    membership = pd.DataFrame(
        links, columns=["signal_date", "training_signal_date", "training_exit_date", "training_stocks"]
    )
    return schedule, membership


def make_target_rows(eligible, fit, calendar, top_n):
    scores = eligible.merge(fit, on="signal_date", validate="many_to_one")
    # Reuse the existing three-factor mean so the fixed control keeps its original ties.
    scores["equal_weight"] = scores["score_without_bias"]
    fitted = scores["fit_status"].eq("fitted")
    scores["rolling_ridge"] = scores["equal_weight"]
    x = scores.loc[fitted, FEATURES].to_numpy(dtype=float) - 0.5
    w = scores.loc[fitted, WEIGHT_COLUMNS].to_numpy(dtype=float)
    scores.loc[fitted, "rolling_ridge"] = np.sum(x * w, axis=1)

    results = []
    for variant in ("equal_weight", "rolling_ridge"):
        ordered = scores[KEYS + [variant]].rename(columns={variant: "score"})
        ordered = ordered.sort_values(
            ["signal_date", "score", "code"], ascending=[True, False, True]
        )
        ordered["rank"] = ordered.groupby("signal_date").cumcount() + 1
        chosen = ordered.loc[ordered["rank"].le(top_n)].copy()
        if not chosen.groupby("signal_date").size().reindex(fit["signal_date"]).eq(top_n).all():
            raise ValueError(f"{variant} has fewer than {top_n} targets on a signal date")
        chosen["strategy"] = variant
        chosen["target_weight"] = 1.0 / top_n
        results.append(chosen)

    targets = pd.concat(results, ignore_index=True).merge(
        calendar, on="signal_date", how="left", validate="many_to_one"
    )
    if not targets["execution_date"].gt(targets["signal_date"]).all():
        raise ValueError("Missing or invalid execution date")

    return targets[[
        "strategy", "signal_date", "execution_date", "code",
        "rank", "score", "target_weight",
    ]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root
    processed = root / "data" / "processed"
    factor_dir = processed / "factors"
    config = tomllib.loads((root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig"))
    top_n = config["portfolio"]["top_n"]

    fixed = pd.read_parquet(
        factor_dir / "csi300_weekly_fixed_scores.parquet",
        columns=KEYS + ["eligible", "score_without_bias", *FEATURES],
    )
    eligible = fixed.loc[fixed["eligible"].fillna(False)].copy()
    if not np.isfinite(eligible[FEATURES + ["score_without_bias"]].to_numpy(dtype=float)).all():
        raise ValueError("Eligible stock has nonfinite factors")
    labels = pd.read_parquet(
        factor_dir / "csi300_weekly_forward_labels.parquet",
        columns=KEYS + ["exit_date", "forward_return"],
    )
    joined = eligible.merge(labels, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some eligible stocks are absent from the label table")
    training = joined.loc[
        joined["exit_date"].notna() & np.isfinite(joined["forward_return"])
    ]
    score_days = pd.DatetimeIndex(fixed["signal_date"].unique()).sort_values()
    fit, membership = fit_schedule(score_days, training)
    if membership.empty:
        raise ValueError("No signal week has enough mature training labels")
    if not membership["training_exit_date"].le(membership["signal_date"]).all():
        raise ValueError("Training schedule uses an unmatured label")

    calendar = pd.read_parquet(
        processed / "universe" / "csi300_weekly_schedule.parquet",
        columns=["signal_date", "execution_date"],
    )
    targets = make_target_rows(eligible, fit, calendar, top_n)
    prior_targets = pd.read_parquet(
        processed / "portfolio" / "csi300_weekly_target_positions.parquet",
        columns=KEYS + ["strategy", "rank"],
    )
    expected = prior_targets.loc[prior_targets["strategy"].eq("without_bias"), KEYS + ["rank"]]
    actual = targets.loc[targets["strategy"].eq("equal_weight"), KEYS + ["rank"]]
    expected = expected.sort_values(KEYS).reset_index(drop=True)
    actual = actual.sort_values(KEYS).reset_index(drop=True)
    if not actual.equals(expected.astype({"rank": actual["rank"].dtype})):
        raise ValueError("Fixed three-factor targets differ from the earlier fixed baseline")

    output = args.output_dir or processed / "portfolio" / "walk_forward"
    output.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(output / "walk_forward_weight_schedule.parquet", index=False)
    membership.to_parquet(output / "walk_forward_training_weeks.parquet", index=False)
    targets.to_parquet(output / "walk_forward_control_targets.parquet", index=False)

    fitted = fit.loc[fit["fit_status"].eq("fitted")]
    print(f"Signal weeks: {len(fit)} | Fitted: {len(fitted)} | Warm-up: {len(fit) - len(fitted)}")
    print(f"Fixed targets match: {len(actual):,} rows | Both variants: {len(targets):,} rows")
    print("First fitted week:")
    print(fitted.head(1)[[
        "signal_date", "training_weeks", "first_training_signal",
        "latest_training_label_end", *WEIGHT_COLUMNS,
    ]].to_string(index=False, float_format="{:.5f}".format))
    overlap = targets.loc[targets["signal_date"].isin(fitted["signal_date"])].groupby(KEYS).size()
    print(f"Mean overlap after warm-up: {overlap.eq(2).sum() / len(fitted):.2f} / {top_n}")
    print("Output:", output)


if __name__ == "__main__":
    main()

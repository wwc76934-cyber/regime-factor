"""Paired h5 experiment with a nonnegative Ridge and a native-monotone Huber tree."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_v3_1_base import DEVELOPMENT_LAST
from train_v3_1_models import COMMON, KEY, TREE, produce, tree_features, weeks_from
from optimize_v3_1_portfolio import run as replay


def train(root, out, workers):
    results = root / "research/v3_1/results"
    inputs = results / "step1"
    spec = json.loads((inputs / "v3_1_selected_factors.json").read_text(encoding="utf-8"))
    factors, states = spec["selected"], spec["states"]
    if spec["status"] != "go" or len(factors) != 6 or len(states) != 5:
        raise ValueError("Expected six admitted oriented factors and five state variables")
    frame = pd.read_parquet(inputs / "factor_panel.parquet", columns=KEY + ["signal_bar_usable", "BETA_120",
                            "VOL_20", *dict.fromkeys([*factors, *COMMON])])
    frame = frame.loc[frame.signal_bar_usable & frame[["BETA_120", "VOL_20", *COMMON]].notna().all(axis=1)]
    labels = pd.read_parquet(inputs / "labels.parquet", filters=[("horizon", "==", 5)])
    market = pd.read_parquet(inputs / "states.parquet", columns=["signal_date", *states])
    if max(frame.signal_date.max(), market.signal_date.max(), labels.exit_date.max()) > DEVELOPMENT_LAST:
        raise ValueError("Development input exceeds the registered end date")
    weeks = weeks_from(frame.sort_values(KEY), labels, market, factors, spec["directions"])
    probe_rows = []

    def probe(tree, current, state):
        # Same 12 cross-sectional names and endpoints for both perturbations, every signal week.
        x = current.x[np.linspace(0, len(current.x)-1, 12, dtype=int)]
        original = tree_features(x, state)
        batch = []
        for i in range(len(factors)):
            for level in (-.25, .25):
                partial = original.copy()
                partial[:, i] = level
                changed = x.copy()
                changed[:, i] = level
                batch.extend([partial, tree_features(changed, state)])
        scores = tree.predict(np.concatenate(batch), num_threads=workers).reshape(len(factors), 2, 2, len(x))
        changes = scores[:, 1] - scores[:, 0]
        for i, factor in enumerate(factors):
            partial, joint = changes[i]
            probe_rows.append((current.day, factor, len(x), int((partial < -1e-12).sum()),
                               int((joint < -1e-12).sum()), float(partial.min()), float(joint.min())))

    scores, windows, _ = produce(weeks, workers, monotone_base=True, prediction_probe=probe)
    probes = pd.DataFrame(probe_rows, columns=["signal_date", "factor", "pairs", "native_decreases",
                                               "joint_decreases", "native_min_change", "joint_min_change"])
    if probes.native_decreases.any():
        raise ValueError("Native monotonicity check failed")
    prior = pd.read_parquet(results / "step4/nonnegative_forecasts/h5/predictions.parquet")
    mask = prior.strategy.eq(TREE)
    keys = pd.MultiIndex.from_frame(prior.loc[mask, KEY])
    candidate = scores.loc[scores.strategy.eq(TREE)].set_index(KEY).score
    if not keys.is_unique or not candidate.index.is_unique or not keys.isin(candidate.index).all():
        raise ValueError("Prior forecast universe cannot be reproduced")
    aligned = candidate.loc[keys].to_numpy()
    if not np.isfinite(aligned).all():
        raise ValueError("Non-finite tree score")
    prior.loc[mask, "score"] = aligned
    destination = out / "forecasts/h5"
    destination.mkdir(parents=True, exist_ok=True)
    prior.to_parquet(destination / "predictions.parquet", index=False)
    windows.to_csv(out / "training_windows.csv", index=False)
    probes.to_csv(out / "monotonicity_probes.csv", index=False)
    feature_names = [*factors, *states, *(f"{s}*{f}" for s in states for f in factors)]
    (out / "feature_constraints.json").write_text(json.dumps(dict(zip(feature_names,
        [1]*len(factors) + [0]*(len(feature_names)-len(factors)), strict=True)), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=["train", "replay", "evaluate", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "research/v3_1/results/step4/symmetric_beta"
    if args.stage in ("train", "all"):
        train(root, out, args.workers)
    if args.stage in ("replay", "all"):
        replay(root, out / "accounts", "symmetric_beta", predictions_dir=out / "forecasts")
    if args.stage in ("evaluate", "all"):
        from evaluate_v3_1_symmetric_beta import evaluate
        evaluate(root, out)


if __name__ == "__main__":
    main()

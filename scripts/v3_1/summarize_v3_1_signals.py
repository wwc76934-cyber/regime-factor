"""Combine the V3.1 signal audits and apply the Step 2 development gate."""
import argparse
import json
from pathlib import Path

import pandas as pd


HORIZONS = (5, 10, 20)
BENCHMARK = "v3_0_state_ridge"


def summarize(root):
    parts = [root / f"h{h}" / "model_signal_audit.csv" for h in HORIZONS]
    optional = root / "gate_h10" / "model_signal_audit.csv"
    if optional.is_file():
        parts.append(optional)
    metrics = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    if metrics.duplicated(["strategy", "horizon", "period"]).any():
        raise ValueError("Duplicate strategy, horizon and period")
    main = metrics.period.eq("all")
    metrics["meets_step2_thresholds"] = (main & (metrics.top30_hit > .10) &
        (metrics.rank_ic > .03) & (metrics.top30_excess_H00300 > 0))
    current = metrics[metrics.strategy.ne(BENCHMARK) & metrics.meets_step2_thresholds]
    qualified = sorted({f"{row.strategy}_h{row.horizon}" for row in current.itertuples(index=False)})
    primary = sorted(current.loc[current.horizon.eq(10), "strategy"].unique())
    status = {
        "status": "go" if qualified else "no_go",
        "eligible_development_variants": qualified,
        "eligible_primary_h10_models": primary,
        "thresholds": {"top30_hit": ">0.10", "rank_ic": ">0.03", "top30_excess_H00300": ">0"},
        "realized_beta_definition": "Cov of holding-period gross top30 and H00300 returns / H00300 variance",
    }
    metrics.to_csv(root / "model_signal_audit.csv", index=False, encoding="utf-8-sig")
    (root / "step2_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(metrics.loc[main, ["strategy", "horizon", "weeks", "rank_ic", "icir", "top30_hit",
        "top30_excess_H00300", "top30_realized_beta", "meets_step2_thresholds"]].to_string(index=False))
    print(json.dumps(status))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2", type=Path, default=Path("research/v3_1/results/step2"))
    summarize(parser.parse_args().step2)

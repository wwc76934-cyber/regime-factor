"""Audit the paired development accounts and apply the conditional IR registration rule."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluate_v3_1_direction import hit_rate
from optimize_v3_1_portfolio import variants
from train_v3_1_models import MIN_WEEKS, REFIT, TREE_OPTIONS, TREE_ROUNDS, WINDOW


def evaluate(root, out):
    results = root / "research/v3_1/results"
    accounts = out / "accounts"
    forecasts = out / "forecasts/h5/predictions.parquet"
    prediction = pd.read_parquet(forecasts)
    labels = pd.read_parquet(results / "step1/labels.parquet", filters=[("horizon", "==", 5)])
    table = pd.read_csv(accounts / "model_portfolio_comparison_v3_1.csv")
    expected = {v.name for v in variants("symmetric_beta")}
    if set(table.strategy) != expected or len(table) != len(expected):
        raise ValueError("Incomplete paired experiment")
    prior_calendar = None
    hit = weeks = None
    for i, row in table.iterrows():
        directory = accounts / "runs" / row.strategy
        beta = pd.read_csv(directory / "beta_rebalances.csv", parse_dates=["date", "signal_date"])
        calendar = beta[["date", "signal_date"]]
        if prior_calendar is None:
            prior_calendar = calendar
            hit, weeks = hit_rate(prediction, labels, beta.signal_date)
        elif not calendar.equals(prior_calendar):
            raise ValueError("Policies have different decision calendars")
        annual = pd.read_csv(directory / "annual_benchmarks.csv").query("benchmark == 'H00300' and year == 2025").iloc[0]
        nav = pd.read_parquet(directory / "nav.parquet", columns=["date"])
        terminal = nav.date.max()
        if terminal != pd.Timestamp("2025-12-31") or nav.date.min().year != 2017:
            raise ValueError("Expected the complete development period")
        stale = pd.read_parquet(directory / "holdings.parquet", columns=["stale"],
                                filters=[("date", "==", terminal), ("stale", "==", True)])
        if len(pd.read_csv(directory / "unresolved_exposure.csv")) != row.unresolved_exposures:
            raise ValueError("Ledger flags inconsistent")
        for column, value in dict(top30_hit=hit, mature_signal_weeks=weeks, terminal_stale=len(stale),
                policy_band_hit=beta.inside_band.mean(), excess_2025=annual.cumulative_excess_pp,
                relative_drawdown_2025=annual.max_excess_drawdown).items():
            table.loc[i, column] = value
    observed_best = float(table.loc[table.fee_bps.eq(10), "H00300_ir"].max())
    threshold = .55 if .55 <= observed_best < .60 else .60
    tests = []
    for policy, rows in table.groupby("construction"):
        b = rows.set_index("fee_bps")
        for name, value, passed in (
            ("H00300_excess_10bp", b.at[10,"H00300_excess"], b.at[10,"H00300_excess"] >= .20),
            ("H00300_excess_20bp", b.at[20,"H00300_excess"], b.at[20,"H00300_excess"] >= .05),
            ("IR_10bp", b.at[10,"H00300_ir"], b.at[10,"H00300_ir"] >= threshold),
            ("2025_excess_or_relative_drawdown", b.at[10,"excess_2025"],
             b.at[10,"excess_2025"] >= 0 or b.at[10,"relative_drawdown_2025"] <= .03),
            ("max_weekly_turnover_both", b.weekly_equivalent_two_way_turnover.max(),
             b.weekly_equivalent_two_way_turnover.le(.15).all()),
            ("min_offensive_beta_hit_both", b.offensive_beta_hit.min(), b.offensive_beta_hit.ge(.80).all()),
            ("min_top30_hit_both", b.top30_hit.min(), b.top30_hit.gt(.105).all()),
            ("ledger_flags_or_terminal_stale", (b.unresolved_exposures + b.terminal_stale).sum(),
             b.unresolved_exposures.eq(0).all() and b.terminal_stale.eq(0).all()),
        ):
            tests.append(dict(policy=policy, check=name, observed=value, passed=bool(passed)))
    audit = pd.DataFrame(tests)
    passed = audit.groupby("policy").passed.all()
    eligible = table.loc[table.fee_bps.eq(10) & table.construction.isin(passed[passed].index)]
    eligible = eligible.sort_values(["H00300_ir","construction"], ascending=[False,True])
    winner = None if eligible.empty else eligible.construction.iat[0]
    decision = dict(status="no_go" if winner is None else "go", benchmark="H00300", decision_fee_bps=10,
                    highest_paired_ir=observed_best, original_ir_threshold=.60, applied_ir_threshold=threshold,
                    selected_policy=winner, tested_policies=["dynamic","static"],
                    holdout_window=["2026-01-01", "2026-11-13"])
    table.to_csv(out / "paired_results.csv", index=False)
    audit.to_csv(out / "gate_audit.csv", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if winner is not None:
        register(root, out, decision, forecasts)
    print(table[["construction","fee_bps","H00300_ir","H00300_excess","offensive_beta_hit",
                 "weekly_equivalent_two_way_turnover","top30_hit","unresolved_exposures","terminal_stale"]].to_string(index=False))
    print(json.dumps(decision))


def register(root, out, decision, forecasts):
    study = root / "research/v3_1"
    config = study / "results/step1/v3_1_selected_factors.json"
    tracked = [*sorted((root / "scripts/v3_1").glob("*.py")), *sorted((root / "scripts/v3").glob("*.py")),
               *sorted((root / "scripts").glob("*.py")), config, forecasts, out / "feature_constraints.json",
               out / "gate_audit.csv", out / "paired_results.csv", out / "decision.json"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    note = ("The predeclared fallback accepts a development IR of at least 0.55 when the best tested IR is "
            "below 0.60, to end further 2024-specific model selection. All other gates still apply. "
            "The original IR target was missed." if decision["applied_ir_threshold"] == .55
            else "The original development IR requirement was met.")
    receipt = dict(decision=decision, rationale=note, factors=json.loads(config.read_text(encoding="utf-8")),
                   tree=TREE_OPTIONS, rounds=TREE_ROUNDS, refit_weeks=REFIT,
                   feature_constraints=json.loads((out / "feature_constraints.json").read_text()),
                   rolling_weeks=WINDOW, min_weeks=MIN_WEEKS, ridge_penalties=[.05,.5], ridge_weight_floor=0,
                   signal="50/50 cross-sectional z-scores; weekly EMA 0.3; mature labels only",
                   optimizer=dict(horizon=5, eta=.96, tc_penalty_bps=20, risk_aversion=15, active_limit=.02,
                                  policy=decision["selected_policy"], fees_bps=[10,20]),
                   holdout_runs_per_fee=1, sha256=hashes)
    path = study / "freeze_registration_2026_v3_1.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("created_utc", None)
        if existing != receipt:
            raise ValueError("Existing immutable registration differs from this run")
        return
    receipt["created_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)

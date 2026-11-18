import json

import numpy as np
import pandas as pd

from evaluate_all_factor_regimes import REPORT_DIR, SPLITS, STATES, registry, require, write_table


LEGACY_CORE = ["RET_5", "VOL_60", "CORR_PV_10"]
REDUNDANCY = 0.75


def same_direction(left, right):
    return left.notna() & right.notna() & left.ne(0) & right.ne(0) & (np.sign(left) == np.sign(right))


def paired(frame, index, values):
    table = frame.pivot(index=index, columns="evaluation_split", values=values)
    table.columns = ["_".join(pair) for pair in table.columns]
    return table.reset_index()


def run(root, output):
    meta = registry(root)[["factor", "family", "hypothesis"]]
    names = meta["factor"].tolist()
    stats = pd.read_parquet(root / REPORT_DIR / "single_factor_by_split.parquet")
    stats = stats.loc[stats["factor"].isin(names) & stats["evaluation_split"].isin(SPLITS)]
    require(len(stats) == 48, "Incomplete unconditional summary")
    audit = meta.merge(paired(stats, "factor", ["mean_rank_ic", "mean_q5_minus_q1", "ic_weeks"]),
                       validate="one_to_one")
    screening_ic = audit["mean_rank_ic_screening"]
    validation_ic = audit["mean_rank_ic_validation"]
    audit["A1"] = screening_ic.abs().ge(0.025)
    audit["A2"] = same_direction(screening_ic, validation_ic) & validation_ic.abs().ge(0.015)
    audit["A_pass"] = audit["A1"] & audit["A2"]
    audit["screening_abs_ic"] = screening_ic.abs()
    audit["legacy_priority"] = audit["factor"].isin(LEGACY_CORE)

    conditional = pd.read_parquet(output / "data/factor_regime_evaluation.parquet")
    interaction = paired(conditional, ["factor", "state_name"],
                         ["delta_ic", "rho_ic_state", "delta_spread_bps", "mean_ic_high", "mean_ic_low",
                          "ic_weeks_high", "ic_weeks_low"])
    delta = interaction["delta_ic_screening"]
    strong_side = interaction[["mean_ic_high_screening", "mean_ic_low_screening"]].abs().max(axis=1)
    interaction["B1"] = delta.abs().ge(0.030) | (strong_side.ge(0.035) & delta.abs().ge(0.020))
    interaction["B2_delta"] = same_direction(delta, interaction["delta_ic_validation"])
    interaction["B2_rho"] = same_direction(interaction["rho_ic_state_screening"], interaction["rho_ic_state_validation"])
    interaction["B2"] = interaction["B2_delta"] & interaction["B2_rho"]
    interaction["B3"] = same_direction(delta, interaction["delta_spread_bps_screening"])
    flags = ["B1", "B2_delta", "B2_rho", "B3"]
    interaction["B_pass"] = interaction[flags].all(axis=1)
    interaction["gate_failures"] = interaction[flags].apply(lambda row: ";".join(row.index[~row]), axis=1)
    interaction["screening_abs_delta_ic"] = delta.abs()

    pairs = pd.read_parquet(root / REPORT_DIR / "redundancy_pairs.parquet")
    lookup = {frozenset([item.factor_a, item.factor_b]): item.median_abs_rank_corr
              for item in pairs.itertuples(index=False)}
    require(len(lookup) == len(names) * (len(names) - 1) // 2 and all(np.isfinite(list(lookup.values()))),
            "Incomplete screening redundancy matrix")

    eligible = interaction.loc[interaction["B_pass"]].copy()
    strongest = eligible.groupby("factor")["screening_abs_delta_ic"].max()
    audit["B_pass_count"] = audit["factor"].map(eligible.groupby("factor").size()).fillna(0).astype(int)
    audit["strongest_passing_delta_ic"] = audit["factor"].map(strongest)
    audit = audit.set_index("factor")
    baseline_order = audit.loc[audit["A_pass"]].sort_values(
        ["legacy_priority", "screening_abs_ic"], ascending=False, kind="stable").index
    conditional_order = strongest.sort_values(ascending=False, kind="stable").index
    selected, base, decisions = [], [], {}

    def consider(factor, track):
        distances = [(other, lookup[frozenset([factor, other])]) for other in selected]
        conflicts = [(other, corr) for other, corr in distances if corr >= REDUNDANCY]
        if conflicts:
            other, corr = max(conflicts, key=lambda pair: pair[1])
            decisions[factor] = ("redundant_dropped", other, corr, track)
            return
        selected.append(factor)
        if track == "base":
            base.append(factor)
        decisions[factor] = ("selected", "", np.nan, track)

    for factor in baseline_order:
        consider(factor, "base")
    for factor in conditional_order:
        if factor not in decisions:
            consider(factor, "conditional")
    for factor in names:
        decisions.setdefault(factor, ("failed_gates", "", np.nan, "none"))
    audit = audit.join(pd.DataFrame.from_dict(decisions, orient="index",
        columns=["decision", "redundant_with", "redundancy_corr", "track"]))
    interaction["factor_decision"] = interaction["factor"].map(audit["decision"])
    interaction["active"] = interaction["B_pass"] & interaction["factor"].isin(selected)
    active = interaction.loc[interaction["active"], ["factor", "state_name"]]
    signs = {
        name: int(np.sign(audit.at[name, "mean_rank_ic_screening"])) for name in selected
    }
    require(all(sign in [-1, 1] for sign in signs.values()), "Cannot orient a selected zero-IC factor")
    spec = {
        "selected_factors": [{"factor": name, "direction": signs[name], "track": decisions[name][3]}
                             for name in selected],
        "base_factors": base,
        "active_interactions": active.to_dict("records"),
        "state_features": STATES,
        "state_thresholds": {name: float(conditional.loc[conditional["state_name"].eq(name), "threshold"].iat[0])
                             for name in STATES},
        "selection_splits": SPLITS,
        "core_A1_rule": "absolute IC >= 0.025",
        "redundancy_threshold": REDUNDANCY,
        "development_evaluation": "retrospective; factor and interaction gates use both development splits",
        "base_orientation_rule": "sign of raw screening mean IC; Ridge coefficients are unconstrained",
    }
    destination = output / "configs/verified_model_spec.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = audit.reset_index()
    display = interaction.pivot(index="factor", columns="state_name",
                                values=[*flags, "B_pass", "active", "delta_ic_screening",
                                        "delta_ic_validation", "rho_ic_state_screening", "rho_ic_state_validation",
                                        "delta_spread_bps_screening", "delta_spread_bps_validation"])
    display.columns = ["__".join(pair) for pair in display.columns]
    write_table(audit.merge(display.reset_index(), validate="one_to_one"),
                output / "reports/factor_selection_audit_table.csv")
    write_table(interaction, output / "reports/factor_interaction_audit.csv")
    print("\nStep 2 | factor decisions", flush=True)
    print(audit[["factor", "A1", "A2", "B_pass_count", "decision", "redundant_with"]].to_string(index=False), flush=True)
    print("Passing pairs before redundancy:", int(interaction["B_pass"].sum()), "| Active:", len(active), flush=True)
    print(active.to_string(index=False), flush=True)
    return spec

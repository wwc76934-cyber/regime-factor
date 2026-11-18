import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from holdout_data import ANCHOR, END, START, KEYS, FACTORS, FEATURES, STATES, build_inputs, corporate_actions, require, write
from holdout_models import fit_all


NAV = "strategy date cash market_value pending_market_value receivables rights_prepaid equity fee gross opening_equity stale_marks".split()
ORDERS = "strategy date code shares_before target_shares requested_change filled_change shares_after open_price signed_gross fee cash_after status".split()


def dev_inputs(project):
    processed = project / "data/processed"
    folder = processed / "factors"
    fixed = pd.read_parquet(folder / "csi300_weekly_fixed_scores.parquet")
    eligible = fixed.loc[fixed["eligible"], KEYS + ["score_without_bias", *FEATURES]]
    labels = pd.read_parquet(folder / "csi300_weekly_forward_labels.parquet",
                             columns=KEYS + ["exit_date", "forward_return", "evaluation_split"])
    states = pd.read_parquet(processed / "market/csi300_weekly_market_states.parquet").set_index("signal_date")[STATES]
    links = pd.read_parquet(processed / "portfolio/walk_forward/walk_forward_training_weeks.parquet")
    groups = {day: [(row.training_signal_date, row.training_exit_date, row.training_stocks)
                    for row in week.sort_values("training_signal_date").itertuples(index=False)]
              for day, week in links.groupby("signal_date", sort=True)}
    require(eligible["signal_date"].le(ANCHOR).all()
            and labels["exit_date"].dropna().le(ANCHOR).all()
            and len(groups) == 202, "Unexpected development cache")
    screening = labels.loc[labels["evaluation_split"].eq("screening"), "signal_date"].unique()
    return eligible, labels[KEYS + ["exit_date", "forward_return"]], states, groups, states.loc[screening].median()


def plot_and_explain(output, weights, gain, margins):
    import matplotlib.pyplot as plt

    for name, table in [("effective_weights", weights), ("weekly_tree_importance", gain),
                        ("weekly_prediction_margins", margins)]:
        write(table, output / f"explain/{name}.csv")
    result = gain.groupby(["sample", "feature"])[["gain", "split_count"]].sum().reset_index()
    for column in ["gain", "split_count"]:
        result[f"{column}_share"] = result[column] / result.groupby("sample")[column].transform("sum")
    write(result, output / "explain/feature_importance.csv")
    summary = margins.groupby(["sample", "state", "regime", "factor"]).agg(
        weeks=("signal_date", "nunique"),
        high_factor_prediction=("high_factor_prediction", "mean"),
        low_factor_prediction=("low_factor_prediction", "mean"),
        high_minus_low=("high_minus_low", "mean")).reset_index()
    write(summary, output / "explain/state_prediction_margins.csv")
    state = weights.loc[weights["strategy"].eq("state_ridge")]
    summary_weights = state.assign(year=state["signal_date"].dt.year).groupby(["year", "factor"])[
        ["effective_weight", "base_weight", "state_tilt", *STATES]].mean().reset_index()
    write(summary_weights, output / "explain/mean_weight_by_year.csv")
    pivot = state.pivot(index="signal_date", columns="factor", values="effective_weight")
    axes = pivot.plot(subplots=True, sharex=True, figsize=(11, 7), grid=True, linewidth=1.1)
    for axis in axes:
        axis.axhline(0, color="0.6", linewidth=.7)
        axis.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), alpha=.09, color="red")
        axis.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), alpha=.09, color="blue")
        axis.axvline(ANCHOR, color="0.3", linestyle="--", linewidth=1)
    axes[0].figure.suptitle("State Ridge effective weights: 2021–2025")
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(output / "explain/effective_weights.png", dpi=170)
    plt.close(axes[0].figure)


def eval_returns(ranks, labels):
    evaluated = ranks.merge(labels, on=KEYS, validate="many_to_one")
    evaluated = evaluated.loc[evaluated["execution_date"].between(START, END)
                              & evaluated["exit_date"].le(END) & np.isfinite(evaluated["forward_return"])]
    rows = []
    for (name, date), week in evaluated.groupby(["strategy", "signal_date"]):
        rows.append((name, date, len(week), week["score"].rank().corr(week["forward_return"].rank())))
    return pd.DataFrame(rows, columns=["strategy", "signal_date", "valid_labels", "rank_ic"])


def run_portfolios(ranks, panel, event_rows, review, days, benchmark, spec, output):
    from run_holdings_buffer_backtest import simulate, make_buffer_selector

    trading = days[(days >= START) & (days <= END)]
    nav_days = trading.insert(0, ANCHOR)
    require(benchmark.index.is_unique and benchmark.index.equals(nav_days) and benchmark.gt(0).all(),
            "H00300 must exactly match the full holdout evaluation calendar")
    benchmark = benchmark / benchmark.iloc[0]
    bench_return = benchmark.pct_change().iloc[1:]
    bars = {day: {row.code: row for row in group.itertuples(index=False)}
            for day, group in panel.loc[panel["date"].isin(trading)].groupby("date", sort=False)}
    configs, navs, summaries, unresolved, target_books, order_books, action_books = [], [], [], [], [], [], []
    for name in spec["models"]:
        configurations = [45, 50] if name == spec["extra_execution"]["strategy"] else [45]
        rankings = ranks.loc[ranks["strategy"].eq(name) & ranks["execution_date"].between(START, END)]
        primary = rankings.loc[rankings["rank"].le(spec["top_n"])]
        schedule = {day: week.set_index("code")["rank"].to_dict()
                    for day, week in primary.groupby("execution_date")}
        require(all(len(week) == 30 for week in schedule.values()), "Incomplete target schedule")
        for buffer in configurations:
            for fee in spec["fee_bps"]:
                audit = []
                selector = make_buffer_selector(name, rankings, buffer, audit)
                metadata = {"strategy": name, "buffer_rank": buffer, "fee_bps": fee}
                records, orders, _, actions, exposure = simulate(
                    name, trading, bars, schedule, event_rows, review, Decimal(spec["capital"]),
                    fee_rate=Decimal(fee) / Decimal(10000), select_targets=selector)
                daily = pd.DataFrame(records, columns=NAV).set_index("date")
                require(daily.index.equals(trading) and daily["equity"].gt(0).all(), "Invalid daily ledger")
                normalized = daily["equity"].reindex(nav_days)
                normalized.loc[ANCHOR] = float(spec["capital"])
                normalized /= float(spec["capital"])
                returns = normalized.pct_change().iloc[1:]
                relative = normalized / benchmark
                active = returns - bench_return
                fees = daily["fee"].sum()
                turnover = daily["gross"] / daily["opening_equity"]
                summaries.append({**metadata, "cumulative_return": normalized.iat[-1]-1,
                      "benchmark_return": benchmark.iat[-1]-1,
                      "excess_pp": (normalized.iat[-1]-benchmark.iat[-1])*100,
                      "annualized_excess": active.mean()*252, "information_ratio": active.mean()/active.std()*np.sqrt(252),
                      "max_drawdown": -(normalized/normalized.cummax()-1).min(),
                      "relative_max_drawdown": -(relative/relative.cummax()-1).min(),
                      "turnover_two_way": turnover.mean(), "fees_paid": fees,
                      "unresolved_exposures": len(exposure), "stale_mark_days": daily["stale_marks"].gt(0).sum(),
                      "rebalance_weeks": len(schedule)})
                navs.append(pd.DataFrame({"date": nav_days, "strategy_nav": normalized, "H00300_nav": benchmark,
                                          **metadata}).reset_index(drop=True))
                configs.extend([(name, buffer, fee, *row[1:]) for row in exposure])
                last = max(schedule)
                picks = pd.DataFrame([row for row in audit if row[1] == last],
                                     columns=["strategy", "date", "code", "model_rank", "already_held", "outside_top30"])
                require(len(picks) == 30, "Last simulated target book must contain 30 names")
                target_books.append(picks.assign(buffer_rank=buffer, fee_bps=fee))
                order_books.append(pd.DataFrame(orders, columns=ORDERS).assign(buffer_rank=buffer, fee_bps=fee))
                action_books.append(pd.DataFrame(actions, columns=[
                    "strategy", "date", "code", "ex_date", "stage", "amount", "shares", "fraction"]).assign(
                    buffer_rank=buffer, fee_bps=fee))
    unresolved = pd.DataFrame(configs, columns=[
        "strategy", "buffer_rank", "fee_bps", "date", "code", "opening_shares", "pending_shares", "category", "reason"])
    write(unresolved, output / "audit/unresolved_exposures.csv")
    if not unresolved.empty:
        print(unresolved[["date", "code", "category", "reason"]].drop_duplicates().to_string(index=False), flush=True)
        raise ValueError("Audit blocked publication. Resolve held corporate action(s) before revealing performance.")
    summary = pd.DataFrame(summaries)
    require(len(summary) == 14, "Expected fourteen frozen result rows")
    return summary, pd.concat(navs, ignore_index=True), pd.concat(target_books), pd.concat(order_books), pd.concat(action_books)


def archive(output, targets, orders, actions, ranks, weights, members, spec_hash):
    key = ["strategy", "buffer_rank", "fee_bps", "date", "code"]
    last = targets["date"].max()
    features = ranks.loc[ranks["execution_date"].eq(last),
                         ["strategy", "execution_date", "signal_date", "code", "score", *FEATURES, *STATES]]
    features = features.rename(columns={"execution_date": "date"})
    names = members[["signal_date", "code", "code_name"]]
    features = features.merge(names, on=KEYS, how="left", validate="many_to_one")
    targets = targets.merge(features, on=["strategy", "date", "code"], validate="many_to_one")
    targets["target_weight"] = 1/30
    targets["inclusion_reason"] = np.where(targets["outside_top30"], "retained_via_buffer", "top30_by_model")
    last_orders = orders.loc[orders["date"].eq(last)].merge(
        targets[key + ["target_weight", "inclusion_reason"]], on=key, how="left", validate="one_to_one")
    last_orders["inclusion_reason"] = last_orders["inclusion_reason"].fillna("out_of_target")
    last_orders = last_orders.merge(features, on=["strategy","date","code"], how="left", validate="many_to_one")
    last_orders["order_side"] = np.select([last_orders.requested_change.gt(0), last_orders.requested_change.lt(0)],
                                          ["BUY", "SELL"], default="NO_ORDER")
    merged_weights = weights.loc[weights["signal_date"].isin(targets["signal_date"])].pivot(
        index=["strategy", "signal_date"], columns="factor", values="effective_weight")
    merged_weights.columns = ["weight_" + field for field in merged_weights.columns]
    targets = targets.merge(merged_weights.reset_index(), on=["strategy","signal_date"], how="left", validate="many_to_one")
    for factor in FACTORS:
        targets[f"ridge_score_contribution_{factor}"] = (targets[f"pct_{factor}"]-.5) * targets[f"weight_{factor}"]
    year_end = ranks.loc[ranks["signal_date"].eq(END) & ranks["rank"].le(30)].copy()
    year_end = year_end.merge(names, on=KEYS, how="left", validate="many_to_one")
    year_end["target_weight_without_buffer"] = 1/30
    year_end["record_type"] = "year_end_signal_only_not_executed_in_holdout"
    write(year_end, output / "simulation/year_end_unbuffered_signal.csv")
    timestamp = datetime.now(timezone.utc).isoformat()
    for table in (targets, last_orders, orders, actions):
        table["record_type"] = "historical_paper_replay"
        table["archived_at_utc"] = timestamp
        table["spec_sha256"] = spec_hash
    for name, table in [("last_rebalance_targets", targets), ("last_rebalance_orders", last_orders),
                        ("full_simulated_orders", orders), ("corporate_action_ledger", actions)]:
        write(table, output / f"simulation/{name}.csv")
    destination = output / "simulation/paper_replay.sqlite"
    temp = destination.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    with closing(sqlite3.connect(temp)) as db, db:
        for name, table in [("last_rebalance_targets", targets), ("last_rebalance_orders", last_orders),
                            ("full_simulated_orders", orders), ("corporate_action_ledger", actions)]:
            table.to_sql(name, db, if_exists="fail", index=False)
        db.execute("CREATE UNIQUE INDEX target_key ON last_rebalance_targets(strategy, buffer_rank, fee_bps, date, code)")
        db.execute("CREATE TABLE metadata (spec_sha256 TEXT, record_type TEXT, archived_at_utc TEXT)")
        db.execute("INSERT INTO metadata VALUES (?, ?, ?)", (spec_hash, "historical_paper_replay", timestamp))
    temp.replace(destination)


def run(root, cache, output, verified):
    spec_path = Path(__file__).with_name("holdout_2025_spec.json")
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    digest = hashlib.sha256(spec_raw).hexdigest()
    require(not (output / "completed.json").exists(), "Holdout already revealed in this directory")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    registration = output / "registration.json"
    if registration.exists():
        require(json.loads(registration.read_text(encoding="utf-8"))["spec_sha256"] == digest
                and (output / "registered_spec.json").read_bytes() == spec_raw,
                "Registered holdout parameters cannot be changed in this directory")
    else:
        (output / "registered_spec.json").write_bytes(spec_raw)
        registration.write_text(json.dumps({"spec_sha256": digest,
            "registered_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    days, schedule, members, panel, eligible, states, labels, codes = build_inputs(cache)
    benchmark = pd.read_parquet(cache / "H00300.parquet").set_index("date").sort_index()
    require(benchmark["source"].nunique() == 1, "H00300 source is not unique")
    write(benchmark.reset_index(), output / "audit/h00300_input.csv")
    if verified:
        write(pd.read_csv(verified), output / "audit/verified_action_overrides.csv")
    event_rows, review, flags = corporate_actions(cache, codes, panel, days, verified)
    for name, table in [("action_review", review), ("price_reference_flags", flags), ("normalized_actions", event_rows)]:
        write(table, output / f"audit/{name}.csv")
    ld, lt, ls, groups, medians = dev_inputs(root)
    require(set(ld.signal_date).isdisjoint(eligible.signal_date), "Development and holdout eligible keys overlap")
    all_inputs = pd.concat([ld, eligible], ignore_index=True)
    all_labels = pd.concat([lt, labels], ignore_index=True)
    all_states = pd.concat([ls, states])
    fit_days = sorted(groups) + schedule["signal_date"].tolist()
    scores, weights, gain, margins, links = fit_all(all_inputs, all_labels, all_states, fit_days, spec,
                                                   set(groups), medians, groups)
    legacy_weights = pd.read_parquet(root / "data/processed/portfolio/walk_forward/walk_forward_state_weights.parquet")
    legacy_weights = legacy_weights.set_index("signal_date")
    recomputed = weights.loc[weights["strategy"].eq("state_ridge") & weights["signal_date"].isin(groups)].pivot(
        index="signal_date", columns="factor", values="effective_weight")
    for factor in FACTORS:
        np.testing.assert_allclose(recomputed[factor], legacy_weights.loc[recomputed.index, f"coef_{factor}"],
                                   atol=1e-12, rtol=0)
    require(links.training_exit_date.le(links.signal_date).all() and links.training_signal_date.lt(links.signal_date).all(),
            "Unmatured holdout labels")
    ranked = scores.merge(eligible[KEYS + FEATURES], on=KEYS, validate="many_to_one").merge(
        states.reset_index(), on="signal_date", validate="many_to_one").merge(schedule, on="signal_date", validate="many_to_one")
    ranked = ranked.sort_values(["strategy","signal_date","score","code"], ascending=[True, True, False, True])
    ranked["rank"] = ranked.groupby(["strategy","signal_date"]).cumcount()+1
    ranked["fit_status"] = "fitted"
    write(links, output / "audit/mature_training_links.parquet")
    plot_and_explain(output, weights, gain, margins)
    summary, nav, targets, orders, actions = run_portfolios(
        ranked, panel, event_rows, review, days, benchmark["close"], spec, output)
    ic = eval_returns(ranked, labels)
    stats = ic.groupby("strategy")["rank_ic"].agg(rank_ic="mean", ic_sd="std", ic_weeks="count").reset_index()
    stats["icir"] = stats["rank_ic"] / stats["ic_sd"]
    summary = summary.merge(stats, on="strategy", validate="many_to_one")
    write(ranked, output / "data/full_rankings.parquet")
    write(ic, output / "reports/weekly_rank_ic.csv")
    write(summary, output / "reports/holdout_summary.csv")
    write(nav, output / "reports/holdout_nav.parquet")
    archive(output, targets, orders, actions, ranked, weights, members, digest)
    (output / "completed.json").write_text(json.dumps({"spec_sha256": digest, "rows": len(summary),
        "holdout_rebalance_weeks": int(summary.rebalance_weeks.iat[0]),
        "record_type": "historical_paper_replay"}, indent=2), encoding="utf-8")
    print(summary.round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verified-actions", type=Path,
                        default=Path(__file__).with_name("verified_actions_2025.csv"))
    options = parser.parse_args()
    project = options.project_root.resolve()
    run(project, options.cache or project / "data/raw/holdout_2025",
        options.output_dir or project / "reports/holdout_2025", options.verified_actions)

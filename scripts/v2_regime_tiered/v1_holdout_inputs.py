from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]
FACTORS = ["RET_5", "VOL_60", "CORR_PV_10"]
FEATURES = [f"pct_{factor}" for factor in FACTORS]
STATES = ["market_return_20d", "market_volatility_20d", "market_breadth_20d"]
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
ANCHOR = pd.Timestamp("2024-12-31")


def require(test, message):
    if not test:
        raise ValueError(message)


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_parquet(path, index=False)


def build_inputs(cache):
    from audit_daily_bars import load_bars, valid_ohlc
    from build_candidate_factors import build_candidate_panel
    from build_fixed_scores import build_scores
    from build_forward_labels import build_holding_windows, join_open_price

    schedule = pd.read_parquet(cache / "schedule.parquet").sort_values("signal_date")
    calendar = pd.read_parquet(cache / "calendar.parquet")
    days = pd.DatetimeIndex(calendar.loc[calendar["is_trading_day"].astype(str).eq("1"), "calendar_date"])
    days = days[(days <= END)].sort_values()
    require(days.is_unique and days[-1] == END and ANCHOR in days, "Incomplete trading calendar")
    members = pd.concat([pd.read_parquet(cache / "members" / f"{day:%Y-%m-%d}.parquet")
                         for day in schedule["signal_date"]], ignore_index=True)
    require(not members.duplicated(KEYS).any() and members.groupby("signal_date").size().eq(300).all()
            and pd.to_datetime(members["updateDate"]).le(members["signal_date"]).all(),
            "Invalid historical member snapshots")
    codes = sorted(members["code"].unique())
    missing = [f"{part}/{code}" for code in codes for part in ("raw", "adjusted", "actions")
               if not (cache / part / f"{code}.parquet").exists()]
    require(not missing, f"Missing cache fragments: {', '.join(missing[:30])}")
    raw = load_bars(cache / "raw", codes, "raw", "3")
    adjusted = load_bars(cache / "adjusted", codes, "adj", "1")
    panel = raw.merge(adjusted, on=["date", "code"], how="outer", validate="one_to_one", indicator=True)
    panel["raw_ohlc_ok"] = valid_ohlc(panel, "raw")
    panel["adj_ohlc_ok"] = valid_ohlc(panel, "adj")
    panel["signal_bar_usable"] = (panel["_merge"].eq("both") & panel["raw_ohlc_ok"]
                                 & panel["adj_ohlc_ok"] & panel["tradestatus"].eq("1"))
    candidates, _ = build_candidate_panel(panel, members[KEYS], days)
    scores = build_scores(candidates, 30)
    scores = scores.loc[scores["eligible"], KEYS + ["score_without_bias", *FEATURES]]
    require(scores.groupby("signal_date").size().reindex(schedule["signal_date"]).ge(280).all(),
            "At least one holdout signal has fewer than 280 eligible member scores")
    close = pd.read_parquet(cache / "price_index.parquet")
    close["date"] = pd.to_datetime(close["date"])
    require(close["date"].is_unique and close["code"].eq("sh.000300").all(), "Invalid price index")
    close = pd.to_numeric(close.set_index("date")["close"]).reindex(days)
    require(np.isfinite(close).all() and close.gt(0).all(), "Missing price index sessions")
    states = pd.DataFrame({STATES[0]: close.pct_change(20), STATES[1]: close.pct_change().rolling(20).std()})
    breadth = candidates["RET_20"].gt(0).where(candidates["RET_20"].notna())
    states[STATES[2]] = breadth.groupby(candidates["signal_date"]).mean()
    states = states.reindex(schedule["signal_date"]).astype(float)
    require(np.isfinite(states).all().all(), "Missing state features")
    windows = build_holding_windows(schedule)
    labels = candidates[KEYS].merge(windows, on="signal_date", validate="many_to_one")
    for side in ("entry", "exit"):
        labels = join_open_price(labels, panel[["date", "code", "adj_open", "tradestatus", "adj_ohlc_ok"]], side)
    valid = labels["exit_date"].le(END)
    for side in ("entry", "exit"):
        valid &= (labels[f"{side}_price_ok"].eq(True) & labels[f"{side}_status"].eq("1")
                  & labels[f"{side}_open_adj"].gt(0))
    labels["forward_return"] = (labels["exit_open_adj"] / labels["entry_open_adj"] - 1).where(valid)
    return days, schedule, members, panel, scores, states, labels[KEYS + ["exit_date", "forward_return"]], codes


def corporate_actions(cache, codes, panel, days, overrides=None):
    from normalize_corporate_actions import normalize_event, EVENT_COLUMNS

    inputs = pd.concat([pd.read_parquet(cache / "actions" / f"{code}.parquet") for code in codes],
                       ignore_index=True)
    inputs["ex_date"] = pd.to_datetime(inputs["dividOperateDate"], errors="coerce")
    inputs = inputs.loc[inputs["ex_date"].between(START, END)]
    events, review = [], []
    verified = {}
    if overrides:
        manual = pd.read_csv(overrides, dtype=str).fillna("")
        for item in manual.to_dict("records"):
            day = pd.Timestamp(item["ex_date"])
            require(item["source_url"].startswith("https://") and item["reason"], "Missing override evidence")
            key = (item["code"], day)
            require(key not in verified and day in days and day >= START, "Invalid or duplicate override date")
            if item["resolution"] == "cash":
                record_date, pay_date = pd.Timestamp(item["record_date"]), pd.Timestamp(item["pay_date"])
                from decimal import Decimal
                require(record_date < day <= pay_date and Decimal(item["cash_per_share"]) > 0,
                        "Invalid override cash terms")
                events.append(dict(code=item["code"], ex_date=day, record_date=record_date, pay_date=pay_date,
                                   stock_listing_date=pd.NaT, cash_per_share=item["cash_per_share"],
                                   bonus_shares_per_share="0", reserve_shares_per_share="0",
                                   event_type="cash_only", source_rows=0))
            else:
                require(item["resolution"] == "no_portfolio_entitlement", "Unknown override resolution")
            verified[key] = item
    for (code, ex_date), group in inputs.groupby(["code", "ex_date"]):
        if (code, ex_date) in verified:
            continue
        try:
            events.append(normalize_event(code, ex_date, group))
        except ValueError as error:
            review.append((code, ex_date, "source_terms_conflict", str(error)))
    events = pd.DataFrame(events, columns=EVENT_COLUMNS)
    prev = panel[["date", "code", "raw_close", "raw_ohlc_ok"]].rename(
        columns={"date": "prior", "raw_close": "previous_close", "raw_ohlc_ok": "previous_valid"})
    current = panel.loc[panel["date"].between(START, END) & panel["raw_ohlc_ok"],
                        ["date", "code", "preclose"]].copy()
    current["prior"] = current["date"].map(pd.Series(days[:-1], index=days[1:]))
    current = current.merge(prev, on=["prior", "code"], how="left", validate="one_to_one")
    current["gap"] = current["preclose"] - current["previous_close"]
    flags = current.loc[current["previous_valid"].eq(True) & current["preclose"].gt(0) & current["gap"].abs().gt(.005)]
    known = set(zip(events["code"], events["ex_date"])) | set(verified) | {(code, day) for code, day, _, _ in review}
    for item in flags.itertuples(index=False):
        if (item.code, item.date) not in known:
            review.append((item.code, item.date, "price_flag_only", f"reference adjustment {item.gap:+.4f}"))
    return events, pd.DataFrame(review, columns=["code", "ex_date", "category", "reason"]), flags

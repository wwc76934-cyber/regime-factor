"""Realized value-minus-trend returns from cohorts formed 60 trading sessions earlier."""
import numpy as np
import pandas as pd


STYLE = "value_minus_trend_60d"
LOOKBACK = 60


def build_style_state(source, dates, members, partitions, output_dates, normalize_value):
    from build_v3_factors_and_labels import stock_history

    quotes = stock_history(source, partitions, "raw", ["date", "code", "amount", "turn", "peTTM", "tradestatus"], "3")
    fields = ["date", "code", "high", "low", "open", "close"]
    adjusted = stock_history(source, partitions, "adjusted", fields, "1")
    ohlc = adjusted[fields[2:]].apply(pd.to_numeric, errors="coerce")
    adjusted["close"] = ohlc.close.where(ohlc.gt(0).all(axis=1) & np.isfinite(ohlc).all(axis=1)
                                        & ohlc.high.ge(ohlc.max(axis=1)) & ohlc.low.le(ohlc.min(axis=1)))

    close = adjusted.pivot(index="date", columns="code", values="close").reindex(dates)
    stock = close.pct_change(fill_method=None)
    vol = stock.rolling(60).std()
    momentum = close.shift(20) / close.shift(120) - 1
    pe = pd.to_numeric(quotes.peTTM, errors="coerce")
    amount = pd.to_numeric(quotes.amount, errors="coerce")
    turn = pd.to_numeric(quotes.turn, errors="coerce")
    active = quotes.tradestatus.astype(str).eq("1")
    valid = active & amount.gt(0) & turn.gt(0) & turn.le(100) & np.isfinite(amount) & np.isfinite(turn)
    daily = quotes[["date", "code"]].assign(mv=(amount / (turn / 100)).where(valid), peTTM=pe, active=active)

    def wide(column):
        return daily.pivot(index="date", columns="code", values=column).reindex(dates)

    cap = wide("mv").rolling(20, min_periods=10).median()
    price = close.ffill()
    pe = wide("peTTM")
    usable = wide("active").eq(True)
    roster = {day: group.code.tolist() for day, group in members.groupby("signal_date")}
    snapshots = pd.DatetimeIndex(sorted(roster))
    rows = []
    for day in output_dates:
        pos = dates.get_loc(day)
        base = dates[pos - LOOKBACK] if pos >= LOOKBACK else pd.NaT
        snapshot_pos = snapshots.searchsorted(base, side="right") - 1 if pd.notna(base) else -1
        if snapshot_pos < 0:
            rows.append((day, base, pd.NaT, pd.NaT, 0, 0, np.nan, np.nan, np.nan, 0, 0, "no_point_in_time_roster"))
            continue
        snapshot = snapshots[snapshot_pos]
        names = roster[snapshot]
        current = close.loc[base].reindex(names)
        available = usable.loc[base].reindex(names, fill_value=False) & current.notna()
        candidates = pd.DataFrame({"signal_date": base, "code": names,
            "peTTM": pe.loc[base].reindex(names).to_numpy(), "VOL_60": vol.loc[base].reindex(names).to_numpy(),
            "float_mv": cap.loc[base].reindex(names).to_numpy(), "signal_bar_usable": available.to_numpy()})
        candidates["value"] = normalize_value(candidates, "peTTM")
        candidates["trend"] = momentum.loc[base].reindex(names).where(available).to_numpy()
        picks = []
        for field in ["value", "trend"]:
            ranked = candidates.dropna(subset=[field]).sort_values([field, "code"], ascending=[False, True])
            picks.append(ranked.code.head(len(ranked) // 3).tolist())
        endpoint = price.loc[day]
        realized = [float((endpoint.reindex(names) / current.reindex(names) - 1).mean()) if names else np.nan
                    for names in picks]
        stale = [int(close.loc[day].reindex(names).isna().sum()) for names in picks]
        status = "ok" if np.isfinite(realized).all() else "insufficient_formation_history"
        rows.append((day, base, snapshot, day, *map(len, picks), *realized,
                     realized[0] - realized[1], *stale, status))
    audit = pd.DataFrame(rows, columns=["signal_date", "formation_date", "roster_snapshot_date", "return_end_date",
        "value_selected", "trend_selected", "value_60d_return", "trend_60d_return", STYLE,
        "value_stale_endpoint", "trend_stale_endpoint", "status"])
    valid = audit.loc[audit[STYLE].notna()]
    safe = (valid.roster_snapshot_date.le(valid.formation_date) & valid.formation_date.lt(valid.return_end_date) &
            valid.return_end_date.eq(valid.signal_date)).all()
    spacing = dates.get_indexer(valid.return_end_date) - dates.get_indexer(valid.formation_date)
    if not safe or not (spacing == LOOKBACK).all():
        raise ValueError("Style cohort uses an invalid formation or endpoint")
    return audit

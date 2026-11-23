"""V3.1 execution snapshot with verified cash and share-exchange handling."""


from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal as D, ROUND_HALF_UP




import numpy as np
import pandas as pd


ZERO = D("0")
CENT = D("0.01")
FEE = D("0.001")
GEM_CHANGE = pd.Timestamp("2020-08-24")
RIGHTS_CODE = "sh.600030"
RIGHTS_RECORD = pd.Timestamp("2022-01-18")
RIGHTS_PAY = pd.Timestamp("2022-01-25")
RIGHTS_EX = pd.Timestamp("2022-01-27")
RIGHTS_LIST = pd.Timestamp("2022-02-15")
RIGHTS_RATIO = D("0.15")
RIGHTS_PRICE = D("14.43")
# CITICS A-share rights issue, announcements 2022-015 and 2022-038.
RIGHTS_KEY = (RIGHTS_CODE, RIGHTS_EX)


@dataclass(frozen=True)
class Exchange:
    old: str
    new: str
    record: pd.Timestamp
    retire: pd.Timestamp
    confirmation: pd.Timestamp
    list_date: pd.Timestamp
    ratio: D


EXCHANGES = (
    Exchange("sh.600005","sh.600019",pd.Timestamp("2017-02-14"),pd.Timestamp("2017-02-15"),
             pd.Timestamp("2017-02-17"),pd.Timestamp("2017-02-27"),D(".56")),
    Exchange("sh.600837","sh.601211",pd.Timestamp("2025-03-03"),pd.Timestamp("2025-03-04"),
             pd.Timestamp("2025-03-13"),pd.Timestamp("2025-03-17"),D(".62")),
)


def verified_cash_event(events, review):
    code, ex_date = "sh.600519", pd.Timestamp("2022-12-27")
    old = events.code.eq(code) & events.ex_date.eq(ex_date)
    unresolved = review.code.eq(code) & review.ex_date.eq(ex_date)
    record = {"code":code, "ex_date":ex_date, "record_date":pd.Timestamp("2022-12-26"),
              "pay_date":ex_date, "cash_per_share":"21.91", "bonus_shares_per_share":"0",
              "reserve_shares_per_share":"0", "stock_listing_date":pd.NaT,
              "event_type":"cash_only", "source_rows":1}
    if old.any():
        if old.sum()!=1 or any(events.loc[old,k].iat[0]!=record[k] for k in ("record_date","pay_date")) \
                or D(events.loc[old,"cash_per_share"].iat[0])!=D(record["cash_per_share"]):
            raise ValueError("Existing dividend conflicts with the Moutai special dividend announcement")
    else:
        events = pd.concat([events,pd.DataFrame([record])],ignore_index=True)
    return events,review.loc[~unresolved].copy()


def money(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def price(value):
    return D(str(value)) if pd.notna(value) and np.isfinite(value) and value > 0 else None


def stock_rules(code, day, quote):
    if code.startswith(("sh.688", "sh.689")):
        return 200, 1, D("0.20")
    if code.startswith(("sz.300", "sz.301", "sz.302")) and day >= GEM_CHANGE:
        return 100, 100, D("0.20")
    is_st = str(quote.isST).strip() == "1"
    return 100, 100, D("0.05" if is_st else "0.10")


def simulate(name, days, bars, targets, events, review, initial_cash, fee_rate=FEE,
             select_targets=None, weight_targets=None, initial_marks=None):
    cash = initial_cash
    positions = {}
    marks = dict(initial_marks or {})
    entitlements = {}
    receivables = {}
    pending_shares = {}
    swap_eligible = {}
    pending_exchanges = {}
    rights_eligible = 0
    rights_paid_shares = 0
    rights_prepaid = ZERO
    nav, orders, holdings, actions, exposure = [], [], [], [], []
    record_events, ex_events = defaultdict(list), defaultdict(list)
    unresolved_by_day = defaultdict(list)
    for event in events.itertuples(index=False):
        record_events[event.record_date].append(event)
        ex_events[event.ex_date].append(event)
    for item in review.itertuples(index=False):
        unresolved_by_day[item.ex_date].append(item)

    def log_action(day, code, event, stage, amount=ZERO, shares=0, fraction=ZERO):
        actions.append((name, day, code, event, stage, float(amount), shares, str(fraction)))

    for day in days:
        quotes = bars.get(day, {})
        day_fee = ZERO
        gross_traded = ZERO
        stale = 0

        for item in unresolved_by_day[day]:
            if (item.code, item.ex_date) == RIGHTS_KEY:
                continue
            shares = positions.get(item.code, 0)
            locked = sum(q for c, _, _, q in pending_shares.values() if c == item.code)
            if shares or locked:
                exposure.append((name, day, item.code, shares, locked, item.category, item.reason))

        for event in EXCHANGES:
            if day == event.retire and event in swap_eligible:
                owned = swap_eligible.pop(event)
                if positions.get(event.old,0)!=owned or any(code==event.old for code,_,_,_ in pending_shares.values()):
                    raise ValueError(f"Unreconciled held entitlement for {event.old}")
                exact = event.ratio*owned
                quantity = int(exact)
                if D(quantity)!=exact:
                    raise ValueError(f"{event.old}: exchange creates an unresolved fractional entitlement")
                if event.old not in marks:
                    raise ValueError(f"{event.old}: missing pre-exchange carry mark")
                carry = money(marks[event.old]*owned)
                del positions[event.old]
                pending_exchanges[event] = (quantity,carry)
                log_action(day,event.old,event.retire,"swap_old_retired",carry,owned)
            if day == event.confirmation and event in pending_exchanges:
                quantity,carry = pending_exchanges[event]
                log_action(day,event.new,event.retire,"swap_confirmed",carry,quantity)
            if day == event.list_date and event in pending_exchanges:
                quantity,carry = pending_exchanges.pop(event)
                quote = quotes.get(event.new)
                if quote is None or price(quote.raw_open) is None or str(quote.tradestatus).strip()!="1":
                    raise ValueError(f"{event.new}: listed replacement shares have no executable opening mark")
                positions[event.new] = positions.get(event.new,0)+quantity
                log_action(day,event.new,event.retire,"swap_new_listed",carry,quantity)

        # Record-date holdings from an earlier close determine the entitlement.
        for event in ex_events[day]:
            key = (event.code, event.ex_date)
            owned = entitlements.pop(key, 0)
            if not owned:
                continue
            dividend = money(D(event.cash_per_share) * owned)
            if dividend:
                receivables[key] = (event.code, event.ex_date, event.pay_date, dividend)
                log_action(day, event.code, event.ex_date, "cash_receivable", dividend)
            ratio = D(event.bonus_shares_per_share) + D(event.reserve_shares_per_share)
            if ratio:
                exact = ratio * owned
                awarded = int(exact)
                log_action(day, event.code, event.ex_date, "stock_entitlement",
                           shares=awarded, fraction=exact - awarded)
                if awarded:
                    pending_shares[key] = (event.code, event.ex_date, event.stock_listing_date, awarded)

        # The paid subscription is carried at cost until this successful ex-date.
        if day == RIGHTS_EX and rights_paid_shares:
            pending_shares[RIGHTS_KEY] = (RIGHTS_CODE, RIGHTS_EX, RIGHTS_LIST, rights_paid_shares)
            log_action(day, RIGHTS_CODE, RIGHTS_EX, "rights_confirmed",
                       rights_prepaid, rights_paid_shares)
            rights_prepaid = ZERO
            rights_paid_shares = 0

        # Shares become tradable at the open on the listing day.
        for key, (code, ex_date, listing, qty) in list(pending_shares.items()):
            if listing <= day:
                positions[code] = positions.get(code, 0) + qty
                del pending_shares[key]
                stage = "rights_listed" if key == RIGHTS_KEY else "shares_listed"
                log_action(day, code, ex_date, stage, shares=qty)

        def mark(code, column):
            quote = quotes.get(code)
            value = price(getattr(quote, column)) if quote is not None else None
            if value is not None:
                return value
            if code not in marks:
                raise ValueError(f"{day.date()} {code} has no price for marking")
            return marks[code]

        def stock_value(column):
            traded = sum((money(mark(code, column) * qty) for code, qty in positions.items()), ZERO)
            locked = sum(
                (money(mark(code, column) * qty) for code, _, _, qty in pending_shares.values()),
                ZERO,
            )
            return traded, locked + sum((carry for _,carry in pending_exchanges.values()),ZERO)

        if day in targets:
            wanted = targets[day]
            if select_targets is not None:
                wanted = select_targets(day, wanted, positions)
            actual_value, locked_value = stock_value("raw_open")
            opening_equity = (
                cash + actual_value + locked_value + rights_prepaid
                + sum((item[3] for item in receivables.values()), ZERO)
            )
            start_cash = cash
            signed_gross_total = ZERO
            weights = weight_targets(day, positions, pending_shares, mark, opening_equity) if weight_targets else None
            if weights is not None:
                wanted = {code: rank for rank, code in enumerate(weights, 1)}
            per_stock = opening_equity / len(wanted)
            locked_by_code = defaultdict(int)
            for code, _, _, qty in pending_shares.values():
                locked_by_code[code] += qty
            codes = set(positions) | set(wanted)
            plans = []
            for code in codes:
                quote = quotes.get(code)
                px = price(quote.raw_open) if quote is not None else None
                previous = positions.get(code, 0)
                desired = None
                minimum, step, limit_rate = (100, 100, D("0.10"))
                if quote is not None:
                    minimum, step, limit_rate = stock_rules(code, day, quote)
                if code not in wanted:
                    desired = 0
                elif px is not None:
                    sizing_cost = px * (1 + fee_rate) if day == days[0] else px
                    budget = per_stock if weights is None else opening_equity * D(str(weights[code]))
                    desired = max(0, int(budget // sizing_cost) - locked_by_code[code])
                change = desired - previous if desired is not None else 0
                if change > 0:
                    change = (change // step) * step
                    if change < minimum:
                        change = 0
                elif change < 0 and desired > 0:
                    # Keep odd residual lots until a full exit when needed.
                    change = -((-change // step) * step)
                    if -change < minimum:
                        change = 0
                plans.append((code, quote, px, previous, desired, change, minimum, step, limit_rate))

            plans.sort(key=lambda p: (0 if p[5] < 0 else 1, wanted.get(p[0], 10**9), p[0]))
            for code, quote, px, old, desired, request, minimum, step, limit_rate in plans:
                fill = 0
                if quote is None or px is None:
                    status = "missing_open"
                elif str(quote.tradestatus).strip() != "1":
                    status = "not_trading"
                elif request == 0:
                    status = "below_minimum_order" if desired != old or (old == 0 and code in wanted) else "unchanged"
                else:
                    ref = price(quote.preclose)
                    if ref is None:
                        status = "missing_reference"
                    elif request > 0 and px >= money(ref * (1 + limit_rate)):
                        status = "buy_at_limit"
                    elif request < 0 and px <= money(ref * (1 - limit_rate)):
                        status = "sell_at_limit"
                    else:
                        fill = request
                        if request > 0:
                            affordable = int(cash // (px * step * (1 + fee_rate))) * step
                            fill = min(affordable, request)
                            if fill < minimum:
                                fill = 0
                        status = "filled" if fill == request else "cash_limited"

                gross = money(px * fill) if fill else ZERO
                fee = money(abs(gross) * fee_rate)
                cash -= gross + fee
                signed_gross_total += gross
                gross_traded += abs(gross)
                day_fee += fee
                new = old + fill
                if cash < 0 or new < 0:
                    raise ValueError(f"{name} {day.date()} {code}: negative cash or shares")
                if new:
                    positions[code] = new
                else:
                    positions.pop(code, None)
                orders.append((name, day, code, old, desired, request, fill, new,
                               float(px) if px is not None else np.nan, float(gross),
                               float(fee), float(cash), status))

            if cash != start_cash - signed_gross_total - day_fee:
                raise ValueError(f"{name} {day.date()}: order cash does not reconcile")
        else:
            opening_equity = None

        # Credit cash only after trading on the pay date.
        for key, (code, ex_date, pay_date, amount) in list(receivables.items()):
            if pay_date <= day:
                cash += amount
                del receivables[key]
                log_action(day, code, ex_date, "cash_paid", amount)

        # Simulation policy: subscribe in full at the close of the final payment
        # day if available cash covers it. Never use later information or borrow.
        if day == RIGHTS_PAY and rights_eligible:
            exact = RIGHTS_RATIO * rights_eligible
            count = int(exact)
            cost = money(RIGHTS_PRICE * count)
            if count and cash >= cost:
                cash -= cost
                rights_prepaid = cost
                rights_paid_shares = count
                log_action(day, RIGHTS_CODE, RIGHTS_EX, "rights_subscribed",
                           cost, count, exact - count)
            else:
                log_action(day, RIGHTS_CODE, RIGHTS_EX, "rights_not_subscribed",
                           cost, count, exact - count)

        locked_by_code = defaultdict(int)
        for code, _, _, qty in pending_shares.values():
            locked_by_code[code] += qty
        for code in set(positions) | set(locked_by_code):
            quote = quotes.get(code)
            fresh = price(quote.raw_close) if quote is not None else None
            close = mark(code, "raw_close")
            stale += fresh is None
            marks[code] = close
            shares = positions.get(code, 0)
            pending_qty = locked_by_code[code]
            holdings.append((name, day, code, shares, pending_qty,
                             float(close), float(money(close * shares) + money(close * pending_qty)),
                             fresh is None))

        traded_value, pending_value = stock_value("raw_close")
        cash_due = sum((item[3] for item in receivables.values()), ZERO)
        equity = cash + traded_value + pending_value + cash_due + rights_prepaid
        nav.append((name, day, float(cash), float(traded_value), float(pending_value),
                    float(cash_due), float(rights_prepaid), float(equity), float(day_fee), float(gross_traded),
                    float(opening_equity) if opening_equity is not None else np.nan, stale))

        # Snapshot at the end of the record date, after any orders on that day.
        for event in record_events[day]:
            entitlements[(event.code, event.ex_date)] = (
                positions.get(event.code, 0) + locked_by_code[event.code]
            )

        for event in EXCHANGES:
            if day == event.record:
                own = positions.get(event.old,0)
                if own:
                    swap_eligible[event] = own
                    log_action(day,event.old,event.retire,"swap_entitlement",shares=own)

        if day == RIGHTS_RECORD:
            rights_eligible = positions.get(RIGHTS_CODE, 0)
            if rights_eligible:
                log_action(day, RIGHTS_CODE, RIGHTS_EX, "rights_record_date",
                           shares=rights_eligible)

    return nav, orders, holdings, actions, exposure



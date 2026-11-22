"""V3 snapshot of V1 cash, lot, limit and corporate-action execution with weighted targets."""


from collections import defaultdict
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
            return traded, locked

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

        if day == RIGHTS_RECORD:
            rights_eligible = positions.get(RIGHTS_CODE, 0)
            if rights_eligible:
                log_action(day, RIGHTS_CODE, RIGHTS_EX, "rights_record_date",
                           shares=rights_eligible)

    return nav, orders, holdings, actions, exposure



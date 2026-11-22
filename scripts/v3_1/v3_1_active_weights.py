"""Structured QP and deterministic lot mapping for the V3.1 optimizer."""
from decimal import Decimal

import numpy as np
from scipy.optimize import brentq


ACTIVE_LIMIT = .03
SINGLE_LIMIT = .06
RISK_AVERSION = 6.


def end_beta(beta, low, high, ascending):
    w = low.copy()
    remaining = 1 - w.sum()
    for i in np.argsort(beta if ascending else -beta, kind="stable"):
        add = min(high[i] - low[i], remaining)
        w[i] += add
        remaining -= add
        if remaining <= 1e-12:
            break
    return float(w @ beta)


def solve_active(anchor, mu, beta, idio_var, market_var, previous, cost, lower_beta, upper_beta):
    """Minimize diagonal plus single-factor risk, minus alpha, plus exact L1 turnover cost."""
    low = np.maximum(0, anchor - ACTIVE_LIMIT)
    high = np.minimum(SINGLE_LIMIT, anchor + ACTIVE_LIMIT)
    if (low > high + 1e-12).any() or low.sum() > 1 + 1e-12 or high.sum() < 1 - 1e-12:
        raise ValueError("Infeasible full-investment active weight bounds")
    feasible_low = end_beta(beta, low, high, True)
    feasible_high = end_beta(beta, low, high, False)
    target_low = float(np.clip(lower_beta, feasible_low, feasible_high))
    target_high = float(np.clip(upper_beta, feasible_low, feasible_high))
    infeasible = upper_beta < feasible_low-1e-9 or lower_beta > feasible_high+1e-9
    diagonal = RISK_AVERSION * idio_var
    market_risk = RISK_AVERSION * market_var
    anchor_beta = float(anchor @ beta)
    kink = np.clip(previous, low, high)

    def weight_at(theta):
        linear = mu - theta * beta + diagonal * (anchor - kink)

        def weight(nu):
            shift = linear - nu
            shrink = np.maximum(shift-cost, 0) + np.minimum(shift+cost, 0)
            return np.clip(kink + shrink / diagonal, low, high)

        minimum = np.min(linear - cost - diagonal*(high-kink)) - 1e-5
        maximum = np.max(linear + cost + diagonal*(kink-low)) + 1e-5
        nu = brentq(lambda v: weight(v).sum() - 1, minimum, maximum, xtol=1e-12)
        return weight(nu)

    limit = market_risk * max(abs(feasible_low-anchor_beta), abs(feasible_high-anchor_beta)) + 1e-6
    theta = brentq(lambda t: t-market_risk*(weight_at(t)@beta-anchor_beta), -limit, limit, xtol=1e-11)
    w = weight_at(theta)
    goal = float(np.clip(w@beta,target_low,target_high))
    if abs(w@beta-goal) > 1e-8:
        direction = 1 if w@beta>goal else -1
        outside = theta + .01*direction
        gap = lambda t: float(weight_at(t)@beta-goal)
        while direction*gap(outside) > 1e-9 and abs(outside-theta) <= 1e4:
            outside = theta+2*(outside-theta)
        theta = outside if direction*gap(outside) >= 0 else brentq(gap,*sorted([theta,outside]),xtol=1e-11)
        w = weight_at(theta)
    if (abs(w.sum()-1) > 1e-7 or (w < low-1e-8).any() or (w > high+1e-8).any()
            or not target_low-1e-7 <= w@beta <= target_high+1e-7):
        raise ValueError("Structured solver exceeded tolerances")
    return w, feasible_low, feasible_high, infeasible


def lot_targets(data, positions, locked, quotes, day, first_day, equity, fee, rules, money):
    """Map continuous weights to orders, then reinvest affordable rounding residuals."""
    reserve = .003
    account = float(equity)
    actual = {}
    upgrades = []
    value = 0.
    fees = 0.
    for code, row in data.items():
        old = positions.get(code, 0)
        barred = locked.get(code, 0)
        quote = quotes.get(code)
        stock = float(quote.raw_open) if quote is not None else np.nan
        if not np.isfinite(stock) or stock <= 0:
            value += (old+barred)*row["mark"]
            if old and row["weight"] > 0:
                actual[code] = row["weight"]  # The legacy engine holds when there is no usable opening quote.
            continue
        minimum, step, limit = rules(code, day, quote)
        base = float(quote.preclose)
        trading = str(quote.tradestatus).strip() == "1" and np.isfinite(base) and base > 0
        can_buy = trading and stock < float(money(Decimal(str(base))*(1+limit)))
        can_sell = trading and stock > float(money(Decimal(str(base))*(1-limit)))
        ideal = max(0.,account*row["weight"]/stock-barred)
        if ideal >= old and can_buy:
            add = int((ideal-old)//step)*step
            target = old + (add if add>=minimum else 0)
        elif ideal < old and can_sell:
            cut = int((old-ideal)//step)*step
            target = 0 if ideal < 1 else old - (cut if cut>=minimum else 0)
        else:
            target = old
        budget_share = stock*(1+fee if day==first_day else 1)
        if target or barred:
            actual[code] = (target+barred+.1)*budget_share/account
        value += (target+barred)*stock
        fees += abs(target-old)*stock*fee
        next_lot = step if target-old>=minimum else (minimum+step-1)//step*step
        maximum = account*row["cap"]/stock-barred
        if can_buy and target>=old and target < ideal and target+next_lot <= maximum+1e-9:
            penalty = ((target+next_lot-ideal)**2 - (target-ideal)**2)*stock**2
            upgrades.append((penalty/(next_lot*stock),code,next_lot,stock,budget_share,target,barred,old))
    room = account*(1-reserve)-value-fees
    for _,code,quantity,stock,budget_share,target,barred,old in sorted(upgrades):
        inc = stock*quantity*(1+fee)
        if inc <= room:
            actual[code] = (target+quantity+barred+.1)*budget_share/account
            room -= inc
    return actual

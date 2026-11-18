import numpy as np
import pandas as pd

from holdout_data import FACTORS, FEATURES, KEYS, STATES, require


def moments_by_week(eligible, labels):
    joined = eligible.merge(labels, on=KEYS, validate="one_to_one")
    joined = joined.loc[joined["exit_date"].notna() & np.isfinite(joined["forward_return"])]
    moments = {}
    for day, group in joined.groupby("signal_date", sort=True):
        if len(group) < 50:
            continue
        require(group["exit_date"].nunique() == 1, "Inconsistent label exits")
        xs = group[FEATURES].to_numpy(dtype=float)
        ys = group["forward_return"].rank(pct=True).to_numpy()
        xc, yc = xs - xs.mean(axis=0), ys - ys.mean()
        moments[day] = (group["exit_date"].iat[0], xs - 0.5, ys, xc.T @ xc / len(xs), xc.T @ yc / len(xs))
    return moments


def fit_all(eligible, labels, states, signals, spec, dev_dates, thresholds, dev_links=None):
    import lightgbm as lgb

    settings = spec["lightgbm"].copy()
    require(settings.pop("version") == lgb.__version__, "Unexpected LightGBM version")
    rounds = settings.pop("rounds")
    training = spec["training"]
    ridge = spec["ridge"]
    moments = moments_by_week(eligible, labels)
    weeks = dict(tuple(eligible.groupby("signal_date", sort=False)))
    full_scores, weights, importance, margins, link_rows = [], [], [], [], []
    for pos, signal in enumerate(signals, start=1):
        selected = [day for day, item in moments.items() if day < signal and item[0] <= signal]
        selected = selected[-training["window_weeks"]:]
        require(len(selected) >= training["min_weeks"], "Insufficient mature training weeks")
        development = signal in dev_dates
        if development and dev_links is not None:
            expected = dev_links[signal]
            actual = [(day, moments[day][0], len(moments[day][1])) for day in selected]
            require(actual == expected, f"Could not replicate original training on {signal.date()}")
        now = states.loc[signal, STATES].to_numpy(dtype=float)
        past = states.loc[selected, STATES].to_numpy(dtype=float)
        center, sd = past.mean(axis=0), past.std(axis=0)
        safe = np.where(sd > 0, sd, 1)
        cap = training["state_z_max"]
        def transform(values):
            return np.clip(np.where(sd > 0, (values - center) / safe, 0), -cap, cap)

        z_past, z_now = transform(past), transform(now)
        normal, response = np.zeros((12, 12)), np.zeros(12)
        train_x, train_y, row_weights = [], [], []
        for day, z in zip(selected, z_past, strict=True):
            exit_date, xs, ys, gram, rhs = moments[day]
            state = np.r_[1.0, z]
            normal += np.kron(np.outer(state, state), gram)
            response += np.kron(state, rhs)
            train_x.append(np.c_[xs, np.broadcast_to(z, (len(xs), 3))])
            train_y.append(ys)
            row_weights.append(np.full(len(xs), 1 / len(xs)))
            if not development:
                link_rows.append((signal, day, exit_date, len(xs)))
        normal /= len(selected)
        response /= len(selected)
        active_week = weeks[signal]
        xs = active_week[FEATURES].to_numpy(dtype=float) - 0.5
        prediction_input = np.c_[xs, np.broadcast_to(z_now, (len(xs), 3))]
        if not development:
            full_scores.append(active_week[KEYS].assign(strategy="equal_weight", score=active_week["score_without_bias"]))
        for name, base_penalty, state_penalty in [
            ("rolling_ridge", ridge["rolling"], None),
            ("state_ridge", ridge["state"], ridge["state"]),
            ("three_diff_reg_050", ridge["diff_base"], ridge["diff_state"]),
        ]:
            if development and name != "state_ridge":
                continue
            length = 3 if state_penalty is None else 12
            penalty = np.r_[np.full(3, base_penalty), np.full(length - 3, state_penalty, dtype=float)]
            coef = np.linalg.solve(normal[:length, :length] + np.diag(penalty), response[:length])
            effective = np.r_[1.0, z_now] @ coef.reshape(4, 3) if length == 12 else coef
            if not development:
                full_scores.append(active_week[KEYS].assign(strategy=name, score=np.sum(xs * effective, axis=1)))
            for factor, base, value in zip(FACTORS, coef[:3], effective, strict=True):
                weights.append({"strategy": name, "signal_date": signal, "factor": factor,
                                "base_weight": base, "effective_weight": value, "state_tilt": value-base,
                                **dict(zip(STATES, now, strict=True))})
        train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
        sample_weight = np.concatenate(row_weights) * (len(train_y) / len(selected))
        for name, width in [("lightgbm_3f",3), ("lightgbm_6f",6)]:
            if development and width == 3:
                continue
            fields = [*FEATURES,*STATES][:width]
            data = lgb.Dataset(train_x[:, :width].copy(), label=train_y, weight=sample_weight, feature_name=fields)
            model = lgb.train(settings, data, num_boost_round=rounds)
            prediction = model.predict(prediction_input[:, :width].copy(), num_threads=settings["num_threads"])
            if not development:
                full_scores.append(active_week[KEYS].assign(strategy=name, score=prediction))
            if width != 6:
                continue
            for j, field in enumerate(fields):
                importance.append({"signal_date": signal, "sample": "development" if development else "holdout",
                                   "feature": field, "gain": model.feature_importance("gain")[j],
                                   "split_count": model.feature_importance("split")[j]})
            for j, factor in enumerate(FACTORS):
                adjusted = xs[:,j]
                high, low = adjusted >= .3, adjusted <= -.3
                require(high.any() and low.any(), "Empty high or low quintile")
                for state in STATES[:2]:
                    margins.append({"signal_date": signal, "sample": "development" if development else "holdout",
                                    "state": state, "regime": "high" if now[STATES.index(state)] >= thresholds[state] else "low",
                                    "factor": factor, "high_factor_prediction": prediction[high].mean(),
                                    "low_factor_prediction": prediction[low].mean(),
                                    "high_minus_low": prediction[high].mean() - prediction[low].mean()})
        if pos % 50 == 0:
            print(f"Completed {pos}/{len(signals)} fit dates", flush=True)
    return (pd.concat(full_scores, ignore_index=True), pd.DataFrame(weights), pd.DataFrame(importance),
            pd.DataFrame(margins), pd.DataFrame(link_rows,
                columns=["signal_date", "training_signal_date", "training_exit_date", "training_stocks"]))

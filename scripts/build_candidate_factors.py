"""构建沪深 300 周度原始候选因子面板。"""

import argparse
from pathlib import Path
import tomllib

import baostock as bs
import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]

# 编号、代码、类别、计算公式、预期方向。
# 预期方向是研究假设，因子面板保存原始值。
REGISTRY = [
    ("F01", "RET_5", "momentum", "C[t]/C[t-5]-1", "negative"),
    ("F02", "RET_20", "momentum", "C[t]/C[t-20]-1", "state-dependent"),
    ("F03", "RET_60", "momentum", "C[t]/C[t-60]-1", "positive"),
    ("F04", "MOM_20_SKIP5", "momentum", "C[t-5]/C[t-20]-1", "positive"),
    ("F05", "INTRA_RET_10", "momentum", "mean10(C/O-1)", "positive"),
    ("F06", "OVERNIGHT_10", "momentum", "mean10(O/C.shift(1)-1)", "negative"),
    ("F07", "VOL_20", "risk", "sample_std20(r)", "negative"),
    ("F08", "VOL_60", "risk", "sample_std60(r)", "negative"),
    ("F09", "DOWN_VOL_20", "risk", "sqrt(mean20(min(r,0)^2))", "negative"),
    ("F10", "AMP_20", "risk", "mean20((H-L)/C.shift(1))", "negative"),
    (
        "F11", "SKEW_20", "risk",
        "mean20((r-window_mean)^3)/sample_std20(r)^3", "negative",
    ),
    ("F12", "MAX_RET_20", "risk", "max20(r)", "negative"),
    ("F13", "TURN_20", "liquidity", "mean20(turn)", "negative"),
    (
        "F14", "VOLUME_RATIO_5_20", "liquidity",
        "mean5(V)/mean20(V)", "undetermined",
    ),
    ("F15", "VOL_CV_20", "liquidity", "sample_std20(V)/mean20(V)", "negative"),
    ("F16", "AMIHUD_20", "liquidity", "mean20(abs(r)/A), A>0", "positive"),
    ("F17", "CORR_PV_10", "liquidity", "corr10(C,log(1+V))", "negative"),
    (
        "F18", "CORD_RV_10", "liquidity",
        "corr10(r,log(V/V.shift(1))), paired V>0", "negative",
    ),
    (
        "F19", "WVMA_20", "liquidity",
        "sample_std20(abs(r)*V)/mean20(abs(r)*V)", "negative",
    ),
    ("F20", "BIAS_20", "price_position", "C/mean20(C)-1", "state-dependent"),
    (
        "F21", "MA_CROSS", "price_position",
        "mean5(C)/mean20(C)-1", "state-dependent",
    ),
    (
        "F22", "RSV_20", "price_position",
        "(C-min20(L))/(max20(H)-min20(L))", "state-dependent",
    ),
    (
        "F23", "PRICE_SLOPE_20", "price_position",
        "OLS slope of last 20 closes divided by current close", "state-dependent",
    ),
    (
        "F24", "RSQR_20", "price_position",
        "R squared of price regressed on time over 20 days", "undetermined",
    ),
    (
        "F25", "UP_MOVE_SHARE_14", "price_position",
        "sum14(max(delta_C,0))/sum14(abs(delta_C))", "negative",
    ),
]


def load_trading_days(path, start, end):
    cached = path.is_file()

    if cached:
        calendar = pd.read_parquet(path)
    else:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败：{login.error_msg}")

        try:
            response = bs.query_trade_dates(start_date=start, end_date=end)
            if response.error_code != "0":
                raise RuntimeError(f"查询交易日历失败：{response.error_msg}")

            rows = []
            while response.next():
                rows.append(response.get_row_data())

            if response.error_code != "0":
                raise RuntimeError(f"读取交易日历失败：{response.error_msg}")

            calendar = pd.DataFrame(rows, columns=response.fields)
        finally:
            bs.logout()

    dates = pd.to_datetime(calendar["calendar_date"], errors="raise")
    flags = calendar["is_trading_day"].astype("string")

    if not pd.DatetimeIndex(dates.sort_values()).equals(pd.date_range(start, end)):
        raise ValueError("交易日历有缺失、重复或越界日期")
    if not flags.isin(["0", "1"]).all():
        raise ValueError("交易日历包含未知的交易状态")

    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        calendar.to_parquet(temporary, index=False)
        temporary.replace(path)

    return pd.DatetimeIndex(dates[flags.eq("1")]).sort_values()


def positive_divide(numerator, denominator):
    """分母为零、负数或缺失时，结果保留为空。"""
    return numerator / denominator.where(denominator.gt(0))


def rolling_corr(left, right, window):
    """窗口内任一序列恒定时，相关系数保留为空。"""
    left_window = left.rolling(window)
    right_window = right.rolling(window)
    valid = left_window.min().lt(left_window.max())
    valid &= right_window.min().lt(right_window.max())
    return left_window.corr(right).where(valid).clip(-1, 1)


def calculate_factors(panel, trading_days):
    def wide(column):
        return (
            panel.pivot(index="date", columns="code", values=column)
            .reindex(trading_days)
        )

    # 行是市场交易日，列是证券。缺少行情的单元格保留为空。
    ohlc_ok = wide("adj_ohlc_ok").eq(True)
    op, high, low, close = (
        wide(f"adj_{field}").where(ohlc_ok)
        for field in ("open", "high", "low", "close")
    )

    volume = wide("volume")
    volume = volume.where(np.isfinite(volume) & volume.ge(0))
    amount = wide("amount")
    amount = amount.where(np.isfinite(amount) & amount.ge(0))

    previous_close = close.shift(1)
    returns = close / previous_close - 1
    change = close - previous_close

    vol20 = returns.rolling(20).std()
    return_mean = returns.rolling(20).mean()

    # E[(r-E[r])^3]，分母使用样本标准差。
    third_moment = (
        returns.pow(3).rolling(20).mean()
        - 3 * return_mean * returns.pow(2).rolling(20).mean()
        + 2 * return_mean.pow(3)
    )
    return_range = returns.rolling(20).max() - returns.rolling(20).min()

    price_low = low.rolling(20).min()
    price_range = high.rolling(20).max() - price_low

    weighted_volume = returns.abs() * volume
    volume20 = volume.rolling(20).mean()
    positive_volume = volume.where(volume.gt(0))

    # 以 0 到 19 为时间自变量，计算 20 日价格的 OLS 斜率。
    # 求和前减去当前收盘价，以降低浮点数抵消误差。
    time_weights = np.arange(20) - 9.5
    time_ss = float(time_weights @ time_weights)
    covariance_sum = sum(
        weight * (close.shift(19 - position) - close)
        for position, weight in enumerate(time_weights)
    )
    slope = covariance_sum / time_ss

    price_window = close.rolling(20)
    price_ss = 19 * price_window.var()
    r_squared = positive_divide(slope.pow(2) * time_ss, price_ss)
    r_squared = r_squared.where(price_window.min().lt(price_window.max()))

    return {
        "RET_5": close / close.shift(5) - 1,
        "RET_20": close / close.shift(20) - 1,
        "RET_60": close / close.shift(60) - 1,
        "MOM_20_SKIP5": close.shift(5) / close.shift(20) - 1,
        "INTRA_RET_10": (close / op - 1).rolling(10).mean(),
        "OVERNIGHT_10": (op / previous_close - 1).rolling(10).mean(),
        "VOL_20": vol20,
        "VOL_60": returns.rolling(60).std(),
        "DOWN_VOL_20": returns.clip(upper=0).pow(2).rolling(20).mean().pow(0.5),
        "AMP_20": ((high - low) / previous_close).rolling(20).mean(),
        "SKEW_20": positive_divide(third_moment, vol20.pow(3)).where(
            return_range.gt(0)
        ),
        "MAX_RET_20": returns.rolling(20).max(),
        "VOLUME_RATIO_5_20": positive_divide(volume.rolling(5).mean(), volume20),
        "VOL_CV_20": positive_divide(volume.rolling(20).std(), volume20),
        "AMIHUD_20": positive_divide(returns.abs(), amount).rolling(20).mean(),
        "CORR_PV_10": rolling_corr(close, np.log1p(volume), 10),
        "CORD_RV_10": rolling_corr(
            returns, np.log(positive_volume / positive_volume.shift(1)), 10
        ),
        "WVMA_20": positive_divide(
            weighted_volume.rolling(20).std(),
            weighted_volume.rolling(20).mean(),
        ),
        "BIAS_20": close / close.rolling(20).mean() - 1,
        "MA_CROSS": close.rolling(5).mean() / close.rolling(20).mean() - 1,
        "RSV_20": positive_divide(close - price_low, price_range).clip(0, 1),
        "PRICE_SLOPE_20": slope / close,
        "RSQR_20": r_squared.clip(0, 1),
        "UP_MOVE_SHARE_14": positive_divide(
            change.clip(lower=0).rolling(14).sum(),
            change.abs().rolling(14).sum(),
        ),
    }


def build_candidate_panel(panel, members, trading_days):
    signal_days = pd.DatetimeIndex(members["signal_date"].unique()).sort_values()

    if not signal_days.isin(trading_days).all():
        raise ValueError("存在不属于市场交易日的信号日")
    if not panel["date"].isin(trading_days).all():
        raise ValueError("日线面板存在不属于市场交易日的日期")

    factor_matrices = calculate_factors(panel, trading_days)

    bars = panel[["date", "code", "signal_bar_usable"]].rename(
        columns={"date": "signal_date"}
    )
    candidates = members.merge(bars, on=KEYS, how="left", validate="one_to_one")
    candidates["signal_bar_usable"] = (
        candidates["signal_bar_usable"].astype("boolean").fillna(False)
    )
    candidates = candidates.set_index(KEYS)

    names = [name for _, name, _, _, _ in REGISTRY if name != "TURN_20"]
    if set(names) != set(factor_matrices):
        raise ValueError("因子字典与实际计算的列不一致")

    for name in names:
        by_signal = factor_matrices[name].reindex(signal_days)
        long = by_signal.rename_axis(index="signal_date", columns="code").stack()
        candidates[name] = long.reindex(candidates.index).replace(
            [np.inf, -np.inf], np.nan
        )

    # 各因子分别统计覆盖率，分母始终是当周全部成分股。
    usable = candidates[names].notna().mul(
        candidates["signal_bar_usable"], axis=0
    )
    weekly_counts = usable.groupby(level="signal_date").sum().astype(int)
    weekly_counts.insert(
        0, "members", candidates.groupby(level="signal_date").size()
    )

    return candidates.sort_index().reset_index(), weekly_counts.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    research = config["research"]
    start = research["bars_start"]
    end = research["development_end"]

    processed = root / "data" / "processed"
    output = args.output_dir or processed / "factors"
    calendar_file = (
        root / "data" / "raw" / "calendar" / f"baostock_{start}_{end}.parquet"
    )

    trading_days = load_trading_days(calendar_file, start, end)
    panel = pd.read_parquet(processed / "csi300_daily_panel.parquet")
    members = pd.read_parquet(
        processed / "universe" / "csi300_weekly_members.parquet",
        columns=KEYS,
    )
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    members["signal_date"] = pd.to_datetime(members["signal_date"], errors="raise")

    candidates, weekly_counts = build_candidate_panel(panel, members, trading_days)

    registry = pd.DataFrame(
        REGISTRY, columns=["id", "factor", "family", "formula", "hypothesis"]
    )
    registry["status"] = np.where(
        registry["id"].eq("F13"), "pending_turn_field", "candidate"
    )
    registry["missing_policy"] = (
        "Unavailable required observation gives NA; "
        "rolling windows require all observations."
    )
    extra_rules = {
        "TURN_20": "Source field turn is not present in the current cache.",
        "SKEW_20": "Constant daily returns give NA.",
        "VOLUME_RATIO_5_20": "Zero mean volume gives NA.",
        "VOL_CV_20": "Zero mean volume gives NA.",
        "AMIHUD_20": "Only positive traded value is admitted; all 20 ratios are required.",
        "CORR_PV_10": "Both series require 10 points and nonzero variation.",
        "CORD_RV_10": (
            "Both volumes must be positive for every volume change; "
            "constant series give NA."
        ),
        "WVMA_20": "Zero mean return-weighted volume gives NA.",
        "RSV_20": "Zero high-low range gives NA.",
        "RSQR_20": "Constant price windows give NA.",
        "UP_MOVE_SHARE_14": "Zero total absolute price movement gives NA.",
    }
    registry["special_rule"] = registry["factor"].map(extra_rules).fillna("")
    registry["price_basis"] = np.where(
        registry["factor"].isin(["TURN_20", "VOLUME_RATIO_5_20", "VOL_CV_20"]),
        "no_price_input",
        "back_adjusted_ohlc",
    )

    names = registry.loc[registry["status"].eq("candidate"), "factor"].tolist()
    ratios = weekly_counts[names].div(weekly_counts["members"], axis=0)
    summary = pd.DataFrame({
        "mean_weekly_coverage": ratios.mean(),
        "worst_weekly_coverage": ratios.min(),
        "weeks_without_observations": weekly_counts[names].eq(0).sum(),
    })

    output.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output / "csi300_weekly_candidates.parquet", index=False)
    weekly_counts.to_parquet(
        output / "csi300_candidate_weekly_coverage.parquet", index=False
    )
    registry.to_parquet(output / "candidate_factor_registry.parquet", index=False)
    summary.rename_axis("factor").reset_index().to_parquet(
        output / "csi300_candidate_coverage_summary.parquet", index=False
    )

    print(
        f"Signal weeks: {len(weekly_counts)} | "
        f"Member rows: {len(candidates):,} | Factors: {len(names)}"
    )
    print("F13 status: pending_turn_field")
    print("\n平均覆盖率最低的 5 个因子")
    print(
        summary.nsmallest(5, "mean_weekly_coverage").to_string(
            formatters={
                "mean_weekly_coverage": "{:.2%}".format,
                "worst_weekly_coverage": "{:.2%}".format,
            }
        )
    )
    print(f"\n输出目录：{output.resolve()}")


if __name__ == "__main__":
    main()
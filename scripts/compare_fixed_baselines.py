"""比较固定四因子和去掉 BIAS_20 的三因子组合。"""

from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
STRATEGIES = ("fixed", "without_bias")
DISPLAY = {
    "fixed": "Four factors",
    "without_bias": "Three factors (no BIAS_20)",
}


def prepare_daily(ledger, initial_cash):
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"], errors="raise")

    if ledger[["strategy", "date"]].isna().any().any():
        raise ValueError("账本存在缺失的策略或日期")
    if ledger.duplicated(["strategy", "date"]).any():
        raise ValueError("账本存在重复的策略和日期")
    if set(ledger["strategy"]) != set(STRATEGIES):
        raise ValueError("账本缺少其中一个策略，或包含额外策略")
    if not np.isfinite(ledger["equity"]).all() or ledger["equity"].le(0).any():
        raise ValueError("账本包含无效资产")

    frames = []
    for strategy in STRATEGIES:
        daily = (
            ledger.loc[ledger["strategy"].eq(strategy)]
            .sort_values("date")
            .copy()
        )

        daily["nav"] = daily["equity"] / initial_cash
        previous_equity = daily["equity"].shift(1, fill_value=initial_cash)
        daily["daily_return"] = daily["equity"] / previous_equity - 1

        # 把建仓前的初始净值 1.0 也算作一个历史高点。
        running_peak = daily["nav"].cummax().clip(lower=1.0)
        daily["drawdown"] = daily["nav"] / running_peak - 1

        # gross_traded 是买入金额与卖出金额绝对值之和。
        rebalance = daily["opening_equity"].notna()
        if not daily.loc[rebalance, "opening_equity"].gt(0).all():
            raise ValueError(f"{strategy} 存在无效的调仓前资产")
        daily["is_rebalance"] = rebalance
        daily["two_way_turnover"] = (
            daily["gross_traded"] / daily["opening_equity"].where(rebalance)
        )
        frames.append(daily)

    if not frames[0]["date"].reset_index(drop=True).equals(
        frames[1]["date"].reset_index(drop=True)
    ):
        raise ValueError("两版的交易日不一致")

    return pd.concat(frames, ignore_index=True)


def summarize(daily):
    rows = []
    for strategy, group in daily.groupby("strategy", sort=False):
        n = len(group)
        final_nav = group["nav"].iloc[-1]
        trough = group.loc[group["drawdown"].idxmin()]
        rows.append({
            "strategy": strategy,
            "trading_days": n,
            "rebalance_days": int(group["is_rebalance"].sum()),
            "final_equity": group["equity"].iloc[-1],
            "total_return": final_nav - 1,
            "annualized_return": final_nav ** (TRADING_DAYS_PER_YEAR / n) - 1,
            "annualized_volatility": (
                group["daily_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            ),
            "max_drawdown": -trough["drawdown"],
            "drawdown_trough_date": trough["date"],
            "mean_rebalance_turnover": group["two_way_turnover"].mean(),
            "fees_paid": group["fee"].sum(),
        })

    return pd.DataFrame(rows).set_index("strategy").reindex(STRATEGIES)


def compare_years(daily):
    work = daily.assign(year=daily["date"].dt.year)
    annual = work.groupby(["year", "strategy"]).agg(
        total_return=("daily_return", lambda values: (1 + values).prod() - 1),
        mean_rebalance_turnover=("two_way_turnover", "mean"),
        fees_paid=("fee", "sum"),
    )
    annual = annual.reset_index()

    comparison = annual.pivot(index="year", columns="strategy", values="total_return")
    comparison = comparison.reindex(columns=STRATEGIES)
    comparison["fixed_minus_without_bias_pp"] = (
        comparison["fixed"] - comparison["without_bias"]
    ) * 100
    return annual, comparison


def plot_comparison(daily, output):
    nav = daily.pivot(index="date", columns="strategy", values="nav")
    drawdown = daily.pivot(index="date", columns="strategy", values="drawdown")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for strategy in STRATEGIES:
        top.plot(nav.index, nav[strategy], label=DISPLAY[strategy], linewidth=1.5)
        bottom.plot(drawdown.index, drawdown[strategy], linewidth=1.2)

    top.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    top.set(title="Fixed factor portfolios | Development period", ylabel="NAV")
    top.legend()
    bottom.set(ylabel="Drawdown", xlabel="Trade date")
    bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    for axis in (top, bottom):
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(
        (root / "configs" / "baseline.toml").read_text(encoding="utf-8-sig")
    )
    initial_cash = float(config["portfolio"]["initial_cash"])
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("初始资金必须为正")

    backtest = root / "reports" / "portfolio" / "development_backtest_v2"
    ledger = pd.read_parquet(backtest / "daily_nav.parquet")
    daily = prepare_daily(ledger, initial_cash)

    summary = summarize(daily)
    annual, comparison = compare_years(daily)

    output = backtest / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_parquet(output / "baseline_summary.parquet", index=False)
    annual.to_parquet(output / "baseline_by_year.parquet", index=False)
    comparison.reset_index().to_parquet(
        output / "baseline_yearly_return_gap.parquet", index=False
    )
    plot_comparison(daily, output / "baseline_nav_drawdown.png")

    percent_columns = [
        "total_return", "annualized_return", "annualized_volatility",
        "max_drawdown", "mean_rebalance_turnover",
    ]
    formats = {column: "{:.2%}".format for column in percent_columns}
    formats.update({
        "final_equity": "{:,.2f}".format,
        "fees_paid": "{:,.2f}".format,
    })

    print("开发期组合汇总")
    print(summary.to_string(formatters=formats))
    print("\n逐年收益对照")
    print(comparison.to_string(formatters={
        "fixed": "{:+.2%}".format,
        "without_bias": "{:+.2%}".format,
        "fixed_minus_without_bias_pp": "{:+.2f}".format,
    }))
    print(f"\n输出目录：{output}")


if __name__ == "__main__":
    main()
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_dir = Path(__file__).resolve().parents[1]
    reports = project_dir / "reports"
    ledger = pd.read_parquet(reports / "sample_backtest_with_fees.parquet")
    ledger = ledger.sort_values("date").set_index("date")

    initial_cash = 100_000.0
    nav = ledger[["equity", "benchmark_equity"]] / initial_cash
    nav.columns = ["Moving average", "Buy and hold"]

    # 初始净值 1.0 也计入历史高点，首笔费用造成的下跌才会被计入。
    running_peak = nav.cummax().clip(lower=1.0)
    drawdown = nav / running_peak - 1

    for name in nav.columns:
        trough_date = drawdown[name].idxmin()
        sample_return = nav[name].iloc[-1] - 1
        max_drawdown = -drawdown.loc[trough_date, name]
        print(
            f"{name}: 样本收益 {sample_return:.2%}，"
            f"最大回撤 {max_drawdown:.2%}，"
            f"回撤低点 {trough_date:%Y-%m-%d}"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    nav.plot(ax=ax, linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Sample backtest: January 2024", xlabel="Date", ylabel="NAV")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    output = reports / "figures" / "sample_nav.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"图表已保存：{output}")


if __name__ == "__main__":
    main()
# 沪深 300 多因子选股与市场状态配权

本地量化课程项目。研究动态市场状态是否能改善沪深 300 历史成分股的周度选股，直接比较三因子等权、滚动 Ridge 和带市场状态交互项的滚动 Ridge。选题、因子评价和回测以开发期为主，2025 年是预设留出期。

- 2020–2024 年开发期按历史时点取沪深 300 成分，覆盖 255 个信号周。
- 计算 24 个候选量价因子，第 25 项换手率因子等待补齐数据。用周度 Rank IC、五分组收益和截面秩相关筛选候选。
- 固定基线使用 `RET_5`、`VOL_60` 和 `CORR_PV_10`，另保留含 `BIAS_20` 的固定组合用于删减对照。
- 滚动模型只用信号时点前已经结束的收益标签。市场状态变量为指数趋势、波动和历史成分上涨广度。
- 模拟成交使用日线开盘价和明确的费用、停牌及涨跌停假设，并保留订单、实际持仓、分红送转和已核实配股流水。

## 开发期结果

评价区间为 2021-01-18 至 2024-12-31，从首个拟合信号执行日开始。下表为无持仓缓冲、买卖各 10 bps 的基线。

| 组合 | 累计收益 | 最大回撤 | 平均每次双边换手 |
| --- | ---: | ---: | ---: |
| 三因子等权 | −5.96% | 31.15% | 139.82% |
| 滚动 Ridge | +2.20% | 33.56% | 109.41% |
| 市场状态 Ridge | +7.04% | 35.65% | 114.19% |

市场状态版的领先主要集中在 2024 年后段。含前 45 名持仓缓冲时，在单边成本 10 bps 下，原三因子状态 Ridge 为 +16.05%，七因子稀疏和分离正则状态 Ridge 分别为 −2.28% 和 +3.98%。同区间沪深 300 全收益指数为 −20.66%，历史成分等权参考为 −10.07%。

[筛选方法、复跑入口与消融结果](research/regime_factor_screening/README.md)：对 24 个候选因子进行双轨筛选，准入 7 个因子和 5 个状态交互。新版本降低换手，但没有替换原三因子状态模型的证据。

[三因子正则化、持仓缓冲与 LightGBM 对照](research/three_factor_experiments/review.md)：同用 Top 30 和前 45 名缓冲，六输入 LightGBM 在单边 10/20 bp 下的开发期累计收益为 +49.81% / +26.88%，原三因子状态 Ridge 为 +16.05% / −0.81%。原状态模型把缓冲放宽至前 50 名后为 +17.82% / +2.71%。[查看复跑说明](research/three_factor_experiments/README.md)与[汇总数据](research/three_factor_experiments/results/)。

## 2025 留出期

[留出期复盘](research/holdout_2025/review.md)与[复跑说明](scripts/holdout_2025/README.md)。冻结六个模型，完成 14 组成本与缓冲对照。同期沪深 300 全收益指数上涨 20.98%，六个模型均落后基准。六输入 LightGBM 在单边 10 bp 下为 +5.01%，与三输入版的 +5.00% 接近；20 bp 下三输入版以 +3.53% 领先。已生成历史模拟账本，实时前向模拟仍待开始。

## V2：波动率分层与组合管理

[V2 完整复盘](research/v2_regime_tiered/review.md) · [复跑说明](research/v2_regime_tiered/README.md)。在已揭盲的 2025 年做回顾性比较：动态分层 LightGBM 在单边 10/20 bp 下为 +14.29% / +9.30%，V1 六输入 LightGBM 仅加得分 EMA 为 +14.13% / +11.78%。两者均落后沪深 300 全收益指数（+20.98%）。V1 留出期结果另行保留。

## V3 长周期研究

V3 将开发样本扩展至 2015–2025 年，增加市场调整收益标签、LambdaRank、市场门控网络与带 Beta 约束的组合优化。已完成开发期消融和 2026 年保留段登记，2026 评价有待采集。

[开发报告](research/v3/DEVELOPMENT_REPORT.md) · [本地流程](research/v3/README.md) · [2026 冻结登记](research/v3/freeze_registration_2026.json)

## 本地运行

在 Windows PowerShell 中运行。开发环境为 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`configs/baseline.toml` 配置开发期、留出期、初始资金和目标持仓数。以项目根目录为当前目录，按下表顺序运行各阶段脚本。通用命令为 `.\.venv\Scripts\python.exe .\scripts\脚本文件名.py`。

| 阶段 | 按顺序运行的脚本 |
| --- | --- |
| 数据采集 | `build_weekly_universe.py` → `fetch_daily_bars.py` → `audit_daily_bars.py` |
| 因子研究 | `build_candidate_factors.py` → `build_forward_labels.py` → `evaluate_single_factors.py` → `analyze_factor_redundancy.py` |
| 固定基线 | `build_fixed_scores.py` → `build_target_positions.py` |
| 公司行动 | `audit_price_reference.py` → `fetch_corporate_actions.py` → `normalize_corporate_actions.py` → `apply_verified_actions.py` |
| 固定组合回测 | `run_development_backtest_v2.py` → `compare_fixed_baselines.py` |
| 状态与滚动权重 | `build_market_states.py` → `evaluate_volatility_regimes.py` → `build_walk_forward_controls.py` → `build_market_state_ridge.py` |

`audit_price_reference.py` 依赖 `build_candidate_factors.py` 缓存的完整交易日历。行情和公司行动采集耗时，可用 `--limit 20` 分批运行。完成采集后可用本地缓存重跑研究流程。

```powershell
.\.venv\Scripts\python.exe .\scripts\run_development_backtest_with_targets.py `
    --targets .\data\processed\portfolio\walk_forward\walk_forward_control_targets.parquet `
    --output-dir .\reports\portfolio\walk_forward_controls

.\.venv\Scripts\python.exe .\scripts\compare_walk_forward_controls.py

.\.venv\Scripts\python.exe .\scripts\run_development_backtest_with_targets.py `
    --targets .\data\processed\portfolio\walk_forward\walk_forward_three_variant_targets.parquet `
    --output-dir .\reports\portfolio\walk_forward_three_variants

.\.venv\Scripts\python.exe .\scripts\audit_state_ridge_robustness.py
```

运行完整因子筛选实验前，在前述流程之后构建基准和全排名：

```powershell
.\.venv\Scripts\python.exe .\scripts\fetch_csi300_total_return.py
.\.venv\Scripts\python.exe .\scripts\build_csi300_equal_weight.py
.\.venv\Scripts\python.exe .\scripts\compare_market_state_ridge.py
.\.venv\Scripts\python.exe .\scripts\build_walk_forward_full_rankings.py
.\.venv\Scripts\python.exe -B .\research\regime_factor_screening\run_verified_factor_pipeline.py --project-root .
```

行情、指数和公司行动主要来自 [BaoStock](https://pypi.org/project/baostock/)。入门样本脚本另使用 [AKShare](https://github.com/akfamily/akshare)。`data/`、`reports/` 和 `.venv/` 由本地运行生成，仓库跟踪代码、配置、项目说明和精选汇总表。

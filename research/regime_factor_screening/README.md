# 沪深 300 因子筛选与状态 Ridge 复跑

从已有本地数据一次跑完四步。本实验只使用截至 2024-12-31 的开发数据，不联网、不覆盖旧回测。先在项目根目录按[仓库说明](../../README.md#本地运行)准备数据，再运行：

    .\.venv\Scripts\python.exe -B .\research\regime_factor_screening\run_verified_factor_pipeline.py --project-root .

默认输出到 reports\verified_factor_pipeline。可传 --output-dir 保存到另一目录；如需对照是否正交化，请传 --breadth-mode raw 并用不同的 --output-dir。auto 模式在筛选段趋势与广度的 Pearson 绝对值超过 0.80 时，用每次拟合的历史窗口估计广度对趋势的回归，再对该窗口和当期广度应用同一估计。阈值筛选始终用原始状态。

| 脚本 | 作用 |
|---|---|
| evaluate_all_factor_regimes.py | 24 个因子 × 3 个状态 × 2 个时期，固定筛选段中位数 |
| select_regime_factors.py | A1/A2、B1–B3 与筛选段秩相关去冗余 |
| build_verified_market_state_ridge.py | 固定股票池与滚动周，稀疏、分离惩罚、同因子仅主效应对照 |
| compare_verified_market_state_ridge.py | 复用项目现有交易引擎和基准读取器，7 × 2 × 2 消融 |
| run_verified_factor_pipeline.py | 一键入口 |

主输出目录内部：

- data/factor_regime_evaluation.parquet、reports/factor_regime_evaluation.csv 与 reports/top15_factor_regimes.csv：条件 IC、ICIR、五分组价差，以及筛选段差异前 15 对。
- reports/factor_selection_audit_table.csv、reports/factor_interaction_audit.csv、configs/verified_model_spec.json：因子与交互的完整准入审计。
- configs/verified_fit_config.json、reports/state_penalty_validation.csv、reports/model_rank_ic_by_split.csv、reports/model_weekly_rank_ic.parquet：拟合设置、惩罚比较及预测指标。
- data/verified_full_rankings.parquet、data/verified_targets.parquet、data/verified_ridge_weights.parquet：排名、目标仓位及每周主效应、交互系数和实际生效权重。
- reports/execution_ablation.csv、reports/execution_ablation_yearly.csv、reports/benchmark_excess_metrics.csv：7 个模型 × 两档单边成本 × 有无缓冲；三个基准的相对指标另列。
- reports/execution_daily_nav.parquet、reports/execution_buffer_selections.parquet、reports/execution_unresolved_exposures.csv：回测核对明细。

使用仓库实际存在的 single_factor_by_split.parquet 和 redundancy_pairs.parquet，注册表的 family / hypothesis 对应规范所称的分类 / 方向假设。轨道一 A1 使用规范允许的固定 0.025 门槛。原三个通过 A1/A2 的非冗余因子优先保留；其余 A 轨道因子按筛选段绝对 IC 排序；B 轨道新增候选按通过门槛的最大绝对 Delta IC 排序。方向只决定输入百分位的朝向，Ridge 系数不锁定符号。B 轨道独有因子也具有可估的主效应，符合规范 Step 3；这不表示其无条件表现过关。

沿用旧模型全部合格股票和训练周；入选因子缺失时用中性截面百分位 0.5（即中心化零）保持可比。缓冲按真实持仓、Top 30 / 保留前 45 执行，仅在拟合启动后生效；预热期各模型沿用原三因子等权做初始化。模型 Rank IC 用选股前的截面分数计算，耗费与缓冲不改变 IC。开发收益从 2021-01-15 收盘锚定，实际评价为 2021-01-18 至 2024-12-31。年化超额收益为日主动收益均值 × 252；相对回撤基于策略净值 / 基准净值；ICIR 为周 IC 均值 / 周 IC 样本标准差，未年化。

因子和交互使用了两个开发分段，惩罚也使用了验证段，因此以上均为回顾性的开发对照。滚动拟合没有用尚未成熟的收益标签，但这不能将整个开发回测解释为严格样本外，更不能把筛选门槛解释为统计显著性。

[实验复盘](review.md)和[汇总结果](results/)已随仓库发布；行情、逐股排名和交易流水由本地生成。

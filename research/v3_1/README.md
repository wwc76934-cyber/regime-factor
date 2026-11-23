# V3.1 沪深 300 指数增强

V3.1 在独立目录复用 V3.0 的 2015–2025 年开发缓存。V3.0 的冻结登记已撤回，登记原件及历史研究保留。2026-01-01 至 2026-11-13 的行情在 V3.1 开发期间隔离，只有新开发方案通过验收并登记后才进入保留段评价。

## Step 0 本地审计与数据交接

用日成交额除以换手率的小数值估算流通市值，在 20 个交易日内取中位数，至少需要 10 个有效观测。指数权重以信号日收盘前可用数据计算，下一交易日开盘按调整后的价格建立分数股组合。等权基准使用同样的点时成分和估值时点。两个篮子基准使用后复权报价，按零成本计算。缺报时沿用最近可得价格，有权重但没有可用价格的标的在当次基准计算中剔除并重新归一。情况与权重均留在审计文件中。它们用于研究对照，实盘组合使用独立的撮合模型。

缓存的 V3.0 数据保存在 `data/raw/v3/development`。补采命令使用独立的 `data/raw/v3_1/development`，只请求 2025 年及以前的开发数据。开发早段的 H00300 缺口与估值覆盖率分开检查，其余行情、点时成分和公司行动复用原缓存。

在项目根目录运行离线审计：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\v3_1\prepare_v3_1_base.py audit
```

以下命令由项目作者在需要时联网运行：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\v3_1\prepare_v3_1_base.py probe
.\.venv\Scripts\python.exe -B .\scripts\v3_1\prepare_v3_1_base.py fetch-index
.\.venv\Scripts\python.exe -B .\scripts\v3_1\prepare_v3_1_base.py fetch-valuations
```

`probe` 对三只股票的 2015、2025 两个样本段测字段；`fetch-index` 补齐 H00300 开发期的早段交易日，支持通过 `--benchmark-csv` 导入经核验的 `date,csi300` 格式全收益指数文件；`fetch-valuations` 按成分股和年份建立缓存，断线后可原样重跑。供应商历史日报字段的可用性、有效值比率与可用正值比率分开记录。

Step 0 的补采复核已通过。H00300 覆盖 2015-01-05 至 2025-12-31，估值分区完整。现金流市盈率的有效正值覆盖低于市销率，两者的逐年结果见 [Step 0 审计](results/step0/REVIEW.md)。

## Step 1 多周期标签、分层准入与第 5 个市场状态

```powershell
.\.venv\Scripts\python.exe -B .\scripts\v3_1\build_v3_1_factors_and_labels.py --reuse-panel .\research\v3_1\results\step1
```

使用后复权股票收盘和 H00300 收盘在完全相同的区间计算标签。收盘信号的标签从下一交易日的收盘开始，标签退出仅在对应退出日的收盘后才成熟。早期以沪深 300 价格指数估计历史 Beta 和状态，正式的远期市场回报来自 H00300。组合阶段会另外计算开盘模拟成交。

快频主筛窗为 5 个交易日，趋势和估值主筛窗为 20 个交易日。按 2015–2018 与 2019–2022 的等权池超额、风险调整 Rank IC 和分层门槛决定六因子主池。H00300 的同期限差额保留在审计表。第 5 个状态使用精确的 60 个交易日收益窗，组合成员来自成组日可用的最近一期点时快照。两组按成组日信号的前 1/3 形成等权篮子。选择规则、统计和时点见 [修订审计](results/step1/REVIEW.md)。

## Step 2 滚动模型与信号门槛

以下命令按 5、10、20 日运行 Ridge 和 Huber 树，对照 V3.0 状态 Ridge。门控网络作为 10 日消融，使用 NumPy、完整小批量训练、时间顺序验证和 Early Stopping。

```powershell
foreach ($h in 5, 10, 20) {
  .\.venv\Scripts\python.exe -B .\scripts\v3_1\train_v3_1_models.py --horizon $h
}
.\.venv\Scripts\python.exe -B .\scripts\v3_1\train_v3_1_gate.py
.\.venv\Scripts\python.exe -B .\scripts\v3_1\summarize_v3_1_signals.py
```

输出见 [Step 2 结果](results/step2/REVIEW.md)。信号指标使用相同的持有期收盘收益。组合交易、费用以及冻结前的各项红线属于后续实验。

## Step 3 指数增强与执行消融

```powershell
foreach ($block in "core", "frequency", "controls", "turnover_sensitivity") {
  .\.venv\Scripts\python.exe -B .\scripts\v3_1\optimize_v3_1_portfolio.py --block $block
}
.\.venv\Scripts\python.exe -B .\scripts\v3_1\summarize_v3_1_portfolio.py
```

[组合与执行审计](results/step3/REVIEW.md) 汇总首轮 32 条账本和 32 条优化器敏感性账本。正式基准为 H00300，并同步保存 FloatMV 与 CSI300_EW 的逐年和全期超额。公司行动已按公告入账。原始账本存于 `results/step3_before_actions`。

复跑优化器网格：

```powershell
foreach ($h in 5, 10) {
  foreach ($fee in 10, 20) {
    .\.venv\Scripts\python.exe -B .\scripts\v3_1\optimize_v3_1_portfolio.py --block optimizer_grid --grid-horizon $h --grid-fee-bps $fee
  }
}
.\.venv\Scripts\python.exe -B .\scripts\v3_1\summarize_v3_1_portfolio.py
```

## Step 4 归因与非负方向复测

[开发报告](DEVELOPMENT_REPORT_V3_1.md) 列出六因子年度有效权重、2024 月度与季度四项归因，以及修复账本前后的方向对照。非负 Ridge 与原始 Ridge 使用相同的融合及组合参数。终验状态由 `results/step4/freeze_gate_status.json` 单独登记，2026 保留段继续隔离。

验收表的每一行都作为新版冻结门槛。多头信号、交易成本、组合 Beta 达标和公司行动账本分别审计。

## 仓库产物

仓库提供脚本、审计表、模型参数和开发报告。原始行情、因子面板及预测 Parquet 在本地生成。已有本地因子和收益面板时可用 Step 1 的 `--reuse-panel`；首次构建时直接运行该脚本。Step 2 的匹配对照也需要本地 V3.0 的 `research/v3/results/step3/predictions.parquet`。
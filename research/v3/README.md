# V3 数据工程和研究边界

V3 代码和研究产物位于 scripts/v3 与 research/v3。2015–2025 开发数据已审计，滚动模型和组合消融已完成。V3.0 的 2026 冻结登记已撤回，原登记与开发结果留档。后续研究进入 [V3.1](../v3_1/README.md)。

## 第一次只跑小样本探测

在 PowerShell 中，从已有项目目录执行：

```powershell
cd D:\QuantProjects\regime-factor
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py probe --project-root .
```

预期产物为 research\v3\results\step1\probe_fields.csv，含 12 行（3 只股票 × 2015/2025 两个日期段 × 两种价格口径）。每行含返回天数、复权标记正确天数以及三个字段的有效值计数。另检查 2015-01-09 的响应是否恰有 300 个不同证券，且 updateDate 不晚于请求日。探测完成后先看输出，不批量下载。

## 探测通过后的批量阶段

探测和正式缓存隔离。正式行情按日历年和证券存放，当天接口请求成功且文件写入完整后才算完成。登录和查询短暂失败会重试，命令停止后原样重跑续传。成功返回的空表也留档，后续以每周覆盖检查区分停牌、退市和真正的数据断档。

```powershell
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py download --project-root . --only universe
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py download --project-root . --only bars
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py download --project-root . --only actions
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py benchmark --project-root .
.\.venv\Scripts\python.exe -B .\scripts\v3\fetch_and_audit_v3_data.py audit --project-root .
```

默认缓存 data\raw\v3\development。2014-06-01 起准备 120 个交易日 Beta 等预热窗口，正式样本 2015–2025。不读、不下载 2026 结果数据；开发阶段最末一周不能凭借下一年行情造标签。2015 年初个股历史不足时，Beta 缺失而不填充。脱离成员后仍可能持仓，行情按首次进入前预热并保留后续年度，不能只取成员当周日线。正式开发基准需要 H00300；默认请求 2016-12 起，主要为 2017 开始的回测提供前一日基准锚点。若早期确实找不到经核实的 H00300，可传 --benchmark-csv 指定含 date,csi300 的可信镜像，或用 --benchmark-start 明确后移可比起点。不得将价格指数作为全收益序列拼入。

审计检查每年字段覆盖、负值与零值、异常复权标记、点时成分、双口径可用信号周及公司行动。复权异常保存到隔离清单；不把原始价填进后复权价，也不把价格参考差直接等同于股息。负 PE 可能代表亏损，不自动视作脏值。字段存在和历史覆盖充分也不等于已证明供应商当时可见；若后续无法证实估值时间口径，必须保留无估值对照并报告限制。持仓涉及未解决公司行动时须先核验，再发布回测。

## 实验口径

1. Step 1：用户完成采集，报告字段覆盖、有效样本周、点时成分检查与公司行动待核验表。
2. Step 2：筛选与去冗余仅使用已成熟开发期标签。用历史 Beta 扣除同区间市场收益的标签称为“市场调整后收益”，不能当成已证实的纯 Alpha；6–8 个因子是期望范围，不作为硬凑名额。
3. Step 3：区分 LightGBM 回归评估指标与 LambdaRank 的训练目标；门控网络是受 MASTER 启发的简化对照。CPU 耗时以实测为准。
4. Step 4：优化器约束的是模型估计的目标 Beta。成交后重新审计股数、换手和暴露，不能保证实现 Beta 精确达标。
5. Step 5：冻结模型和成本参数后才读隔离的 2026 评价数据。本项目曾为 V1 前向尝试下载过一部分 2026 行情，因此发布时称“V3 参数冻结后的保留段评价”，不声称整个团队从未接触过 2026 数据。

## 2026 冻结登记状态

[原 V3.0 冻结登记](freeze_registration_2026.json)于 2026-11-22 撤回，作为历史记录保留。此前的 V3.0 数据采集交接随该登记撤回。开发状态见 [V3.1 说明](../v3_1/README.md)和[撤回记录](../v3_1/v3_0_freeze_status.json)。

## GitHub 发布内容与冻结复现

仓库发布脚本、开发报告、汇总表、净值图和原始冻结登记。原始行情、预测 Parquet 及逐笔模拟账本保存在本地。冻结核验按原始字节校验，运行时需要原开发环境及登记对应的本地研究产物。若在独立副本中恢复冻结环境，[指数读取依赖快照](reproduction/fetch_csi300_total_return.py) 对应登记中的 scripts/fetch_csi300_total_return.py。V1 的仓库版本保留其独立的评估入口。
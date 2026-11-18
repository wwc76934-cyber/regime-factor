# V2：波动率分层研究

第一阶段输出为逐周分层特征、未来标签，以及三个波动率层的年度收益和相对沪深 300 全收益指数的实现 Beta。2025 已在 V1 中揭盲，并用于提出 V2；这里将其明确标为回顾性样本。

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\tiered_data.py --project-root .
```

保留原缓存和 V1 结果。V2 读取已有 2020–2024 和 2025 数据。按信号日原始 VOL_60 排序，完全同分时按代码排序。组内采用中点百分位再减 0.5，使各层特征与有效未来收益标签分别精确中心化。非有限值不进入排名；2025 年最后两个无法在当年观察到完整下一周收益的信号仍可保留因子，不构造跨期标签。

层级收益是诊断用的无摩擦等权参考组合：沿用上一信号日的名单，在约定执行日收盘用复权价重新等权。它不模拟开盘成交、费率或交易限制；缺价数量随每日净值一起记录。2021 从 1 月 15 日收盘归一化，2025 从 2024 年 12 月 31 日收盘归一化；Beta 由同区间按自然周复合的收益率相对 H00300 回归得到。低、中、高波分层不保证组合拥有指定 Beta，也不是对个股的 Beta 排序。

[最终复盘](review.md)给出 V1/V2、两档成本和两个基准的对照。

从项目根目录依次运行第二至第四阶段：

```powershell
.\.venv\Scripts\python.exe -m pip install lightgbm==4.6.0 scipy==1.17.1
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\tiered_models.py --project-root .
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\evaluate_score_ema.py --project-root .
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\tiered_models.py --project-root . --include-review --output-dir .\research\v2_regime_tiered\results\stage3_scores
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\evaluate_allocator.py --project-root .
.\.venv\Scripts\python.exe -B .\scripts\v2_regime_tiered\run_v2_evaluation.py --project-root .
```

第四阶段复用历史 V1 的全排名与分数，不重新调参。有项目内 stage4/inputs 缓存时直接读取；否则使用三因子实验和 V1 盲测产出的 data/full_rankings.parquet。也可通过 --v1-dev-rankings / --v1-2025-rankings 指定。若尚无文件，分别运行 [V1 开发期脚本](../three_factor_experiments/README.md)和[V1 留出期脚本](../../scripts/holdout_2025/README.md)；复跑留出期请用新的输出目录，保留已登记的结果。行情、逐股排名和每日账本由本地生成。

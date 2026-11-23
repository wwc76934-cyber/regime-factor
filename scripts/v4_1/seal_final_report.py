"""Release a single set of paired out-of-sample results, with fingerprints for the four native ledgers."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_final_inputs import ROOT, DEST, WORK, ANCHOR, FIRST, LAST, market, seal
from run_final_ledgers import engine


def drawdown(curve):
    return float((1-curve/curve.cummax()).max())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reg = seal(ROOT)
    receipt = DEST/"holdout_seal.json"
    if receipt.exists():
        signed = json.loads(receipt.read_text(encoding="utf-8"))
        for f,h in signed["sha256"].items():
            if digest(ROOT/f)!=h:
                raise ValueError("Changed sealed artifact: "+f)
        print("Existing release verified")
        return
    audit = pd.read_csv(DEST/"audit/ledger_verification.csv").set_index(["model","fee_bps"])
    accounts = reg["configuration"]["accounts"]
    if (set(audit.index)!={(a["model"],a["fee_bps"]) for a in accounts} or len(audit)!=4 or
        not audit.all_ledgers_verified.eq(True).all() or audit[["unresolved","terminal_stale"]].ne(0).any().any()):
        raise ValueError("Four verified ledgers are required")
    bench = pd.read_parquet(WORK/"benchmarks.parquet").set_index("date")
    metrics,daily,ends,fills,actions = [],[],[],[],[]
    native = []
    for a in accounts:
        model,fee = a["model"],a["fee_bps"]
        folder = WORK/model/f"{fee}bp"
        native.extend(sorted(folder.glob("*.parquet")))
        n = pd.read_parquet(folder/"nav.parquet").set_index("date")
        nav = n.equity / a["end_balances"]["equity"]
        if nav.iloc[0]!=1 or nav.index[0]!=ANCHOR or not n.index.equals(bench.index):
            raise ValueError("Benchmark calendar or anchor mismatch")
        days = n.loc[FIRST:]
        turn = (days.gross_traded/days.opening_equity).sum()*5/len(days)
        daily.append(n.assign(model=model,fee_bps=fee).join(bench[list(engine.BENCHMARKS.values())]).reset_index())
        h = pd.read_parquet(folder/"holdings.parquet")
        ends.append(h.loc[h.date.eq(LAST)].assign(fee_bps=fee))
        for name,acc in [("orders",fills),("actions",actions)]:
            t = pd.read_parquet(folder/(name+".parquet"))
            acc.append(t.loc[t.date.ge(FIRST)].assign(fee_bps=fee))
        for b,col in engine.BENCHMARKS.items():
            bnav = bench[col]/bench[col].iloc[0]
            active = nav.pct_change().iloc[1:] - bnav.pct_change().iloc[1:]
            if not np.isfinite(active).all() or active.std()<=0:
                raise ValueError("Invalid active return")
            metrics.append(dict(model=model,beta=a["policy"],fee_bps=fee,benchmark=b,sessions=len(days),
                net_return=nav.iat[-1]-1,benchmark_return=bnav.iat[-1]-1,excess_pp=(nav.iat[-1]-bnav.iat[-1])*100,
                IR=np.sqrt(252)*active.mean()/active.std(),max_drawdown=drawdown(nav),
                relative_max_drawdown=drawdown(nav/bnav),weekly_two_way_turnover=turn,fees=days.fee.sum(),
                anchor_equity=n.equity.iat[0],terminal_equity=n.equity.iat[-1]))
    result = pd.DataFrame(metrics)
    for name,df in [("benchmark_metrics",result),("daily_ledgers",pd.concat(daily)),("holdings_at_cutoff",pd.concat(ends)),
                    ("order_ledgers",pd.concat(fills)),("action_ledgers",pd.concat(actions))]:
        market.save(df, DEST/(name+".csv"))
    compare = []
    for fee,g in result.loc[result.benchmark.eq("H00300")].groupby("fee_bps"):
        pair = g.set_index("model")
        diff = pair.loc["v4_1_moe_huber","net_return"]-pair.loc["v3_1_champion","net_return"]
        compare.append((fee,100*diff,"V4.1" if diff>0 else "V3.1" if diff<0 else "持平"))
    content = ["# 2026 年沪深 300 指数增强冻结盲测","",
        "冻结提交为 `d4a05d9`。以 2025 年 12 月 31 日四本账各自的收盘资产为起点，评价 2026 年 1 月 1 日至 11 月 13 日的 208 个交易日。此次检验是对历史留出期的一次性回放。早前 V1 项目曾进行前向数据采集。","",
        "## 两个模型的同期表现","",
        "|方案|单边费率|净收益|相对 H00300 超额|IR|最大回撤|周等效双边换手|","|---|---:|---:|---:|---:|---:|---:|"]
    for r in result[result.benchmark.eq("H00300")].itertuples():
        content.append(f"|{r.model} / {r.beta}|{r.fee_bps} bp|{r.net_return:+.2%}|{r.excess_pp:+.2f} pp|{r.IR:.3f}|{r.max_drawdown:.2%}|{r.weekly_two_way_turnover:.2%}|")
    content += ["","按期末净收益比较，"+"，".join(f"{fee} bp 下 V4.1 减 V3.1 为 {d:+.2f} 个百分点（{winner}）" for fee,d,winner in compare)+"。两套方案同时改变了评分模型和组合 Beta 政策，同期差值反映整套方案的差异。","",
        "## 三个参照组合","",
        "|方案|费率|基准|同期基准收益|累计超额（百分点）|IR|相对净值最大回撤|","|---|---:|---|---:|---:|---:|---:|"]
    for r in result.itertuples():
        content.append(f"|{r.model}|{r.fee_bps} bp|{r.benchmark}|{r.benchmark_return:+.2%}|{r.excess_pp:+.2f}|{r.IR:.3f}|{r.relative_max_drawdown:.2%}|")
    content += ["","H00300 为正式全收益基准。FloatMV 使用流通市值代理权重，CSI300_EW 使用成分等权，后两者沿用开发期零成本分数股组合口径。累计超额取同区间收益率之差，IR 取日主动收益均值除以样本标准差并乘 √252。回撤含年末起点，周等效双边换手为逐日成交绝对金额除以调仓前资产后求和，再乘 5/208。","",
        "## 跨年、训练和交易核验","",
        "沿用原始股票持仓、现金、EMA 记忆和每五个交易日调仓的相位。逐日核对了开发期权益，并逐股核对年末股数和估值。六只已调出成分的跨年行情补入历史风格组合。证券池采用当周成分。重训仅使用退出日不晚于当期信号日的标签。","",
        "四本账均通过核验。开盘成交、现金、费用、收盘资产、登记日权益、分红及送转均已对账，未决敞口和期末陈旧估值均为零。现金红利沿用开发期的税前口径。公司行动补录和换股条款的原始公告链接保存在 audit 中。","",
        "逐日账、买卖单、公司行动分录、期末实际持仓及分基准指标随报告保存。期末持仓是历史回放结果。后续模拟单独归档。",""]
    report = DEST/"HOLDOUT_REPORT_2026.md"
    report.write_text("\n".join(content),encoding="utf-8")
    files = [report,*sorted(DEST.glob("*.csv")),*sorted((DEST/"audit").glob("*.*")),*native,
             *sorted((ROOT/"scripts/v4_1").glob("*final*.py")),
             *(WORK/f for f in ("baseline_predictions.parquet","primary_predictions.parquet","benchmarks.parquet"))]
    signed = {"sealed_utc":datetime.now(timezone.utc).isoformat(),"through":str(LAST.date()),
              "configuration_sha256":reg["configuration_sha256"],"all_ledgers_verified":True,
              "unresolved":0,"terminal_stale":0,"sha256":{f.relative_to(ROOT).as_posix():digest(f) for f in files}}
    with receipt.open("x",encoding="utf-8") as f:
        json.dump(signed,f,indent=2)
    print(result[result.benchmark.eq("H00300")].to_string(index=False))


if __name__=="__main__":
    main()

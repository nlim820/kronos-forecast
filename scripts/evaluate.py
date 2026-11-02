# -*- coding: utf-8 -*-
"""
evaluate.py — Kronos 预测能力客观评估（Phase 2，缩规模验证版）

设计（先讲清楚，避免方法漏洞）：
- 对每只标的，在历史上取 N 个「滚动时点」(cutoff)。每个时点：
    context = 截至 cutoff 的最近 LOOKBACK 根 K 线
    actual  = cutoff 之后真实的 PRED_LEN 个交易日（从同一份数据切出）
- 用 Kronos 预测未来 PRED_LEN 日收盘价路径；同时构造 3 个解析基线：
    1. 动量延续  momentum   : 用截止日近 20 日收益外推（趋势延续假设）
    2. 均值回归  mean_revert : 预测价向 LOOKBACK 窗口均值收敛（Phase1 发现 Kronos 实质即此）
    3. 常数零收益 zero       : 预测未来无涨跌
- 对比指标（两个粒度）：
    A. 终点累计收益率 R = close[P]/close[0] - 1
       → 方向准确率 sign(R_pred)==sign(R_act)；IC=Pearson(R_pred,R_act)；RankIC=Spearman
    B. 逐日收益率
       → 同样算 IC / RankIC（样本量更大，更稳健）
  关键：若 Kronos 的 IC 不显著高于 mean_revert 基线，则证明它没有超越"向窗口均值回归"这一
        trivial 统计效应——这与 ashare_quant 双均线"无稳定 alpha"的结论一脉相承。

用法：
    python scripts/evaluate.py --symbols 600519,510300 --pred-len 20 --rollings 6 --samples 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Kronos, KronosPredictor, KronosTokenizer
import data_utils as du

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_ohlcv(symbol: str, days: int = 1100) -> pd.DataFrame:
    """取前复权日线，统一列名（与 predict_demo 一致）。"""
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    df = du.get_price(symbol, start, end, source="auto")
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = df[cols].astype({c: "float64" for c in cols[1:]})
    return df.dropna().reset_index(drop=True)


def forecast_paths(predictor, df_cut, lookback, pred_len, samples, seed_base=0):
    """对截断到 cutoff 的 df_cut 预测未来路径（sampling 取分位）。"""
    x_df = df_cut.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(df_cut["date"].iloc[-lookback:].values)
    y_dates = du.next_trade_days(df_cut["date"].iloc[-1], pred_len)
    y_ts = pd.Series(y_dates.values)

    paths = []
    for i in range(samples):
        np.random.seed(seed_base + i)
        try:
            import torch
            torch.manual_seed(seed_base + i)
        except Exception:
            pass
        pred = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )
        paths.append(pred["close"].to_numpy())
    return np.vstack(paths), y_dates


def rolling_cutoffs(df: pd.DataFrame, n: int, pred_len: int, gap: int = 80):
    """返回 n 个 cutoff 行索引：每个 cutoff 之后至少有 pred_len 个真实交易日。
    cutoff_i 位于 len - (pred_len+5) - i*gap ，即越往后时点越靠近今天。"""
    cutoffs = []
    for i in range(n):
        idx = len(df) - (pred_len + 5) - i * gap
        if idx < lookback_min:
            break
        cutoffs.append(idx)
    return sorted(cutoffs)  # 时间从早到晚


# 基线用的窗口长度（与 Kronos 内部 z-score 窗口尺度接近）
LOOKBACK = 400
lookback_min = LOOKBACK + 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="600519,510300")
    ap.add_argument("--pred-len", type=int, default=20)
    ap.add_argument("--rollings", type=int, default=6)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--gap", type=int, default=80)
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = ap.parse_args()

    global lookback_min
    lookback_min = args.lookback + 50

    names = {"600519": "贵州茅台", "510300": "沪深300ETF",
             "601611": "中国核建", "002600": "领益智造",
             "000001": "平安银行", "002594": "比亚迪"}
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"加载模型 {args.model} ...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

    # 收集所有 (方法, 终点累计收益, 实际累计收益) 与逐日 (预测收益, 实际收益)
    end_runs = {m: {"pred": [], "act": []} for m in
                ["kronos", "momentum", "mean_revert", "zero"]}
    daily_runs = {m: {"pred": [], "act": []} for m in
                  ["kronos", "momentum", "mean_revert", "zero"]}

    detail_rows = []

    for sym in symbols:
        print(f"\n=== {sym} {names.get(sym, sym)} ===")
        df = load_ohlcv(sym)
        cutoffs = rolling_cutoffs(df, args.rollings, args.pred_len, args.gap)
        print(f"  数据 {len(df)} 根；滚动时点 {len(cutoffs)} 个")

        for ci, cidx in enumerate(cutoffs):
            cutoff_date = df["date"].iloc[cidx]
            df_cut = df.iloc[:cidx + 1].reset_index(drop=True)
            actual = df.iloc[cidx + 1: cidx + 1 + args.pred_len].reset_index(drop=True)
            if len(actual) < args.pred_len:
                print(f"  时点 {cutoff_date:%Y-%m-%d}: 未来数据不足，跳过")
                continue

            last_close = float(df_cut["close"].iloc[-1])
            # 真实未来累计/逐日收益
            act_cum = actual["close"].to_numpy() / last_close - 1.0
            act_daily = actual["close"].to_numpy()[1:] / actual["close"].to_numpy()[:-1] - 1.0

            # ---- 基线（解析，无需模型）----
            # 动量：近 20 日累计收益外推
            past20 = df_cut["close"].iloc[-20:].to_numpy()
            mom_ret = past20[-1] / past20[0] - 1.0
            # 均值回归：窗口均值 / 最新收盘 - 1（向窗口均值收敛的幅度）
            win = df_cut["close"].iloc[-args.lookback:].to_numpy()
            revert_ret = win.mean() / last_close - 1.0
            # 逐日：动量=常数(单日近似)，均值回归=几何收敛日收益，zero=0
            mom_daily = (1.0 + mom_ret) ** (1.0 / args.pred_len) - 1.0
            rev_daily = (win.mean() / last_close) ** (1.0 / args.pred_len) - 1.0

            for m, pr, dr in [("momentum", mom_ret, mom_daily),
                              ("mean_revert", revert_ret, rev_daily),
                              ("zero", 0.0, 0.0)]:
                end_runs[m]["pred"].append(pr)
                end_runs[m]["act"].append(float(act_cum[-1]))
                daily_runs[m]["pred"].extend([dr] * len(act_daily))
                daily_runs[m]["act"].extend(list(act_daily))

            # ---- Kronos ----（带磁盘缓存：同 (标的,cutoff,参数) 只推理一次）
            cache_key = f"{sym}_{cutoff_date:%Y%m%d}_{args.pred_len}_{args.samples}"
            cache_path = OUT / ".kronos_cache" / f"{cache_key}.npz"
            if cache_path.exists():
                arr = np.load(cache_path)
                paths = arr["paths"]
                y_dates = pd.DatetimeIndex(arr["dates"])
            else:
                paths, y_dates = forecast_paths(
                    predictor, df_cut, args.lookback, args.pred_len, args.samples, seed_base=ci * 100)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, paths=paths, dates=y_dates.values)
            med = np.percentile(paths, 50, axis=0)
            kronos_cum = med / last_close - 1.0
            kronos_daily = med[1:] / med[:-1] - 1.0
            end_runs["kronos"]["pred"].append(float(kronos_cum[-1]))
            end_runs["kronos"]["act"].append(float(act_cum[-1]))
            daily_runs["kronos"]["pred"].extend(list(kronos_daily))
            daily_runs["kronos"]["act"].extend(list(act_daily))

            detail_rows.append({
                "symbol": sym, "cutoff": cutoff_date.strftime("%Y-%m-%d"),
                "last_close": last_close,
                "kronos_ret": float(kronos_cum[-1]), "act_ret": float(act_cum[-1]),
                "mom_ret": mom_ret, "revert_ret": revert_ret,
                "kronos_dir_hit": int(np.sign(kronos_cum[-1]) == np.sign(act_cum[-1])),
            })
            print(f"  时点 {cutoff_date:%Y-%m-%d}: Kronos {kronos_cum[-1]:+.2%} | "
                  f"实际 {act_cum[-1]:+.2%} | 回归基线 {revert_ret:+.2%}")

    # ---------- 指标计算 ----------
    def metrics(pred_list, act_list):
        pred = np.array(pred_list, float)
        act = np.array(act_list, float)
        n = len(pred)
        dir_acc = float(np.mean(np.sign(pred) == np.sign(act))) if n else float("nan")
        # IC：去除 NaN
        m = ~(np.isnan(pred) | np.isnan(act))
        ic = float(stats.pearsonr(pred[m], act[m])[0]) if m.sum() > 2 else float("nan")
        ric = float(stats.spearmanr(pred[m], act[m])[0]) if m.sum() > 2 else float("nan")
        return n, dir_acc, ic, ric

    def bootstrap_ic(pred_list, act_list, n_boot=2000, seed=42):
        """对 IC 做 bootstrap 2000 次重采样，返回 95% 置信区间 [lo, hi]。"""
        pred = np.array(pred_list, float)
        act = np.array(act_list, float)
        m = ~(np.isnan(pred) | np.isnan(act))
        pred, act = pred[m], act[m]
        n = len(pred)
        if n < 5:
            return float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        ics = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                r = stats.pearsonr(pred[idx], act[idx])[0]
                if not np.isnan(r):
                    ics.append(r)
            except Exception:
                pass
        ics = np.array(ics)
        if len(ics) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))

    print("\n================ 评估结果 ================")
    print(f"{'方法':<12}{'粒度':<8}{'n':>4}{'方向准确率':>10}{'IC':>8}{'IC_95%CI':>18}{'RankIC':>9}")
    rows = []
    for m in ["kronos", "momentum", "mean_revert", "zero"]:
        n_e, da_e, ic_e, ric_e = metrics(end_runs[m]["pred"], end_runs[m]["act"])
        n_d, da_d, ic_d, ric_d = metrics(daily_runs[m]["pred"], daily_runs[m]["act"])
        lo_e, hi_e = bootstrap_ic(end_runs[m]["pred"], end_runs[m]["act"])
        lo_d, hi_d = bootstrap_ic(daily_runs[m]["pred"], daily_runs[m]["act"])
        ci_e = f"[{lo_e:+.3f},{hi_e:+.3f}]" if not np.isnan(lo_e) else "    nan"
        ci_d = f"[{lo_d:+.3f},{hi_d:+.3f}]" if not np.isnan(lo_d) else "    nan"
        print(f"{m:<12}{'终点':<8}{n_e:>4}{da_e:>10.2%}{ic_e:>8.3f}{ci_e:>18}{ric_e:>9.3f}")
        print(f"{'':<12}{'逐日':<8}{n_d:>4}{da_d:>10.2%}{ic_d:>8.3f}{ci_d:>18}{ric_d:>9.3f}")
        rows.append({"method": m, "granularity": "end", "n": n_e,
                     "dir_acc": da_e, "ic": ic_e, "ic_ci_lo": lo_e, "ic_ci_hi": hi_e,
                     "rankic": ric_e})
        rows.append({"method": m, "granularity": "daily", "n": n_d,
                     "dir_acc": da_d, "ic": ic_d, "ic_ci_lo": lo_d, "ic_ci_hi": hi_d,
                     "rankic": ric_d})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "eval_result.csv", index=False, float_format="%.4f")
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "eval_detail.csv", index=False, float_format="%.4f")
    print(f"\n结果已保存: outputs/eval_result.csv, outputs/eval_detail.csv")

    # 关键结论提示
    ke, _, ic_k, _ = metrics(end_runs["kronos"]["pred"], end_runs["kronos"]["act"])
    _, _, ic_r, _ = metrics(end_runs["mean_revert"]["pred"], end_runs["mean_revert"]["act"])
    print(f"\n[Kronos 终点 IC = {ic_k:+.3f}] vs [均值回归基线 IC = {ic_r:+.3f}]")
    if abs(ic_k - ic_r) < 0.05:
        print("→ 两者 IC 接近：Kronos 未显著超越'向窗口均值回归'这一 trivial 效应。")
    else:
        print("→ Kronos 与均值回归基线存在差异，需进一步放大样本确认。")


if __name__ == "__main__":
    main()

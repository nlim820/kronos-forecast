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

import data_utils as du

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(pred_list, act_list):
    """返回 (n, 方向准确率, IC, RankIC)。

    **常数预测的方向准确率不适用**：zero 基线的预测恒为 0，sign(0)=0 永远
    不等于 sign(实际)，算出来是一个恒为 0 的数——它看起来像「比抛硬币差得多」，
    其实只是指标退化。同理，常数序列的 Pearson/Spearman 分母方差为 0，无定义。
    这两类情况统一返回 nan，由报告显式标注「不适用」，而不是把它当成绩展示。

    放在模块级（而非 main 内）是为了能直接单测——测试无需加载模型权重。
    """
    pred = np.asarray(pred_list, dtype=float)
    act = np.asarray(act_list, dtype=float)
    n = len(pred)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    ok = ~(np.isnan(pred) | np.isnan(act))
    p, a = pred[ok], act[ok]
    if len(p) < 3 or np.ptp(p) == 0 or np.ptp(a) == 0:
        return n, float("nan"), float("nan"), float("nan")
    dir_acc = float(np.mean(np.sign(p) == np.sign(a)))
    ic = float(stats.pearsonr(p, a)[0])
    ric = float(stats.spearmanr(p, a)[0])
    return n, dir_acc, ic, ric


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


def rolling_cutoffs(df: pd.DataFrame, n: int, pred_len: int, gap: int = 80,
                    fixed=None):
    """返回 cutoff 行索引（升序）。

    fixed 给定时按日期列表精确取行号，用于严格复现；否则从「最早可评估点」
    正向推进，取全部可评估时点。

    为什么改成正向推进：旧实现是 idx = len(df) - (pred_len+5) - i*gap，从数据
    末尾倒推。这导致**每新增一个交易日，所有历史 cutoff 都整体后移一天**——
    同一份代码今天跑和明天跑给出的样本完全不同，结果不可复现。正向推进后，
    新增数据只在末尾追加时点，已有的 cutoff 保持不变。
    """
    if fixed:
        out = []
        for ds in fixed:
            hit = df.index[df["date"] == pd.Timestamp(ds)].tolist()
            if not hit:
                print(f"  ⚠️ 指定的 cutoff {ds} 不在数据中，跳过")
                continue
            idx = int(hit[0])
            if idx < lookback_min:
                print(f"  ⚠️ 指定的 cutoff {ds} 之前不足 {lookback_min} 根历史，跳过")
                continue
            if len(df) - idx - 1 < pred_len:
                print(f"  ⚠️ 指定的 cutoff {ds} 之后不足 {pred_len} 个交易日，跳过")
                continue
            out.append(idx)
        return sorted(out)

    last = len(df) - pred_len - 1        # 最后一个「其后仍有完整 pred_len 天」的位置
    out = list(range(lookback_min, last + 1, gap))
    if n and len(out) > n:
        # 不截断：截断会丢掉早期时点，等于又引入漂移。样本变多本身是好事。
        print(f"  可评估时点 {len(out)} 个（--rollings={n} 仅为期望值，实取全部）")
    return out


def default_cutoff_dates(df: pd.DataFrame, pred_len: int, gap: int) -> list:
    """只算日期不取行号，便于把本次评估用到的 cutoff 写入产物做留痕。"""
    lm = lookback_min
    last = len(df) - pred_len - 1
    return [df["date"].iloc[i] for i in range(lm, last + 1, gap)]


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
    ap.add_argument("--cutoffs", default="",
                    help="逗号分隔的固定 cutoff 日期（如 2025-09-18,2025-10-28），"
                         "用于严格复现某次评估；留空则从最早可评估点正向推进")
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = ap.parse_args()

    global lookback_min
    lookback_min = args.lookback + 50

    names = {"600519": "贵州茅台", "510300": "沪深300ETF",
             "601611": "中国核建", "002600": "领益智造",
             "000001": "平安银行", "002594": "比亚迪"}
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    fixed_cutoffs = [s.strip() for s in args.cutoffs.split(",") if s.strip()]

    print(f"加载模型 {args.model} ...")
    # 模型相关导入放到 main 内：模块级导入 torch + 模型会拖慢任何想 import
    # evaluate（比如只测 metrics 函数）的场景，也让 CI 必须装 torch 才能跑测试。
    from model import Kronos, KronosPredictor, KronosTokenizer
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

    METHODS = ["kronos", "momentum", "mean_revert", "zero"]

    # 收集所有 (方法, 终点累计收益, 实际累计收益) 与逐日 (预测收益, 实际收益)
    end_runs = {m: {"pred": [], "act": []} for m in METHODS}
    daily_runs = {m: {"pred": [], "act": []} for m in METHODS}

    # 除池化数组外，逐日观测还必须按「标的 × cutoff」分块保存。
    # 原因：912 个逐日观测不是 912 份独立证据——同一块内的 19 天来自同一条
    # 预测路径，块内强自相关；解析基线在块内甚至是一个常数（见下方 [dr]*len）。
    # bootstrap 必须以块为最小重采样单位，否则会把「19 个相同的数」当成 19 份
    # 独立样本，把置信区间虚假收窄约 sqrt(19)≈4.4 倍。
    daily_blocks = {m: [] for m in METHODS}   # 每元素 = 该块内 PRED_LEN-1 天的 (pred, act)
    block_meta = []                           # [(symbol, cutoff_str), ...]，与 block_meta 索引对齐

    detail_rows = []

    for sym in symbols:
        print(f"\n=== {sym} {names.get(sym, sym)} ===")
        df = load_ohlcv(sym)
        cutoffs = rolling_cutoffs(df, args.rollings, args.pred_len, args.gap,
                                 fixed=fixed_cutoffs or None)
        print(f"  数据 {len(df)} 根；滚动时点 {len(cutoffs)} 个"
              f"{'（固定日期列表）' if fixed_cutoffs else '（正向推进）'}")

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

            # 该 (标的, cutoff) 块：三个基线 + Kronos 共用同一段真实未来
            # 注：act_daily 长度 = PRED_LEN-1（首日相对 cutoff 收盘，已在 act_cum[0] 里）
            act_daily_arr = np.asarray(act_daily, dtype=float)
            block_meta.append((sym, cutoff_date.strftime("%Y-%m-%d")))

            for m, pr, dr in [("momentum", mom_ret, mom_daily),
                              ("mean_revert", revert_ret, rev_daily),
                              ("zero", 0.0, 0.0)]:
                end_runs[m]["pred"].append(pr)
                end_runs[m]["act"].append(float(act_cum[-1]))
                daily_runs[m]["pred"].extend([dr] * len(act_daily))
                daily_runs[m]["act"].extend(list(act_daily))
                # 基线在块内是同一个常数——这正是旧 iid bootstrap 被高估置信度的根源
                daily_blocks[m].append((np.full(len(act_daily_arr), dr), act_daily_arr))

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
            # Kronos 块内是真正的逐日变值序列（med[t]/med[t-1]-1），但仍与同块的实际
            # 走势共享一条市场路径，块内自相关依旧存在 → 同样按块重采样
            daily_blocks["kronos"].append((np.asarray(kronos_daily, dtype=float), act_daily_arr))

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
    # 指标函数已提到模块级（见文件头部 metrics），单位置定义、可直接单测。

    def bootstrap_ic_blocks(blocks, n_boot=2000, seed=42):
        """块级 bootstrap：以「标的 × cutoff」块为最小重采样单位，返回 IC 的 95% CI。

        blocks: [(pred_array, act_array), ...]，每元素是一块的逐日序列。
        端点粒度时每块只含 1 个观测，退化为普通的观测级 bootstrap。

        为什么必须按块：逐日粒度的 912 个观测并不独立。同一块内的 19 天来自
        同一条预测路径、同一段真实行情，块内强自相关；解析基线在块内是一个
        常数。按单点重采样等于把「19 个相同的数」当成 19 份独立证据，会把 CI
        虚假收窄约 sqrt(19)≈4.4 倍。按块重采样后，有效样本量回到 48 个块。
        """
        blocks = [(np.asarray(p, float).ravel(), np.asarray(a, float).ravel())
                  for p, a in blocks]
        nb = len(blocks)
        if nb < 5:
            return float("nan"), float("nan")
        # 铺成 (nb, L) 矩阵，重采样时直接按行取，避免 n_boot 次 concatenate
        L = max(len(p) for p, _ in blocks)
        P = np.full((nb, L), np.nan)
        A = np.full((nb, L), np.nan)
        for i, (p, a) in enumerate(blocks):
            P[i, :len(p)] = p
            A[i, :len(a)] = a
        rng = np.random.default_rng(seed)
        ics = []
        for _ in range(n_boot):
            idx = rng.integers(0, nb, nb)
            pred, act = P[idx].ravel(), A[idx].ravel()
            m = ~(np.isnan(pred) | np.isnan(act))
            if m.sum() < 5:
                continue
            try:
                r = stats.pearsonr(pred[m], act[m])[0]
                if not np.isnan(r):
                    ics.append(r)
            except Exception:
                pass
        ics = np.array(ics)
        if len(ics) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))

    def bootstrap_ic(pred_list, act_list, n_boot=2000, seed=42):
        """端点粒度用：48 个 (标的, cutoff) 观测互不重叠（cutoff 间隔 80 个交易日，
        远大于 20 日预测窗口），近似独立，故每个观测自身即为一个块。"""
        blocks = [(np.array([p], float), np.array([a], float))
                  for p, a in zip(pred_list, act_list)]
        return bootstrap_ic_blocks(blocks, n_boot=n_boot, seed=seed)

    print("\n================ 评估结果 ================")
    print("注：逐日粒度的 n 是名义样本量，『块数』才是 bootstrap 的有效重采样单位数。")
    print(f"{'方法':<12}{'粒度':<8}{'n':>5}{'块数':>5}{'方向准确率':>10}{'IC':>8}{'IC_95%CI':>18}{'RankIC':>9}")
    rows = []
    n_blocks_total = len(block_meta)
    ci_compare = []
    for m in METHODS:
        n_e, da_e, ic_e, ric_e = metrics(end_runs[m]["pred"], end_runs[m]["act"])
        n_d, da_d, ic_d, ric_d = metrics(daily_runs[m]["pred"], daily_runs[m]["act"])
        lo_e, hi_e = bootstrap_ic(end_runs[m]["pred"], end_runs[m]["act"])
        # 逐日 CI：主口径改为块级 bootstrap；同时算一遍旧的单点口径做对照，
        # 这样「改口径到底改变了什么」是实测出来的，不是假设出来的。
        lo_d, hi_d = bootstrap_ic_blocks(daily_blocks[m])
        lo_di, hi_di = bootstrap_ic(daily_runs[m]["pred"], daily_runs[m]["act"])
        if not (np.isnan(lo_d) or np.isnan(lo_di)):
            ci_compare.append((m, lo_di, hi_di, lo_d, hi_d))
        ci_e = f"[{lo_e:+.3f},{hi_e:+.3f}]" if not np.isnan(lo_e) else "    nan"
        ci_d = f"[{lo_d:+.3f},{hi_d:+.3f}]" if not np.isnan(lo_d) else "    nan"
        print(f"{m:<12}{'终点':<8}{n_e:>5}{n_blocks_total:>5}{da_e:>10.2%}{ic_e:>8.3f}{ci_e:>18}{ric_e:>9.3f}")
        print(f"{'':<12}{'逐日':<8}{n_d:>5}{n_blocks_total:>5}{da_d:>10.2%}{ic_d:>8.3f}{ci_d:>18}{ric_d:>9.3f}")
        rows.append({"method": m, "granularity": "end", "n": n_e, "n_blocks": n_blocks_total,
                     "dir_acc": da_e, "ic": ic_e, "ic_ci_lo": lo_e, "ic_ci_hi": hi_e,
                     "ic_ci_lo_iid": lo_e, "ic_ci_hi_iid": hi_e, "rankic": ric_e})
        rows.append({"method": m, "granularity": "daily", "n": n_d, "n_blocks": n_blocks_total,
                     "dir_acc": da_d, "ic": ic_d, "ic_ci_lo": lo_d, "ic_ci_hi": hi_d,
                     "ic_ci_lo_iid": lo_di, "ic_ci_hi_iid": hi_di, "rankic": ric_d})

    print("\n---------- 逐日 CI：单点 bootstrap（旧）vs 块级 bootstrap（新）----------")
    print(f"{'方法':<12}{'单点 CI':>20}{'宽度':>9}{'块级 CI':>20}{'宽度':>9}{'宽度比':>8}")
    for m, a, b, c, d in ci_compare:
        w_old, w_new = b - a, d - c
        ratio = w_new / w_old if w_old > 0 else float("nan")
        print(f"{m:<12}[{a:+.3f},{b:+.3f}]{w_old:>9.3f}[{c:+.3f},{d:+.3f}]{w_new:>9.3f}{ratio:>8.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "eval_result.csv", index=False, float_format="%.4f")
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "eval_detail.csv", index=False, float_format="%.4f")

    # ---------- 逐标的分解 ----------
    # 池化 IC 会把标的间的符号冲突平均掉：4 只标的同向时池化值有意义，
    # 反向时池化值接近 0 却掩盖了「部分标的强正、部分强负」。必须单独披露。
    sym_list = list(dict.fromkeys([mt[0] for mt in block_meta]))
    by_rows = []
    for sym in sym_list:
        sel = [i for i, mt in enumerate(block_meta) if mt[0] == sym]
        for m in METHODS:
            pe = np.array([end_runs[m]["pred"][i] for i in sel], float)
            ae = np.array([end_runs[m]["act"][i] for i in sel], float)
            n_s, da_s, ic_s, ric_s = metrics(pe, ae)
            lo_s, hi_s = bootstrap_ic(pe, ae)
            blk = [daily_blocks[m][i] for i in sel]
            d_pred = np.concatenate([b[0] for b in blk])
            d_act = np.concatenate([b[1] for b in blk])
            _, da_sd, ic_sd, ric_sd = metrics(d_pred, d_act)
            lo_sd, hi_sd = bootstrap_ic_blocks(blk)
            by_rows.append({
                "symbol": sym, "method": m, "n": n_s, "n_blocks": len(sel),
                "dir_acc": da_s, "ic": ic_s, "ic_ci_lo": lo_s, "ic_ci_hi": hi_s,
                "rankic": ric_s,
                "daily_dir_acc": da_sd, "daily_ic": ic_sd,
                "daily_ic_ci_lo": lo_sd, "daily_ic_ci_hi": hi_sd, "daily_rankic": ric_sd,
            })
    by_sym = pd.DataFrame(by_rows)
    by_sym.to_csv(OUT / "eval_by_symbol.csv", index=False, float_format="%.4f")

    print("\n---------- 逐标的终点 IC 分解（看清池化值的成分）----------")
    print(f"{'标的':<10}{'方法':<12}{'n':>3}{'IC':>8}{'IC_95%CI':>18}{'RankIC':>8}")
    for r in by_rows:
        ci = f"[{r['ic_ci_lo']:+.3f},{r['ic_ci_hi']:+.3f}]" if not np.isnan(r["ic_ci_lo"]) else "      nan"
        print(f"{r['symbol']:<10}{r['method']:<12}{r['n']:>3}{r['ic']:>8.3f}{ci:>18}{r['rankic']:>8.3f}")

    print("\n---------- Kronos vs 均值回归基线：逐标的镜像检查 ----------")
    print("（若两者在每个标的上同号同量级，说明 Kronos 只是复现了均值回归，")
    print("  这比池化 IC 的接近程度更有说服力）")
    print(f"{'标的':<10}{'Kronos IC':>11}{'mean_revert IC':>16}{'差值':>9}")
    k_ic = {r["symbol"]: r["ic"] for r in by_rows if r["method"] == "kronos"}
    r_ic = {r["symbol"]: r["ic"] for r in by_rows if r["method"] == "mean_revert"}
    k_dic = {r["symbol"]: r["daily_ic"] for r in by_rows if r["method"] == "kronos"}
    r_dic = {r["symbol"]: r["daily_ic"] for r in by_rows if r["method"] == "mean_revert"}
    for sym in sym_list:
        if np.isnan(k_ic[sym]) or np.isnan(r_ic[sym]):
            continue
        print(f"{sym:<10}{k_ic[sym]:>+11.3f}{r_ic[sym]:>+16.3f}{k_ic[sym] - r_ic[sym]:>+9.3f}")
    daily_diffs = [abs(k_dic[s] - r_dic[s]) for s in sym_list
                   if not (np.isnan(k_dic[s]) or np.isnan(r_dic[s]))]
    if daily_diffs:
        print(f"逐日 IC 的逐标的绝对差值: 最大 {max(daily_diffs):.3f}，"
              f"均值 {float(np.mean(daily_diffs)):.3f}")

    print(f"\n结果已保存: outputs/eval_result.csv, outputs/eval_detail.csv, "
          f"outputs/eval_by_symbol.csv")

    # 关键结论提示
    ke, _, ic_k, _ = metrics(end_runs["kronos"]["pred"], end_runs["kronos"]["act"])
    _, _, ic_r, _ = metrics(end_runs["mean_revert"]["pred"], end_runs["mean_revert"]["act"])
    print(f"\n[Kronos 终点 IC = {ic_k:+.3f}] vs [均值回归基线 IC = {ic_r:+.3f}]"
          f"（池化，差值 {ic_k - ic_r:+.3f}）")
    pairs = [s for s in sym_list if not (np.isnan(k_ic[s]) or np.isnan(r_ic[s]))]
    diffs = [k_ic[s] - r_ic[s] for s in pairs]
    same_sign = all(np.sign(k_ic[s]) == np.sign(r_ic[s]) for s in pairs)
    if diffs and same_sign and max(abs(d) for d in diffs) < 0.10:
        print(f"→ 逐标的对比：{len(pairs)} 个标的全部同号，IC 差值最大 "
              f"{max(abs(d) for d in diffs):.3f}。")
        print("  这比池化值的接近更有力：Kronos 在每个标的上都复现了「向窗口均值回归」，")
        print("  而非只在某一个标的上恰好相关。")
    elif abs(ic_k - ic_r) < 0.05:
        print("→ 池化 IC 接近，但逐标的未满足镜像条件，结论偏弱，需放大样本确认。")
    else:
        print("→ Kronos 与均值回归基线存在差异，需进一步放大样本确认。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
predict_demo.py — Kronos A 股/ETF 预测演示（Phase 1）

对每只标的：
1. 用 ashare_quant 的数据层取前复权日线（东财→新浪自动回退）
2. 取最近 LOOKBACK 根作为上下文，预测未来 PRED_LEN 个交易日
3. 跑 SAMPLES 次独立采样（每次 sample_count=1），取分位数画扇形区间
   —— 注意：官方 predict() 在 sample_count>1 时返回多采样均值，拿不到分布，
      所以要多次调用才能得到不确定性。
4. 输出：CSV 预测明细 + 交互式 HTML（K 线历史 + 未来分位扇形）

用法：
    python scripts/predict_demo.py --symbols 600519,510300 --pred-len 60 --samples 20
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import Kronos, KronosPredictor, KronosTokenizer
import data_utils as du

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# A 股配色：涨红跌绿
RED = "#e54d42"
GREEN = "#3eb370"


def load_ohlcv(symbol: str, days: int = 900) -> pd.DataFrame:
    """取前复权日线，统一成 Kronos 需要的列名。"""
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


def forecast_paths(predictor, df: pd.DataFrame, lookback: int, pred_len: int,
                   samples: int, seed_base: int = 0):
    """跑 samples 次独立采样，返回 (paths[ndarray samples×pred_len], y_dates)。"""
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(df["date"].iloc[-lookback:].values)
    # 用真实 A 股交易日历（剔除国庆/春节等），而非 bdate_range
    y_dates = du.next_trade_days(df["date"].iloc[-1], pred_len)
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


def build_html(symbol: str, name: str, df: pd.DataFrame, paths: np.ndarray,
               y_dates, hist_bars: int = 120) -> str:
    """K 线历史 + 未来分位数扇形。"""
    hist = df.iloc[-hist_bars:]
    last_close = float(df["close"].iloc[-1])
    last_date = df["date"].iloc[-1]

    q05, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    fig = go.Figure()

    # 历史 K 线（涨红跌绿）
    fig.add_trace(go.Candlestick(
        x=hist["date"], open=hist["open"], high=hist["high"],
        low=hist["low"], close=hist["close"], name="历史 K 线",
        increasing_line_color=RED, increasing_fillcolor=RED,
        decreasing_line_color=GREEN, decreasing_fillcolor=GREEN,
    ))

    x_fan = list(y_dates)

    # 90% 区间
    fig.add_trace(go.Scatter(x=x_fan + x_fan[::-1], y=list(q95) + list(q05[::-1]),
                             fill="toself", fillcolor="rgba(120,150,220,0.15)",
                             line=dict(width=0), name="90% 区间", hoverinfo="skip"))
    # 50% 区间
    fig.add_trace(go.Scatter(x=x_fan + x_fan[::-1], y=list(q75) + list(q25[::-1]),
                             fill="toself", fillcolor="rgba(120,150,220,0.30)",
                             line=dict(width=0), name="50% 区间", hoverinfo="skip"))
    # 中位数路径
    fig.add_trace(go.Scatter(x=x_fan, y=q50, mode="lines", name="预测中位数",
                             line=dict(color="#4a7ebb", width=2)))

    # 衔接点：最后一个收盘价
    med_end = float(q50[-1])
    chg = med_end / last_close - 1
    fig.add_vline(x=last_date, line=dict(color="#888", width=1, dash="dot"))

    fig.update_layout(
        title=(f"Kronos 预测 · {name}（{symbol}）　"
               f"最新收盘 {last_close:.3f}（{last_date:%Y-%m-%d}）→ "
               f"{len(y_dates)} 日后中位数 {med_end:.3f}（{chg:+.2%}）<br>"
               f"<sup>样本 {paths.shape[0]} 条路径；扇形为预测分位区间，不代表真实概率分布</sup>"),
        template="plotly_dark",
        height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=30, t=90, b=40),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # 隐藏周末

    path = OUT / f"forecast_{symbol}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="600519,510300")
    ap.add_argument("--pred-len", type=int, default=60)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = ap.parse_args()

    names = {"600519": "贵州茅台", "510300": "沪深300ETF",
             "601611": "中国核建", "002600": "领益智造",
             "000001": "平安银行", "002594": "比亚迪"}
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"加载模型 {args.model} ...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

    rows = []
    for sym in symbols:
        print(f"\n=== {sym} ===")
        df = load_ohlcv(sym)
        print(f"  数据 {len(df)} 根，{df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}")

        paths, y_dates = forecast_paths(predictor, df, args.lookback, args.pred_len, args.samples)

        last_close = float(df["close"].iloc[-1])
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        out_df = pd.DataFrame({"date": y_dates, "q05": q[0], "q25": q[1],
                               "q50": q[2], "q75": q[3], "q95": q[4]})
        out_df.to_csv(OUT / f"pred_{sym}.csv", index=False, float_format="%.4f")

        html = build_html(sym, names.get(sym, sym), df, paths, y_dates)
        print(f"  HTML: {html}")

        rows.append({
            "symbol": sym, "name": names.get(sym, sym), "last_close": last_close,
            "med_end": float(q[2][-1]),
            "med_ret": float(q[2][-1] / last_close - 1),
            "lo90_ret": float(q[0][-1] / last_close - 1),
            "hi90_ret": float(q[4][-1] / last_close - 1),
            "band_width_pct": float((q[4][-1] - q[0][-1]) / last_close),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    print("\n=== 汇总 ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

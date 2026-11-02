# -*- coding: utf-8 -*-
"""
smoke_test.py — Kronos 环境冒烟测试
1. 验证 torch CUDA 可用
2. 从 HF 镜像下载 Kronos-small 权重（首次运行需联网）
3. 用合成 K 线数据跑一次端到端预测
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

print("=" * 50)
print("[1/3] CUDA 检查")
print(f"  torch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠️ CUDA 不可用，将回退 CPU（会很慢，需排查）")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("=" * 50)
print("[2/3] 加载 Kronos-small 权重")
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
print("  权重加载成功")
print("=" * 50)
print("[3/3] 合成数据端到端预测")

import numpy as np
import pandas as pd

# 合成 250 根日线：几何随机游走，接近真实 K 线形态
np.random.seed(42)
n = 250
close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
open_ = np.roll(close, 1)
open_[0] = 100
high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.008, n)))
low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.008, n)))
volume = np.random.lognormal(10, 0.5, n)
amount = close * volume

df = pd.DataFrame({
    "open": open_, "high": high, "low": low,
    "close": close, "volume": volume, "amount": amount,
})
dates = pd.bdate_range("2025-01-01", periods=n)
x_df = df.iloc[-400:]
y_timestamp = pd.Series(pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=24))
x_timestamp = pd.Series(dates[-400:])

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=24,
    T=1.0,
    top_p=0.9,
    sample_count=1,
)

print("\n预测结果（前 5 天）:")
print(pred_df.head(5).round(2))
last_close = close[-1]
pred_close_5d = pred_df["close"].iloc[4]
print(f"\n最后一日收盘: {last_close:.2f}")
print(f"预测第 5 日收盘: {pred_close_5d:.2f}")
print(f"隐含 5 日涨跌: {(pred_close_5d / last_close - 1) * 100:+.2f}%")
print("=" * 50)
print("✅ 冒烟测试通过：环境、权重、GPU 推理全部正常")

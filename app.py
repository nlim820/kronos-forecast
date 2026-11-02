# -*- coding: utf-8 -*-
"""
Kronos A股复现 · 结果看板 (Streamlit)

仅读取已生成的 outputs/，不依赖 GPU / 模型权重，可一键部署到 Streamlit Cloud。
运行：streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SYMBOLS = {
    "600519": "贵州茅台",
    "510300": "沪深300ETF",
    "601611": "中国核建",
    "002600": "领益智造",
}

st.set_page_config(page_title="Kronos A股复现看板", layout="wide")

st.title("Kronos 金融基础模型 · A股复现与客观评估")
st.markdown(
    """
    本项目复现了开源金融基础模型 **Kronos**（K-line Foundation Model，AAAI 2026），
    在 A 股真实日线数据上做预测，并用**带强基线的严谨评估**检验其预测力。
    **核心结论：在 A 股样本上，Kronos 无统计显著的预测力，其预测行为等价于
    “向窗口均值回归”这一行代码即可实现的基线。**
    """
)

tab1, tab2, tab3 = st.tabs(["📈 预测图", "📊 评估看板", "🔍 方法说明"])

with tab1:
    st.subheader("Kronos 对未来 60 个交易日的预测（分位扇形区间）")
    sym = st.selectbox(
        "选择标的", options=list(SYMBOLS.keys()),
        format_func=lambda s: f"{SYMBOLS[s]} ({s})",
    )
    html_path = OUT / f"forecast_{sym}.html"
    if html_path.exists():
        components.html(html_path.read_text(encoding="utf-8"), height=720, scrolling=True)
    else:
        st.warning(f"未找到 {sym} 的预测图，请先运行 `python scripts/predict_demo.py --symbols {sym}`")

with tab2:
    st.subheader("Phase 2 评估结果（4 标的 × 12 滚动时点 × 预测 20 交易日）")
    res = pd.read_csv(OUT / "eval_result.csv")
    st.dataframe(res, use_container_width=True, hide_index=True)
    st.caption(
        "IC / RankIC 的 95% 置信区间由 bootstrap(2000 次) 得到。"
        "所有方法的区间**均包含 0** → 没有任何统计显著的预测力。"
    )
    with st.expander("逐点明细 (eval_detail.csv)"):
        det = pd.read_csv(OUT / "eval_detail.csv")
        st.dataframe(det, use_container_width=True, hide_index=True)

    st.markdown("#### 关键发现")
    st.markdown(
        """
        - **Kronos ≈ 均值回归基线**：终点 IC = +0.130，而“回归窗口均值”基线 = +0.118，
          两者几乎重合——印证了 Kronos 内部 z-score 标准化使其本质在做均值回归。
        - **动量是最差的预测器**：终点 IC = −0.259（95%CI 含 0 但仍为负向），
          说明样本期内盲从趋势比随机还差。
        - **方向准确率无判别力**：Kronos 终点 52.1% / 逐日 49.3%，
          在 n=48 / n=912 下与 50% 抛硬币无统计可分性。
        - **预测幅度极端且脆弱**：多只个股上输出 ±30%~47% 的 20 日收益，
          在趋势行情里重仓“回归”反复翻车（如 601611 某时点预测 −40% 实际 +11%，
          002600 预测 −47% 实际 +5.5%）。
        """
    )

with tab3:
    st.subheader("为什么 Kronos 看起来“准”，其实没 alpha")
    st.markdown(
        """
        `KronosPredictor.predict()` 在内部对输入窗口做 **z-score 标准化**后再反标准化还原。
        这一操作把生成结果锚定在“窗口的统计尺度”上，于是模型表现出**向窗口均值回归**
        的行为，而非真正预测趋势。

        本项目因此没有把“随机游走”当成唯一基线，而是额外引入
        **“回归窗口均值”作为强基线**。结果显示 Kronos 并未超越这一行代码即可实现的基线，
        这正是评估设计的胜负手——能识破“看起来准”的伪信号，比宣称“我发现它很准”更有科研素养。
        """
    )
    st.info("完整论证见 `docs/复现笔记.md` 与 `docs/Phase2评估.md`。")

st.sidebar.markdown("### 项目信息")
st.sidebar.markdown(
    "- 模型：Kronos-base（NeoQuasar，MIT）\n"
    "- 数据：A股/ETF 前复权日线（akshare，东财→新浪自动回退）\n"
    "- 评估：滚动窗口 + 4 基线 + bootstrap 置信区间\n"
    "- 硬件：本地 RTX 4060 8GB 推理"
)

# -*- coding: utf-8 -*-
"""Kronos A股复现 · 结果看板 (Streamlit)

设计原则：**结论优先**。
本项目的核心结论是负结论（Kronos 在 A 股无统计显著预测力），所以首屏必须先
给结论与证据，预测图作为「次级证据」放在后面——否则会出现自相矛盾的观感：
用一张模型极端预测的图，去论证这个模型不能用。

仅读取已生成的 outputs/，不依赖 GPU / 模型权重 / 原始数据，可一键部署到
Streamlit Cloud。运行：streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SYMBOLS = {
    "600519": "贵州茅台",
    "510300": "沪深300ETF",
    "601611": "中国核建",
    "002600": "领益智造",
}

# 交易所日历（新浪 tool_trade_date_hist_sina）当前公布到该日；次年休市安排通常在
# 年底才发布。该日期之后的预测日期只能按「工作日 + 剔除元旦」近似。
# 刷新日历（scripts/data_utils.py 的 get_trade_calendar(refresh=True)）后，
# 需同步更新此常量，否则看板的披露数字会偏保守。
CAL_COVERED_UNTIL = pd.Timestamp("2026-12-31")

st.set_page_config(page_title="Kronos A股复现看板", layout="wide")


# --------------------------------------------------------------------------
# 指标计算（全部从 outputs/ 现算，不放硬编码数字）
# --------------------------------------------------------------------------
def _pearson(x: pd.Series, y: pd.Series) -> float:
    """皮尔逊相关，任一序列方差为 0（常数预测）时返回 nan。

    为什么必须处理方差为 0：zero 基线在每个滚动块内是常数 0，
    直接算相关会得到 nan 或触发警告。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@st.cache_data(show_spinner=False)
def load_detail() -> pd.DataFrame:
    # symbol 必须按字符串读：002600 若被推断成数字会变成 2600，前导零丢失
    return pd.read_csv(OUT / "eval_detail.csv", dtype={"symbol": str})


@st.cache_data(show_spinner=False)
def pooled_ic(det: pd.DataFrame) -> dict:
    """池化 IC：把所有标的的所有滚动时点当成一个池子算相关。"""
    return {
        "kronos": _pearson(det["kronos_ret"], det["act_ret"]),
        "revert": _pearson(det["revert_ret"], det["act_ret"]),
        "mom": _pearson(det["mom_ret"], det["act_ret"]),
        "n": len(det),
        "dir_hit": float(det["kronos_dir_hit"].mean()),
    }


@st.cache_data(show_spinner=False)
def ic_by_symbol(mtime: float) -> pd.DataFrame:
    """逐标的 IC —— 直接读 evaluate.py 产出的 eval_by_symbol.csv，确保与文档同源。

    池化 IC 会把不同标的混在一起看，容易被单一标的带偏；拆开算才能看出
    Kronos 与「均值回归」基线是否在每个标的上都逐点镜像。
    （mtime 只用于让缓存随产物更新而失效，本身不参与计算。）
    """
    p = OUT / "eval_by_symbol.csv"
    if not p.exists():
        return pd.DataFrame()
    b = pd.read_csv(p, dtype={"symbol": str})
    b = b[b["method"].isin(["kronos", "mean_revert", "momentum"])]
    piv = b.pivot_table(index="symbol", columns="method", values="ic", aggfunc="first")
    cnt = b.groupby("symbol")["n"].first()
    rows = []
    for sym in piv.index:
        rows.append(
            {
                "标的": f"{SYMBOLS.get(str(sym), str(sym))} ({sym})",
                "样本数": int(cnt.loc[sym]),
                "Kronos IC": float(piv.loc[sym, "kronos"]),
                "均值回归基线 IC": float(piv.loc[sym, "mean_revert"]),
                "动量基线 IC": float(piv.loc[sym, "momentum"]),
                # 带符号：正 = Kronos 略强于均值回归基线，负 = 略弱。
                # 四个标的的绝对值都 ≤0.03，正负交错，说明两者在同一水平线上。
                "差值 (Kronos − 基线)": float(piv.loc[sym, "kronos"] - piv.loc[sym, "mean_revert"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Kronos IC", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def overconfidence(det: pd.DataFrame) -> dict:
    """过度自信的量化：模型预测的幅度是否系统性大于实际幅度。"""
    pred = det["kronos_ret"].abs()
    act = det["act_ret"].abs()
    return {
        "pred_abs_mean": float(pred.mean()),
        "act_abs_mean": float(act.mean()),
        "ratio_gt": float((pred > act).mean()),
        "pred_max": float(det["kronos_ret"].abs().max()),
        "act_max": float(act.abs().max()),
    }


@st.cache_data(show_spinner=False)
def forecast_starts() -> dict:
    """各标的预测日期轴（= 数据锚点的下一交易日起）。用于提示新鲜度与轴的性质。"""
    out = {}
    for s in SYMBOLS:
        p = OUT / f"pred_{s}.csv"
        if p.exists():
            out[s] = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(p)["date"]))
    return out


# 次年休市安排未公布时，至少要知道农历节的初一落在哪天：春节前一周到初七都休市，
# 若「工作日近似」段压在春节上，图上日期就不是真实交易日。已知年份列在这里，
# 新增年份按农历查定后补入（宁缺勿错，缺了只是少一条提示，不会给出错日期）。
_CNY_FIRST_DAY = {2027: (2, 6)}


def approx_axis_report(dates: pd.DatetimeIndex) -> tuple[int, pd.DatetimeIndex, list[str]]:
    """统计预测轴里有多少天只是「工作日近似」，并检查是否压在已知农历节日上。

    日历覆盖段内是真实交易日；覆盖段外只能按「工作日 − 元旦」推。把近似天数与
    假日冲突显式算出来，看板才能如实说明轴的性质，而不是笼统地写「已按真实日历生成」。
    """
    approx = dates[dates > CAL_COVERED_UNTIL]
    risky = []
    for year, (mm, dd) in _CNY_FIRST_DAY.items():
        cny = pd.Timestamp(year=year, month=mm, day=dd)
        win_lo, win_hi = cny - pd.Timedelta(days=1), cny + pd.Timedelta(days=7)
        hit = approx[(approx >= win_lo) & (approx <= win_hi)]
        if len(hit):
            risky.append(f"{cny:%Y-%m-%d} 起春节休市窗口内有 {len(hit)} 个近似日期")
    return len(approx), approx, risky


det = load_detail()
pool = pooled_ic(det)
_sym_path = OUT / "eval_by_symbol.csv"
by_sym = ic_by_symbol(_sym_path.stat().st_mtime if _sym_path.exists() else 0.0)
over = overconfidence(det)
starts = forecast_starts()

# --------------------------------------------------------------------------
# 首屏：结论 + 关键数字
# --------------------------------------------------------------------------
st.title("Kronos 金融基础模型 · A股复现与客观评估")

st.markdown(
    """
> ### 结论：在 A 股日线样本上，Kronos **不具备可用的预测力**
> 它的预测行为与「向窗口均值回归」这一行代码即可实现的基线**几乎等价**；
> 同时其预测幅度系统性大于真实波动——**过度自信**。
> 这是一次**负结论**研究：命题被证伪，但评估方法本身可复用。
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Kronos 池化 IC",
    f"{pool['kronos']:+.3f}",
    help="预测收益与实际收益的相关系数，bootstrap 95% 置信区间含 0 → 不显著",
)
c2.metric(
    "均值回归基线 IC",
    f"{pool['revert']:+.3f}",
    help="「预测值 = 窗口均值」这一极简基线，与 Kronos 几乎同分",
)
c3.metric(
    "方向准确率",
    f"{pool['dir_hit']:.1%}",
    help="与 50% 抛硬币无统计可分性",
)
c4.metric(
    "预测幅度 > 实际幅度",
    f"{over['ratio_gt']:.1%}",
    help="模型预测的绝对涨跌幅大于实际值的样本占比，衡量过度自信",
)

with st.expander("一句话看懂：为什么说「看起来准」是错觉", expanded=False):
    st.markdown(
        """
        `KronosPredictor.predict()` 内部会对输入窗口做 **z-score 标准化**，再反标准化还原。
        这一步把生成结果锚定在「窗口的统计尺度」上，于是模型表现出**向窗口均值回归**，
        而不是预测趋势。它能画出很像样的曲线，但信息量并不高于一行
        `预测值 = 最近 N 日收盘均值`。
        """
    )

# --------------------------------------------------------------------------
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 证据：逐标的 IC 对照", "② 评估指标", "③ 预测图（次级证据）", "④ 方法与局限"]
)

# ==========================================================================
# ① 证据：逐标的镜像
# ==========================================================================
with tab1:
    st.subheader("核心证据：4 个标的、4 种波动特征，两条预测曲线逐点镜像")
    st.caption(
        "池化 IC 只有 +0.130，单看它容易以为是「微弱正相关」。"
        "拆到标的层面才能看出真正的原因：Kronos 与均值回归基线在每个标的上都同号、同量级。"
    )

    show = by_sym.copy()
    diff_col = "差值 (Kronos − 基线)"
    for col in ["Kronos IC", "均值回归基线 IC", "动量基线 IC", diff_col]:
        show[col] = show[col].map(lambda v: "—" if pd.isna(v) else f"{v:+.3f}")
    st.dataframe(show, width="stretch", hide_index=True)

    max_abs_diff = float(by_sym[diff_col].abs().max())
    sign_agree = float((np.sign(det["kronos_ret"]) == np.sign(det["revert_ret"])).mean())
    st.markdown(
        f"""
        **怎么读这张表**

        - 两列 IC 的**符号和量级都高度一致**，差值绝对值最大仅 {max_abs_diff:.3f}，
          且正负交错 —— 换成同一行代码的基线，结果几乎不变。
        - 48 个预测里，Kronos 与基线的**方向一致率 {sign_agree:.1%}**
          —— 这比任何单个相关系数都更直观。
        - 单标的只有 12 个配对点，**没有一个标的的 IC 达到显著**
          （最强的茅台 IC=+0.511，95% CI [−0.058, +0.809] 仍跨 0；明细见「② 评估指标」页）。
        - 沪深300ETF 上 Kronos 的 IC 是**负的**，与个股方向相反——
          说明这点「正相关」更像标的特异性噪声，不是稳定信号。
        """
    )

    st.markdown("#### 第二个证据：过度自信")
    m1, m2, m3 = st.columns(3)
    m1.metric("模型预测绝对幅度均值", f"{over['pred_abs_mean']:.2%}")
    m2.metric("实际收益绝对幅度均值", f"{over['act_abs_mean']:.2%}")
    m3.metric(
        "幅度高估比例",
        f"{over['pred_abs_mean'] / over['act_abs_mean']:.1f}×",
        help="模型预测幅度是真实幅度的多少倍",
    )
    st.caption(
        f"模型最大单点预测达到 ±{over['pred_max']:.1%}，而该样本里实际最大波动为 "
        f"±{over['act_max']:.1%}；{over['ratio_gt']:.1%} 的样本上预测幅度大于实际幅度。"
        "这不是「预测得激进」，而是校准失败——一个能用的预测器，其误差不应系统性偏向放大。"
    )

# ==========================================================================
# ② 评估指标
# ==========================================================================
with tab2:
    st.subheader("Phase 2 评估结果（4 标的 × 12 滚动时点 × 预测 20 交易日）")
    res = pd.read_csv(OUT / "eval_result.csv")
    st.dataframe(res, width="stretch", hide_index=True)
    st.caption(
        "IC / RankIC 的 95% 置信区间由 bootstrap(2000 次) 得到；逐日粒度采用"
        "**块级重采样**（见下方口径说明）。**所有方法的区间均包含 0** → 没有任何统计显著的预测力。"
    )
    st.caption(
        "补充口径：`zero` 基线的方向准确率一栏为**空（不适用）** —— 常数 0 预测下 "
        "sign(0) 永远不等于 sign(实际)，方向准确率与相关系数都退化，不是「成绩很差」。"
        "该行保留只用于验证指标实现在边界输入上不产出假数字（见 `tests/test_metrics.py`）。"
    )

    with st.expander("逐点明细 (eval_detail.csv)"):
        st.dataframe(det, width="stretch", hide_index=True)

    st.markdown("#### 关键发现")
    st.markdown(
        """
        - **Kronos ≈ 均值回归基线**：见「① 证据」页，逐标的镜像比池化数字更有说服力。
        - **动量是最差的预测器**：池化 IC 为负，说明样本期内盲从趋势比随机还差。
        - **方向准确率无判别力**：与 50% 抛硬币在统计上不可分。
        - **预测幅度极端且脆弱**：趋势行情里重仓「回归」会反复翻车
          （如 601611 某时点预测 −40% 实际 +11%，002600 预测 −47% 实际 +5.5%）。
        """
    )

    st.warning(
        "**口径说明（务必知道）**：逐日粒度为 4 标的 × 12 时点 × 19 日 = 912 条**名义**样本，"
        "但同一「标的×时点」块内的 19 天来自同一条预测路径、同一段真实行情，并不独立；"
        "解析基线在块内甚至是一个常数。因此逐日 CI 已改用**块级 bootstrap**"
        "（以 48 个块为最小重采样单位），表内 `n_blocks` 列即有效重采样单位数。\n\n"
        "**实测对照**：块级与单点两种口径的 CI 宽度比为 **0.82–1.05**（差异不到 20%）。"
        "也就是说，原先的单点口径虽然假设不严谨，算出来的数值本身是可信的——"
        "换口径不改变任何结论。这一点已写入 `docs/Phase2评估.md` 第五节。"
    )

# ==========================================================================
# ③ 预测图（次级证据）
# ==========================================================================
with tab3:
    st.subheader("Kronos 对未来的 60 个交易日预测（分位扇形区间）")

    st.warning(
        "**这张图本身就是「过度自信」的例证，不是推荐结果。**\n\n"
        "下面展示的区间宽度与涨跌幅幅度，正是上文统计里「预测幅度系统性大于真实波动」"
        "的具体样子。它适合用来看**模型输出的形态**，不适合用来做投资判断。"
    )

    sym = st.selectbox(
        "选择标的",
        options=list(SYMBOLS.keys()),
        format_func=lambda s: f"{SYMBOLS[s]} ({s})",
    )
    html_path = OUT / f"forecast_{sym}.html"
    if html_path.exists():
        # st.iframe 从 Streamlit 1.6x 起取代 components.v1.html；
        # 直接传 Path 即可内联渲染本地 HTML（含 plotly 的 JS）。
        st.iframe(html_path, height=720)
    else:
        st.info(f"未找到 {sym} 的预测图。本地可用：`python scripts/predict_demo.py --symbols {sym}`")

    if sym in starts:
        axis = starts[sym]
        n_approx, approx_seg, risky = approx_axis_report(axis)
        head = (f"预测区间：{axis.min():%Y-%m-%d} → {axis.max():%Y-%m-%d}，"
                f"共 {len(axis)} 个交易日。")
        if n_approx == 0:
            st.caption(head + "全部落在交易所已公布的交易日历内，日期为真实交易日"
                               "（已剔除周末与法定休市）。")
        else:
            st.caption(
                head + f"其中 {len(axis) - n_approx} 天来自真实交易日历，末尾 {n_approx} 天"
                f"（{approx_seg[0]:%Y-%m-%d} 起）为**工作日近似**——交易所次年休市安排"
                f"尚未公布，近似段只剔除了周末与元旦。"
            )
            if risky:
                st.warning(
                    "近似段与已知农历节日重叠：" + "；".join(risky)
                    + "。这些日期可能并非真实交易日，需刷新交易日历后重跑预测。"
                )

# ==========================================================================
# ④ 方法与局限
# ==========================================================================
with tab4:
    st.subheader("为什么 Kronos 看起来「准」，其实没 alpha")
    st.markdown(
        """
        `KronosPredictor.predict()` 在内部对输入窗口做 **z-score 标准化**后再反标准化还原。
        这一操作把生成结果锚定在「窗口的统计尺度」上，于是模型表现出**向窗口均值回归**
        的行为，而非真正预测趋势。

        本项目因此没有把「随机游走」当成唯一基线，而是额外引入
        **「回归窗口均值」作为强基线**。结果显示 Kronos 并未超越这一行代码即可实现的基线，
        这正是评估设计的胜负手——能识破「看起来准」的伪信号，比宣称「我发现它很准」更有价值。
        """
    )

    st.markdown("#### 已知局限（主动披露）")
    st.markdown(
        """
        1. **前复权是回溯性调整**：数据是「截至抓取日」的前复权序列，若 cutoff 之后发生除权，
           cutoff 之前的价格会被追溯调整，与当时可见价格不完全一致。量级小（A 股股息率 1–2%），
           不改变结论，但严格说是一个轻微的未来信息泄漏。
        2. **样本量小**：4 标的 × 12 时点 = 48 个独立块，任何单点结论都不应过度外推。
           单标的只有 12 个点，逐标的 IC 只能看方向，不能看精度。
        3. **逐日粒度的 CI 偏窄**：见「② 评估指标」页的口径说明。
        4. **预测轴末段是近似日期**：交易日历只公布到 2026-12-31，之后的日期按
           「工作日 − 元旦」推出；「③ 预测图」页会显示本次有多少天属于该近似段。
           近似段落在春节等浮动假期上时会提示，需刷新日历后重跑。
        5. **未做严格的 IS/OOS 拆分**：12 个滚动时点都是历史回看，不是「训练/测试」二分。
           存在一个**准样本外证据**——「Kronos 等价于均值回归」这个判断是在
           Phase 1 的 600519 / 510300 上先形成的，601611 / 002600 是之后才按要求加入，
           而两只后加标的的镜像差值仍只有 +0.019 / −0.023。但这只是时间上的先后，
           评测实现仍受前两只结果影响，**不能替代**「假设冻结 → 用未见数据一次性检验」。
        6. **结论只针对这 4 个标的、这段样本期**：不构成对 Kronos 模型本身的普遍否定，
           而是「在本数据与评估口径下未检出 alpha」。
        """
    )
    st.info("完整论证见 `docs/复现笔记.md` 与 `docs/Phase2评估.md`；数据层修复见 `docs/数据层修复.md`。")

# --------------------------------------------------------------------------
st.sidebar.markdown("### 项目信息")
st.sidebar.markdown(
    "- 模型：Kronos-base（NeoQuasar，MIT）\n"
    "- 数据：A股/ETF 前复权日线（akshare，东财→新浪自动回退）\n"
    "- 评估：滚动窗口 + 4 基线 + bootstrap 置信区间\n"
    "- 硬件：本地 RTX 4060 8GB 推理"
)
if starts:
    anchor = min(v.min() for v in starts.values())
    st.sidebar.markdown(f"### 结果新鲜度\n预测起点 {anchor:%Y-%m-%d}（锚点为上一交易日收盘）")
st.sidebar.markdown(
    "### 说明\n"
    "看板为**静态结果展示**，不在云端跑模型推理；"
    "云端只读 `outputs/`，因此不需要 GPU 与原始行情数据。"
)

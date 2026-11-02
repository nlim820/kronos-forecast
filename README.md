# Kronos A股复现与客观评估

> 复现开源金融基础模型 **Kronos**（K-line Foundation Model，AAAI 2026），
> 在 A 股真实日线数据上做预测，并用**带强基线的严谨评估**检验其预测力。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%2Bcu126-orange.svg)](https://pytorch.org)

## 一句话结论

**在 A 股真实样本上，Kronos 没有统计显著的预测力；它的预测行为等价于
“向窗口均值回归”这一行代码即可实现的基线。**

这不是贬低模型，而是一次诚实的复现 + 证伪——比“宣称它很准”更有科研价值，
也更能体现金融工程的方法论素养。

## 项目结构

```
kronos-forecast/
├── README.md                 # 本文件（项目说明与结论）
├── requirements.txt          # 轻量依赖（仅看板：streamlit+pandas，供 Streamlit Cloud 自动读取）
├── requirements_full.txt     # 完整复现环境（含 torch CUDA，供本地推理/评估）
├── app.py                    # Streamlit 结果看板（读 outputs/，无需 GPU）
├── model/                    # 上游推理代码（MIT，见 NOTICE）
├── scripts/
│   ├── data_utils.py         # 数据层（东财→新浪回退 + 真实交易日历）
│   ├── smoke_test.py         # 环境冒烟测试
│   ├── predict_demo.py       # 预测 + 分位扇形可视化
│   └── evaluate.py           # 滚动窗口评估（4 基线 + bootstrap 置信区间）
├── docs/
│   ├── 环境搭建.md           # 环境踩坑与固化
│   ├── 复现笔记.md           # z-score 均值回归机制发现
│   ├── Phase2评估.md         # 评估方法 + 全量结果 + 诚实结论
│   └── 简历_项目经历.md      # 简历项目经历段落
└── outputs/                  # 预测图(HTML) + 评估明细(CSV)
```

## 快速开始

```bash
# 1. 建独立 venv（Python 3.13，避开系统 3.14 的兼容坑）
python -m venv .venv && .venv\Scripts\activate

# 2. 装 PyTorch（CUDA 版，从专用索引）
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. 装其余依赖（完整环境，含 torch）
pip install -r requirements_full.txt

# 4. 冒烟测试（首次会联网从 HF 镜像拉权重）
set HF_ENDPOINT=https://hf-mirror.com
python scripts/smoke_test.py

# 5. 跑预测（生成 outputs/forecast_*.html）
python scripts/predict_demo.py --symbols 600519,510300,601611,002600

# 6. 跑评估（4 标的 × 12 滚动时点）
python scripts/evaluate.py --symbols 600519,510300,601611,002600 --pred-len 20 --rollings 12 --samples 8 --gap 22

# 7. 看板（只需轻量依赖，可单独装）
pip install -r requirements.txt
streamlit run app.py
```

> 国内网络提示：`pip` 若卡在代理/镜像，清掉 `HTTP_PROXY/HTTPS_PROXY` 环境变量并改用
> 阿里云 `https://mirrors.aliyun.com/pypi/simple`；HuggingFace 权重走 `HF_ENDPOINT=https://hf-mirror.com`。

## 评估结果（摘要，4 标的 × 12 滚动时点）

| 方法 | 粒度 | n | 方向准确率 | IC | IC 95% CI |
|---|---|---|---|---|---|
| **kronos** | 终点 | 48 | 52.1% | **+0.130** | [−0.17, +0.42] |
| mean_revert（回归窗口均值） | 终点 | 48 | 54.2% | **+0.118** | [−0.17, +0.41] |
| momentum（动量延续） | 终点 | 48 | 41.7% | **−0.259** | [−0.48, +0.03] |
| kronos | 逐日 | 912 | 49.3% | −0.076 | [−0.16, +0.02] |

- 所有方法的 IC 置信区间**都包含 0** → 没有统计显著的预测力。
- **Kronos（IC +0.130）≈ 均值回归基线（+0.118）** → 预测本质就是向窗口均值回归。
- 动量是最差的预测器（−0.259），盲从趋势在样本期内比随机还差。

完整论证见 [`docs/Phase2评估.md`](docs/Phase2评估.md) 与 [`docs/复现笔记.md`](docs/复现笔记.md)。

## 文档

- [`docs/环境搭建.md`](docs/环境搭建.md)：环境、版本、踩坑固化
- [`docs/复现笔记.md`](docs/复现笔记.md)：z-score 标准化 → 均值回归的机制发现
- [`docs/Phase2评估.md`](docs/Phase2评估.md)：评估方法设计 + 全量结果 + 诚实结论
- [`docs/简历_项目经历.md`](docs/简历_项目经历.md)：可直接粘贴进简历的项目经历
- [`docs/部署到StreamlitCloud.md`](docs/部署到StreamlitCloud.md)：Streamlit Cloud 部署 checklist 与避坑

## 🚀 在线看板

看板已部署到 Streamlit Cloud（三页：预测图 / 评估指标 / 方法说明），为**静态结果展示**
（读取已生成的 `outputs/`），不在云端跑模型推理，无需 GPU。

**两种打开方式：**

| 方式 | 链接 / 操作 | 适用场景 |
|---|---|---|
| **GitHub Codespaces（推荐，免登录）** | 仓库页面 `Code → Codespaces → Create codespace on main`，约 2 分钟自动装依赖并启动看板 | 现场演示、面试投屏 |
| **Streamlit Cloud** | 部署实例已开启平台登录校验，**访客需登录才可查看** | 自己登录后演示 |

> Codespaces 方式依赖仓库内的 `.devcontainer/devcontainer.json`（已配置自动 `streamlit run app.py` + 转发 8501 端口）。
> Streamlit Cloud 免费实例会休眠，首次打开需冷启动 30–60 秒。
> **注意：不要把带登录校验的 Streamlit 链接直接放进简历**，读者点开只会看到登录页。

## 🛠 部署到 Streamlit Cloud（自行复现）

1. 在 <https://share.streamlit.io> 关联本仓库 `nlim820/kronos-forecast`。
2. **主文件填 `app.py`**，Branch 选 `main`，其余保持默认，**直接点 Deploy 即可**。
3. 平台会自动读取仓库根目录的 `requirements.txt`（已为轻量版：仅 streamlit+pandas，
   不含 PyTorch），因此无需、也无法在界面指定其他依赖文件。
4. ⚠️ **部署后务必关掉 "Require authentication"**（否则链接对访客只是登录页）——
   详见 [`docs/部署到StreamlitCloud.md`](docs/部署到StreamlitCloud.md) 第 5 步。

> 若界面有 Python version 选项，保持默认 3.14 即可；如构建因 Python 版本失败，可降到 3.12。
> 完整本地复现环境在 `requirements_full.txt`（含 PyTorch CUDA）。

## 致谢与版权

- 模型 **Kronos**（`NeoQuasar/Kronos-*`，MIT）：<https://huggingface.co/NeoQuasar>
- 推理代码（本仓库 `model/` 等）：来自 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)（MIT），
  上游 README 留存于 [`docs/upstream_README.md`](docs/upstream_README.md)
- 数据：akshare（东财/新浪行情），仅供研究学习，**不构成投资建议**
- 本项目新增内容（脚本/评估/文档/网页）以 MIT 协议发布，见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)

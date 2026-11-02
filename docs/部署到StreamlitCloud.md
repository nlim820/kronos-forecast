# 部署到 Streamlit Cloud（傻瓜式 Checklist）

本项目的 `app.py` 是一个**纯展示型看板**：它只读取 `outputs/` 目录下已经生成好的预测图和评估结果，**不加载模型、不依赖 GPU、不安装 torch**。因此部署几乎没有技术坑，照着下面勾完即可上线。

预计耗时：5 分钟（含 1–2 分钟部署等待）。

---

## ⚠️ 一个必须记住的坑

免费版 Streamlit Cloud 默认会去读仓库根目录的 `requirements.txt`，而那个文件里有一行装 PyTorch（约 2.5GB）。免费实例的构建超时会被它拖崩。

**解决办法：部署时把依赖清单从 `requirements.txt` 改成 `requirements_app.txt`。** 后者只装 `streamlit` 和 `plotly`，几百 KB，秒过。

这个坑已经帮你绕开了——`requirements_app.txt` 已经在仓库里，你只要在第 4 步选对它就行。

---

## 步骤清单

- [ ] **1. 打开部署页**
  浏览器访问 <https://share.streamlit.io/>，用你的 **GitHub 账号**登录（首次会请求授权读取你的仓库，点 Accept）。

- [ ] **2. 新建应用**
  点页面上的 **"New app"**（或 "Deploy an app"）按钮。

- [ ] **3. 选择本仓库**
  在 Repository 下拉里选 **`nlim820/kronos-forecast`**。
  - Branch：`main`
  - Main file path：**`app.py`**（默认就会被识别到，确认一下即可）

- [ ] **4. 改依赖清单（关键）**
  展开 **"Advanced settings"**（高级设置），找到 Dependencies 一项，把默认的 `requirements.txt` **改成 `requirements_app.txt`**。
  > 不改这一步 = 免费实例被 torch 拖超时 = 部署失败。

- [ ] **5. 点 Deploy**
  点 **"Deploy"**，等待 1–2 分钟。看到 "Your app is live" 和一个 `https://xxxx.streamlit.app` 的链接，就成功了。

- [ ] **6. 验证**
  打开链接，确认三页都能打开：
  1. **预测图**：四只标的（茅台 / 沪深300ETF / 中国核建 / 领益智造）的下拉能切换，扇形区间图能显示。
  2. **评估看板**：指标表（IC / 方向准确率 / bootstrap 区间）能显示。
  3. **方法说明**：结论文字完整。

---

## 常见问题

**Q：打开页面是空的 / 报错？**
A：多半是第 4 步没改依赖清单，部署时去装 torch 超时了。删掉这个 app 重来，记得改 `requirements_app.txt`。

**Q：图表里 plotly 不显示？**
A：`app.py` 用的是 `st.components.v1.html` 内联嵌入 HTML，`requirements_app.txt` 里已含 `plotly`。若仍不显示，确认 `outputs/forecast_*.html` 和 `outputs/eval_*.csv` 已随仓库推送（已在仓库内，正常情况不用管）。

**Q：免费版每次打开很慢？**
A：免费实例会"休眠"，长时间无人访问后首次打开要冷启动几十秒，这是正常的。点了之后等它转完即可。

**Q：想改看板内容？**
A：改 `app.py` 后 `git commit && git push` 推到 `main` 分支，Streamlit Cloud 会自动重新部署。注意 `app.py` 只读 `outputs/` 的结果，如果要换标的/换结论，得先在本机跑 `scripts/predict_demo.py` / `scripts/evaluate.py` 重新生成 `outputs/`，再把新文件推送上去。

---

## 本地预览（可选，不部署也能看）

不想上线、只想在自己电脑看效果：

```bash
# 用项目自带的 venv
cd /d/projects/kronos-forecast
.venv/Scripts/python.exe -m pip install -i https://mirrors.aliyun.com/pypi/simple streamlit plotly
.venv/Scripts/python.exe -m streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

---

## 仓库里相关文件

| 文件 | 作用 |
|---|---|
| `app.py` | 看板主程序（纯展示，不加载模型） |
| `requirements_app.txt` | **部署用**依赖（streamlit + plotly，轻量） |
| `requirements.txt` | 本地复现用完整依赖（含 torch，勿用于部署） |
| `outputs/` | 预测图 + 评估结果（看板的数据来源） |

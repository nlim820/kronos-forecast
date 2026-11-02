# 部署到 Streamlit Cloud（傻瓜式 Checklist）

本项目的 `app.py` 是一个**纯展示型看板**：它只读取 `outputs/` 目录下已经生成好的预测图和评估结果，**不加载模型、不依赖 GPU、不安装 torch**。因此部署几乎没有技术坑，照着下面勾完即可上线。

预计耗时：5 分钟（含 1–2 分钟部署等待）。

---

## 好消息：依赖已经帮你配好，不用改

仓库根目录的 `requirements.txt` **已经是轻量版**（只有 `streamlit` + `pandas`，几百 KB），
Streamlit Cloud 部署时会**自动读取它**，不会去装 2.5GB 的 PyTorch。所以这个 checklist
**不需要你去找"依赖清单"字段**——之前那个版本里"把 requirements.txt 改成 requirements_app.txt"
的步骤已经作废，因为那个字段在这个版本的界面里根本不存在。

如果你在 Advanced settings 里看到 Python version，保持 3.14 即可（如构建因 Python 版本失败，可降到 3.12）。
Secrets 框留空（我们的 app 不需要任何密钥）。

---

## 步骤清单

- [ ] **1. 打开部署页**
  浏览器访问 <https://share.streamlit.io/>，用你的 **GitHub 账号**登录（首次会请求授权读取你的仓库，点 Accept）。

- [ ] **2. 新建应用**
  点页面上的 **"New app"**（或 "Deploy an app"）按钮。

- [ ] **3. 填写主表单**
  - Repository：**`nlim820/kronos-forecast`**（注意是 `nlim820`，多一个 `l`）
  - Branch：**`main`**
  - **Main file path：`app.py`**（⚠️ 不是 `streamlit_app.py`，填错会报 "This file does not exist"）
  - App URL：默认即可（显示 "Domain is available"）

- [ ] **4. 直接点 Deploy**
  没有需要改的依赖字段——`requirements.txt` 已是轻量版，平台自动读取。
  点 **"Deploy"**，等待 1–2 分钟。看到 "Your app is live" 和一个 `https://xxxx.streamlit.app` 链接就成功了。

- [ ] **5. ⚠️ 关掉登录墙（必做，否则链接对别人无效）**
  Streamlit Cloud 新版部署表单里有一项 **"Require authentication"（要求登录）**，默认可能被勾上。
  勾上时，**匿名访客（面试官、同学、甚至搜索引擎）打开链接都会被 303 重定向到登录页**，app 等于没上线。
  - 部署页 / app 页面右下角点 **"Manage app"**（齿轮图标）→ 进入管理面板 → 找到 **Settings** 或 **Advanced settings**
  - 取消勾选 **"Require authentication"**（或把 "Viewer access" 改成 Public / Anyone with the link）
  - 保存后等待自动重载（约 30–60 秒）

  **如何自查是否真的关掉了**：拿一个**未登录**的浏览器（或无痕窗口）打开链接。
  若被弹到 `share.streamlit.io/-/auth/app` 或出现 "sign in to continue"，就是没关成功。
  命令行自查：`curl -sI "https://<你的app>.streamlit.app/"` —— 期望 `HTTP/2 200`，
  若返回 `303` 且 `Location` 指向 `share.streamlit.io/-/auth/...` 则说明登录墙仍在。

- [ ] **6. 验证**
  打开链接，确认三页都能打开：
  1. **预测图**：四只标的（茅台 / 沪深300ETF / 中国核建 / 领益智造）的下拉能切换，扇形区间图能显示。
  2. **评估看板**：指标表（IC / 方向准确率 / bootstrap 区间）能显示。
  3. **方法说明**：结论文字完整。

  > 注意：如果你自己是**登录状态**打开的，看到内容正常**不代表**别人也能看到。
  > 第 5 步没做的话，你本地永远一切正常，但发给别人的链接是个登录页。

---

## 常见问题

**Q：页面报 "This file does not exist"（Main file path 处）？**
A：Main file path 填成了 `streamlit_app.py`。改成 **`app.py`** 即可——仓库里的看板文件就叫 `app.py`。

**Q：打开页面是空的 / 部署超时？**
A：正常不会。若真超时，先确认 Main file path 是 `app.py`（不是 `streamlit_app.py`），且 Repository 拼写是 `nlim820/kronos-forecast`。`requirements.txt` 已是轻量版，不会去装 torch。

**Q：我自己打开一切正常，但同学说打不开？**
A：99% 是**登录墙**（Require authentication）没关。你是 app 拥有者，登录态下永远能看；
访客会被 303 重定向到 `share.streamlit.io/-/auth/app`。按上面第 5 步关掉即可，
并用无痕窗口验证。命令行自查：`curl -sI "<你的app>.streamlit.app/"` 看是不是 200 而不是 303。

**Q：图表里 plotly 不显示？**
A：预测图 HTML 是自包含文件（plotly 已内联），`app.py` 用 `st.components.v1.html` 嵌入，不依赖云端装 plotly。若仍不显示，确认 `outputs/forecast_*.html` 和 `outputs/eval_*.csv` 已随仓库推送（已在仓库内）。

**Q：免费版每次打开很慢？**
A：免费实例会"休眠"，长时间无人访问后首次打开要冷启动几十秒，正常。点了之后等它转完即可。

**Q：想改看板内容？**
A：改 `app.py` 后 `git commit && git push` 推到 `main` 分支，Streamlit Cloud 会自动重新部署。注意 `app.py` 只读 `outputs/` 的结果，要换标的/换结论，得先在本机跑 `scripts/predict_demo.py` / `scripts/evaluate.py` 重新生成 `outputs/`，再推送。

---

## 本地预览（可选，不部署也能看）

不想上线、只想在自己电脑看效果：

```bash
# 用项目自带的 venv
cd /d/projects/kronos-forecast
.venv/Scripts/python.exe -m pip install -i https://mirrors.aliyun.com/pypi/simple streamlit pandas
.venv/Scripts/python.exe -m streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

---

## 仓库里相关文件

| 文件 | 作用 |
|---|---|
| `app.py` | 看板主程序（纯展示，不加载模型） |
| `requirements.txt` | **部署用**轻量依赖（streamlit + pandas，Streamlit Cloud 自动读取） |
| `requirements_full.txt` | 本地复现完整依赖（含 torch CUDA，仅供本机跑模型） |
| `outputs/` | 预测图 + 评估结果（看板的数据来源） |

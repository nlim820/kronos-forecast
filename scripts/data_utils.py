"""数据工具模块：被 Phase 1 / Phase 2 共用，避免重复代码。

提供两个核心函数：
- fetch(): 拉取单只 A股日线行情（东方财富优先，限流自动回退新浪）
- add_ma(): 在行情上计算 MA5 / MA20 均线

金融/工程知识点：
- 把"取数"和"画图/回测"解耦，是真实量化项目的标准做法：
  数据层只管产出干净的 DataFrame，上游（可视化、回测）各自消费。
"""
import akshare as ak
import pandas as pd
from pathlib import Path

try:
    from akshare.utils.http import Push2RateLimitError
except Exception:  # 老版本 akshare 可能未单独导出该异常
    Push2RateLimitError = Exception

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def fetch(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """拉取单只 A股日线行情，返回 DataFrame（列名为中文）。

    优先东方财富；若被限流，自动切换新浪并统一列名，下游无需改动。
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust=adjust,
        )
        return df
    except Push2RateLimitError:
        print("   ⚠️ 东方财富被限流，自动切换新浪数据源...")
        prefix = "sh" if symbol.startswith("6") else "sz"
        sina = ak.stock_zh_a_daily(
            symbol=f"{prefix}{symbol}", start_date=start,
            end_date=end, adjust=adjust,
        )
        sina = sina.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高",
            "low": "最低", "close": "收盘", "volume": "成交量",
        })
        return sina


def is_fund(symbol: str) -> bool:
    """判断 6 位代码是否为场内基金（ETF/LOF/封基）。

    股票代码段：60/68（沪）、00/30（深）；基金代码段：50/51/56/58（沪）、
    15/16/18（深）。用代码前缀区分，无需新增显式开关。
    """
    s = symbol.strip()
    return len(s) == 6 and s.isdigit() and s[:2] in (
        "50", "51", "56", "58", "15", "16", "18",
    )


def _fund_sina_symbol(symbol: str) -> str:
    """把 6 位基金代码转成新浪格式（sh/sz + 代码）。

    基金代码段：50/51/56/58 为沪市（sh），15/16/18 为深市（sz）。
    """
    s = symbol.strip()
    prefix = "sh" if s.startswith("5") else "sz"
    return f"{prefix}{s}"


def fund_fetch(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """拉取单只场内基金（ETF/LOF）日线行情，返回与股票同构的 DataFrame。

    优先东方财富（支持 qfq 前复权 + 起止日期）；若被限流，自动切换新浪。
    注意：新浪接口仅接受代码、返回全量且不复权，这里在代码内按日期切片对齐，
    仅作限流兜底，正常情况仍走东方财富 qfq 数据。
    """
    rename_map = {
        "date": "日期", "日期": "日期",
        "open": "开盘", "开盘": "开盘", "开盘价": "开盘",
        "close": "收盘", "收盘": "收盘", "收盘价": "收盘",
        "high": "最高", "最高": "最高",
        "low": "最低", "最低": "最低",
        "volume": "成交量", "成交量": "成交量",
    }
    keep = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]

    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust=adjust,
        )
    except Push2RateLimitError:
        print("   ⚠️ 东方财富基金接口被限流，自动切换新浪数据源...")
        raw = ak.fund_etf_hist_sina(symbol=_fund_sina_symbol(symbol))
        # 新浪：仅 symbol 入参，全量历史且不复权 → 统一列名后按日期切片
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
        raw["日期"] = pd.to_datetime(raw["日期"])
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        df = raw[(raw["日期"] >= s) & (raw["日期"] <= e)].copy()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df[[c for c in keep if c in df.columns]]


def _to_market_symbol(symbol: str) -> str:
    """6 位代码 → 带市场前缀的代码（腾讯 / 新浪接口需要）。

    沪市：6/9（A股）、5（ETF/LOF）→ sh；深市：0/2/3（A股）、1（ETF/LOF）→ sz。
    已带 sh/sz 前缀的直接原样返回。
    """
    s = symbol.strip()
    if len(s) == 6 and s.isdigit():
        return f"sh{s}" if s[0] in ("6", "9", "5") else f"sz{s}"
    return s


def _fetch_sina(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """显式使用新浪数据源（股票走 stock_zh_a_daily，基金走 fund_etf_hist_sina）。

    仅当用户在「数据源」中选择新浪时直接调用；默认路径仍走东财 + 限流回退。
    注意：新浪基金接口返回全量且不复权，这里按日期切片，仅作备用源。
    """
    m = _to_market_symbol(symbol)
    rename_map = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
    }
    keep = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    if is_fund(symbol):
        raw = ak.fund_etf_hist_sina(symbol=m)
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
        raw["日期"] = pd.to_datetime(raw["日期"])
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        raw = raw[(raw["日期"] >= s) & (raw["日期"] <= e)].copy()
        return raw[[c for c in keep if c in raw.columns]]
    df = ak.stock_zh_a_daily(symbol=m, start_date=start, end_date=end, adjust=adjust)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df[[c for c in keep if c in df.columns]]


def fetch_tx(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """腾讯证券日线（stock_zh_a_hist_tx）：支持 A股个股与场内 ETF，支持 qfq/hfq。

    腾讯接口入参需带市场前缀（如 sh510300 / sz000001），返回英文列名，
    这里统一成项目标准中文列，下游可视化 / 回测无需任何改动。
    注意：不同数据源的复权因子算法不同，跨源结果不宜直接混用比较。
    """
    m = _to_market_symbol(symbol)
    df = ak.stock_zh_a_hist_tx(
        symbol=m, start_date=start, end_date=end, adjust=adjust,
    )
    rename_map = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    return df[[c for c in keep if c in df.columns]]


def add_ma(df: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    """基于收盘价计算 MA5 / MA20 等均线列。"""
    for w in windows:
        df[f"MA{w}"] = df["收盘"].rolling(window=w).mean()
    return df


def get_price(symbol: str, start: str, end: str, adjust: str = "qfq",
              source: str = "auto") -> pd.DataFrame:
    """取行情，带本地缓存：已下载过就直接读 CSV，避免重复联网/触发限流。

    source 数据源选择：
      - auto / em（默认）：东方财富（基金走 ETF 接口），限流时自动回退新浪；
      - tx：腾讯证券日线（stock_zh_a_hist_tx）；
      - sina：新浪日线（显式指定）。
    缓存文件按「复权方式 + 数据源」分别存储，避免不同来源数据互相污染。
    缓存放在 data/raw/（已被 .gitignore 忽略，不会进仓库）。
    """
    cache_dir = DATA / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # auto/em 沿用旧缓存名（{code}_{start}_{end}_{adjust}.csv），保持历史缓存可用；
    # 其他源追加 _source 后缀，独立缓存、互不覆盖。
    suffix = "" if source in ("auto", "em") else f"_{source}"
    path = cache_dir / f"{symbol}_{start}_{end}_{adjust}{suffix}.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")

    if source == "tx":
        df = fetch_tx(symbol, start, end, adjust)
    elif source == "sina":
        df = _fetch_sina(symbol, start, end, adjust)
    else:  # auto / em
        df = fund_fetch(symbol, start, end, adjust) if is_fund(symbol) else fetch(symbol, start, end, adjust)

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def get_index(symbol: str = "sh000300", start: str = "20200101",
              end: str = "20231231") -> pd.DataFrame:
    """取指数日线（默认沪深300 sh000300），带本地缓存。

    返回含「日期 / 收盘」两列的 DataFrame，用于把策略收益与「市场基准」比较，
    从而判断策略是否真有 alpha（跑赢个股买入持有 ≠ 跑赢市场）。
    """
    cache_dir = DATA / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"idx_{symbol}_{start}_{end}.csv"
    if path.exists():
        idx = pd.read_csv(path, encoding="utf-8-sig")
        # 缓存读取时「日期」是字符串，必须转 datetime，否则下方与行情
        # DatetimeIndex 对齐时全部错位、沪深300 基准算不出来（仅首次联网能算）
        idx["日期"] = pd.to_datetime(idx["日期"])
        return idx[["日期", "收盘"]]
    idx = ak.stock_zh_index_daily(symbol=symbol)
    idx = idx.rename(columns={"date": "日期", "close": "收盘"})
    idx["日期"] = pd.to_datetime(idx["日期"])
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    idx = idx[(idx["日期"] >= s) & (idx["日期"] <= e)][["日期", "收盘"]]
    idx.to_csv(path, index=False, encoding="utf-8-sig")
    return idx



def get_trade_calendar(refresh: bool = False) -> pd.DatetimeIndex:
    """A 股交易日历（1990 至今），带本地缓存。

    为什么必须用它：`pd.bdate_range` 只剔除周末，不认中国法定节假日。
    预测未来 N 个交易日时若用 bdate_range，会把国庆/春节算成交易日，
    导致预测日期与真实交易日错位——Phase 2 拿预测对齐真实收益时会直接失效。
    """
    cache_dir = DATA / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "trade_calendar.csv"

    if path.exists() and not refresh:
        cal = pd.read_csv(path, encoding="utf-8-sig")
        return pd.DatetimeIndex(pd.to_datetime(cal["trade_date"]))

    cal = ak.tool_trade_date_hist_sina()
    cal.columns = ["trade_date"]
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    cal.to_csv(path, index=False, encoding="utf-8-sig")
    return pd.DatetimeIndex(cal["trade_date"])


def next_trade_days(last_date, n: int) -> pd.DatetimeIndex:
    """返回 last_date 之后的 n 个真实交易日（不含 last_date）。"""
    cal = get_trade_calendar()
    last = pd.Timestamp(last_date)
    fut = cal[cal > last]
    if len(fut) < n:
        # 日历尚未覆盖未来（跨年后未刷新），退化用工作日并提示
        print(f"   ⚠️ 交易日历仅剩 {len(fut)} 天，不足 {n} 天，退回工作日近似")
        return pd.bdate_range(start=last + pd.Timedelta(days=1), periods=n)
    return fut[:n]

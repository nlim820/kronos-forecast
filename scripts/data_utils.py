"""数据工具模块：被 Phase 1 / Phase 2 共用，避免重复代码。

提供两个核心函数：
- fetch(): 拉取单只 A股日线行情（东方财富优先，限流自动回退新浪）
- add_ma(): 在行情上计算 MA5 / MA20 均线

金融/工程知识点：
- 把"取数"和"画图/回测"解耦，是真实量化项目的标准做法：
  数据层只管产出干净的 DataFrame，上游（可视化、回测）各自消费。
"""
import json

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


# --------------------------------------------------------------- 缓存新鲜度
# 踩过的坑（2026-11 审计）：旧版缓存只做「文件是否存在」判断，不校验内容新鲜度。
# 事故链条：11-02 08:33（开盘前）用 end=20261102 抓数 → 只拿到 10-30 的数据 →
# 存成 xxx_20261102_qfq.csv；当日 16:19 重跑预测，end 仍是 20261102 → 命中早上
# 那份缓存 → 11-02 的 bar 永久丢失，预测锚点错位一个交易日。
# 结论：缓存键里带 end 不等于缓存内容真的到了 end，必须校验「最后一行日期」。


def _session_settled(now: pd.Timestamp) -> pd.Timestamp:
    """当日日线在收盘后还要一段时间才在数据源落库，取 15:40 作为安全分界。"""
    return now.replace(hour=15, minute=40, second=0, microsecond=0)


def _last_completed_trade_day(now: pd.Timestamp = None) -> pd.Timestamp:
    """返回「其数据理应已经可得」的最近一个交易日。

    交易日 d 的日线在 d 当日 15:40 之后才算可得；不到 15:40 则 d 尚未完成，
    退到 d 之前的最近交易日。用于避免「盘中要求当天数据 → 永远不满足 → 反复联网」。
    """
    now = now or pd.Timestamp.now()
    cal = get_trade_calendar()
    cutoff = now.normalize() if now >= _session_settled(now) \
        else now.normalize() - pd.Timedelta(days=1)
    prior = cal[cal <= cutoff]
    return prior[-1] if len(prior) else None


def _required_last_date(end: str, now: pd.Timestamp = None) -> pd.Timestamp:
    """给定 end，返回缓存「必须至少覆盖到」的最后一个交易日。

    不能简单要求「最后一行 == end」，因为：
      1) end 可能根本不是交易日（周末 / 法定节假日，如 2026-10-31 是周六）；
      2) end 可能指向今天而今天还没收盘，数据源此时给不出当天 bar。
    """
    now = now or pd.Timestamp.now()
    end_ts = pd.Timestamp(str(end)).normalize()
    cal = get_trade_calendar()
    if end_ts <= cal.max():
        cand = cal[cal <= end_ts]
        expected = cand[-1] if len(cand) else end_ts
    else:
        # 日历尚未覆盖 end（跨年未刷新），退回工作日近似，同时提示
        print(f"   ⚠️ 交易日历未覆盖 {end_ts.date()}（日历止于 {cal.max().date()}），退回工作日近似")
        expected = pd.bdate_range(end=end_ts, periods=1)[-1]
    completed = _last_completed_trade_day(now)
    if completed is not None and expected > completed:
        return completed
    return expected


def _meta_path(path: Path) -> Path:
    """缓存旁挂的元数据文件路径（记录抓取时刻与当时要求的日期）。"""
    return path.parent / (path.name + ".meta.json")


def _read_meta(path: Path) -> dict:
    mp = _meta_path(path)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(path: Path, df: pd.DataFrame, source: str, required) -> None:
    """记录本次抓取的时刻、结果最后日期、当时要求的最后日期。

    为什么必须记 required_last_date：若某标的停牌导致数据源压根给不出某天数据，
    光看「缓存没到 required」会判定陈旧 → 每次调用都重新联网。记下「今天已经为
    这个 required 抓过一次」，即可区分「缓存旧」和「数据源确实没有」。
    """
    last = None
    if "日期" in getattr(df, "columns", []):
        try:
            s = pd.to_datetime(df["日期"], errors="coerce").dropna()
            last = str(s.max().date()) if len(s) else None
        except Exception:
            last = None
    meta = {
        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "last_date": last,
        "required_last_date": str(pd.Timestamp(required).date()) if required is not None else None,
        "source": source,
    }
    try:
        _meta_path(path).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass  # 元数据写失败不应阻断主流程


def _cached_last_date(df: pd.DataFrame):
    """取缓存数据的最后一根 bar 日期，取不到返回 None。"""
    if "日期" not in getattr(df, "columns", []) or df.empty:
        return None
    try:
        s = pd.to_datetime(df["日期"], errors="coerce").dropna()
        return s.max() if len(s) else None
    except Exception:
        return None


def _cache_is_fresh(path: Path, df: pd.DataFrame, required) -> bool:
    """已有缓存能否直接复用。

    命中情形有且只有两种：
      1) 最后一行已覆盖 required 交易日 → 数据够新；
      2) 未覆盖，但元数据显示「今天已为同一个 required 抓过」→ 数据源当日确实
         到此为止（如停牌），不要再反复联网。
    """
    if required is None:
        return True
    last = _cached_last_date(df)
    if last is None:
        return False
    if last.normalize() >= pd.Timestamp(required).normalize():
        return True

    meta = _read_meta(path)
    if str(meta.get("required_last_date")) == str(pd.Timestamp(required).date()):
        try:
            if pd.Timestamp(meta.get("fetched_at")).normalize() == pd.Timestamp.now().normalize():
                return True
        except Exception:
            pass
    return False


def get_price(symbol: str, start: str, end: str, adjust: str = "qfq",
              source: str = "auto", refresh: bool = False) -> pd.DataFrame:
    """取行情，带本地缓存（会校验新鲜度，不会把旧快照当新数据）。

    source 数据源选择：
      - auto / em（默认）：东方财富（基金走 ETF 接口），限流时自动回退新浪；
      - tx：腾讯证券日线（stock_zh_a_hist_tx）；
      - sina：新浪日线（显式指定）。
    refresh=True 强制忽略已有缓存重新联网抓取。

    缓存文件按「复权方式 + 数据源」分别存储，避免不同来源数据互相污染。
    缓存放在 data/raw/（已被 .gitignore 忽略，不会进仓库）。

    新鲜度规则见 _cache_is_fresh：缓存最后一行必须覆盖到 end 对应的真实交易日
    （end 非交易日自动前推；今天未收盘则只要求到上一交易日）。
    """
    cache_dir = DATA / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # auto/em 沿用旧缓存名（{code}_{start}_{end}_{adjust}.csv），保持历史缓存可用；
    # 其他源追加 _source 后缀，独立缓存、互不覆盖。
    suffix = "" if source in ("auto", "em") else f"_{source}"
    path = cache_dir / f"{symbol}_{start}_{end}_{adjust}{suffix}.csv"
    required = _required_last_date(end)

    if path.exists() and not refresh:
        cached = pd.read_csv(path, encoding="utf-8-sig")
        if _cache_is_fresh(path, cached, required):
            return cached
        last = _cached_last_date(cached)
        print(f"   ♻️ 缓存陈旧（最后 {last.date() if last is not None else '空'} "
              f"< 应有 {pd.Timestamp(required).date()}），重新抓取 {symbol}")

    # 抓取终点必须用 required（最近一个「数据应已完成」的交易日），而不是调用方
    # 传的 end。否则盘中调用会把当天**未收盘的不完整 K 线**写进缓存；而 qfq 前复权
    # 以最新价为基准，一根盘中脏数据会让整段历史价格被重算——实测 510300 因此在
    # 13:41 抓取后整体偏移 9.3%，cuts 之后 20 日收益从 -0.80% 变成 +1.83%。
    fetch_end = pd.Timestamp(required).strftime("%Y%m%d")

    if source == "tx":
        df = fetch_tx(symbol, start, fetch_end, adjust)
    elif source == "sina":
        df = _fetch_sina(symbol, start, fetch_end, adjust)
    else:  # auto / em
        df = fund_fetch(symbol, start, fetch_end, adjust) if is_fund(symbol) else fetch(symbol, start, fetch_end, adjust)

    # 双保险：即便数据源无视 end 参数、仍返回 required 之后的行（ETF 接口实测会在
    # 盘中返回当天未完成的 bar），落盘前也一律剔除。qfq 以最新价为基准，一根盘中
    # 脏 bar 会把整段历史价格重算，代价远大于"少一天数据"。
    if df is not None and len(df) > 0 and "日期" in df.columns:
        _dates = pd.to_datetime(df["日期"])
        over = _dates > pd.Timestamp(required)
        if over.any():
            print(f"   🧹 {symbol} 剔除 {int(over.sum())} 行晚于 "
                  f"{pd.Timestamp(required).date()} 的数据（未完成交易日）")
            df = df.loc[~over].reset_index(drop=True)

    # 数据源偶发返回空表：宁可沿用旧缓存，也不要把缓存写坏
    if df is None or len(df) == 0:
        if path.exists():
            print(f"   ⚠️ {symbol} 本次返回空数据，沿用旧缓存")
            return pd.read_csv(path, encoding="utf-8-sig")
        raise RuntimeError(f"{symbol} 取数失败：数据源返回空表（start={start}, end={fetch_end}）")

    # 防止「重抓反而更旧」把好缓存覆盖掉（数据源抖动 / 接口临时降级时会发生）
    if path.exists():
        old_last = _cached_last_date(pd.read_csv(path, encoding="utf-8-sig"))
        new_last = _cached_last_date(df)
        if old_last is not None and new_last is not None and new_last < old_last:
            print(f"   ⚠️ {symbol} 新抓数据({new_last.date()}) 比缓存({old_last.date()}) 更旧，保留原缓存")
            return pd.read_csv(path, encoding="utf-8-sig")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    _write_meta(path, df, source, required)
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
    required = _required_last_date(end)
    if path.exists():
        idx = pd.read_csv(path, encoding="utf-8-sig")
        # 缓存读取时「日期」是字符串，必须转 datetime，否则下方与行情
        # DatetimeIndex 对齐时全部错位、沪深300 基准算不出来（仅首次联网能算）
        idx["日期"] = pd.to_datetime(idx["日期"])
        if _cache_is_fresh(path, idx, required):
            return idx[["日期", "收盘"]]
        print(f"   ♻️ 指数缓存陈旧（最后 {idx['日期'].max().date()} "
              f"< 应有 {pd.Timestamp(required).date()}），重新抓取 {symbol}")
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


# 交易所日历（新浪）只公布到当年年底，次年休市安排未公布前，覆盖不到的日期
# 只能按工作日近似。已知的确定性休市必须剔除，否则会把节假日当成交易日
# （2026-11 踩过：60 日预测轴把 2027-01-01 元旦当成了交易日）。
_FIXED_MARKET_HOLIDAYS = ((1, 1),)  # 元旦固定休市；其余随国务院年度安排变动


def _approx_trade_days(start, n: int) -> pd.DatetimeIndex:
    """日历未覆盖时的兜底：工作日序列，剔除固定休市日（元旦）。"""
    out, cur = [], pd.Timestamp(start) - pd.Timedelta(days=1)
    while len(out) < n:
        cur += pd.Timedelta(days=1)
        if cur.weekday() >= 5:
            continue
        if (cur.month, cur.day) in _FIXED_MARKET_HOLIDAYS:
            continue
        out.append(cur)
    return pd.DatetimeIndex(out)


def next_trade_days_split(last_date, n: int):
    """返回 (日期索引, 其中来自真实日历的天数)。

    为什么要把「真实/近似」的天数显式返回：调用方（预测图、看板、文档）需要
    这个数字才能**如实披露**轴的性质。此前只返回日期，展示层就笼统写成
    「已按真实交易日历生成」——而实际上 60 日预测里有十几天的日期是工作日
    近似出来的，这种表述属于过度承诺。
    """
    cal = get_trade_calendar()
    last = pd.Timestamp(last_date)
    fut = cal[cal > last]
    if len(fut) >= n:
        return fut[:n], n

    n_real = len(fut)
    print(f"   ⚠️ 交易日历仅覆盖到 {cal.max().date()}，剩余 {n - n_real} 天按工作日近似"
          f"（已剔除元旦；其余法定休市以次年安排公布为准）")
    if n_real == 0:
        return _approx_trade_days(last + pd.Timedelta(days=1), n), 0
    tail = _approx_trade_days(fut[-1] + pd.Timedelta(days=1), n - n_real)
    return pd.DatetimeIndex(list(fut) + list(tail)), n_real


def next_trade_days(last_date, n: int) -> pd.DatetimeIndex:
    """返回 last_date 之后的 n 个交易日（不含 last_date）。

    日历覆盖范围内的日期用真实交易日历（已含春节/国庆等法定休市）；
    超出覆盖范围（次年休市安排尚未公布）的部分按工作日近似并剔除元旦，
    近似段会在日志明确告警——这是已知局限，不是静默降级。
    需要知道「几天是近似」时用 next_trade_days_split。
    """
    return next_trade_days_split(last_date, n)[0]

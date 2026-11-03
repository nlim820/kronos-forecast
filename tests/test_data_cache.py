"""数据层回归测试：缓存新鲜度判定 + 预测日期轴合法性。

这两个 bug 都是实际踩过的（2026-11 审计发现），在此固化成回归测试，
避免以后重构 data_utils 时无声回退：

1. 缓存只判断「文件是否存在」，不校验内容是否覆盖到 end —— 导致 11-02 的数据
   被 10-30 的旧快照永久遮蔽（早盘抓数后，收盘再用同一个 end 重跑会命中旧缓存）。
2. 交易日历只覆盖到当年年底，跨年预测退化为 pd.bdate_range（只剔周末），
   把 2027-01-01 元旦当成交易日。

这些测试不依赖 torch / akshare 联网，可直接 `pytest tests/test_data_cache.py` 跑。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import data_utils as du  # noqa: E402


# --------------------------------------------------------------------------
# 1. required 日期的推导：end 可能是周末 / 节假日 / 今天未收盘 / 未来
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "end, now, expect, why",
    [
        ("20261102", "2026-11-03 13:22", "2026-11-02", "end=周一，今天周二未收盘"),
        ("20261103", "2026-11-03 13:22", "2026-11-02", "end=今天未收盘，只能要求到上一交易日"),
        ("20261103", "2026-11-03 15:50", "2026-11-03", "end=今天且已收盘，应要求今天"),
        ("20261031", "2026-11-03 13:22", "2026-10-30", "end=周六（非交易日），前推到周五"),
        ("20261001", "2026-11-03 13:22", "2026-09-30", "end=国庆休市，前推到节前"),
        ("20270122", "2026-11-03 13:22", "2026-11-02", "end=未来，只能要求到最近已完成交易日"),
    ],
)
def test_required_last_date(end, now, expect, why):
    got = du._required_last_date(end, pd.Timestamp(now))
    assert str(got.date()) == expect, f"{why}：期望 {expect}，实际 {got.date()}"


# --------------------------------------------------------------------------
# 2. 缓存命中判定
# --------------------------------------------------------------------------
@pytest.fixture()
def stale_cache(tmp_path):
    """模拟事故现场：缓存文件在、但只到 10-30。"""
    p = tmp_path / "600519_20231029_20261102_qfq.csv"
    p.write_text("日期,收盘\n2026-10-29,100\n2026-10-30,101\n", encoding="utf-8")
    return p, pd.read_csv(p, encoding="utf-8-sig")


def test_cache_hit_when_covers_required(stale_cache):
    path, df = stale_cache
    assert du._cache_is_fresh(path, df, pd.Timestamp("2026-10-30")) is True


def test_cache_miss_when_stale(stale_cache):
    """核心回归：缓存到 10-30，但 required 是 11-02，必须判为陈旧。"""
    path, df = stale_cache
    assert du._cache_is_fresh(path, df, pd.Timestamp("2026-11-02")) is False


def test_cache_meta_prevents_infinite_refetch(stale_cache, monkeypatch):
    """停牌场景：数据源压根给不出某天数据，同一天内不应反复联网。

    元数据记录「今天已为同一个 required 抓过」后，应视为命中。
    """
    path, df = stale_cache
    du._write_meta(path, df, "auto", pd.Timestamp("2026-11-02"))
    assert du._cache_is_fresh(path, df, pd.Timestamp("2026-11-02")) is True


def test_cache_meta_from_other_day_does_not_help(stale_cache):
    """元数据若是昨天写的，不能豁免今天的新鲜度检查。"""
    path, df = stale_cache
    du._write_meta(path, df, "auto", pd.Timestamp("2026-11-02"))
    meta_file = du._meta_path(path)
    content = meta_file.read_text(encoding="utf-8").replace(
        pd.Timestamp.now().strftime("%Y-%m-%d"), "2026-11-01"
    )
    meta_file.write_text(content, encoding="utf-8")
    assert du._cache_is_fresh(path, df, pd.Timestamp("2026-11-02")) is False


# --------------------------------------------------------------------------
# 3. 预测日期轴
# --------------------------------------------------------------------------
def test_next_trade_days_skips_national_day():
    """覆盖范围内的法定假期走真实日历（国庆 10-01~10-07 休市）。"""
    days = du.next_trade_days("2026-09-30", 4)
    assert str(days[0].date()) == "2026-10-08"


def test_next_trade_days_excludes_new_year_beyond_calendar():
    """核心回归：日历覆盖不到的 2027-01-01 元旦不能被当成交易日。"""
    days = du.next_trade_days("2026-11-02", 60)
    assert len(days) == 60
    assert pd.Timestamp("2027-01-01") not in set(days)
    assert not any(d.weekday() >= 5 for d in days)


def test_next_trade_days_cross_year_first_day():
    """跨年首个交易日应为 2027-01-04（元旦 + 周末连休）。"""
    days = du.next_trade_days("2026-12-31", 3)
    assert str(days[0].date()) == "2027-01-04"


# --------------------------------------------------------------------------
# 4. 盘中不得把未完成的 K 线写进缓存
#    事故：2026-11-03 13:41 跑评估，load_ohlcv 的 end 传的是「今天」，ETF 接口
#    返回了当天未收盘的 bar。qfq 前复权以最新价为基准，这根脏 bar 让整段历史
#    价格被重算——510300 与同期其它缓存相比最大相对差达 9.30%，cutoff 之后
#    20 日收益从 -0.80% 变成 +1.83%，直接改写了评估结果。
# --------------------------------------------------------------------------
def _fake_ohlcv(last_date: str = "2026-10-30") -> pd.DataFrame:
    dates = pd.date_range(end=last_date, periods=3, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "日期": dates,
        "开盘": [1.0] * 3, "最高": [1.0] * 3, "最低": [1.0] * 3,
        "收盘": [1.0] * 3, "成交量": [1.0] * 3, "成交额": [1.0] * 3,
    })


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """把缓存目录指到临时路径，避免测试动到项目里的 data/raw。"""
    monkeypatch.setattr(du, "DATA", tmp_path)
    monkeypatch.setattr(du, "_required_last_date",
                        lambda end, now=None: pd.Timestamp("2026-11-02"))
    return tmp_path


@pytest.mark.parametrize("is_etf", [False, True], ids=["stock", "etf"])
def test_fetch_end_is_required_not_today(isolated, monkeypatch, is_etf):
    """数据源收到的 end 必须是 required（已完成交易日），不能是调用方传的今天。"""
    monkeypatch.setattr(du, "is_fund", lambda s: is_etf)
    captured = {}

    def fake(symbol, start, end, adjust):
        captured["end"] = end
        return _fake_ohlcv()

    monkeypatch.setattr(du, "fetch", fake)
    monkeypatch.setattr(du, "fund_fetch", fake)

    du.get_price("510300" if is_etf else "600519", "20261001", "20261103", source="em")

    assert captured["end"] == "20261102", (
        f"抓取终点应收敛到 required=20261102，实际传了 {captured['end']}"
    )


def test_source_returning_intraday_bar_is_truncated(isolated, monkeypatch):
    """数据源无视 end、硬返回今天未收盘的 bar 时，落盘前必须被剔除。"""
    monkeypatch.setattr(du, "is_fund", lambda s: True)
    # 模拟 ETF 接口行为：返回了 2026-11-03（今天，未收盘）这一行
    monkeypatch.setattr(du, "fund_fetch", lambda *a, **k: _fake_ohlcv("2026-11-03"))

    df = du.get_price("510300", "20261001", "20261103", source="em")

    assert pd.to_datetime(df["日期"]).max() == pd.Timestamp("2026-11-02"), \
        "返回值里不应含 2026-11-03 的盘中 bar"
    saved = pd.read_csv(isolated / "raw" / "510300_20261001_20261103_qfq.csv",
                        encoding="utf-8-sig")
    assert pd.to_datetime(saved["日期"]).max() == pd.Timestamp("2026-11-02"), \
        "落盘缓存里也不应含盘中 bar（否则 qfq 复权基准会被污染）"

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

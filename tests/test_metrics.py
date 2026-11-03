"""指标层回归测试：常数预测不能产出「假成绩」。

背景（2026-11-03 审计）：`zero` 基线的方向准确率原先是 0.0000%，报告里被
读成「这条基线比抛硬币差得多」。实际原因是 sign(0)=0 永远不等于 sign(实际)，
属于**指标退化**，不是预测能力差。现在统一返回 nan，并在此固化成测试。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import evaluate as ev  # noqa: E402


def test_constant_prediction_has_no_direction_accuracy():
    """预测恒为 0 → 方向准确率 / IC / RankIC 全部不适用（nan）。"""
    n, dir_acc, ic, ric = ev.metrics([0.0] * 10, [0.01, -0.02] * 5)
    assert n == 10
    assert np.isnan(dir_acc) and np.isnan(ic) and np.isnan(ric)


def test_normal_prediction_keeps_real_numbers():
    """非退化输入不受影响：方向全中时准确率=1，同序时相关为正。"""
    pred = [0.01, -0.02, 0.03, -0.01, 0.02]
    act = [0.02, -0.01, 0.05, -0.03, 0.01]
    n, dir_acc, ic, ric = ev.metrics(pred, act)
    assert n == 5
    assert dir_acc == 1.0
    assert ic > 0.7 and ric > 0.7


def test_constant_actual_is_also_not_applicable():
    """实际收益全相同（如长期停牌）时相关无定义，应返回 nan 而不是抛异常。"""
    _, dir_acc, ic, _ = ev.metrics([0.01, 0.02, 0.03], [0.0, 0.0, 0.0])
    assert np.isnan(dir_acc) and np.isnan(ic)


def test_too_few_points_returns_nan():
    """2 个点无法谈相关性，返回 nan 但 n 仍如实报告。"""
    n, dir_acc, _, _ = ev.metrics([0.01, 0.02], [0.03, -0.01])
    assert n == 2 and np.isnan(dir_acc)

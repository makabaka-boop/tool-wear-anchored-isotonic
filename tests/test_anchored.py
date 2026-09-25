"""固定观测校正的领域算法测试。

最优性用独立神谕核对：在全部区间加权均值候选值上做单调序列 DP
（tests/oracle.py 的 oracle_anchored_fit），与"PAVA + 常数裁剪"的
求解器逐点对拍；另覆盖相等锚点、端点锚点、全固定、冲突定位与
确定性等场景。全程不允许 float。
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from math import gcd

import pytest

from app.isotonic import (
    InfeasibleAnchorsError,
    fit_isotonic,
    fit_isotonic_anchored,
)

from .oracle import oracle_anchored_fit


def _fitted(result):
    return [
        Fraction(num, den)
        for num, den in zip(result.fitted_num, result.fitted_den)
    ]


def _total_error(result):
    return Fraction(result.total_error_num, result.total_error_den)


def _block_ranges(result):
    return tuple((b.start, b.end) for b in result.blocks)


def _maximal_equal_runs(values):
    """由逐点 fitted 直接构造最大连续等值区间（独立于求解器）。"""

    runs = []
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[start]:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(values) - 1))
    return tuple(runs)


def _is_feasible(readings, anchors):
    ordered = sorted(anchors)
    return all(
        readings[a] <= readings[b] for a, b in zip(ordered, ordered[1:])
    )


# ---------------------------------------------------------------------------
# 与独立有理数候选枚举神谕对拍
# ---------------------------------------------------------------------------


def _anchor_subsets(n, max_size=3):
    subsets = []
    for size in range(1, min(max_size, n) + 1):
        subsets.extend(itertools.combinations(range(n), size))
    return subsets


def test_anchored_matches_candidate_dp_oracle_exhaustive_small():
    # n<=4：全部 reading 模式 × 全部 1..3 元锚点子集，等权。
    for n in range(2, 5):
        for pattern in itertools.product(range(3), repeat=n):
            readings = list(pattern)
            weights = [1] * n
            for anchors in _anchor_subsets(n):
                if not _is_feasible(readings, anchors):
                    continue
                result = fit_isotonic_anchored(readings, weights, list(anchors))
                expected_fitted, expected_error = oracle_anchored_fit(
                    readings, weights, list(anchors)
                )
                assert _fitted(result) == expected_fitted, (
                    readings,
                    anchors,
                )
                assert _total_error(result) == expected_error
                # 分块 = 唯一最优 fitted 的最大连续等值区间
                assert _block_ranges(result) == _maximal_equal_runs(
                    expected_fitted
                )


def test_anchored_matches_candidate_dp_oracle_random_weighted():
    import random

    rng = random.Random(20260925)
    for _ in range(300):
        n = rng.randint(2, 8)
        readings = [rng.randint(0, 6) for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        size = rng.randint(1, min(3, n))
        anchors = sorted(rng.sample(range(n), size))
        if not _is_feasible(readings, anchors):
            continue
        result = fit_isotonic_anchored(readings, weights, anchors)
        expected_fitted, expected_error = oracle_anchored_fit(
            readings, weights, anchors
        )
        assert _fitted(result) == expected_fitted, (readings, weights, anchors)
        assert _total_error(result) == expected_error
        assert _block_ranges(result) == _maximal_equal_runs(expected_fitted)


# ---------------------------------------------------------------------------
# 结构不变量
# ---------------------------------------------------------------------------


def _check_anchored_invariants(readings, weights, anchors, result):
    n = len(readings)
    fitted = _fitted(result)

    # 保留全部原始点、曲线非递减
    assert len(fitted) == n
    for a, b in zip(fitted, fitted[1:]):
        assert a <= b

    # 锚点 fitted 精确等于其 reading（原值保留）
    for index in anchors:
        assert fitted[index] == readings[index]

    # 块连续覆盖全部点且无重叠；块内等值；分块最大（相邻块值不同）
    assert result.blocks[0].start == 0
    assert result.blocks[-1].end == n - 1
    for prev, curr in zip(result.blocks, result.blocks[1:]):
        assert curr.start == prev.end + 1
        prev_mean = Fraction(prev.mean_num, prev.mean_den)
        curr_mean = Fraction(curr.mean_num, curr.mean_den)
        assert prev_mean < curr_mean

    total = Fraction(0, 1)
    for block in result.blocks:
        mean = Fraction(block.mean_num, block.mean_den)
        assert block.mean_den > 0
        assert gcd(abs(block.mean_num), block.mean_den) == 1
        block_error = Fraction(0, 1)
        for i in range(block.start, block.end + 1):
            assert fitted[i] == mean
            residual = mean - readings[i]
            block_error += weights[i] * residual * residual
        # 块误差精确且不可约
        assert Fraction(block.error_num, block.error_den) == block_error
        assert block.error_den > 0
        assert gcd(abs(block.error_num), block.error_den) == 1
        total += block_error

    assert _total_error(result) == total
    assert result.total_error_den > 0
    assert gcd(abs(result.total_error_num), result.total_error_den) == 1


def test_anchored_invariants_on_random_sequences():
    import random

    rng = random.Random(925)
    checked = 0
    while checked < 120:
        n = rng.randint(2, 30)
        readings = [rng.randint(0, 50) for _ in range(n)]
        weights = [rng.randint(1, 10) for _ in range(n)]
        size = rng.randint(1, min(12, n))
        anchors = sorted(rng.sample(range(n), size))
        if not _is_feasible(readings, anchors):
            continue
        result = fit_isotonic_anchored(readings, weights, anchors)
        _check_anchored_invariants(readings, weights, anchors, result)
        checked += 1


# ---------------------------------------------------------------------------
# 规定的场景：相等锚点、端点锚点、全固定、跨锚点等值合并
# ---------------------------------------------------------------------------


def test_equal_anchors_force_plateau_between_them():
    # 两个锚点读数相等：之间所有点被夹到同一常数
    result = fit_isotonic_anchored([5, 0, 5], [1, 1, 1], [0, 2])
    assert _fitted(result) == [Fraction(5)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total_error(result) == Fraction(25)

    # 读数相同的锚点本身合并为一个等值块
    result = fit_isotonic_anchored([4, 4, 4], [2, 3, 4], [0, 1, 2])
    assert _fitted(result) == [Fraction(4)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total_error(result) == 0


def test_endpoint_anchors():
    # 首点为锚：左侧无段，其余点受其读数的下界约束
    result = fit_isotonic_anchored([2, 0, 0, 1], [1, 1, 1, 1], [0])
    assert _fitted(result) == [Fraction(2)] * 4
    assert _block_ranges(result) == ((0, 3),)
    assert _total_error(result) == Fraction(4 + 4 + 1)

    # 末点为锚：右侧无段，其余点受其读数的上界约束；
    # 跨锚点的等值合并为一个分块
    result = fit_isotonic_anchored([1, 9, 9, 2], [1, 1, 1, 1], [3])
    assert _fitted(result) == [Fraction(1), Fraction(2), Fraction(2), Fraction(2)]
    assert _block_ranges(result) == ((0, 0), (1, 3))
    assert _total_error(result) == Fraction(49 + 49)

    # 首尾同时为锚
    result = fit_isotonic_anchored([1, 9, 0, 3], [1, 1, 1, 1], [0, 3])
    assert _fitted(result) == [
        Fraction(1),
        Fraction(3),
        Fraction(3),
        Fraction(3),
    ]
    assert _total_error(result) == Fraction(36 + 9)


def test_fully_fixed_returns_readings_unchanged():
    # 全部点固定：fitted 逐点等于 reading，误差为 0；
    # 相邻等读数锚点合并为最大等值块
    readings = [1, 1, 2, 2, 5]
    weights = [3, 1, 4, 1, 5]
    result = fit_isotonic_anchored(readings, weights, [0, 1, 2, 3, 4])
    assert _fitted(result) == [Fraction(r) for r in readings]
    assert _block_ranges(result) == ((0, 1), (2, 3), (4, 4))
    assert _total_error(result) == 0


def test_equal_values_merge_across_anchors_into_single_block():
    # 等值跨锚点也只能作为一个分块
    readings = [3, 1, 3, 1, 3]
    weights = [1, 1, 1, 1, 1]
    result = fit_isotonic_anchored(readings, weights, [1, 3])
    assert _fitted(result) == [Fraction(1)] * 4 + [Fraction(3)]
    assert _block_ranges(result) == ((0, 3), (4, 4))
    block = result.blocks[0]
    assert (block.mean_num, block.mean_den) == (1, 1)
    assert _total_error(result) == Fraction(4 + 4)


def test_anchor_preserves_signed_human_reviewed_value_exactly():
    # 人工复测点原值保留：即使两侧数据都把它往别处拉
    readings = [0, 0, 7, 0, 0]
    weights = [5, 5, 1, 5, 5]
    result = fit_isotonic_anchored(readings, weights, [2])
    fitted = _fitted(result)
    assert fitted[2] == Fraction(7)
    assert fitted == [Fraction(0), Fraction(0), Fraction(7), Fraction(7), Fraction(7)]


# ---------------------------------------------------------------------------
# 冲突：最早相邻下降锚点对
# ---------------------------------------------------------------------------


def test_infeasible_anchors_raise_earliest_conflict():
    # 锚点读数 5, 4, 2：最早冲突为 (0, 2)
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([5, 1, 4, 2], [1, 1, 1, 1], [0, 2, 3])
    assert (excinfo.value.first_index, excinfo.value.second_index) == (0, 2)

    # 前一对可行、后一对冲突：报告 (1, 2)
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([2, 2, 1], [1, 1, 1], [0, 1, 2])
    assert (excinfo.value.first_index, excinfo.value.second_index) == (1, 2)

    # 全固定但读数下降：同样不可行
    with pytest.raises(InfeasibleAnchorsError):
        fit_isotonic_anchored([3, 1], [1, 1], [0, 1])


def test_infeasible_check_scans_all_adjacent_pairs_in_order():
    import random

    rng = random.Random(31)
    for _ in range(200):
        n = rng.randint(2, 9)
        readings = [rng.randint(0, 9) for _ in range(n)]
        weights = [1] * n
        size = rng.randint(2, min(4, n))
        anchors = sorted(rng.sample(range(n), size))
        expected_pair = None
        for prev, curr in zip(anchors, anchors[1:]):
            if readings[prev] > readings[curr]:
                expected_pair = (prev, curr)
                break
        if expected_pair is None:
            result = fit_isotonic_anchored(readings, weights, anchors)
            assert len(result.fitted_num) == n
        else:
            with pytest.raises(InfeasibleAnchorsError) as excinfo:
                fit_isotonic_anchored(readings, weights, anchors)
            assert (excinfo.value.first_index, excinfo.value.second_index) == expected_pair


# ---------------------------------------------------------------------------
# 与原有无锚点拟合的关系
# ---------------------------------------------------------------------------


def test_non_binding_anchors_match_unanchored_fit():
    # 锚点不改变已满足其约束的拟合：无锚点最优解在锚点处恰好等于
    # reading 时，锚定结果与无锚点结果逐项一致
    unanchored = fit_isotonic([3, 1, 2], [1, 1, 1])
    anchored = fit_isotonic_anchored([3, 1, 2], [1, 1, 1], [2])
    assert anchored == unanchored

    import random

    rng = random.Random(77)
    for _ in range(100):
        n = rng.randint(2, 10)
        readings = [rng.randint(0, 8) for _ in range(n)]
        weights = [rng.randint(1, 5) for _ in range(n)]
        plain = fit_isotonic(readings, weights)
        plain_fitted = _fitted(plain)
        candidates = [
            i for i in range(n) if plain_fitted[i] == readings[i]
        ]
        if not candidates:
            continue
        anchors = sorted(
            rng.sample(candidates, rng.randint(1, len(candidates)))
        )
        anchored = fit_isotonic_anchored(readings, weights, anchors)
        assert _fitted(anchored) == plain_fitted
        assert _total_error(anchored) == _total_error(plain)
        assert _block_ranges(anchored) == _block_ranges(plain)


def test_anchored_solver_input_validation():
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1], [0])  # 长度不齐
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [])  # 至少一个锚点
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [0, 0])  # 重复锚点
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [2])  # 越界
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [-1])  # 越界


def test_anchored_deterministic_for_identical_input():
    import random

    rng = random.Random(5150)
    for _ in range(20):
        n = rng.randint(2, 15)
        readings = [rng.randint(0, 20) for _ in range(n)]
        weights = [rng.randint(1, 6) for _ in range(n)]
        anchors = sorted(rng.sample(range(n), rng.randint(1, min(4, n))))
        if not _is_feasible(readings, anchors):
            continue
        first = fit_isotonic_anchored(readings, weights, anchors)
        for _ in range(3):
            assert fit_isotonic_anchored(readings, weights, anchors) == first

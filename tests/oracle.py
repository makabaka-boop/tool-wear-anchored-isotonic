"""保序回归的"神谕"实现：枚举全部连续分区，用 Fraction 精确计算。

用于与 PAVA 对拍：n 个点的连续分区共有 2^(n-1) 种（每个相邻间隙
选择断开或合并）。对每个分区，块内 fitted 取加权均值，总误差为各块
误差之和。我们取总误差最小的分区；最优分块唯一（PAVA 的最粗分块），
故直接断言枚举选出的分块与 PAVA 完全一致。
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple


def _cut_masks(n: int):
    """生成 n 个点的全部 2^(n-1) 个连续分区（用切点位掩码表示）。"""

    for mask in range(1 << (n - 1)):
        yield mask


def _block_error(
    readings: Sequence[int], weights: Sequence[int], start: int, end: int
) -> Fraction:
    sw = sum(weights[i] for i in range(start, end + 1))
    swr = sum(weights[i] * readings[i] for i in range(start, end + 1))
    swr2 = sum(
        weights[i] * readings[i] * readings[i]
        for i in range(start, end + 1)
    )
    return Fraction(sw * swr2 - swr * swr, sw)


def _partition_blocks(
    readings: Sequence[int], weights: Sequence[int], mask: int
) -> Tuple[List[Tuple[int, int]], Fraction, bool]:
    """按切分掩码构造分块，返回（区间列表, 总误差, 是否可行）。

    可行当且仅当各块加权均值非递减——单点分块对降序输入误差为 0，
    但拟合曲线下降，不属于保序回归的可行域。
    """

    n = len(readings)
    ranges: List[Tuple[int, int]] = []
    error = Fraction(0, 1)
    prev_mean: Fraction | None = None
    feasible = True
    start = 0
    for i in range(n - 1):
        if mask & (1 << i):
            ranges.append((start, i))
            sw = sum(weights[start : i + 1])
            swr = sum(
                weights[j] * readings[j] for j in range(start, i + 1)
            )
            mean = Fraction(swr, sw)
            if prev_mean is not None and mean < prev_mean:
                feasible = False
            prev_mean = mean
            error += _block_error(readings, weights, start, i)
            start = i + 1
    ranges.append((start, n - 1))
    sw = sum(weights[start:n])
    swr = sum(weights[j] * readings[j] for j in range(start, n))
    mean = Fraction(swr, sw)
    if prev_mean is not None and mean < prev_mean:
        feasible = False
    error += _block_error(readings, weights, start, n - 1)
    return ranges, error, feasible


def oracle_best_partition(
    readings: Sequence[int], weights: Sequence[int]
) -> Tuple[Tuple[Tuple[int, int], ...], Fraction]:
    """枚举全部连续分区，返回（最优可行分块区间序列，最小总误差）。"""

    n = len(readings)
    best_ranges: Tuple[Tuple[int, int], ...] | None = None
    best_error = Fraction(10**100, 1)

    for mask in _cut_masks(n):
        ranges, error, feasible = _partition_blocks(readings, weights, mask)
        if not feasible:
            continue
        if error < best_error:
            best_error = error
            best_ranges = tuple(ranges)

    assert best_ranges is not None
    return best_ranges, best_error


def oracle_anchored_fit(
    readings: Sequence[int], weights: Sequence[int], anchor_indices: Sequence[int]
) -> Tuple[List[Fraction], Fraction]:
    """固定观测保序回归的独立神谕：有理数候选枚举 + 单调序列 DP。

    最优 fitted 的每个取值必为某区间的加权均值（无界块的均值，或锚点
    读数即单点区间均值），故候选值集合有限：全部 O(n^2) 个区间加权均值。
    在候选值上枚举非递减序列（锚点位置只允许取其读数），用 DP 精确求得
    最小加权平方误差。目标关于 fitted 严格凸（权重为正），最优 fitted
    唯一，因此可直接与求解器逐点对比。

    仅用于可行输入（锚点读数按加工顺序非递减）；返回（逐点 fitted,
    最小总误差）。
    """

    n = len(readings)
    anchors = set(anchor_indices)

    candidates = set()
    for u in range(n):
        sum_w = 0
        sum_wr = 0
        for v in range(u, n):
            sum_w += weights[v]
            sum_wr += weights[v] * readings[v]
            candidates.add(Fraction(sum_wr, sum_w))
    levels = sorted(candidates)
    m = len(levels)

    # dp[j]：处理到第 i 个点且 f_i = levels[j] 时的最小误差；
    # choice 记录最优前驱下标用于回溯。None 表示不可达。
    dp: List[Fraction | None] = [None] * m
    choices: List[List[int]] = []
    for i in range(n):
        row: List[Fraction | None] = [None] * m
        row_choice = [-1] * m
        for j in range(m):
            value = levels[j]
            if i in anchors and value != readings[i]:
                continue
            cost = weights[i] * (value - readings[i]) ** 2
            if i == 0:
                row[j] = cost
                continue
            best_prev: Fraction | None = None
            best_j = -1
            for jp in range(m):
                if levels[jp] > value:
                    break
                if dp[jp] is None:
                    continue
                if best_prev is None or dp[jp] < best_prev:
                    best_prev = dp[jp]
                    best_j = jp
            if best_prev is not None:
                row[j] = best_prev + cost
                row_choice[j] = best_j
        dp = row
        choices.append(row_choice)

    best_total: Fraction | None = None
    best_j = -1
    for j in range(m):
        if dp[j] is not None and (best_total is None or dp[j] < best_total):
            best_total = dp[j]
            best_j = j
    assert best_total is not None, "infeasible anchor configuration"

    fitted = [Fraction(0, 1)] * n
    j = best_j
    for i in range(n - 1, -1, -1):
        fitted[i] = levels[j]
        j = choices[i][j]
    return fitted, best_total

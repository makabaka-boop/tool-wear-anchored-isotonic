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

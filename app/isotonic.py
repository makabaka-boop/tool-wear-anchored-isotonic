"""加权最小二乘保序回归（Pool Adjacent Violators Algorithm）。

输入点按加工顺序排列，输出唯一的非递减拟合曲线。整个计算只使用
``int`` 与分数（整数分子/整数分母），不引入 ``float``：

* 块均值的比较通过交叉相乘完成；
* 每个块的加权均值与块内平方误差在合并时即时约分；
* 总误差由各块分数以公分母精确累加后再约分。

当相邻块均值相等时也进行合并，因此不存在"在何处断开并列块"的人为
选择空间——算法始终返回所有最优解中最粗（块数最少）的规范分块，
相同输入必然得到相同分块与逐点 fitted。

:func:`fit_isotonic_anchored` 是带固定观测（锚点）的变体：锚点的
fitted 被精确固定为其 reading，其余点在非递减约束下最小化加权平方
误差；锚点之间与两端按常数界做有界精确拟合（PAVA + 常数裁剪），
不使用极大有限权重近似固定。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Block:
    """一个连续合并块。"""

    start: int
    end: int
    mean_num: int
    """块加权均值的分子（不可约）。"""
    mean_den: int
    """块加权均值的分母（不可约，恒为正）。"""
    error_num: int
    """块内总误差 sum(w*(fitted-reading)^2) 的分子（不可约）。"""
    error_den: int
    """块内总误差的分母（不可约，恒为正）。"""


@dataclass(frozen=True)
class FitResult:
    """保序回归结果，分块索引均为含端点的 0 基索引。"""

    blocks: Tuple[Block, ...]
    fitted_num: Tuple[int, ...]
    """逐点 fitted 分子（与 observations 等长、同序）。"""
    fitted_den: Tuple[int, ...]
    """逐点 fitted 分母（恒为正）。"""
    total_error_num: int
    """总误差分子（不可约）。"""
    total_error_den: int
    """总误差分母（不可约，恒为正）。"""


class InfeasibleAnchorsError(ValueError):
    """锚点读数按加工顺序出现下降：不存在满足固定约束的非递减曲线。

    携带按加工顺序最早冲突的一对相邻锚点的 0 基下标
    （``first_index`` 在前、``second_index`` 在后）。
    """

    def __init__(self, first_index: int, second_index: int) -> None:
        self.first_index = first_index
        self.second_index = second_index
        super().__init__(
            "anchor readings decrease between adjacent anchors at "
            f"indices {first_index} and {second_index}"
        )


@dataclass
class _Pool:
    """PAVA 运行时的中间合并块（可变）。"""

    start: int
    end: int
    sum_wr: int  # sum(weight * reading)
    sum_w: int  # sum(weight)
    sum_wr2: int  # sum(weight * reading^2)
    error_num: int
    error_den: int


def _merge(left: _Pool, right: _Pool) -> _Pool:
    """合并两个相邻块，并精确重算合并块的平方误差。

    合并块均值 m = sum(w*r)/sum(w)，块误差为::

        sum(w*r^2) - (sum(w*r))^2 / sum(w)
                  = (sum_w*sum_wr2 - sum_wr^2) / sum_w
    """

    sum_w = left.sum_w + right.sum_w
    sum_wr = left.sum_wr + right.sum_wr
    sum_wr2 = left.sum_wr2 + right.sum_wr2
    error_num = sum_w * sum_wr2 - sum_wr * sum_wr
    error = Fraction(error_num, sum_w)
    return _Pool(
        start=left.start,
        end=right.end,
        sum_wr=sum_wr,
        sum_w=sum_w,
        sum_wr2=sum_wr2,
        error_num=error.numerator,
        error_den=error.denominator,
    )


def fit_isotonic(
    readings: Sequence[int], weights: Sequence[int]
) -> FitResult:
    """计算加权最小二乘保序回归。

    ``readings[i]`` / ``weights[i]`` 为非负整数（权重为正整数）。
    返回规范（最粗）分块、逐点 fitted 与总误差，全部以不可约分数表示。
    """

    n = len(readings)
    if n != len(weights):
        raise ValueError("readings and weights must have equal length")
    if n == 0:
        return FitResult((), (), (), 0, 1)

    stacks: List[_Pool] = []
    for i, (reading, weight) in enumerate(zip(readings, weights)):
        pool = _Pool(
            start=i,
            end=i,
            sum_wr=weight * reading,
            sum_w=weight,
            sum_wr2=weight * reading * reading,
            error_num=0,
            error_den=1,
        )
        # 栈顶块均值 <= 新块均值 时违反非递减约束（== 也合并，
        # 以消除并列处断开位置的任意性），比较用交叉相乘，无浮点。
        while stacks:
            top = stacks[-1]
            if top.sum_wr * pool.sum_w < pool.sum_wr * top.sum_w:
                break
            pool = _merge(stacks.pop(), pool)
        stacks.append(pool)

    blocks: List[Block] = []
    fitted_num: List[int] = [0] * n
    fitted_den: List[int] = [1] * n
    total = Fraction(0, 1)

    for pool in stacks:
        mean = Fraction(pool.sum_wr, pool.sum_w)
        error = Fraction(pool.error_num, pool.error_den)
        blocks.append(
            Block(
                start=pool.start,
                end=pool.end,
                mean_num=mean.numerator,
                mean_den=mean.denominator,
                error_num=error.numerator,
                error_den=error.denominator,
            )
        )
        for idx in range(pool.start, pool.end + 1):
            fitted_num[idx] = mean.numerator
            fitted_den[idx] = mean.denominator
        total += error

    total = Fraction(total.numerator, total.denominator)
    return FitResult(
        blocks=tuple(blocks),
        fitted_num=tuple(fitted_num),
        fitted_den=tuple(fitted_den),
        total_error_num=total.numerator,
        total_error_den=total.denominator,
    )


def _run_block(start: int, end: int, value: Fraction, error: Fraction) -> Block:
    """由公共 fitted 值与块内误差构造一个不可约分数的等值分块。"""

    return Block(
        start=start,
        end=end,
        mean_num=value.numerator,
        mean_den=value.denominator,
        error_num=error.numerator,
        error_den=error.denominator,
    )


def fit_isotonic_anchored(
    readings: Sequence[int],
    weights: Sequence[int],
    anchor_indices: Sequence[int],
) -> FitResult:
    """带固定观测的加权最小二乘保序回归。

    锚点（``anchor_indices`` 给出的 0 基下标）的 fitted 必须逐点等于其
    ``reading``；其余点在整条非递减曲线上最小化加权平方误差。

    锚点把序列切成独立段（两端与相邻锚点之间）：段内拟合值只需非递减
    并落在相邻锚点读数给出的常数界 [L, U] 内。常数界下的有界保序回归
    有精确闭式解——由最小-最大公式，常数裁剪与保序投影可交换，因此
    先对段内做（无界）PAVA，再把各块均值裁剪到 [L, U] 即得有界问题的
    精确最优解；锚点本身直接取其读数。全程不借助任何"极大有限权重"
    之类的近似固定手段。

    锚点读数按加工顺序（下标升序）必须非递减，否则无可行曲线，抛出
    :class:`InfeasibleAnchorsError`，并携带最早冲突的相邻锚点对。

    返回的 ``blocks`` 为整条曲线上的**最大连续等值**分块：等值跨越
    锚点时也合并为同一块，块均值即该块的公共 fitted 值。
    """

    n = len(readings)
    if n != len(weights):
        raise ValueError("readings and weights must have equal length")
    if not anchor_indices:
        raise ValueError("at least one anchor is required")
    anchors = sorted(anchor_indices)
    if anchors[0] < 0 or anchors[-1] >= n:
        raise ValueError("anchor index out of range")
    if any(curr <= prev for prev, curr in zip(anchors, anchors[1:])):
        raise ValueError("duplicate anchor indices")

    # 可行性：锚点读数按加工顺序非递减；相邻锚点对的第一个下降即
    # 最早冲突（非相邻的下降必然蕴含某相邻对下降）。
    for prev, curr in zip(anchors, anchors[1:]):
        if readings[prev] > readings[curr]:
            raise InfeasibleAnchorsError(prev, curr)

    fitted: List[Fraction] = [Fraction(0, 1)] * n

    # 段边界：(-1, anchors..., n)；每段为开区间 (left, right) 内的点。
    cuts = [-1, *anchors, n]
    for left, right in zip(cuts, cuts[1:]):
        lo = left + 1
        hi = right - 1
        if lo > hi:
            continue
        lower = Fraction(readings[left]) if left >= 0 else None
        upper = Fraction(readings[right]) if right < n else None
        segment = fit_isotonic(readings[lo : hi + 1], weights[lo : hi + 1])
        for block in segment.blocks:
            mean = Fraction(block.mean_num, block.mean_den)
            if lower is not None and mean < lower:
                mean = lower
            if upper is not None and mean > upper:
                mean = upper
            for offset in range(block.start, block.end + 1):
                fitted[lo + offset] = mean

    for index in anchors:
        fitted[index] = Fraction(readings[index])

    # 汇总：逐点误差精确累加；最大连续等值分块（等值跨锚点也合并）。
    total = Fraction(0, 1)
    blocks: List[Block] = []
    run_start = 0
    run_error = Fraction(0, 1)
    for i in range(n):
        residual = fitted[i] - readings[i]
        point_error = weights[i] * residual * residual
        total += point_error
        if fitted[i] != fitted[run_start]:
            blocks.append(
                _run_block(run_start, i - 1, fitted[run_start], run_error)
            )
            run_start = i
            run_error = Fraction(0, 1)
        run_error += point_error
    blocks.append(_run_block(run_start, n - 1, fitted[run_start], run_error))

    return FitResult(
        blocks=tuple(blocks),
        fitted_num=tuple(value.numerator for value in fitted),
        fitted_den=tuple(value.denominator for value in fitted),
        total_error_num=total.numerator,
        total_error_den=total.denominator,
    )

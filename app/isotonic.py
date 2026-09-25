"""加权最小二乘保序回归（Pool Adjacent Violators Algorithm）。

输入点按加工顺序排列，输出唯一的非递减拟合曲线。整个计算只使用
``int`` 与分数（整数分子/整数分母），不引入 ``float``：

* 块均值的比较通过交叉相乘完成；
* 每个块的加权均值与块内平方误差在合并时即时约分；
* 总误差由各块分数以公分母精确累加后再约分。

当相邻块均值相等时也进行合并，因此不存在"在何处断开并列块"的人为
选择空间——算法始终返回所有最优解中最粗（块数最少）的规范分块，
相同输入必然得到相同分块与逐点 fitted。
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

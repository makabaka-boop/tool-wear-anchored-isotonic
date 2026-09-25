"""请求/响应模型与输入校验。

约束：
* 每次请求 2 至 5000 个观测；
* 每项有唯一 ASCII ``id``、整数 ``reading`` ∈ [0, 10^9]、
  整数 ``weight`` ∈ [1, 10^6]；
* 未知字段、重复 id、越界值、类型错误一律令整次请求返回 422。

固定观测校正（``/correct/anchored``）在此基础上额外接受 1 至 12 个
锚点 id；锚点 id 重复或指向不存在的观测同样按原校验契约返回 422。
"""

from __future__ import annotations

from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MAX_OBSERVATIONS = 5000
MIN_OBSERVATIONS = 2
MAX_READING = 10**9
MAX_WEIGHT = 10**6
MIN_ANCHORS = 1
MAX_ANCHORS = 12


class Observation(BaseModel):
    """单个刀具磨损观测。拒绝一切未知字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    reading: int = Field(ge=0, le=MAX_READING)
    weight: int = Field(ge=1, le=MAX_WEIGHT)

    @field_validator("id")
    @classmethod
    def _ascii_only(cls, value: str) -> str:
        if not value.isascii():
            raise ValueError("id must contain ASCII characters only")
        return value


class CorrectionRequest(BaseModel):
    """整批观测；额外的顶层未知字段同样拒绝。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    observations: List[Observation] = Field(
        min_length=MIN_OBSERVATIONS, max_length=MAX_OBSERVATIONS
    )

    @field_validator("observations")
    @classmethod
    def _unique_ids(cls, value: List[Observation]) -> List[Observation]:
        seen: set[str] = set()
        for item in value:
            if item.id in seen:
                raise ValueError(f"duplicate id: {item.id!r}")
            seen.add(item.id)
        return value


class FractionModel(BaseModel):
    """不可约分数。"""

    model_config = ConfigDict(extra="forbid")

    numerator: int
    denominator: int


class BlockModel(BaseModel):
    """最大连续等值分块：含端点的 0 基索引与块内公共 fitted 值。

    无锚点校正中等值块即 PAVA 合并块，``mean`` 为块内加权均值；
    固定观测校正中 ``mean`` 为该等值块的公共 fitted 值（可能因
    锚点或常数界裁剪而不同于块内读数的加权均值）。
    """

    model_config = ConfigDict(extra="forbid")

    start_index: int
    end_index: int
    mean: FractionModel


class FittedPoint(BaseModel):
    """逐点拟合结果，保留全部原始点。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    fitted: FractionModel


class CorrectionResponse(BaseModel):
    """保序校正结果。"""

    model_config = ConfigDict(extra="forbid")

    blocks: List[BlockModel]
    fitted: List[FittedPoint]
    total_error: FractionModel


class AnchoredCorrectionRequest(CorrectionRequest):
    """固定观测校正请求：整批观测之外，指定 1 至 12 个锚点 id。

    锚点的 fitted 被固定为其 reading。锚点 id 重复或指向不存在的
    观测时，与原有校验失败一样整次请求返回 422。
    """

    anchors: List[str] = Field(min_length=MIN_ANCHORS, max_length=MAX_ANCHORS)

    @field_validator("anchors")
    @classmethod
    def _unique_anchor_ids(cls, value: List[str]) -> List[str]:
        seen: set[str] = set()
        for item in value:
            if item in seen:
                raise ValueError(f"duplicate anchor id: {item!r}")
            seen.add(item)
        return value

    @model_validator(mode="after")
    def _anchors_must_exist(self) -> "AnchoredCorrectionRequest":
        known = {item.id for item in self.observations}
        for anchor in self.anchors:
            if anchor not in known:
                raise ValueError(f"unknown anchor id: {anchor!r}")
        return self


class AnchoredFittedPoint(BaseModel):
    """逐点拟合结果，并明确标出该点是否为固定观测（锚点）。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    fitted: FractionModel
    fixed: bool


class AnchoredCorrectionResponse(BaseModel):
    """固定观测校正结果：分块为最大连续等值块（等值跨锚点也合并）。"""

    model_config = ConfigDict(extra="forbid")

    blocks: List[BlockModel]
    fitted: List[AnchoredFittedPoint]
    total_error: FractionModel

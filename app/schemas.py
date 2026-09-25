"""请求/响应模型与输入校验。

约束：
* 每次请求 2 至 5000 个观测；
* 每项有唯一 ASCII ``id``、整数 ``reading`` ∈ [0, 10^9]、
  整数 ``weight`` ∈ [1, 10^6]；
* 未知字段、重复 id、越界值、类型错误一律令整次请求返回 422。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_OBSERVATIONS = 5000
MIN_OBSERVATIONS = 2
MAX_READING = 10**9
MAX_WEIGHT = 10**6


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
    """连续合并块：含端点的 0 基索引与块加权均值。"""

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

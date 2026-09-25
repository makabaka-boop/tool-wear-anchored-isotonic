"""FastAPI 入口：接收普通 JSON，返回保序校正结果。

* ``POST /correct``：加权最小二乘保序回归；
* ``POST /correct/anchored``：固定观测校正——锚点 fitted 精确等于其
  reading，其余点仍在整条非递减曲线上最小化加权平方误差。锚点读数
  按加工顺序出现下降时返回 409 与 ``INFEASIBLE_ANCHORS``（含最早冲突
  的相邻锚点 id），不发布部分拟合。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .isotonic import InfeasibleAnchorsError, fit_isotonic, fit_isotonic_anchored
from .schemas import (
    AnchoredCorrectionRequest,
    AnchoredCorrectionResponse,
    AnchoredFittedPoint,
    BlockModel,
    CorrectionRequest,
    CorrectionResponse,
    FittedPoint,
    FractionModel,
)

app = FastAPI(
    title="Tool Wear Isotonic Correction Service",
    version="1.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/correct", response_model=CorrectionResponse)
def correct(request: CorrectionRequest) -> CorrectionResponse:
    observations = request.observations
    readings = [o.reading for o in observations]
    weights = [o.weight for o in observations]

    result = fit_isotonic(readings, weights)

    blocks = [
        BlockModel(
            start_index=block.start,
            end_index=block.end,
            mean=FractionModel(
                numerator=block.mean_num,
                denominator=block.mean_den,
            ),
        )
        for block in result.blocks
    ]
    fitted = [
        FittedPoint(
            id=observations[i].id,
            fitted=FractionModel(
                numerator=result.fitted_num[i],
                denominator=result.fitted_den[i],
            ),
        )
        for i in range(len(observations))
    ]
    return CorrectionResponse(
        blocks=blocks,
        fitted=fitted,
        total_error=FractionModel(
            numerator=result.total_error_num,
            denominator=result.total_error_den,
        ),
    )


@app.post("/correct/anchored", response_model=AnchoredCorrectionResponse)
def correct_anchored(
    request: AnchoredCorrectionRequest,
) -> AnchoredCorrectionResponse | JSONResponse:
    observations = request.observations
    readings = [o.reading for o in observations]
    weights = [o.weight for o in observations]
    id_to_index = {o.id: i for i, o in enumerate(observations)}
    anchor_indices = sorted(id_to_index[anchor] for anchor in request.anchors)

    try:
        result = fit_isotonic_anchored(readings, weights, anchor_indices)
    except InfeasibleAnchorsError as exc:
        # 锚点读数按加工顺序下降：不发布任何部分拟合，只报告最早冲突。
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "error": "INFEASIBLE_ANCHORS",
                    "conflicting_anchor_ids": [
                        observations[exc.first_index].id,
                        observations[exc.second_index].id,
                    ],
                }
            },
        )

    anchor_set = set(anchor_indices)
    blocks = [
        BlockModel(
            start_index=block.start,
            end_index=block.end,
            mean=FractionModel(
                numerator=block.mean_num,
                denominator=block.mean_den,
            ),
        )
        for block in result.blocks
    ]
    fitted = [
        AnchoredFittedPoint(
            id=observations[i].id,
            fitted=FractionModel(
                numerator=result.fitted_num[i],
                denominator=result.fitted_den[i],
            ),
            fixed=i in anchor_set,
        )
        for i in range(len(observations))
    ]
    return AnchoredCorrectionResponse(
        blocks=blocks,
        fitted=fitted,
        total_error=FractionModel(
            numerator=result.total_error_num,
            denominator=result.total_error_den,
        ),
    )

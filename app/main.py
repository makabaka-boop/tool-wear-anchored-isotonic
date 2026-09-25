"""FastAPI 入口：接收普通 JSON，返回保序校正结果。"""

from __future__ import annotations

from fastapi import FastAPI

from .isotonic import fit_isotonic
from .schemas import (
    BlockModel,
    CorrectionRequest,
    CorrectionResponse,
    FittedPoint,
    FractionModel,
)

app = FastAPI(
    title="Tool Wear Isotonic Correction Service",
    version="1.0.0",
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

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from qmshe.api.dependencies import get_pipeline
from qmshe.pipeline import QMSHEPipeline

router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=12, ge=1, le=100)
    return_debug: bool = False


@router.post("/query")
def query(request: QueryRequest, pipeline: QMSHEPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.query(request.question, request.top_k, request.return_debug).__dict__


@router.get("/metrics")
def metrics(pipeline: QMSHEPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.metrics.snapshot()

from qmshe.pipeline import QMSHEPipeline

_pipeline: QMSHEPipeline | None = None


def set_pipeline(pipeline: QMSHEPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> QMSHEPipeline:
    if _pipeline is None:
        raise RuntimeError("index is not built; call POST /v1/index/build first")
    return _pipeline


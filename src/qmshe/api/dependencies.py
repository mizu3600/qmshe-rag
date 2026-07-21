from qmshe.pipeline import QMSHEPipeline
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline

_pipeline: QMSHEPipeline | None = None
_graph_pipelines: dict[GraphProfile, QMSGEGraphPipeline] = {}


def set_pipeline(pipeline: QMSHEPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> QMSHEPipeline:
    if _pipeline is None:
        raise RuntimeError("index is not built; call POST /v1/index/build first")
    return _pipeline


def set_graph_pipeline(pipeline: QMSGEGraphPipeline) -> None:
    _graph_pipelines[pipeline.profile] = pipeline


def get_graph_pipeline(
    profile: GraphProfile | str = GraphProfile.REIFIED_FACT,
) -> QMSGEGraphPipeline:
    selected = GraphProfile(profile)
    if selected not in _graph_pipelines:
        raise RuntimeError(
            f"ordinary graph index ({selected.value}) is not built; "
            "call POST /v1/index/build with mode=graph first"
        )
    return _graph_pipelines[selected]

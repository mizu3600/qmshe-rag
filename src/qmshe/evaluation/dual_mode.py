from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.benchmarks.schemas import BenchmarkSuite
from qmshe.evaluation.experiment import LocalBenchmarkEncoder
from qmshe.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


@dataclass(frozen=True)
class DualModeRecord:
    example_id: str
    mode: str
    profile: str
    hop_count: int
    query_type: str
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    bridge_recall_at_20: float
    latency_ms: float


class DualModeExperimentRunner:
    """Fair comparison: one corpus, encoder, top-k and generator policy across all modes."""

    def __init__(self, encoder_dimension: int = 128):
        self.encoder = LocalBenchmarkEncoder(encoder_dimension)

    def run(self, suite: BenchmarkSuite, output_dir: str | Path) -> list[DualModeRecord]:
        records: list[DualModeRecord] = []
        for example in suite.examples:
            built = build_example_corpus(example)
            hypergraph_pipeline = QMSHEPipeline(built.corpus, text_encoder=self.encoder)
            pipelines = [
                ("vanilla_rag", "dense_bge_compatible", hypergraph_pipeline),
                ("hypergraph", "evidence_hypergraph", hypergraph_pipeline),
                ("graph", GraphProfile.ENTITY_RELATION.value, QMSGEGraphPipeline(
                    built.corpus, text_encoder=self.encoder, profile=GraphProfile.ENTITY_RELATION
                )),
                ("graph", GraphProfile.REIFIED_FACT.value, QMSGEGraphPipeline(
                    built.corpus, text_encoder=self.encoder, profile=GraphProfile.REIFIED_FACT
                )),
            ]
            for mode, profile, pipeline in pipelines:
                pipeline.generator.client = None
                started = perf_counter()
                if mode == "vanilla_rag":
                    query_vector = self.encoder.encode([example.question])[0]
                    hits = pipeline.raw_index.search(query_vector, 60, "vanilla-dense")
                    facts = [hit.object_id for hit in hits if hit.object_id.startswith("fact_")][:20]
                    entities = [hit.object_id for hit in hits if hit.object_id.startswith("ent_")][:20]
                else:
                    result = pipeline.query(example.question, top_k=20, return_debug=False)
                    if mode == "hypergraph":
                        facts = result.retrieved_hyperedges
                        entities = result.retrieved_entities
                    else:
                        facts = result.retrieved_facts
                        entities = [item for item in result.retrieved_nodes if item.startswith("ent_")]
                latency = (perf_counter() - started) * 1000
                records.append(DualModeRecord(
                    example_id=example.example_id, mode=mode, profile=profile,
                    hop_count=example.hop_count, query_type=example.query_type,
                    recall_at_10=recall_at_k(facts, built.gold_fact_ids, 10),
                    recall_at_20=recall_at_k(facts, built.gold_fact_ids, 20),
                    mrr=reciprocal_rank(facts, built.gold_fact_ids),
                    ndcg_at_10=ndcg_at_k(facts, built.gold_fact_ids, 10),
                    bridge_recall_at_20=recall_at_k(entities, built.bridge_entity_ids, 20),
                    latency_ms=latency,
                ))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "records.json").write_text(
            json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8"
        )
        summary = _summarize(records)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_render_report(suite, summary), encoding="utf-8")
        return records


def _summarize(records: list[DualModeRecord]) -> dict:
    groups: dict[str, list[DualModeRecord]] = {}
    for record in records:
        groups.setdefault(f"{record.mode}:{record.profile}", []).append(record)
    metrics = ("recall_at_10", "recall_at_20", "mrr", "ndcg_at_10", "bridge_recall_at_20", "latency_ms")
    return {
        name: {metric: mean(getattr(record, metric) for record in items) for metric in metrics}
        for name, items in groups.items()
    }


def _render_report(suite: BenchmarkSuite, summary: dict) -> str:
    rows = [
        f"# {suite.name} Graph/Hypergraph fair comparison", "",
        f"Examples: {len(suite.examples)}", "",
        "| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        rows.append(
            f"| {name} | {item['recall_at_10']:.4f} | {item['recall_at_20']:.4f} | "
            f"{item['mrr']:.4f} | {item['ndcg_at_10']:.4f} | "
            f"{item['bridge_recall_at_20']:.4f} | {item['latency_ms']:.2f} |"
        )
    return "\n".join(rows) + "\n"

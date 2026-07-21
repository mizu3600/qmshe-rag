from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def __init__(self):
        self.encoder = DeterministicEmbedder(64)

    def encode(self, texts):
        return self.encoder.embed(texts)


def test_end_to_end_query_returns_traceable_evidence():
    pipeline = QMSHEPipeline(make_synthetic_corpus(), text_encoder=LocalEncoder())
    result = pipeline.query("How does PEAI improve Voc in inverted PSCs?", top_k=4, return_debug=True)
    assert result.retrieved_hyperedges
    assert all(item.startswith("fact_") for item in result.retrieved_hyperedges)
    assert result.citations
    assert abs(sum(result.band_weights.values()) - 1.0) < 1e-6
    assert abs(sum(result.relation_weights.values()) - 1.0) < 1e-6
    assert result.evidence_path

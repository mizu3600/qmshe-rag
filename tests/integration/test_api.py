from fastapi.testclient import TestClient

from qmshe.api.dependencies import set_pipeline
from qmshe.api.main import app
from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def test_health_and_query_api():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    set_pipeline(QMSHEPipeline(make_synthetic_corpus(), text_encoder=LocalEncoder()))
    response = client.post("/v1/query", json={"question": "How does PEAI improve Voc?", "top_k": 3, "return_debug": True})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert set(body["band_weights"]) == {"raw", "low", "mid", "high"}
    assert body["relation_weights"]
    evaluation = client.post("/v1/evaluate/run")
    assert evaluation.status_code == 200
    run_id = evaluation.json()["run_id"]
    assert client.get(f"/v1/evaluate/{run_id}").json()["status"] == "completed"
    metrics = client.get("/v1/metrics").json()
    assert metrics["queries"] >= 1
    assert metrics["p95_latency_ms"] >= 0

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from qmshe.api.dependencies import set_pipeline
from qmshe.extraction.canonicalizer import canonicalize_entities
from qmshe.extraction.entity_extractor import extract_entities_rule_based
from qmshe.extraction.fact_extractor import extract_facts_rule_based, extract_facts_with_llm
from qmshe.ingest.chunker import chunk_document
from qmshe.ingest.pdf_parser import parse_document
from qmshe.ingest.schemas import Corpus
from qmshe.pipeline import QMSHEPipeline, load_corpus, save_corpus
from qmshe.providers import DeepSeekClient, ProviderError

router = APIRouter(prefix="/v1", tags=["index"])


class IngestRequest(BaseModel):
    source_uri: str
    domain: str = "PSC"
    rebuild_embeddings: bool = True


class BuildRequest(BaseModel):
    corpus_version: str = "psc-v1"
    corpus_path: str = "data/processed/corpus.json"
    build_evidence_graph: bool = True
    build_semantic_graph: bool = True
    build_spectral_embeddings: bool = True


class IncrementalRequest(BaseModel):
    corpus_path: str


@router.post("/documents/ingest")
def ingest(request: IngestRequest) -> dict:
    path = Path(request.source_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="source file not found")
    parsed = parse_document(path, request.domain)
    chunks = chunk_document(parsed)
    entities = canonicalize_entities(extract_entities_rule_based(chunks))
    try:
        facts = extract_facts_with_llm(chunks, entities, DeepSeekClient())
    except ProviderError:
        facts = extract_facts_rule_based(chunks, entities)
    corpus = Corpus(documents=[parsed.document], chunks=chunks, entities=entities, evidence_hyperedges=facts)
    output = Path("data/processed/corpus.json")
    save_corpus(corpus, output)
    return {"document_id": parsed.document.document_id, "chunks": len(chunks), "entities": len(entities),
            "facts": len(facts), "corpus_path": str(output)}


@router.post("/index/build")
def build_index(request: BuildRequest) -> dict:
    _ = (request.build_evidence_graph, request.build_semantic_graph, request.build_spectral_embeddings)
    corpus = load_corpus(request.corpus_path)
    set_pipeline(QMSHEPipeline(corpus))
    return {"status": "ready", "corpus_version": request.corpus_version,
            "entities": len(corpus.entities), "hyperedges": len(corpus.evidence_hyperedges)}


@router.post("/index/incremental")
def incremental_index(request: IncrementalRequest) -> dict:
    from qmshe.api.dependencies import get_pipeline

    pipeline = get_pipeline()
    updated = load_corpus(request.corpus_path)
    return pipeline.incremental_update(updated)

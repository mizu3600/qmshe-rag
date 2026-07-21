import json
from pathlib import Path

import typer

from qmshe.extraction.canonicalizer import canonicalize_entities
from qmshe.extraction.entity_extractor import extract_entities_rule_based
from qmshe.extraction.fact_extractor import extract_facts_rule_based, extract_facts_with_llm
from qmshe.ingest.chunker import chunk_document
from qmshe.ingest.pdf_parser import parse_document
from qmshe.ingest.schemas import Corpus
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline, load_corpus, save_corpus
from qmshe.providers import DeepSeekClient, ProviderError
from qmshe.synthetic import make_synthetic_corpus

app = typer.Typer(help="QMSHE-RAG research prototype")


@app.command()
def ingest(source: Path, output: Path = Path("data/processed/corpus.json")) -> None:
    parsed = parse_document(source)
    chunks = chunk_document(parsed)
    entities = canonicalize_entities(extract_entities_rule_based(chunks))
    try:
        facts = extract_facts_with_llm(chunks, entities, DeepSeekClient())
    except ProviderError:
        facts = extract_facts_rule_based(chunks, entities)
    corpus = Corpus(documents=[parsed.document], chunks=chunks, entities=entities, evidence_hyperedges=facts)
    save_corpus(corpus, output)
    typer.echo(f"saved {len(chunks)} chunks, {len(entities)} entities, {len(facts)} facts to {output}")


@app.command()
def build(
    corpus_path: Path = Path("data/processed/corpus.json"),
    mode: str = "hypergraph", graph_profile: str = "reified_fact",
) -> None:
    corpus = load_corpus(corpus_path)
    if mode == "graph":
        pipeline = QMSGEGraphPipeline(corpus, profile=GraphProfile(graph_profile))
        typer.echo(
            f"built ordinary graph ({pipeline.profile.value}) with {len(pipeline.node_ids)} nodes"
        )
        return
    if mode != "hypergraph":
        raise typer.BadParameter("mode must be hypergraph or graph")
    pipeline = QMSHEPipeline(corpus)
    typer.echo(f"built hypergraph index with {len(pipeline.object_ids)} objects")


@app.command()
def query(
    question: str, corpus_path: Path = Path("data/processed/corpus.json"),
    mode: str = "hypergraph", graph_profile: str = "reified_fact",
) -> None:
    corpus = load_corpus(corpus_path)
    if mode == "graph":
        result = QMSGEGraphPipeline(corpus, profile=GraphProfile(graph_profile)).query(
            question, return_debug=True
        )
    elif mode == "hypergraph":
        result = QMSHEPipeline(corpus).query(question, return_debug=True)
    else:
        raise typer.BadParameter("mode must be hypergraph or graph")
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command()
def demo(output: Path = Path("data/processed/synthetic.json")) -> None:
    corpus = make_synthetic_corpus()
    save_corpus(corpus, output)
    result = QMSHEPipeline(corpus).query("How does PEAI improve Voc in inverted PSCs?", return_debug=True)
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command()
def evaluate() -> None:
    corpus = make_synthetic_corpus()
    result = QMSHEPipeline(corpus).query("How does PEAI improve Voc?", return_debug=True)
    gold = {"fact_1", "fact_2", "fact_3"}
    recall = len(set(result.retrieved_hyperedges[:20]) & gold) / len(gold)
    typer.echo(json.dumps({"supporting_fact_recall@20": recall, "retrieved": result.retrieved_hyperedges}, indent=2))

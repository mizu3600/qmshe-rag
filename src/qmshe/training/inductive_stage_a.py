from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.embedding.text_encoder import encode_queries
from qmshe.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


@dataclass(frozen=True)
class InductiveStageAConfig:
    mode: str
    output_path: str
    profile: str = "evidence_hypergraph"
    epochs: int = 3
    learning_rate: float = 2e-4
    bridge_loss_weight: float = 0.5
    margin_weight: float = 0.2
    seed: int = 42
    device: str = "cuda"


@dataclass(frozen=True)
class InductiveStageAReport:
    mode: str
    profile: str
    seed: int
    train_examples: int
    validation_examples: int
    input_dim: int
    roles: list[str]
    loss_history: list[float]
    validation_history: list[dict[str, float]]
    selected_epoch: int
    elapsed_seconds: float
    device: str


@dataclass
class _PreparedGraph:
    question: str
    query_vector: torch.Tensor
    raw_features: torch.Tensor
    operator: torch.Tensor
    role_operators: dict[str, torch.Tensor]
    node_ids: list[str]
    positive_indices: list[int]
    bridge_indices: list[int]
    facts_by_entity: dict[str, set[str]]
    fact_by_node: dict[str, str]
    gold_fact_ids: set[str]


def train_inductive_stage_a(
    training_examples, validation_examples, text_encoder, config: InductiveStageAConfig,
) -> InductiveStageAReport:
    if config.mode not in {"hypergraph", "graph"}:
        raise ValueError("mode must be hypergraph or graph")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    started = perf_counter()
    train_graphs, template, roles = _prepare(
        training_examples, text_encoder, config.mode, config.profile, config.seed
    )
    validation_graphs, _, validation_roles = _prepare(
        validation_examples, text_encoder, config.mode, config.profile, config.seed
    )
    if roles != validation_roles:
        raise ValueError(f"training roles {roles} do not match validation roles {validation_roles}")
    model = template.model.to(device)
    relation_gate = template.relation_gate.to(device) if config.mode == "hypergraph" else None
    parameters = list(model.parameters()) + (
        list(relation_gate.parameters()) if relation_gate is not None else []
    )
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    loss_history, validation_history = [], []
    best_state, best_score, selected_epoch = None, (-1.0, -1.0), 0
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = list(range(len(train_graphs)))
        rng.shuffle(order)
        model.train()
        if relation_gate is not None:
            relation_gate.train()
        losses = []
        for index in order:
            graph = train_graphs[index]
            query = graph.query_vector.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = _score_graph(model, relation_gate, graph, query, roles, device)
            positive = torch.tensor(graph.positive_indices, device=device)
            positives = scores[positive]
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(positives, dim=0)
            negative_mask = torch.ones(len(scores), dtype=torch.bool, device=device)
            negative_mask[positive] = False
            if negative_mask.any():
                loss = loss + config.margin_weight * torch.relu(
                    0.2 - positives.max() + scores[negative_mask].max()
                )
            if graph.bridge_indices and config.bridge_loss_weight > 0:
                bridge = scores[torch.tensor(graph.bridge_indices, device=device)]
                loss = loss + config.bridge_loss_weight * (
                    torch.logsumexp(scores, dim=0) - torch.logsumexp(bridge, dim=0)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        loss_history.append(mean(losses))
        recall, mrr = _evaluate(model, relation_gate, validation_graphs, roles, device)
        validation_history.append({
            "epoch": float(epoch + 1), "recall_at_20": recall, "mrr": mrr,
        })
        if (recall, mrr) > best_score:
            best_score, selected_epoch = (recall, mrr), epoch + 1
            best_state = {
                "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "relation_gate": (
                    {name: value.detach().cpu().clone() for name, value in relation_gate.state_dict().items()}
                    if relation_gate is not None else None
                ),
            }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "mode": config.mode, "profile": config.profile,
        "input_dim": train_graphs[0].raw_features.shape[1], "roles": roles,
        "seed": config.seed, "selected_epoch": selected_epoch,
        "model": best_state["model"],
    }
    if best_state["relation_gate"] is not None:
        payload["relation_gate"] = best_state["relation_gate"]
    torch.save(payload, output)
    report = InductiveStageAReport(
        mode=config.mode, profile=config.profile, seed=config.seed,
        train_examples=len(train_graphs), validation_examples=len(validation_graphs),
        input_dim=train_graphs[0].raw_features.shape[1], roles=roles,
        loss_history=loss_history, validation_history=validation_history,
        selected_epoch=selected_epoch, elapsed_seconds=perf_counter() - started,
        device=str(device),
    )
    output.with_suffix(".json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report


def _prepare(examples, encoder, mode, profile, seed):
    output, template, role_names = [], None, None
    for example in examples:
        built = build_example_corpus(example)
        if mode == "hypergraph":
            pipeline = QMSHEPipeline(
                built.corpus, text_encoder=encoder, seed=seed, enable_remote_reranker=False
            )
            roles = pipeline.role_names
            operator = pipeline.laplacian
            role_operators = pipeline.role_laplacians
            node_ids = pipeline.object_ids
            facts_by_entity, fact_by_node = {}, {}
        else:
            pipeline = QMSGEGraphPipeline(
                built.corpus, text_encoder=encoder, profile=GraphProfile(profile), seed=seed,
                enable_remote_reranker=False,
            )
            roles = []
            operator = pipeline.propagation
            role_operators = {}
            node_ids = pipeline.node_ids
            facts_by_entity = pipeline.artifacts.facts_by_entity
            fact_by_node = pipeline.artifacts.fact_by_node
        if role_names is None:
            role_names = roles
        elif role_names != roles:
            raise ValueError(f"inconsistent role vocabulary: {role_names} vs {roles}")
        if template is None:
            template = pipeline
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        positives = set(built.gold_fact_ids)
        if mode == "graph" and GraphProfile(profile) is GraphProfile.ENTITY_RELATION:
            for entity_id, fact_ids in facts_by_entity.items():
                if fact_ids & built.gold_fact_ids:
                    positives.add(entity_id)
        positive_indices = [node_index[item] for item in positives if item in node_index]
        if not positive_indices:
            continue
        output.append(_PreparedGraph(
            question=example.question,
            query_vector=torch.tensor(
                encode_queries(encoder, [example.question])[0], dtype=torch.float32
            ),
            raw_features=pipeline.raw_features.detach().cpu(),
            operator=operator.cpu(),
            role_operators={name: item.cpu() for name, item in role_operators.items()},
            node_ids=node_ids, positive_indices=positive_indices,
            bridge_indices=[node_index[item] for item in built.bridge_entity_ids if item in node_index],
            facts_by_entity=facts_by_entity, fact_by_node=fact_by_node,
            gold_fact_ids=built.gold_fact_ids,
        ))
    if not output or template is None:
        raise ValueError("no usable graphs were prepared")
    return output, template, role_names or []


def _score_graph(model, relation_gate, graph, query, roles, device):
    raw = graph.raw_features.to(device)
    operator = graph.operator.to(device)
    bands = model.encode_nodes(raw, operator)
    if relation_gate is not None:
        weights = relation_gate(query)
        role_bands = {
            role: model.encode_nodes(raw, graph.role_operators[role].to(device)) for role in roles
        }
        conditioned = {"raw": bands["raw"]}
        for name in ("low", "mid", "high"):
            stacked = torch.stack([role_bands[role][name] for role in roles])
            conditioned[name] = torch.einsum("r,rnd->nd", weights, stacked)
        bands = conditioned
    query_vector, _ = model.encode_query(query, raw, bands)
    full = torch.cat([bands[name] for name in ("raw", "low", "mid", "high")], dim=-1)
    return full @ query_vector


def _evaluate(model, relation_gate, graphs, roles, device):
    model.eval()
    if relation_gate is not None:
        relation_gate.eval()
    recalls, reciprocal_ranks = [], []
    with torch.no_grad():
        for graph in graphs:
            # Validation query embeddings are frozen during Stage A.
            query = graph.query_vector.to(device)
            scores = _score_graph(model, relation_gate, graph, query, roles, device)
            ranking = torch.argsort(scores, descending=True).cpu().tolist()
            ranked_nodes = [graph.node_ids[index] for index in ranking]
            ranked_facts = _rank_facts(ranked_nodes, graph)
            gold = graph.gold_fact_ids
            recalls.append(recall_at_k(ranked_facts, gold, 20))
            reciprocal_ranks.append(reciprocal_rank(ranked_facts, gold))
    return mean(recalls), mean(reciprocal_ranks)


def _rank_facts(ranked_nodes, graph):
    output = []
    for node_id in ranked_nodes:
        if node_id.startswith("fact_"):
            output.append(node_id)
        if node_id in graph.fact_by_node:
            output.append(graph.fact_by_node[node_id])
        output.extend(sorted(graph.facts_by_entity.get(node_id, set())))
    return list(dict.fromkeys(output))

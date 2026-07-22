# QMSxE-RAG 核心消融结论

## 1. GraphRAG token 为什么低于 BM25

统一基准表中的 `Observed tokens` 不是所有方法的端到端总 token。BM25 的
845.17 tokens 是它的 Top-5 上下文加共享答案生成；GraphRAG 的 669.12
tokens 主要只覆盖共享答案生成，它历史建索引和图查询阶段的 LLM/embedding
token 无法从已有 trace 恢复。GraphRAG 返回给生成器的上下文更短，所以当前
可观测 prompt token（632.72）低于 BM25（807.89）是可能的，但不能推出
GraphRAG 的真实总 token 更少。

报告已经把列名从 `Total tokens` 改为 `Observed tokens`，并为 GraphRAG 标为
`generation_measured_index_unavailable`。

## 2. 实验设计

- 固定 HotpotQA 500 条哈希划分：362 train、75 validation、63 test；
- seeds：13、42、73；
- Stage A 组件消融全部独立重训 3 epochs；
- 内在轨道直接对 Fact 做谱排序，不使用 BM25、图重排或神经 reranker；
- 运行时轨道固定 Full Stage A 与已微调 reranker，只切换无需重训的组件；
- 运行时每个条件都清空 query embedding/reranker 缓存并 CUDA synchronize；
- Recall、Hit（检索 accuracy）、Complete 均保存 K=1/2/5/10/20/30/40；
- 报告给出 mean±sample SD、配对 bootstrap 95% CI 和随机化检验。

## 3. 必须重训的 Stage A 消融

下表是 reranker 前的内在 Fact 排序，主指标为 Recall@20。

| 配置 | R@20 mean±SD | 相对 Full | p |
|---|---:|---:|---:|
| Full | 0.7693±0.0105 | — | — |
| Raw only | 0.7601±0.0204 | -0.0093 | 0.3368 |
| No Low | 0.7640±0.0196 | -0.0053 | 0.7255 |
| No Mid | 0.7587±0.0150 | -0.0106 | 0.2812 |
| No High | 0.7640±0.0196 | -0.0053 | 0.7259 |
| Fixed Gate | 0.7415±0.0046 | -0.0278 | 0.0559 |
| No Role Gate | 0.7640±0.0139 | -0.0053 | 0.6940 |
| No Semantic Graph | 0.7640±0.0139 | -0.0053 | 0.6947 |
| No Bridge Loss | 0.7515±0.0053 | -0.0178 | 0.3525 |
| No Hard Negatives | 0.7287±0.0189 | -0.0406 | 0.0002 |

结论：困难负例是当前唯一通过显著性检验的训练组件；动态 Query Gate 有较大且
接近显著的贡献。Mid 的点估计贡献高于 Low/High，但三种单频带删除、Role
Gate、Semantic Graph 和 Bridge Loss 在当前 63 条测试题上均不足以拒绝零效应。
这不等于这些组件无效，只说明当前样本量和 Hotpot 图构造还不能形成强证据。

## 4. Entity / Fact / Hyperedge 结构控制

同一内在谱排序轨道的 Full 模型结果为：

| 结构 | R@20 mean±SD | MRR mean±SD | ms/query |
|---|---:|---:|---:|
| Entity-Relation | 0.4372±0.0306 | 0.1639±0.0023 | 2.03 |
| Reified-Fact | 0.6142±0.0375 | 0.3817±0.0448 | 2.10 |
| Evidence-Hypergraph | 0.7693±0.0105 | 0.4133±0.0095 | 9.42 |

因此在神经 reranker 和 BM25 介入前，Hyperedge 表达的内在召回最高；此前最终
系统中普通图更高，主要发生在候选融合、Entity→Fact 展开和 reranker 阶段，
不能解释为普通图的 Stage A 谱表示本身更强。

## 5. 无需重训的运行时消融

| 结构 | 配置 | Fact R@20 | MRR | s/query |
|---|---|---:|---:|---:|
| Entity-Relation | Full | 0.9453 | 0.8846 | 0.2488 |
| Entity-Relation | Dense only | 0.9272 | 0.8732 | 0.1619 |
| Entity-Relation | No BM25 | 0.9427 | 0.8846 | 0.2414 |
| Entity-Relation | No Graph Rerank | 0.9453 | 0.8846 | 0.2386 |
| Reified-Fact | Full | 0.9405 | 0.8846 | 0.2259 |
| Reified-Fact | Dense only | 0.9087 | 0.8732 | 0.1533 |
| Reified-Fact | No BM25 | 0.9422 | 0.8899 | 0.2324 |
| Reified-Fact | No Graph Rerank | 0.9405 | 0.8846 | 0.2244 |
| Evidence-Hypergraph | Full | 0.8193 | 0.8714 | 0.1359 |
| Evidence-Hypergraph | Dense only | 0.6770 | 0.8663 | 0.0749 |
| Evidence-Hypergraph | No BM25 | 0.5957 | 0.6861 | 0.0990 |
| Evidence-Hypergraph | No Graph Rerank | 0.8193 | 0.8714 | 0.1323 |

超图对多源融合和 BM25 明显依赖：Dense only 的 R@20 下降 0.1423，No BM25
下降 0.2236，二者 p=0.0001。Reified-Fact 的 Dense only 下降 0.0317，p=0.0265；
普通图 No BM25 的变化不显著。No Graph Rerank 在三个结构中都完全不改变排序，
说明当前实现的结构重排贡献被后续 reranker 覆盖，或实际没有改变最终候选顺序。

普通图索引策略中，Entity-Relation 的 Hybrid 点估计最好；Reified-Fact 的
Single 在 R@20/MRR 上略高，但相对 Hybrid 不显著。因此不能宣称一种索引策略
稳定胜出，Hybrid 仍是覆盖率与鲁棒性更稳妥的默认值。

## 6. 边界

本轮消融关闭答案生成，以隔离检索组件，所以不重复宣称 Answer EM/F1、citation
或 token 成本变化；这些指标仍由统一端到端 benchmark 报告。本轮的 accuracy
指 Hit@K 检索命中率，不等同于生成答案准确率。

# Graph Memory

A neurobiologically inspired episodic and semantic graph memory engine and evaluation harness for long-term conversational agents.

## Dialogue & Ingestion

**Turn**:
A single timestamped utterance from either the user or the assistant within a conversation.
_Avoid_: Message, utterance, chunk

**Conversation**:
A chronologically ordered sequence of turns spanning one or more dialogue sessions.
_Avoid_: Dialog, chat history, transcript

**Session**:
A contiguous block of interaction within a conversation separated from other sessions by an elapsed time boundary.
_Avoid_: Episode, interaction

## Memory & Representation

**Memory Record**:
An extracted unit of information—either an atomic proposition or an OpenIE relation triple—anchored to its source turn and temporal metadata.
_Avoid_: Memory item, passage, factoid, entry

**Proposition**:
A self-contained natural language factual statement extracted from dialogue.
_Avoid_: Fact, sentence, claim, assertion

**Relation Triple**:
A structured assertion in `(subject, relation, object)` format extracted via OpenIE.
_Avoid_: Edge, link, RDF statement, fact

**Bi-Temporal Event Graph (BTEG)**:
A graph memory representation that distinguishes valid time (when an event occurred in the real world) from transaction time (when the dialogue turn was observed).
_Avoid_: Temporal KG, time graph

**Concept Graph (Canonical)**:
An entity-and-synonym graph memory representation ($G = (N, E \cup E')$) implementing the strict, unadulterated HippoRAG 1 specification (NeurIPS 2024) without stop-node filtering or structural intervention, traversed via Personalized PageRank with linear inverse occurrence specificity ($s_i = 1 / |P_i|$).
_Avoid_: Pure HippoRAG, unpruned KG, raw entity graph

**Concept Graph (Calibrated)**:
A conversational adaptation of the Concept Graph that incorporates data-driven occurrence filtering ($|P_v| / |T| > \theta$) to suppress structural dialogue super-hubs before traversal via Personalized PageRank.
_Avoid_: Tampered HippoRAG, hacked KG, stopword graph

**Flattened Graph**:
A zero-edge ablation of a Concept Graph ($E = \emptyset, E' = \emptyset$) that preserves entity-to-turn occurrence mapping and specificity weighting while eliminating relational traversal.
_Avoid_: Isolated nodes graph, 0-edge graph

**Rewired Graph**:
A degree-preserving randomized control of a Concept Graph that applies double-edge swaps to randomize relational connectivity while holding exact node degrees, density, and degree inequality invariant.
_Avoid_: Randomized graph, shuffled KG

**Bipartite Concept Graph**:
A heterogeneous graph memory representation that integrates dialogue turns and extracted entities as co-equal graph vertices, linking turns to their constituent entities via occurrence edges and entities to entities via relation triples and synonymy bridges, traversed via joint bi-directional Personalized PageRank.
_Avoid_: HippoRAG 2 graph, passage-entity graph, two-tier graph

**Dual-Level Graph**:
A hierarchical graph memory representation that stratifies memory into low-level entity-relation triples and high-level thematic concept nodes derived from semantic clustering of subgraphs, supporting dual-level query routing.
_Avoid_: LightRAG graph, hierarchical KG, community graph

**Multi-Graph Agentic Memory (MAGMA)**:
A conversational agent memory representation that stratifies dialogue events across four typed, orthogonal relational layers (Temporal, Causal, Entity, and Semantic) over segmented event nodes, supporting multi-view associative traversal.
_Avoid_: Flat multi-graph, loose relation graph

**EDU Heterogeneous Graph (EMem-G)**:
A conversational memory representation that segments dialogue turns into elementary discourse units (EDUs) and extracted arguments (entities, temporal expressions, events) as distinct heterogeneous vertices, traversed via Personalized PageRank (or ablated via flat EDU retrieval in EMem).
_Avoid_: Raw turn graph, homogeneous EDU graph

**Dynamic Zettelkasten Graph (A-MEM)**:
An agentic memory representation that models conversational experiences as structured memory note vertices, linking notes via LLM-evolved associative connections, semantic tags, and context descriptions.
_Avoid_: Static note list, unlinked memo store

**Topological Sidecar**:
A read-only diagnostic observer that extracts and projects persisted graph memory representations from upstream agent runtimes into simple undirected graphs, computing non-parametric topological metrics without modifying upstream indexing or retrieval logic.
_Avoid_: Intrusive graph wrapper, modified memory engine

## Retrieval & Evaluation

**Provenance**:
The minimal subset of source dialogue turns containing the ground truth evidence required to answer a question.
_Avoid_: Ground truth turns, gold evidence, context

**Flat Memory**:
A non-graph control baseline that embeds and retrieves over atomic propositions without entity links or topological traversal.
_Avoid_: Naive memory, raw memory, unlinked facts

**Vector Baseline**:
Dense semantic retrieval executed directly over raw dialogue turns without fact or entity extraction.
_Avoid_: Vanilla RAG, naive RAG, dense baseline

**Evidence Coverage**:
The fraction of gold ground-truth provenance turns retrieved within the top-$k$ context items ($|\text{retrieved} \cap \text{provenance}| / |\text{provenance}|$).
_Avoid_: Fractional recall, turn overlap, evidence hit rate

**Precision@k**:
The fraction of unique retrieved context turns that belong to the ground-truth provenance set ($|\text{retrieved} \cap \text{provenance}| / |\text{retrieved}|$).
_Avoid_: Context purity, turn precision, hit accuracy

## Topological Diagnostics (H2)

**Degree Gini ($G_{\text{deg}}$)**:
A non-parametric Lorenz-curve inequality metric ($G_{\text{deg}} \in [0, 1]$) computed over the graph's node degree distribution, measuring super-hub concentration where generic entities divert associative traversal away from specific evidence.
_Avoid_: Hub Gini, degree skewness, hubness score

**Max Degree Ratio**:
The ratio of the maximum node degree to the average node degree ($\max_v \deg(v) / \bar{d}$), capturing extreme single-entity hub dominance.
_Avoid_: Peak degree ratio, hub ratio, maximum connectivity ratio

**Largest Connected Component Ratio ($\rho_{\text{LCC}}$)**:
The fraction of total graph nodes contained in the largest connected component ($|V_{\text{LCC}}| / |V|$), measuring global topological reachability versus component fragmentation.
_Avoid_: LCC fraction, giant component size, connectivity score

**Bridge Edge Ratio**:
The fraction of simple undirected edges whose removal increases the number of connected components ($|E_{\text{bridges}}| / |E_{\text{simple}}|$), measuring fragile single-path topological bottlenecks.
_Avoid_: Cut-edge fraction, bottleneck ratio, bridge frequency


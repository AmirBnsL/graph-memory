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

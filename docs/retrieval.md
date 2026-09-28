# Retrieval

Four components give the Coder a focused prompt. Each is a flag in
`RetrievalConfig` (`retrieval/pipeline.py`) and a row in
`benchmarks/ablations.yaml`, so each one's effect is measured on its own.

| Component | Flag | Job | Code |
|---|---|---|---|
| Schema linking | `schema_k` | Keep the k columns most related to the question, plus join keys and bridge tables | `retrieval/schema_linking.py` |
| Value retrieval | `values` | Map question phrases to stored values ('Led Zepelin' → `Led Zeppelin`); matched columns are never pruned | `retrieval/values.py` |
| Query memory | `memory_k` | Few-shot examples from similar solved questions (Spider **train** only) | `retrieval/memory.py` |
| Semantic layer | `semantic` | Metric definitions and column descriptions from YAML | `semantic_layer/` |

## Design

**Hybrid ranking.** Columns are ranked by BM25 over a per-column document
(table and column names split into words, description, sample values) and by
embedding cosine over the same text. The two rankings are fused with reciprocal
rank fusion, `score = Σ 1/(60 + rank)`, which needs no calibration between
BM25's unbounded scores and cosine similarity.

**Pruning must not break joins.** After taking the top-k columns, every kept
table keeps its primary and foreign keys, and a table that links two kept but
otherwise unconnected tables (a many-to-many bridge like `PlaylistTrack`) is
added back. Schemas with 30 or fewer columns are never pruned.

**Retrieval is measured separately from generation.** `benchmarks/retrieval_eval.py`
parses each gold query with sqlglot, resolves aliases per scope, and checks
whether every gold column survived pruning. It needs no LLM.

**No leakage.** Query memory refuses any id from an evaluation split (Spider
dev, BIRD mini-dev, the Chinook golden set), so few-shot examples can only come
from train data.

**Storage.** Indexes are in-memory numpy arrays built per database and cached;
embeddings are cached on disk by content hash, so re-runs make no embedding
calls. At this scale (hundreds of columns per database, ~8.6k memory items) an
external vector database adds infrastructure without adding speed. pgvector
becomes worthwhile when memory grows from user feedback in the deployed service
(P4/P8), where Postgres already exists.

## Results so far (offline, no LLM)

Schema-linking recall: share of questions for which **all** gold columns
survived pruning (95% bootstrap CI), and prompt size relative to the full
schema. BM25 only unless stated.

| Benchmark | k | All gold columns kept | + value hints | Prompt size |
|---|---|---|---|---|
| Spider dev (n=992) | 5 | 77.8% (75.2–80.3) | 82.8% (80.3–85.1) | 35% |
| Spider dev | 10 | 91.9% (90.2–93.5) | 93.5% (92.0–95.1) | 59% |
| Spider dev | 20 | 97.5% (96.5–98.4) | 98.3% (97.5–99.0) | 84% |
| BIRD mini-dev (n=500) | 10 | 68.6% (64.6–72.6) | 72.0% (68.0–75.8) | 34% |
| BIRD mini-dev | 20 | 82.8% (79.4–86.0) | 85.0% (81.8–88.0) | 51% |
| BIRD mini-dev | 30 | 89.6% (86.8–92.2) | 90.2% (87.6–92.8) | 61% |
| BIRD mini-dev | 50 | 95.6% (93.8–97.4) | 95.8% (94.0–97.4) | 76% |

What this says:

- On Spider, schemas are small enough that pruning buys little: keeping 98% of
  questions whole still needs ~84% of the prompt. Schema linking is a BIRD tool.
- On BIRD, pruning to k=30 cuts the prompt to 61% but drops a needed column for
  ~1 question in 10. Whether the shorter prompt's accuracy gain beats that loss
  is exactly what the end-to-end ablation (`bird-flash-schema30`) measures.
- The offline hashing embedder adds nothing over BM25 (within CI; slightly
  worse at small k). Semantic embeddings (Gemini) are not yet measured.
- Value hints help most at small k (+5 pts on Spider at k=5); the intervals
  above are unpaired, so the paired test on the end-to-end ablation is the one
  to cite.

End-to-end EX lift per component: pending the ablation run
(`python -m benchmarks.ablate --only bird-flash-values,bird-flash-schema30,...`).

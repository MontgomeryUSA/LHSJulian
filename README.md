# Local RAG Memory Engine for Therapy Session Transcripts

A fully local, offline semantic search engine over therapy session
transcripts. No cloud APIs, no external services -- everything runs on
a single consumer laptop (Windows, macOS, or Linux).

```
"sleep problems" ──► embed query ──► FAISS search ──► SQLite hydrate ──► ranked results
                                                                │
                                                                └──► retrieve_context() expands
                                                                     each hit into a ±2-turn window
```

## Architecture

```
memory_engine/
├── config.py        EngineConfig — all paths/settings in one place
├── models.py         Segment / SearchResult / ContextWindow dataclasses
├── database.py        SQLite layer (transcript_segments, source_files)
├── ingestion.py       Phase 1/5 — scan, parse, idempotently load transcripts
├── embeddings.py       Phase 2 — sentence-transformers wrapper (Embedder protocol)
├── vector_store.py     Phase 2 — FAISS IndexIDMap wrapper
├── search.py            Phase 3/4 — search() and retrieve_context()
├── engine.py             Phase 6 — MemoryEngine facade (the only class you should use)
└── cli.py                 Phase 7 — one-shot CLI + interactive REPL
```

**Only `MemoryEngine` is a public dependency.** Everything else is an
internal collaborator, swappable independently (different embedding
model, different vector backend, etc.) without touching call sites.

## Install

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

`faiss-cpu` ships prebuilt wheels for Windows/macOS/Linux on recent
versions (>=1.8) for common Python versions, so `pip install` is
normally sufficient. If `pip` can't find a wheel for your exact Python
version on Windows, install via conda instead:
`conda install -c pytorch faiss-cpu`.

## Quick start

```bash
python -m memory_engine.cli ingest ./transcripts
python -m memory_engine.cli search "sleep problems"
python -m memory_engine.cli context "sleep problems"
python -m memory_engine.cli session session_001
```

Or interactively:

```
$ python -m memory_engine.cli
memory> search sleep problems
Results:
[1] Score: 0.94  Speaker: Patient  (13.4s-19.0s)  Session: session_001
    "I've been waking up around 3am almost every night and can't get back to sleep."
[2] Score: 0.91  Speaker: Patient  (19.3s-25.7s)  Session: session_001
    "Sleep problems began after I changed jobs a few months ago."
memory> exit
```

Or from Python:

```python
from memory_engine import MemoryEngine

with MemoryEngine() as engine:
    engine.ingest_directory("./transcripts")
    for r in engine.search("sleep problems", top_k=5):
        print(r["score"], r["speaker"], r["text"])

    for window in engine.retrieve_context("sleep problems", top_k=3):
        print(window["formatted_text"])

    session = engine.get_session("session_001")
    print(session["transcript"])
```

## Going fully offline (air-gapped deployment)

Running inference is **always** local -- `MemoryEngine.search()` /
`ingest_directory()` never make a network call. The one exception is
the **first** time a given embedding model name is used on a machine:
`sentence-transformers` will download the model weights (~130 MB for
`bge-small-en-v1.5`) from Hugging Face if they aren't already cached.

For a HIPAA-conscious, air-gapped deployment:

1. On a machine with internet access, run the engine once (e.g. `python
   -m memory_engine.cli rebuild` against an empty `data_dir`) to
   populate the local Hugging Face cache, or use
   `huggingface-cli download BAAI/bge-small-en-v1.5`.
2. Copy that cache folder onto the target laptop (or set
   `EngineConfig.model_cache_dir` to a folder you ship alongside the
   app installer).
3. Construct the engine with `local_files_only=True`:
   ```python
   from memory_engine import MemoryEngine, EngineConfig
   engine = MemoryEngine(EngineConfig(local_files_only=True, model_cache_dir="./model_cache"))
   ```
   With this set, the engine will *never* attempt a network call and
   will fail loudly instead if the weights are missing -- which is what
   you want for a verifiable, air-gapped clinical workstation.

## Data model

**`transcript_segments`** (SQLite) -- one row per transcript segment:

| column | notes |
|---|---|
| `id` | primary key; also used directly as the FAISS vector id |
| `conversation_id` | derived from the filename (e.g. `session_001.json` → `session_001`) |
| `patient_id` | auto-detected from `Patients/<id>/sessions/...` paths; `NULL` for today's flat layout |
| `session_id` | same as `conversation_id` today; kept distinct so it can diverge later |
| `file_path` | absolute path to the source file (immutable source data) |
| `segment_index` | 0-based position within the file, used to reconstruct order/context |
| `speaker`, `start_time`, `end_time`, `text` | copied as-is from the transcript JSON |
| `created_at` | ingestion timestamp |

**`source_files`** tracks a content hash per ingested file so
`ingest_directory()` is idempotent: re-running it on an unchanged
directory does no work, and editing a file causes only that file's
segments (and embeddings) to be replaced.

**FAISS** stores vectors completely separately from SQLite, as
required. The FAISS↔SQLite mapping is the identity by construction:
vectors are always added with `add_with_ids(vector, segment_id)`, so
there's no secondary id-translation table that could drift out of
sync.

### Phase 5 — migrating to `Patients/<id>/sessions/<file>.json`

No schema migration is needed. `patient_id` and `session_id` columns
already exist and are populated automatically once transcripts live
under a `Patients/<patient_id>/sessions/` directory — ingest that
directory tree as-is and the columns backfill themselves. (`get_session`
already keys off `session_id`, not file path, so it works unchanged
once that layout is the norm.)

One caveat worth knowing about: today, `conversation_id` is derived
purely from the filename stem. If two different patients ever have a
file with the same name (e.g. both have `session_001.json`), their
`conversation_id` values will collide, though their rows remain fully
distinguishable via `patient_id` + `file_path`. If this matters for
your deployment, consider namespacing `conversation_id` by patient at
ingestion time — it's a one-line change in `ingestion.py`.

## Testing

The test suite uses a deterministic `FakeEmbedder` (see
`tests/fakes.py`) so it runs in well under a second with no model
download or network access required:

```bash
python tests/test_engine.py
# or: pytest tests/test_engine.py -v
```

This exercises ingestion (including malformed files, idempotency, and
file-change re-indexing), search, context windows, session retrieval,
patient-path detection, index rebuild, and persistence across process
restarts.

## Known limitations / next steps

- **Single-writer SQLite.** Fine for a desktop app; if a future
  service has multiple writer processes, switch to a proper
  client/server DB or add file-level locking around writes.
- **FAISS `IndexFlatIP` is exact, brute-force search.** This is the
  right choice for a single clinician's transcript archive (thousands
  to low tens-of-thousands of segments) on a laptop. If volumes grow
  much larger, swap in an approximate index (e.g. `IndexHNSWFlat`)
  behind the same `FaissVectorStore` interface.
- **No PHI redaction.** This engine indexes and stores transcript text
  verbatim, as instructed. Encrypting `data_dir` at rest (e.g. via OS
  full-disk encryption) and restricting filesystem permissions on it
  is recommended for any real clinical deployment.
- **No de-identification of logs.** Logging deliberately avoids ever
  writing transcript text to logs (only counts, ids, file paths) --
  keep it that way if you extend logging elsewhere.

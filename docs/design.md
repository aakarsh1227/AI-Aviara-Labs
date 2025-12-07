# Design Doc 



## Architecture


- FastAPI application exposes endpoints for ingest, extract, ask, audit, stream, metrics, health.
- Storage is filesystem-based:
    - `data/docs`: original PDFs and extracted text per `document_id`.
    - `data/index`: TF-IDF vectorizer, normalized matrix, FAISS index, chunk texts, chunk metadata.
- Retrieval:
    - TF-IDF vectorization over chunk texts.
    - FAISS `IndexFlatIP` over L2-normalized vectors for inner-product similarity.
- LLM integration via Groq (optional) for extraction and audit; rule-based fallback.


```
Client → FastAPI → Core (extract/chunk/index) → Filesystem (docs/index)
                                     └→ Groq LLM (optional)
```


## Data Model


- Document metadata (`docs.json`):
    - `id`, `filename`, `path_pdf`, `path_txt`, `chunks_count`, `size_bytes`, `mime_type`, `created_at`, `sha256`.
- Chunk metadata (`chunk_meta.joblib` via `index_faiss`):
    - `text`, `doc_id`, `start`, `end`, `page?` (optional).
- Index state:
    - `vectorizer.joblib`, `matrix.npy` (float32, normalized), `faiss.index`.


## Chunking Rationale


- Default `CHUNK_SIZE=700`, `CHUNK_OVERLAP=100` to balance recall and precision for contract text.
- Span-aware chunks carry `start/end` offsets for accurate citations in `/ask`.
- Optional per-page mapping can be added to populate `page` for citations.


## Fallback Behavior


- `/ask`: If LLM not configured or disabled, use rule-based synthesis over top-k retrieved chunks. Citations still include spans.
- `/extract`: If Groq LLM returns invalid output or times out, fallback to regex-based extraction.
- `/audit`: If Groq LLM unavailable, fallback to regex heuristics with configurable thresholds.


## Security Notes


- PDF validation checks `%PDF` header and content-type to avoid non-PDF uploads.
- SHA256 duplicate detection to prevent re-ingesting identical documents.
- No PII exfiltration; data stored locally under `data/`.
- LLM calls constrained and not given raw PDFs; only text context is used.
- Prometheus metrics endpoint exposed; place behind auth or restrict in production.
- Consider CORS and rate limiting for public deployments.


## Deployment


- Containerized via `Dockerfile` and `docker-compose.yml`.
- Health checks: `/healthz` liveness; readiness can be extended.
- Reindex endpoint to migrate legacy data to span-aware chunks.
- PII redaction counter increments on each redaction.


##  Edge Cases Handled
- Empty corpus returns rule-based message.
- No keywords -> message signaling that.
- PDF extraction failure -> UTF-8 decode fallback.


##  Risks
- Rebuilding index on every ingestion may not scale (optimize with incremental updates later).
- Large PDFs could raise memory usage; consider streaming chunking.


(End)






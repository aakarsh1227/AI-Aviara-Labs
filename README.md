# PDF Q&A Service

Ingest PDF documents, ask questions with retrieval + (placeholder) LLM synthesis or deterministic rule-engine fallback, and audit corpus health.

## Features
- Upload multiple PDFs (`/extract`)
- Ask questions (`/ask`) with retrieval over chunked text
- Force rule engine via query param `force_rule=true` or header `X-Force-Rule: 1`
- Audit stats (`/audit`)
- Prometheus metrics (`/metrics`)
- PII redaction in logs (emails, phone numbers, SSNs)
## Tech Stack
FastAPI, FAISS (faiss-cpu) + TF-IDF (scikit-learn), Prometheus client, pdfminer.six. Local storage for PDFs and index.

## Quick Start (Docker)
```powershell
# cd into the project root
# Copy env file and adjust if needed
Copy-Item .env.example .env

# Build & start
docker compose up --build

# App available at http://localhost:8000/docs
```

## Local Dev (No Docker)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.app.main:app --reload
```
No database required. Files are stored under `data/docs` and the FAISS index under `data/index`.
### POST /extract
Multipart file upload. Saves PDFs to `data/docs`, extracted text to `.txt`, rebuilds FAISS index. Returns number of chunks per file.
Alias: `POST /ingest` (same behavior).
```powershell
Invoke-WebRequest -Uri http://localhost:8000/extract -Method POST -InFile sample.pdf -ContentType multipart/form-data
```
(PowerShell note: Use a tool like curl for simpler multi-file: `curl -F "files=@sample.pdf" http://localhost:8000/extract`).

### GET /ask
Params: `question`, optional `force_rule=true`. Header alternative: `X-Force-Rule: 1`.
```powershell
curl "http://localhost:8000/ask?question=How%20can%20I%20force%20the%20rule%20engine?"
curl "http://localhost:8000/ask?question=How%20can%20I%20force%20the%20rule%20engine?&force_rule=true"
```

### GET /audit
Corpus statistics.
```powershell
curl http://localhost:8000/audit
```

### GET /metrics
Prometheus exposition format.
```powershell
curl http://localhost:8000/metrics
```
- No LLM key configured
Scores sentences containing keywords; returns top unique lines.

```powershell
pytest -q
```
## Trade-offs & Notes
- TF-IDF + FAISS chosen for zero warm-up complexity and fast approximate similarity.
- No external DB; simpler local persistence suitable for assignment.
- Placeholder LLM step avoids external calls (safe for interview environment).

## Security
- Logs redact PII via regex filter.
- No secrets stored in repo.

## Future Improvements
- Replace TF-IDF with embeddings (sentence-transformers + pgvector).
- Add authentication & rate limiting.
- Streaming responses & better answer synthesis.

## License
MIT (adjust as needed).

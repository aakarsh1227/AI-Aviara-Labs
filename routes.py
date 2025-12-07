from fastapi import APIRouter, UploadFile, File, Query, Header
from fastapi.responses import PlainTextResponse
from ..core.extract import pdf_to_text, chunk_text_iter
from ..core.index_faiss import rebuild_index, query as query_index
from ..core.rule_engine import rule_engine_answer
from ..core.config import settings
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# Removed rapidfuzz fuzzy scoring (unused) to keep dependencies minimal.
import os, json, datetime


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DOCS_DIR = os.path.join(DATA_DIR, 'docs')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
CHUNKS_PATH = os.path.join(INDEX_DIR, 'chunks.json')
DOCS_META_PATH = os.path.join(INDEX_DIR, 'docs.json')
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


router = APIRouter()


REQ_COUNTER = Counter('api_requests_total', 'Total API requests', ['endpoint'])
LATENCY = Histogram('api_latency_seconds', 'Request latency', ['endpoint'])


logger = logging.getLogger(__name__)


@router.post('/extract')
@router.post('/ingest')
async def extract(files: list[UploadFile] = File(...)):
    REQ_COUNTER.labels(endpoint='extract').inc()
    with LATENCY.labels(endpoint='extract').time():
        # Load existing corpus and docs metadata once
        existing: list[str] = []


        docs_meta: list[dict] = []
        next_id = 1
        if os.path.exists(CHUNKS_PATH):
            try:
                with open(CHUNKS_PATH, 'r', encoding='utf-8') as rf:
                    existing = json.load(rf)
            except Exception:
                existing = []
        if os.path.exists(DOCS_META_PATH):
            try:
                with open(DOCS_META_PATH, 'r', encoding='utf-8') as df:
                    docs_meta = json.load(df)
                    if isinstance(docs_meta, list) and docs_meta:
                        next_id = max(d.get('id', 0) for d in docs_meta) + 1
            except Exception:
                docs_meta = []
        ingested = []
        document_ids: list[int] = []
        total_new = []
        for f in files:
            content_bytes = await f.read()
            text = pdf_to_text(content_bytes)
            pdf_path = os.path.join(DOCS_DIR, f.filename)
            with open(pdf_path, 'wb') as out_pdf:
                out_pdf.write(content_bytes)
            with open(pdf_path + '.txt', 'w', encoding='utf-8') as out_txt:
                out_txt.write(text)
            # Assign document id and record metadata
            doc_id = next_id
            next_id += 1
            document_ids.append(doc_id)
            count = 0
            batch = []
            for ch in chunk_text_iter(text):
                batch.append(ch)
                count += 1
                if len(batch) >= settings.CHUNK_BATCH_SIZE:
                    total_new.extend(batch)
                    batch = []
            if batch:
                total_new.extend(batch)
            created_at = datetime.datetime.utcnow().isoformat() + 'Z'
            size_bytes = len(content_bytes)
            mime_type = f.content_type or 'application/pdf'
            meta = {
                'id': doc_id,
                'filename': f.filename,
                'path_pdf': pdf_path,
                'path_txt': pdf_path + '.txt',
                'chunks_count': count,
                'size_bytes': size_bytes,
                'mime_type': mime_type,
                'created_at': created_at
            }
            docs_meta.append(meta)
            ingested.append({'document_id': doc_id})
        # Enforce MAX_CHUNKS cap to avoid pathological memory usage (log if exceeded)
        combined = existing + total_new
        if len(combined) > settings.MAX_CHUNKS:
            logger.warning('Combined chunks exceed MAX_CHUNKS; truncating tail.')
            combined = combined[:settings.MAX_CHUNKS]
        # Rebuild index once for all new content
        rebuild_index(combined)
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as wf:
            json.dump(combined, wf)
        with open(DOCS_META_PATH, 'w', encoding='utf-8') as df:
            json.dump(docs_meta, df, indent=2)
    return {'status': 'ok', 'document_ids': document_ids, 'count': len(document_ids)}


@router.get('/ask')
async def ask(
    question: str = Query(...),
    force_rule: bool = Query(False),
    x_force_rule: str | None = Header(None),
):
    REQ_COUNTER.labels(endpoint='ask').inc()
    with LATENCY.labels(endpoint='ask').time():
        use_rule = force_rule or (x_force_rule == '1')
        retrieved = query_index(question, top_k=settings.MAX_TOP_CHUNKS)
        top_context = '\n'.join([r['text'] for r in retrieved])
        llm_answer = None
        reason = ''
        if not use_rule and retrieved:
            if settings.OPENAI_API_KEY:
                # Placeholder LLM logic: concatenate top chunks
                llm_answer = f"Synthesized answer based on context:\n{top_context[:1000]}"
                reason = 'llm'
            else:
                llm_answer = None
                reason = 'llm_not_configured'
        if llm_answer is None:
            # fallback rule engine (explicit or due to missing LLM)
            corpus_chunks = [r['text'] for r in retrieved]
            rule_ans = rule_engine_answer(question, corpus_chunks)
            reason = 'rule_fallback' if reason != 'llm' else reason
            answer = rule_ans
        else:
            answer = llm_answer
        match_score = 0.0
        if retrieved:
            best = max(retrieved, key=lambda r: r['score'])
            match_score = best['score']
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved,
            'reason': reason,
            'similarity_top': match_score,
        }


@router.get('/audit')
async def audit():
    REQ_COUNTER.labels(endpoint='audit').inc()
    with LATENCY.labels(endpoint='audit').time():
        # Audit from local files and stored chunks
        doc_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
        chunk_count = 0
        lengths = []
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks)
                lengths = [len(t) for t in chunks[:500]]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        potential_issues = []
        if avg_len < 200:
            potential_issues.append('Chunks may be too small, consider increasing CHUNK_SIZE.')
        if chunk_count == 0:
            potential_issues.append('No chunks ingested.')
        return {
            'documents': len(doc_files),
            'chunks': chunk_count,
            'avg_chunk_length': avg_len,
            'issues': potential_issues
        }


@router.get('/metrics')
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode('utf-8'), media_type=CONTENT_TYPE_LATEST)






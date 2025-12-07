from fastapi import APIRouter, UploadFile, File, Query, Header
from fastapi import Body
from pydantic import BaseModel, Field
import base64, hashlib
from fastapi.responses import PlainTextResponse, StreamingResponse
from ..core.extract import pdf_to_text, chunk_text_iter, chunk_text_iter_with_spans
from ..core.index_faiss import rebuild_index, query as query_index
from ..core.rule_engine import rule_engine_answer
from ..core.config import settings
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# Removed rapidfuzz fuzzy scoring (unused) to keep dependencies minimal.
import os, json, datetime
import re
from typing import Optional
from src.app.core.extract import extract_fields as parse_fields, audit_risky_clauses, llm_extract_fields, llm_audit_risky_clauses


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


@router.post('/ingest')
async def ingest(files: list[UploadFile] = File(...)):
    REQ_COUNTER.labels(endpoint='ingest').inc()
    with LATENCY.labels(endpoint='ingest').time():
        # Load existing corpus and docs metadata once
        existing_chunks: list = []
        docs_meta: list[dict] = []
        next_id = 1
        if os.path.exists(CHUNKS_PATH):
            try:
                with open(CHUNKS_PATH, 'r', encoding='utf-8') as rf:
                    existing_chunks = json.load(rf)
            except Exception:
                existing_chunks = []
        if os.path.exists(DOCS_META_PATH):
            try:
                with open(DOCS_META_PATH, 'r', encoding='utf-8') as df:
                    docs_meta = json.load(df)
                    if isinstance(docs_meta, list) and docs_meta:
                        next_id = max(d.get('id', 0) for d in docs_meta) + 1
            except Exception:
                docs_meta = []


        # Map of existing hashes for duplicate detection
        hashes = {d.get('sha256'): d for d in docs_meta if d.get('sha256')}


        document_ids: list[int] = []
        new_chunks = []
        for f in files:
            content_bytes = await f.read()
            # Basic PDF validation by magic header and content-type
            if not content_bytes.startswith(b"%PDF"):
                return {'status': 'error', 'message': f'{f.filename} is not a PDF (missing %PDF header)'}
            if f.content_type and 'pdf' not in (f.content_type or '').lower():
                return {'status': 'error', 'message': f'{f.filename} content-type {f.content_type} not accepted'}


            sha256 = hashlib.sha256(content_bytes).hexdigest()
            if sha256 in hashes:
                # Duplicate: return existing id, skip processing
                document_ids.append(hashes[sha256]['id'])
                continue


            # Assign document id and persist
            doc_id = next_id
            next_id += 1
            document_ids.append(doc_id)


            pdf_path = os.path.join(DOCS_DIR, f'{doc_id}.pdf')
            txt_path = os.path.join(DOCS_DIR, f'{doc_id}.txt')
            with open(pdf_path, 'wb') as out_pdf:
                out_pdf.write(content_bytes)
            text = pdf_to_text(content_bytes)
            with open(txt_path, 'w', encoding='utf-8') as out_txt:
                out_txt.write(text)


            # Chunk
            count = 0
            for ch in chunk_text_iter_with_spans(text):
                ch['doc_id'] = doc_id
                new_chunks.append(ch)
                count += 1
                if count >= settings.MAX_CHUNKS:
                    break


            # Record metadata
            meta = {
                'id': doc_id,
                'filename': f.filename,
                'path_pdf': pdf_path,
                'path_txt': txt_path,
                'chunks_count': count,
                'size_bytes': len(content_bytes),
                'mime_type': f.content_type or 'application/pdf',
                'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
                'sha256': sha256,
            }
            docs_meta.append(meta)
            hashes[sha256] = meta


        # Combine and enforce cap
        combined = existing_chunks + new_chunks
        if len(combined) > settings.MAX_CHUNKS:
            logger.warning('Combined chunks exceed MAX_CHUNKS; truncating to MAX_CHUNKS.')
            combined = combined[:settings.MAX_CHUNKS]


        # Rebuild index only if new chunks added
        if new_chunks:
            rebuild_index(combined)
            with open(CHUNKS_PATH, 'w', encoding='utf-8') as wf:
                json.dump(combined, wf)
        else:
            # still ensure chunks file exists
            if not os.path.exists(CHUNKS_PATH):
                with open(CHUNKS_PATH, 'w', encoding='utf-8') as wf:
                    json.dump(existing_chunks, wf)


        # Persist docs metadata
        with open(DOCS_META_PATH, 'w', encoding='utf-8') as df:
            json.dump(docs_meta, df, indent=2)
    return {'status': 'ok', 'document_ids': document_ids, 'count': len(document_ids)}


# -------- JSON Ingest (optional, avoids multipart) --------


class IngestJsonRequest(BaseModel):
    filename: str = Field(..., description='Original filename e.g., contract.pdf')
    content_base64: str = Field(..., description='Base64-encoded file bytes')


@router.post('/ingest_json')
async def ingest_json(payload: IngestJsonRequest):
    REQ_COUNTER.labels(endpoint='ingest').inc()
    with LATENCY.labels(endpoint='ingest').time():
        # Decode
        try:
            file_bytes = base64.b64decode(payload.content_base64)
        except Exception:
            return {'status': 'error', 'message': 'invalid base64 content'}


        # Basic PDF validation by header ("%PDF")
        if not file_bytes.startswith(b"%PDF"):
            return {'status': 'error', 'message': 'only PDF files are accepted'}


        # Ensure dirs
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)


        # Load corpus and docs
        existing_chunks = []
        if os.path.exists(CHUNKS_PATH):
            try:
                with open(CHUNKS_PATH, 'r', encoding='utf-8') as cf:
                    existing_chunks = json.load(cf)
            except Exception:
                existing_chunks = []


        existing_docs = []
        if os.path.exists(DOCS_META_PATH):
            try:
                with open(DOCS_META_PATH, 'r', encoding='utf-8') as df:
                    existing_docs = json.load(df)
            except Exception:
                existing_docs = []


        next_id = (max([d.get('id', 0) for d in existing_docs]) + 1) if existing_docs else 1


        # Duplicate detection via SHA256
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        for d in existing_docs:
            if d.get('sha256') == sha256:
                return {'status': 'ok', 'document_ids': [d['id']], 'count': 1, 'message': 'duplicate detected'}


        # Persist file and text
        doc_id = next_id
        pdf_path = os.path.join(DOCS_DIR, f'{doc_id}.pdf')
        txt_path = os.path.join(DOCS_DIR, f'{doc_id}.txt')
        with open(pdf_path, 'wb') as pf:
            pf.write(file_bytes)


        from src.app.core.extract import pdf_to_text, chunk_text_iter_with_spans
        text = pdf_to_text(file_bytes)
        with open(txt_path, 'w', encoding='utf-8') as tf:
            tf.write(text)


        # Chunk and limit
        new_chunks = []
        produced = 0
        for ch in chunk_text_iter_with_spans(text):
            ch['doc_id'] = doc_id
            new_chunks.append(ch)
            produced += 1
            if produced >= settings.MAX_CHUNKS:
                break


        all_chunks = existing_chunks + new_chunks
        if len(all_chunks) > settings.MAX_CHUNKS:
            all_chunks = all_chunks[-settings.MAX_CHUNKS:]


        # Rebuild index once (state is persisted internally)
        from src.app.core.index_faiss import rebuild_index
        rebuild_index(all_chunks)


        # Append doc metadata
        meta = {
            'id': doc_id,
            'filename': payload.filename,
            'path_pdf': pdf_path,
            'path_txt': txt_path,
            'size_bytes': len(file_bytes),
            'mime_type': 'application/pdf',
            'created_at': datetime.datetime.utcnow().isoformat(),
            'chunks_count': len(new_chunks),
            'sha256': sha256,
        }
        existing_docs.append(meta)


        with open(DOCS_META_PATH, 'w', encoding='utf-8') as df:
            json.dump(existing_docs, df, ensure_ascii=False, indent=2)
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as cf:
            json.dump(all_chunks, cf, ensure_ascii=False, indent=2)


        return {'status': 'ok', 'document_ids': [doc_id], 'count': 1}


# -------- Contract Field Extraction --------


def _load_doc_text_by_id(doc_id: int) -> Optional[str]:
    if not os.path.exists(DOCS_META_PATH):
        return None
    try:
        with open(DOCS_META_PATH, 'r', encoding='utf-8') as df:
            docs = json.load(df)
    except Exception:
        return None
    if not isinstance(docs, list):
        return None
    for d in docs:
        if d.get('id') == doc_id:
            path_txt = d.get('path_txt')
            if path_txt and os.path.exists(path_txt):
                try:
                    with open(path_txt, 'r', encoding='utf-8') as tf:
                        return tf.read()
                except Exception:
                    return None
    return None




@router.get('/extract')
async def extract_get(document_id: int = Query(...)):
    REQ_COUNTER.labels(endpoint='extract').inc()
    with LATENCY.labels(endpoint='extract').time():
        text = _load_doc_text_by_id(document_id)
        if not text:
            return {'status': 'error', 'message': 'document not found or text unavailable', 'document_id': document_id}
        fields = parse_fields(text)
        return {'status': 'ok', 'document_id': document_id, **fields}


class DocumentIdRequest(BaseModel):
    document_id: int


@router.post('/extract')
async def extract_post(payload: DocumentIdRequest, use_llm: bool = Query(False)):
    REQ_COUNTER.labels(endpoint='extract').inc()
    with LATENCY.labels(endpoint='extract').time():
        document_id = payload.document_id
        text = _load_doc_text_by_id(document_id)
        if not text:
            return {'status': 'error', 'message': 'document not found or text unavailable', 'document_id': document_id}
        if use_llm:
            fields = llm_extract_fields(text)
        else:
            fields = parse_fields(text)
        return {'status': 'ok', 'document_id': document_id, **fields}


@router.post('/audit')
async def audit_post(payload: DocumentIdRequest, use_llm: bool = Query(False)):
    REQ_COUNTER.labels(endpoint='audit').inc()
    with LATENCY.labels(endpoint='audit').time():
        document_id = payload.document_id
        text = _load_doc_text_by_id(document_id)
        if not text:
            return {'status': 'error', 'message': 'document not found or text unavailable', 'document_id': document_id}
        findings = llm_audit_risky_clauses(text) if use_llm else audit_risky_clauses(text)
        return {'status': 'ok', 'document_id': document_id, 'findings': findings}


class AskRequest(BaseModel):
    question: str
    force_rule: bool = False


@router.post('/ask')
async def ask(payload: AskRequest, x_force_rule: str | None = Header(None)):
    REQ_COUNTER.labels(endpoint='ask').inc()
    with LATENCY.labels(endpoint='ask').time():
        question = payload.question
        use_rule = payload.force_rule or (x_force_rule == '1')
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
        # Simple citations: doc_id and excerpt
        citations = []
        for r in retrieved[:3]:
            citations.append({
                'document_id': r.get('doc_id'),
                'page': r.get('page'),
                'char_start': r.get('start'),
                'char_end': r.get('end'),
                'evidence': r.get('text', '')[:200]
            })
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'reason': reason,
            'similarity_top': match_score,
        }


@router.get('/audit')
async def audit_summary():
    REQ_COUNTER.labels(endpoint='audit').inc()
    with LATENCY.labels(endpoint='audit').time():
        doc_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
        chunk_count = 0
        lengths = []
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                chunk_count = len(chunks)
                for t in chunks[:500]:
                    if isinstance(t, dict):
                        lengths.append(len(t.get('text', '')))
                    elif isinstance(t, str):
                        lengths.append(len(t))
                    else:
                        lengths.append(0)
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


@router.get('/ask/stream')
async def ask_stream(question: str = Query(...)):
    REQ_COUNTER.labels(endpoint='ask_stream').inc()
    with LATENCY.labels(endpoint='ask_stream').time():
        retrieved = query_index(question, top_k=settings.MAX_TOP_CHUNKS)
        top_context = '\n'.join([r['text'] for r in retrieved])
        synthetic = f"Synthesized answer based on context: {top_context[:2000]}"
        async def event_generator():
            for i in range(0, len(synthetic), 64):
                chunk = synthetic[i:i+64]
                yield f"data: {chunk}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get('/metrics')
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode('utf-8'), media_type=CONTENT_TYPE_LATEST)


@router.get('/healthz')
async def healthz():
    """Basic health check for liveness/readiness."""
    try:
        # Minimal filesystem checks
        docs_ok = os.path.exists(DOCS_DIR)
        index_ok = os.path.exists(INDEX_DIR)
        # Lightweight index state check
        from src.app.core.index_faiss import load_state
        load_state()
        status = 'ok'
    except Exception as e:
        status = 'error'
    return {
        'status': status,
        'docs_dir': DOCS_DIR,
        'index_dir': INDEX_DIR,
        'version': 'v1',
    }


@router.post('/reindex')
async def reindex():
    """Rebuild all chunks with span-aware metadata for existing docs and rebuild the index.
    Use this to migrate legacy ingestions that lack `doc_id/start/end` in citations.
    """
    REQ_COUNTER.labels(endpoint='reindex').inc()
    with LATENCY.labels(endpoint='reindex').time():
        # Load existing docs metadata
        if not os.path.exists(DOCS_META_PATH):
            return {'status': 'error', 'message': 'no docs metadata found'}
        try:
            with open(DOCS_META_PATH, 'r', encoding='utf-8') as df:
                docs_meta = json.load(df)
        except Exception:
            return {'status': 'error', 'message': 'failed to read docs metadata'}
        if not isinstance(docs_meta, list) or not docs_meta:
            return {'status': 'error', 'message': 'no documents to reindex'}


        # Build span-aware chunks for all docs
        from src.app.core.extract import chunk_text_iter_with_spans
        all_chunks = []
        processed = 0
        for d in docs_meta:
            doc_id = d.get('id')
            path_txt = d.get('path_txt')
            if not doc_id or not path_txt or not os.path.exists(path_txt):
                continue
            try:
                with open(path_txt, 'r', encoding='utf-8') as tf:
                    text = tf.read()
            except Exception:
                continue
            for ch in chunk_text_iter_with_spans(text):
                ch['doc_id'] = doc_id
                all_chunks.append(ch)
                if len(all_chunks) >= settings.MAX_CHUNKS:
                    break
            processed += 1
            if len(all_chunks) >= settings.MAX_CHUNKS:
                break


        if not all_chunks:
            return {'status': 'error', 'message': 'no chunks produced'}


        # Persist chunks and rebuild index
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as cf:
            json.dump(all_chunks, cf, ensure_ascii=False, indent=2)
        from src.app.core.index_faiss import rebuild_index
        rebuild_index(all_chunks)


        return {'status': 'ok', 'documents_processed': processed, 'chunks_count': len(all_chunks)}




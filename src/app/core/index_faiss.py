import os
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from typing import List, Dict, Iterable


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DOCS_DIR = os.path.join(DATA_DIR, 'docs')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
VECTORIZER_PATH = os.path.join(INDEX_DIR, 'vectorizer.joblib')
MATRIX_PATH = os.path.join(INDEX_DIR, 'matrix.npy')
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, 'faiss.index')
CHUNK_MAP_PATH = os.path.join(INDEX_DIR, 'chunk_texts.joblib')
CHUNK_META_PATH = os.path.join(INDEX_DIR, 'chunk_meta.joblib')


_vectorizer: TfidfVectorizer | None = None
_matrix: np.ndarray | None = None
_index: faiss.IndexFlatIP | None = None
_chunk_texts: List[str] = []
_chunk_meta: List[Dict] = []


os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)




def save_state():
    if _vectorizer is not None:
        dump(_vectorizer, VECTORIZER_PATH)
    if _matrix is not None:
        np.save(MATRIX_PATH, _matrix)
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
    dump(_chunk_texts, CHUNK_MAP_PATH)
    dump(_chunk_meta, CHUNK_META_PATH)




def load_state():
    global _vectorizer, _matrix, _index, _chunk_texts, _chunk_meta
    if os.path.exists(VECTORIZER_PATH):
        _vectorizer = load(VECTORIZER_PATH)
    if os.path.exists(MATRIX_PATH):
        _matrix = np.load(MATRIX_PATH)
    if os.path.exists(FAISS_INDEX_PATH):
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if os.path.exists(CHUNK_MAP_PATH):
        _chunk_texts = load(CHUNK_MAP_PATH)
    if os.path.exists(CHUNK_META_PATH):
        _chunk_meta = load(CHUNK_META_PATH)




def rebuild_index(chunks: List[Dict]):
    global _vectorizer, _matrix, _index, _chunk_texts, _chunk_meta
    # Accept list of dicts {'text':..., 'doc_id':..., 'start':..., 'end':..., 'page':...}
    _chunk_meta = [c if isinstance(c, dict) else {'text': str(c)} for c in chunks]
    _chunk_texts = [m.get('text', '') for m in _chunk_meta]
    if not _chunk_texts:
        _vectorizer = None
        _matrix = None
        _index = None
        save_state()
        return
    _vectorizer = TfidfVectorizer(stop_words='english')
    mat = _vectorizer.fit_transform(_chunk_texts)
    # Convert to dense; normalize for inner product
    dense = mat.toarray().astype('float32')
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms
    _matrix = dense
    _index = faiss.IndexFlatIP(dense.shape[1])
    _index.add(dense)
    save_state()


def add_chunks(chunks_iter: Iterable[Dict]):
    """Incrementally add chunks to index without loading all into RAM at once.
    If index/vectorizer not initialized, will initialize using first batch.
    """
    global _vectorizer, _matrix, _index, _chunk_texts, _chunk_meta
    load_state()
    new_chunks = list(chunks_iter)
    if not new_chunks:
        return
    if _vectorizer is None or _index is None or _matrix is None or not _chunk_texts:
        rebuild_index(new_chunks)
        return
    combined_meta = _chunk_meta + [c if isinstance(c, dict) else {'text': str(c)} for c in new_chunks]
    combined_texts = [m.get('text', '') for m in combined_meta]
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(combined_texts)
    dense = mat.toarray().astype('float32')
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms
    _vectorizer = vec
    _matrix = dense
    _index = faiss.IndexFlatIP(dense.shape[1])
    _index.add(dense)
    _chunk_texts = combined_texts
    _chunk_meta = combined_meta
    save_state()




def query(question: str, top_k: int = 5) -> List[Dict]:
    load_state()
    if _vectorizer is None or _index is None or _matrix is None or not _chunk_texts:
        return []
    q_vec = _vectorizer.transform([question]).toarray().astype('float32')
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q_vec = q_vec / q_norm
    D, I = _index.search(q_vec, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    results = []
    for rank, (i, s) in enumerate(zip(idxs, scores)):
        if i < 0 or i >= len(_chunk_texts):
            continue
        meta = _chunk_meta[i] if i < len(_chunk_meta) else {}
        out = {'chunk_index': i, 'text': _chunk_texts[i], 'score': float(s)}
        out.update({
            'doc_id': meta.get('doc_id'),
            'page': meta.get('page'),
            'start': meta.get('start'),
            'end': meta.get('end'),
        })
        results.append(out)
    return results








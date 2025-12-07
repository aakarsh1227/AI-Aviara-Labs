from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.orm import Session
from ..db.models import Chunk
from .config import settings


_vectorizer: TfidfVectorizer | None = None
_matrix = None
_chunk_cache: List[Tuple[int, str]] = []  # (chunk_id, text)




def rebuild_index(db: Session):
    global _vectorizer, _matrix, _chunk_cache
    chunks = db.query(Chunk).order_by(Chunk.id).all()
    texts = [c.text for c in chunks]
    _chunk_cache = [(c.id, c.text) for c in chunks]
    if not texts:
        _vectorizer = None
        _matrix = None
        return
    _vectorizer = TfidfVectorizer(stop_words='english')
    _matrix = _vectorizer.fit_transform(texts)




def query_index(question: str, top_k: int | None = None):
    if top_k is None:
        top_k = settings.MAX_TOP_CHUNKS
    if _vectorizer is None or _matrix is None or not _chunk_cache:
        return []
    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _matrix)[0]
    order = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in order:
        chunk_id, text = _chunk_cache[idx]
        results.append({
            'chunk_id': chunk_id,
            'text': text,
            'score': float(sims[idx])
        })
    return results






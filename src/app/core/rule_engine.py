from collections import Counter
import re


KEYWORD_PATTERN = re.compile(r'[A-Za-z]{4,}')


def rule_engine_answer(question: str, corpus_chunks: list[str], max_sentences: int = 5):
    """Simple heuristic rule engine: keyword overlap + frequency scoring.
    Returns top sentences containing most frequent question keywords.
    """
    keywords = KEYWORD_PATTERN.findall(question.lower())
    if not keywords:
        return 'No actionable keywords found.'
    freq = Counter(keywords)
    scored = []
    for chunk in corpus_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        for s in sentences:
            s_lower = s.lower()
            score = sum(freq[w] for w in keywords if w in s_lower)
            if score > 0:
                scored.append((score, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    unique = []
    seen = set()
    for score, sent in scored:
        if sent not in seen:
            seen.add(sent)
            unique.append(sent)
        if len(unique) >= max_sentences:
            break
    if not unique:
        return 'No rule-based match.'
    return '\n'.join(unique)






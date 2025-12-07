import json, requests, os, statistics
from rapidfuzz import fuzz


BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')


with open(os.path.join(os.path.dirname(__file__), 'questions.json'), 'r', encoding='utf-8') as f:
    items = json.load(f)


scores = []
results = []
for item in items:
    q = item['question']
    r = requests.get(f"{BASE_URL}/ask", params={'question': q})
    data = r.json()
    answer = data.get('answer','')
    expected_keywords = item.get('expected_keywords', [])
    keyword_hits = sum(1 for k in expected_keywords if k.lower() in answer.lower())
    fuzzy = fuzz.partial_ratio(' '.join(expected_keywords), answer)
    composite = (keyword_hits / max(1,len(expected_keywords))) * 0.5 + (fuzzy/100) * 0.5
    scores.append(composite)
    results.append({"question": q, "answer": answer, "score": composite})


summary = {
  'avg_score': statistics.mean(scores) if scores else 0.0,
  'details': results
}
print(json.dumps(summary, indent=2))





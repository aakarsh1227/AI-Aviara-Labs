from fastapi.testclient import TestClient
from src.app.main import app


client = TestClient(app)


def setup_module():
    # Ensure at least one document exists
    fake_pdf = b"Rule engine can be forced via query param force_rule or header X-Force-Rule. This service handles PDF extraction."
    client.post('/extract', files={'files': ('sample.pdf', fake_pdf, 'application/pdf')})


def test_ask_default():
    resp = client.get('/ask', params={'question': 'How can I force the rule engine?'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'answer' in data
    assert data['reason'] in ['rule_fallback','llm_not_configured']  # LLM not configured




def test_ask_forced_rule():
    resp = client.get('/ask', params={'question': 'How can I force the rule engine?', 'force_rule': True})
    assert resp.status_code == 200
    data = resp.json()
    assert data['reason'] == 'rule_fallback'
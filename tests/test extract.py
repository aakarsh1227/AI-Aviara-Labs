from fastapi.testclient import TestClient
from src.app.main import app


client = TestClient(app)


def test_extract_and_chunks():
    fake_pdf = b"Fake PDF content about PDF Q&A extraction service."  # Will fallback to utf-8 decode
    files = {
        'files': ('sample.pdf', fake_pdf, 'application/pdf')
    }
    resp = client.post('/extract', files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'ok'
    assert data['ingested'][0]['chunks'] >= 1


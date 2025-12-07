import io
from pdfminer.high_level import extract_text
import re
from typing import List, Dict, Optional
import json


def pdf_to_text(file_bytes: bytes) -> str:
    bio = io.BytesIO(file_bytes)
    text = extract_text(bio)
    if not text or not text.strip():
        try:
            return file_bytes.decode('utf-8', 'ignore')
        except Exception:
            return text or ""
    return text
from .config import settings


def chunk_text(text: str):
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    result = []
    start = 0
    produced = 0
    while start < len(text):
        end = min(len(text), start + size)
        result.append(text[start:end])
        produced += 1
        if produced >= settings.MAX_CHUNKS:
            break
        start = max(0, end - overlap)
        if start >= len(text):
            break
    return result
def chunk_text_iter(text: str):
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    start = 0
    produced = 0
    while start < len(text):
        end = min(len(text), start + size)
        yield text[start:end]
        produced += 1
        if produced >= settings.MAX_CHUNKS:
            break
        start = max(0, end - overlap)
        if start >= len(text):
            break


def chunk_text_iter_with_spans(text: str):
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    start = 0
    produced = 0
    while start < len(text):
        end = min(len(text), start + size)
        yield {'start': start, 'end': end, 'text': text[start:end]}
        produced += 1
        if produced >= settings.MAX_CHUNKS:
            break
        start = max(0, end - overlap)
        if start >= len(text):
            break


# Contract field extraction utilities


def extract_fields(text: str) -> Dict:
    parties: List[str] = []
    def _clean_party(s: str) -> str:
        s = s.strip()
        # Remove role labels in parentheses like ("Client"), (Vendor)
        s = re.sub(r"\(\s*[‘’'\"]?(Client|Customer|Vendor|Supplier|Party)[’’'\"]?\s*\)", "", s, flags=re.IGNORECASE)
        # Trim trailing commas and whitespace
        s = re.sub(r"[\s,]+$", "", s)
        return s
    parties_patterns = [
        # Capture lazily until 'and', across newlines, then stop at comma/period/newline/end
        r"between\s+(.*?)\s+and\s+(.*?)(?:[,.\n]|$)",
        r"parties?\s*:\s*(.*?)\s*;\s*(.*?)\s*(?:[\n]|$)"
    ]
    for pat in parties_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            p1 = _clean_party(m.group(1))
            p2 = _clean_party(m.group(2))
            if p1 and p2:
                parties = [p1, p2]
                break


    eff_date: Optional[str] = None
    m = re.search(r"effective\s+date\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m:
        eff_date = m.group(1).strip()


    term: Optional[str] = None
    m = re.search(r"\bterm\b\s*[:\-]?\s*([A-Za-z0-9\s]{1,80})\b", text, re.IGNORECASE)
    if m:
        term = m.group(1).strip()


    gov_law: Optional[str] = None
    m = re.search(r"governing\s+law\s*[:\-]?\s*([A-Za-z\s]{1,80})", text, re.IGNORECASE)
    if m:
        gov_law = m.group(1).strip()


    payment_terms: Optional[str] = None
    m = re.search(r"payment\s+terms?\s*[:\-]?\s*([\s\S]{0,500})\n", text, re.IGNORECASE)
    if m:
        payment_terms = m.group(1).strip()


    termination: Optional[str] = None
    m = re.search(r"termination\s*[:\-]?\s*([\s\S]{0,500})\n", text, re.IGNORECASE)
    if m:
        termination = m.group(1).strip()


    auto_renewal: Optional[str] = None
    m = re.search(r"auto[-\s]?renew(al)?\s*[:\-]?\s*(yes|no|true|false|enabled|disabled)", text, re.IGNORECASE)
    if m:
        auto_renewal = m.group(2).strip().lower()


    confidentiality: Optional[str] = None
    m = re.search(r"confidentiality\s*[:\-]?\s*([\s\S]{0,400})\n", text, re.IGNORECASE)
    if m:
        confidentiality = m.group(1).strip()


    indemnity: Optional[str] = None
    m = re.search(r"indemnity\s*[:\-]?\s*([\s\S]{0,400})\n", text, re.IGNORECASE)
    if m:
        indemnity = m.group(1).strip()


    liability_cap: Optional[Dict[str, Optional[str]]] = None
    m = re.search(r"liability\s+cap\s*[:\-]?\s*([\$€£]?\s?\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)\s*([A-Za-z]{3})?", text, re.IGNORECASE)
    if m:
        amount = m.group(1).replace(' ', '')
        currency = m.group(2)
        liability_cap = { 'amount': amount, 'currency': currency }


    signatories: List[Dict[str, str]] = []
    for m in re.finditer(r"\b(Signatory|Signer|By):\s*([A-Za-z .'-]+)\s*,\s*(Title|Designation):\s*([A-Za-z .'-]+)", text, re.IGNORECASE):
        signatories.append({'name': m.group(2).strip(), 'title': m.group(4).strip()})


    return {
        'parties': parties,
        'effective_date': eff_date,
        'term': term,
        'governing_law': gov_law,
        'payment_terms': payment_terms,
        'termination': termination,
        'auto_renewal': auto_renewal,
        'confidentiality': confidentiality,
        'indemnity': indemnity,
        'liability_cap': liability_cap,
        'signatories': signatories,
    }


def llm_extract_fields(text: str) -> Dict:
    """Use an LLM (OpenAI) to extract contract fields.
    Falls back to regex if API key is not configured or on error.
    """
    from .config import settings
    if not settings.GROQ_API_KEY:
        return extract_fields(text)
    try:
        # Lazy import Groq SDK
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        system_prompt = (
            "You extract structured contract metadata. Return strict JSON with keys: "
            "parties (array of strings), effective_date, term, governing_law, payment_terms, "
            "termination, auto_renewal, confidentiality, indemnity, liability_cap (object with amount and currency), "
            "signatories (array of {name,title}). If unsure, use null or empty array."
        )
        user_prompt = (
            "Contract text:\n" + text[:20000] + "\n\nReturn only JSON, no prose."
        )
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        # ensure all expected keys exist
        for k in [
            'parties','effective_date','term','governing_law','payment_terms','termination',
            'auto_renewal','confidentiality','indemnity','liability_cap','signatories']:
            data.setdefault(k, None if k!='parties' and k!='signatories' else [])
        return data
    except Exception:
        return extract_fields(text)


def audit_risky_clauses(text: str) -> List[Dict]:
    findings: List[Dict] = []


    # Normalize for robust matching
    normalized = text


    def add_finding(kind: str, severity: str, match: re.Match, note: Optional[str] = None):
        # Expand evidence to sentence window for better context
        start = match.start()
        end = match.end()
        left = normalized.rfind('.', 0, start)
        right = normalized.find('.', end)
        left = 0 if left == -1 else left + 1
        right = len(normalized) if right == -1 else right + 1
        span_text = normalized[left:right].strip()
        findings.append({
            'clause': kind,
            'severity': severity,
            'evidence': span_text,
            'start': start,
            'end': end,
            'note': note
        })


    # Auto-renewal with < threshold days notice (configurable)
    from .config import settings
    for m in re.finditer(r"auto[-\s]?renew\w*[^\n]{0,300}?notice[^\n]{0,80}?(\d{1,2})\s*day", normalized, re.IGNORECASE):
        try:
            days = int(m.group(1))
            if days < settings.NOTICE_DAYS_THRESHOLD:
                add_finding('auto_renewal_short_notice', 'medium', m, note=f'Notice {days} days < {settings.NOTICE_DAYS_THRESHOLD}')
        except Exception:
            continue


    # Unlimited liability indicators
    unlimited_patterns = [
        r"unlimited\s+liability",
        r"without\s+limit\s+of\s+liability",
        r"no\s+cap\s+on\s+liability",
        r"liability\s+shall\s+not\s+be\s+limited"
    ]
    for pat in unlimited_patterns:
        for m in re.finditer(pat, normalized, re.IGNORECASE):
            add_finding('unlimited_liability', 'high', m)


    # Broad indemnity indicators (non-exhaustive)
    broad_indemnity_patterns = [
        r"indemnif\w+[^\n]{0,300}?(any|all)\s+claims",
        r"indemnif\w+[^\n]{0,300}?third\s+parties",
        r"hold\s+harmless[^\n]{0,300}?(any|all)\s+losses",
        r"defend\s+and\s+indemnif\w+"
    ]
    for pat in broad_indemnity_patterns:
        for m in re.finditer(pat, normalized, re.IGNORECASE):
            add_finding('broad_indemnity', 'medium', m)


    # Liability cap overly high or missing currency (heuristic)
    m = re.search(r"liability\s+cap[^\n]{0,150}?([\$€£]?\s?\d{1,3}(?:[,\s]?\d{3})*(?:\.\d{2})?)\s*([A-Za-z]{3})?", normalized, re.IGNORECASE)
    if not m:
        # If liability cap section missing, flag informational
        for mm in re.finditer(r"liability\s+cap", normalized, re.IGNORECASE):
            add_finding('liability_cap_unclear', 'low', mm, note='Liability cap referenced but not found')
    else:
        amount_raw = m.group(1).replace(' ', '')
        digits = re.sub(r"[^0-9]", "", amount_raw)
        try:
            val = int(digits)
            if val >= settings.LIABILITY_CAP_THRESHOLD:
                add_finding('liability_cap_high', 'medium', m, note=f'Cap ~{val} >= {settings.LIABILITY_CAP_THRESHOLD}')
        except Exception:
            pass


    return findings


def llm_audit_risky_clauses(text: str) -> List[Dict]:
    """Use an LLM to detect risky clauses and return findings with severity and evidence spans.
    Falls back to regex audit when API is unavailable or errors.
    """
    from .config import settings
    if not settings.GROQ_API_KEY:
        return audit_risky_clauses(text)
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        system_prompt = (
            "You are a contract risk auditor. Identify risky clauses: "
            "auto-renewal with short notice (<" + str(settings.NOTICE_DAYS_THRESHOLD) + " days), "
            "unlimited liability, broad indemnity. Return ONLY JSON array 'findings'. "
            "Each finding must have: clause (string), severity (high|medium|low), "
            "evidence (short quoted text), start (int), end (int), note (optional). "
            "Character offsets are based on the provided text."
        )
        user_prompt = (
            "Text to audit (first 20k chars):\n" + text[:20000] + "\n\nReturn only JSON for 'findings'."
        )
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if isinstance(data, dict) and 'findings' in data:
            findings = data['findings']
        elif isinstance(data, list):
            findings = data
        else:
            findings = []
        # basic normalization
        norm = []
        for f in findings:
            try:
                norm.append({
                    'clause': f.get('clause'),
                    'severity': f.get('severity'),
                    'evidence': f.get('evidence'),
                    'start': int(f.get('start', 0)),
                    'end': int(f.get('end', 0)),
                    'note': f.get('note'),
                })
            except Exception:
                continue
        return norm
    except Exception:
        return audit_risky_clauses(text)


import logging, re
from .config import settings
from prometheus_client import Counter


PII_REDACTIONS = Counter('pii_redactions_total','Number of PII redactions applied')


EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,2}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')


class PiiRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        original = record.getMessage()
        redacted = EMAIL_PATTERN.sub('[REDACTED_EMAIL]', original)
        redacted = PHONE_PATTERN.sub('[REDACTED_PHONE]', redacted)
        redacted = SSN_PATTERN.sub('[REDACTED_SSN]', redacted)
        if redacted != original:
            record.msg = redacted
            PII_REDACTIONS.inc()
        return True


def configure_logging():
    handler = logging.StreamHandler()
    handler.addFilter(PiiRedactionFilter())
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    root.addHandler(handler)
    return root




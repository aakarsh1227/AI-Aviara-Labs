import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    # Tuned defaults for better context: ~700 chars with ~14% overlap
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '700'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '100'))
    # Audit tuning thresholds
    NOTICE_DAYS_THRESHOLD: int = int(os.getenv('NOTICE_DAYS_THRESHOLD', '30'))
    LIABILITY_CAP_THRESHOLD: int = int(os.getenv('LIABILITY_CAP_THRESHOLD', '1000000'))
    # LLM provider (Groq)
    GROQ_API_KEY: str | None = os.getenv('GROQ_API_KEY')
    GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')
    OPENAI_API_KEY: str | None = os.getenv('OPENAI_API_KEY')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    MAX_TOP_CHUNKS: int = int(os.getenv('MAX_TOP_CHUNKS', '5'))
    RULE_ENGINE_FORCE_HEADER: str = 'X-Force-Rule'
    RULE_ENGINE_QUERY_PARAM: str = 'force_rule'
    MAX_RAW_CHARS: int = int(os.getenv('MAX_RAW_CHARS', '2000000'))  # 2M characters (~2MB)
    MAX_CHUNKS: int = int(os.getenv('MAX_CHUNKS', '20000'))  # safety cap
    CHUNK_BATCH_SIZE: int = int(os.getenv('CHUNK_BATCH_SIZE', '1000'))


settings = Settings()




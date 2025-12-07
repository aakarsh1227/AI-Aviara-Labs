from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base
from ..core.config import settings
import os, logging


engine = create_engine(settings.DATABASE_URL, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))


logger = logging.getLogger(__name__)


_applied = False


def apply_migrations_once():
    global _applied
    if _applied:
        return
    migration_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'migrations', '001_init.sql')
    if os.path.exists(migration_path):
        with engine.connect() as conn, open(migration_path, 'r', encoding='utf-8') as f:
            sql = f.read()
            for statement in sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        logger.info('Migrations applied')
    _applied = True




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()






"""One-shot schema creation, run as a Kubernetes Job before the Deployment rolls out so that
replicas never race to CREATE TABLE.   python -m app.db.init
NOTE: this is create-if-missing, not a migration tool. Schema changes need Alembic migrations."""

from app.core.config import get_settings
from app.db.models import make_session_factory

if __name__ == "__main__":
    make_session_factory(get_settings().database_url)
    print("schema ready")

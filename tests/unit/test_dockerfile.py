"""Guards the container's non-root + writable-database setup (a plain `docker run` once crashed on startup
because the default SQLite file lived in a root-owned directory)."""

from pathlib import Path

DOCKERFILE = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text()


def test_runs_as_non_root_user():
    assert "USER 10001" in DOCKERFILE


def test_default_database_is_in_a_directory_the_app_user_owns():
    assert "ENV DATABASE_URL=sqlite:////data/" in DOCKERFILE
    assert "mkdir -p /data" in DOCKERFILE and "chown 10001:10001 /data" in DOCKERFILE


def test_data_dir_is_prepared_before_dropping_privileges():
    assert DOCKERFILE.index("chown 10001:10001 /data") < DOCKERFILE.index("USER 10001")

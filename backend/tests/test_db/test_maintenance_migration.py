import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command
from app.config import settings


def test_maintenance_upgrade_downgrade_preserves_original_source_schema(
    tmp_path, monkeypatch
):
    backend = Path(__file__).resolve().parents[2]
    database = tmp_path / "maintenance.db"
    monkeypatch.setattr(settings, "database_url", "sqlite:///" + database.as_posix())
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    command.upgrade(config, "b72c1a")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO papers(id,title,type,status,authors,created_at,updated_at) "
            "VALUES('p','Preserved','research_paper','active','[]','2026-11-03','2026-11-03')"
        )
    command.upgrade(config, "c73b7a")
    with sqlite3.connect(database) as db:
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "embedding_memos",
            "source_deletion_jobs",
            "source_sync_states",
        } <= tables
        assert db.execute("SELECT title FROM papers WHERE id='p'").fetchone() == (
            "Preserved",
        )
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    command.downgrade(config, "b72c1a")
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT title FROM papers WHERE id='p'").fetchone() == (
            "Preserved",
        )
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

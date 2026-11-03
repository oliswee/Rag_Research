"""The source schema upgrade must preserve legacy IDs and evidence links."""

import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command
from app.config import settings


def test_upgrade_keeps_legacy_evidence_and_rebuilds_keyword_index(
    tmp_path, monkeypatch
):
    backend = Path(__file__).resolve().parents[2]
    database = tmp_path / "legacy.db"
    monkeypatch.setattr(settings, "database_url", "sqlite:///" + database.as_posix())
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    command.upgrade(config, "8c21f0")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO papers(id,title,type,status,authors,created_at,updated_at) "
            "VALUES('paper','Legacy','research_paper','active','[]',"
            "'2026-10-23','2026-10-23')"
        )
        db.execute(
            "INSERT INTO paper_chunks(id,paper_id,chunk_index,text,content_scope,"
            "checksum) VALUES('child','paper',0,'Legacy original evidence.',"
            "'full_text','checksum')"
        )
        db.execute(
            "INSERT INTO patterns(id,paper_id,name,claim,status,created_at,updated_at) "
            "VALUES('pattern','paper','Finding','Claim','approved',"
            "'2026-10-23','2026-10-23')"
        )
        db.execute(
            "INSERT INTO pattern_evidence(pattern_id,chunk_id) "
            "VALUES('pattern','child')"
        )
    command.upgrade(config, "head")
    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_xinfo(paper_chunks)")}
        assert "index_version" in columns, "Versioned source migration is missing"
        row = db.execute(
            "SELECT id,text,search_text,index_version,document_id,parent_id "
            "FROM paper_chunks"
        ).fetchone()
        assert row == (
            "child",
            "Legacy original evidence.",
            "Legacy original evidence.",
            "legacy",
            None,
            None,
        )
        assert db.execute("SELECT * FROM pattern_evidence").fetchall() == [
            ("pattern", "child")
        ]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute(
            "SELECT text FROM paper_chunks_fts WHERE paper_chunks_fts MATCH 'legacy'"
        ).fetchall() == [("Legacy original evidence.",)]

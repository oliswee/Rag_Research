"""Frozen SQLite FTS schema v2 shared by migration and fresh test databases.

Future schema changes must add a new version instead of changing this contract.
"""


def active_chunk_sql(alias: str = "c") -> str:
    return (
        f"(EXISTS (SELECT 1 FROM source_index_generations g "
        f"WHERE g.index_version={alias}.index_version AND g.is_active=1) "
        f"OR ({alias}.index_version='legacy' AND NOT EXISTS "
        f"(SELECT 1 FROM source_index_generations g "
        f"WHERE g.paper_id={alias}.paper_id AND g.is_active=1)))"
    )


CREATE_STATEMENTS = (
    "CREATE VIEW IF NOT EXISTS source_search_content AS "
    "SELECT c.rowid AS rowid,c.search_text AS text FROM paper_chunks c WHERE "
    + active_chunk_sql(),
    "CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunks_fts "
    "USING fts5(text, content='source_search_content', content_rowid='rowid')",
    "CREATE TRIGGER IF NOT EXISTS paper_chunks_ai AFTER INSERT ON paper_chunks WHEN "
    + active_chunk_sql("new")
    + " BEGIN INSERT INTO paper_chunks_fts(rowid,text) "
    "VALUES(new.rowid,new.search_text); END",
    "CREATE TRIGGER IF NOT EXISTS paper_chunks_ad AFTER DELETE ON paper_chunks WHEN "
    + active_chunk_sql("old")
    + " BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
    "VALUES('delete',old.rowid,old.search_text); END",
    "CREATE TRIGGER IF NOT EXISTS paper_chunks_au AFTER UPDATE ON paper_chunks BEGIN "
    "INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
    "SELECT 'delete',old.rowid,old.search_text WHERE " + active_chunk_sql("old") + "; "
    "INSERT INTO paper_chunks_fts(rowid,text) SELECT new.rowid,new.search_text WHERE "
    + active_chunk_sql("new")
    + "; END",
    "CREATE TRIGGER IF NOT EXISTS source_generation_ai AFTER INSERT "
    "ON source_index_generations WHEN new.is_active=1 BEGIN "
    "INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild'); END",
    "CREATE TRIGGER IF NOT EXISTS source_generation_au AFTER UPDATE OF is_active "
    "ON source_index_generations WHEN new.is_active<>old.is_active BEGIN "
    "INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild'); END",
    "CREATE TRIGGER IF NOT EXISTS source_generation_ad AFTER DELETE "
    "ON source_index_generations WHEN old.is_active=1 BEGIN "
    "INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild'); END",
)

DROP_STATEMENTS = tuple(
    f"DROP TRIGGER IF EXISTS {name}"
    for name in (
        "paper_chunks_ai",
        "paper_chunks_ad",
        "paper_chunks_au",
        "source_generation_ai",
        "source_generation_au",
        "source_generation_ad",
    )
) + (
    "DROP TABLE IF EXISTS paper_chunks_fts",
    "DROP VIEW IF EXISTS source_search_content",
)

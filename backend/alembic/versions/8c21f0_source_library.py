"""Source evidence, import progress, and persistent keyword index.

Revision ID: 8c21f0
Revises: d80764b2a617
"""

import sqlalchemy as sa

from alembic import op

revision = "8c21f0"
down_revision = "d80764b2a617"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("papers") as batch:
        batch.add_column(sa.Column("import_source", sa.String(40)))
        batch.add_column(sa.Column("imported_at", sa.DateTime()))
        batch.add_column(sa.Column("open_access", sa.Boolean()))
        batch.add_column(sa.Column("content_scope", sa.String(20)))
    op.create_table(
        "paper_identifiers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(36),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("value", sa.String(500), nullable=False),
        sa.UniqueConstraint("kind", "value", name="uq_paper_identifier"),
    )
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(36),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_scope", sa.String(20), nullable=False),
        sa.Column("section", sa.String(255)),
        sa.Column("page_start", sa.Integer()),
        sa.Column("page_end", sa.Integer()),
        sa.Column("start_char", sa.Integer()),
        sa.Column("end_char", sa.Integer()),
        sa.Column("source_url", sa.String(2000)),
        sa.Column("checksum", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.UniqueConstraint("paper_id", "chunk_index", name="uq_paper_chunk_index"),
    )
    op.create_index("idx_paper_chunks_paper_id", "paper_chunks", ["paper_id"])
    op.create_table(
        "pattern_evidence",
        sa.Column(
            "pattern_id",
            sa.String(36),
            sa.ForeignKey("patterns.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "chunk_id",
            sa.String(36),
            sa.ForeignKey("paper_chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    op.create_table(
        "import_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("completed_at", sa.DateTime()),
    )
    op.create_table(
        "import_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("import_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(40), nullable=False),
        sa.Column("provider_id", sa.String(500), nullable=False),
        sa.Column("candidate", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column(
            "paper_id", sa.String(36), sa.ForeignKey("papers.id", ondelete="SET NULL")
        ),
        sa.Column("content_scope", sa.String(20)),
        sa.Column("warning", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.UniqueConstraint(
            "job_id", "provider", "provider_id", name="uq_import_job_item"
        ),
    )
    op.create_index("idx_import_items_job_id", "import_items", ["job_id"])
    op.execute(
        "CREATE VIRTUAL TABLE paper_chunks_fts "
        "USING fts5(text, content='paper_chunks', content_rowid='rowid')"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_ai AFTER INSERT ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(rowid,text) "
        "VALUES (new.rowid,new.text); END"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_ad AFTER DELETE ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
        "VALUES ('delete',old.rowid,old.text); END"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_au AFTER UPDATE ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
        "VALUES ('delete',old.rowid,old.text); "
        "INSERT INTO paper_chunks_fts(rowid,text) "
        "VALUES (new.rowid,new.text); END"
    )


def downgrade() -> None:
    for trigger in ("paper_chunks_ai", "paper_chunks_ad", "paper_chunks_au"):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    op.execute("DROP TABLE IF EXISTS paper_chunks_fts")
    op.drop_index("idx_import_items_job_id", "import_items")
    op.drop_table("import_items")
    op.drop_table("import_jobs")
    op.drop_table("pattern_evidence")
    op.drop_index("idx_paper_chunks_paper_id", "paper_chunks")
    op.drop_table("paper_chunks")
    op.drop_table("paper_identifiers")
    with op.batch_alter_table("papers") as batch:
        for column in ("content_scope", "open_access", "imported_at", "import_source"):
            batch.drop_column(column)

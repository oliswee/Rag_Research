"""Versioned source documents, parent/child generations and active-only FTS.

Revision ID: b72c1a
Revises: 8c21f0
"""

import sqlalchemy as sa

from alembic import op
from app.db.source_fts_v2 import CREATE_STATEMENTS, DROP_STATEMENTS

revision = "b72c1a"
down_revision = "8c21f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for statement in DROP_STATEMENTS:
        op.execute(statement)
    op.create_table(
        "source_documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("paper_id", sa.String(length=36), nullable=False),
        sa.Column("document_version", sa.String(length=80), nullable=False),
        sa.Column("parser_version", sa.String(length=150), nullable=False),
        sa.Column("snapshot_digest", sa.String(length=80), nullable=False),
        sa.Column("raw_pages", sa.JSON(), nullable=False),
        sa.Column("content_scope", sa.String(length=20), nullable=False),
        sa.Column("source_origin", sa.String(length=30), nullable=False),
        sa.Column("source_url", sa.String(length=2000), nullable=True),
        sa.Column("original_pdf", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "paper_id", name="uq_source_document_paper"),
    )
    op.create_table(
        "source_index_generations",
        sa.Column("index_version", sa.String(length=80), nullable=False),
        sa.Column("paper_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("cleaner_version", sa.String(length=100), nullable=False),
        sa.Column("chunker_version", sa.String(length=100), nullable=False),
        sa.Column("embedding_model_id", sa.String(length=200), nullable=False),
        sa.Column("child_size", sa.Integer(), nullable=False),
        sa.Column("parent_size", sa.Integer(), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("vector_status", sa.String(length=20), nullable=False),
        sa.Column("physical_revision", sa.String(length=80), nullable=True),
        sa.Column("index_manifest", sa.JSON(), nullable=True),
        sa.Column("cleaning_manifest", sa.JSON(), nullable=False),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
        sa.Column("error_type", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "paper_id"],
            ["source_documents.id", "source_documents.paper_id"],
            name="fk_generation_document_paper",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("index_version"),
        sa.UniqueConstraint(
            "index_version", "document_id", name="uq_generation_document"
        ),
        sa.UniqueConstraint(
            "index_version", "paper_id", "document_id", name="uq_generation_source"
        ),
    )
    with op.batch_alter_table("source_index_generations", schema=None) as batch_op:
        batch_op.create_index(
            "uq_active_generation_paper",
            ["paper_id"],
            unique=True,
            sqlite_where=sa.text("is_active = 1"),
        )

    op.create_table(
        "source_parents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("generation_id", sa.String(length=80), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("section_id", sa.String(length=36), nullable=True),
        sa.Column("parent_index", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("raw_spans", sa.JSON(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["generation_id", "document_id"],
            [
                "source_index_generations.index_version",
                "source_index_generations.document_id",
            ],
            name="fk_parent_generation_document",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "id", "generation_id", "document_id", name="uq_parent_generation_source"
        ),
    )
    with op.batch_alter_table("source_parents", schema=None) as batch_op:
        batch_op.create_index(
            "idx_source_parents_generation", ["generation_id"], unique=False
        )

    with op.batch_alter_table("paper_chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("cleaned_text", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "search_text",
                sa.Text(),
                sa.Computed(
                    "coalesce(cleaned_text, text)",
                ),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "index_version",
                sa.String(length=80),
                server_default="legacy",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("document_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(sa.Column("parent_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("source_spans", sa.JSON(), nullable=True))
        batch_op.create_check_constraint(
            "ck_chunk_provenance",
            "(document_id IS NULL AND parent_id IS NULL AND index_version = 'legacy') "
            "OR (document_id IS NOT NULL AND parent_id IS NOT NULL "
            "AND index_version <> 'legacy')",
        )
        batch_op.drop_constraint(batch_op.f("uq_paper_chunk_index"), type_="unique")
        batch_op.create_unique_constraint(
            "uq_paper_chunk_version_index", ["paper_id", "index_version", "chunk_index"]
        )
        batch_op.create_foreign_key(
            "fk_chunk_generation_source",
            "source_index_generations",
            ["index_version", "paper_id", "document_id"],
            ["index_version", "paper_id", "document_id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_chunk_parent_source",
            "source_parents",
            ["parent_id", "index_version", "document_id"],
            ["id", "generation_id", "document_id"],
            ondelete="CASCADE",
        )

    for statement in CREATE_STATEMENTS:
        op.execute(statement)
    op.execute("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild')")


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM source_documents")):
        raise RuntimeError("Cannot downgrade while structured source documents exist")
    for statement in DROP_STATEMENTS:
        op.execute(statement)
    with op.batch_alter_table("paper_chunks", schema=None) as batch_op:
        batch_op.drop_constraint("ck_chunk_provenance", type_="check")
        batch_op.drop_constraint("fk_chunk_parent_source", type_="foreignkey")
        batch_op.drop_constraint("fk_chunk_generation_source", type_="foreignkey")
        batch_op.drop_constraint("uq_paper_chunk_version_index", type_="unique")
        batch_op.create_unique_constraint(
            batch_op.f("uq_paper_chunk_index"), ["paper_id", "chunk_index"]
        )
        batch_op.drop_column("source_spans")
        batch_op.drop_column("parent_id")
        batch_op.drop_column("document_id")
        batch_op.drop_column("index_version")
        batch_op.drop_column("search_text")
        batch_op.drop_column("cleaned_text")

    with op.batch_alter_table("source_parents", schema=None) as batch_op:
        batch_op.drop_index("idx_source_parents_generation")

    op.drop_table("source_parents")
    with op.batch_alter_table("source_index_generations", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_active_generation_paper", sqlite_where=sa.text("is_active = 1")
        )

    op.drop_table("source_index_generations")
    op.drop_table("source_documents")

    op.execute(
        "CREATE VIRTUAL TABLE paper_chunks_fts "
        "USING fts5(text, content='paper_chunks', content_rowid='rowid')"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_ai AFTER INSERT ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(rowid,text) "
        "VALUES(new.rowid,new.text); END"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_ad AFTER DELETE ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
        "VALUES('delete',old.rowid,old.text); END"
    )
    op.execute(
        "CREATE TRIGGER paper_chunks_au AFTER UPDATE ON paper_chunks "
        "BEGIN INSERT INTO paper_chunks_fts(paper_chunks_fts,rowid,text) "
        "VALUES('delete',old.rowid,old.text); "
        "INSERT INTO paper_chunks_fts(rowid,text) VALUES(new.rowid,new.text); END"
    )
    op.execute("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild')")

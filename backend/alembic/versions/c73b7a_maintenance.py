"""Embedding memo and durable scoped source maintenance metadata.

Revision ID: c73b7a
Revises: b72c1a
"""

import sqlalchemy as sa

from alembic import op

revision = "c73b7a"
down_revision = "b72c1a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "embedding_memos",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_id", sa.String(200), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("vector_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "source_deletion_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("dataset_id", sa.String(100), nullable=False),
        sa.Column("point_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_type", sa.String(100)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_source_deletion_jobs_dataset_id", "source_deletion_jobs", ["dataset_id"]
    )
    op.create_table(
        "source_sync_states",
        sa.Column(
            "paper_id",
            sa.String(36),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("content_hash", sa.String(80), nullable=False),
        sa.Column("index_version", sa.String(80), nullable=False),
        sa.Column("parser_version", sa.String(150), nullable=False),
        sa.Column("chunker_version", sa.String(100), nullable=False),
        sa.Column("embedding_model_id", sa.String(200), nullable=False),
        sa.Column("checked_at", sa.DateTime(), nullable=False),
        sa.Column("outcome", sa.String(30), nullable=False),
        sa.Column("error_type", sa.String(100)),
    )


def downgrade():
    op.drop_table("source_sync_states")
    op.drop_index(
        "ix_source_deletion_jobs_dataset_id", table_name="source_deletion_jobs"
    )
    op.drop_table("source_deletion_jobs")
    op.drop_table("embedding_memos")

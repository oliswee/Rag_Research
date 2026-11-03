"""Minimal public publication-event provenance.

Revision ID: d84b1a
Revises: c73b7a
"""

import sqlalchemy as sa

from alembic import op

revision = "d84b1a"
down_revision = "c73b7a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "publication_states",
        sa.Column(
            "paper_id",
            sa.String(36),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("checked_at", sa.DateTime()),
        sa.Column("last_attempt_at", sa.DateTime()),
        sa.Column("refresh_error", sa.String(100)),
    )


def downgrade():
    op.drop_table("publication_states")

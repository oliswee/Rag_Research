"""Minimal local run references, feedback and labelled regression guards.

Revision ID: e95c2a
Revises: d84b1a
"""

import sqlalchemy as sa

from alembic import op

revision = "e95c2a"
down_revision = "d84b1a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "answer_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("config_version", sa.String(80), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column("minimal_trace", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "answer_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(64),
            sa.ForeignKey("answer_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reason", sa.String(40), nullable=False),
        sa.Column("source", sa.JSON()),
        sa.Column("expected_paper_id", sa.String(36)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_answer_feedback_run_id", "answer_feedback", ["run_id"])
    op.create_table(
        "feedback_regressions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "feedback_id",
            sa.String(36),
            sa.ForeignKey("answer_feedback.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("fixture", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("fix_reference", sa.String(160)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade():
    op.drop_table("feedback_regressions")
    op.drop_index("ix_answer_feedback_run_id", table_name="answer_feedback")
    op.drop_table("answer_feedback")
    op.drop_table("answer_runs")

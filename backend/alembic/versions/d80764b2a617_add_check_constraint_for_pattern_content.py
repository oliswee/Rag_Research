"""Add CHECK constraint for pattern content

Revision ID: d80764b2a617
Revises: 41bbec096be1
Create Date: 2026-01-05 21:17:41.682841

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d80764b2a617"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = "41bbec096be1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add CHECK constraint ensuring at least one content field is non-null."""
    # SQLite requires batch mode to add constraints to existing tables
    with op.batch_alter_table("patterns", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "ck_patterns_has_content",
            "claim IS NOT NULL OR evidence IS NOT NULL OR context IS NOT NULL",
        )


def downgrade() -> None:
    """Remove CHECK constraint."""
    with op.batch_alter_table("patterns", schema=None) as batch_op:
        batch_op.drop_constraint("ck_patterns_has_content", type_="check")

"""allow zero in fslsm dimension checks for baseline

Revision ID: 2a31f84c6cdc
Revises: 9dd44b55cb91
Create Date: 2026-03-14 22:29:01.664789

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2a31f84c6cdc'
down_revision: Union[str, Sequence[str], None] = '9dd44b55cb91'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_DIMS = ["act_ref", "sen_int", "vis_ver", "seq_glo"]


def upgrade() -> None:
    """Relax CHECK constraints to allow 0 (for baseline profile)."""
    for dim in _DIMS:
        op.drop_constraint(f"fslsm_profiles_{dim}_check", "fslsm_profiles", type_="check")
        op.create_check_constraint(
            f"fslsm_profiles_{dim}_check",
            "fslsm_profiles",
            f"{dim} IN (-1, 0, 1)",
        )


def downgrade() -> None:
    """Restore CHECK constraints to only allow -1 and 1."""
    for dim in _DIMS:
        op.drop_constraint(f"fslsm_profiles_{dim}_check", "fslsm_profiles", type_="check")
        op.create_check_constraint(
            f"fslsm_profiles_{dim}_check",
            "fslsm_profiles",
            f"{dim} IN (-1, 1)",
        )

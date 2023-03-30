"""empty message

Revision ID: fe2267333895
Revises: d6f91ac1e960
Create Date: 2023-03-29 13:26:13.467982

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fe2267333895'
down_revision = 'd6f91ac1e960'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ccfdata',
    sa.Column('apogeeID', sa.String(), nullable=False),
    sa.Column('nvisits', sa.Integer(), nullable=True),
    sa.Column('mjd', sa.ARRAY(sa.Integer()), nullable=True),
    sa.Column('ccf', sa.ARRAY(sa.Float()), nullable=True),
    sa.PrimaryKeyConstraint('apogeeID')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ccfdata')
    # ### end Alembic commands ###

import asyncio
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context

# Import models so autogenerate can detect them
from app.models import KnowledgeBase, Document, DocumentImage, DocumentTable, ChatMessage
from app.core.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using a sync engine.

    We use a sync engine here because Alembic's context.begin_transaction()
    is a sync context manager incompatible with async connections.
    The sync engine is only used for migration runs; the app continues
    using the async engine from app.core.database.
    """
    synchronous_url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(synchronous_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

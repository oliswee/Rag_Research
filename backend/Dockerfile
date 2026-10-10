FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev dependencies in production)
RUN uv sync --frozen --no-dev

# Copy alembic configuration and migrations
COPY alembic.ini ./
COPY alembic ./alembic

# Copy application code
COPY app ./app

# Create data directory for SQLite
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Default command - run migrations then start server
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]

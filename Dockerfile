FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
COPY solver ./solver
COPY routing ./routing
COPY dynamic ./dynamic
COPY api ./api
COPY benchmarks ./benchmarks
COPY dashboard/dist ./dashboard/dist

RUN pip install --no-cache-dir -e .

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

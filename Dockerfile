# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    build-essential ninja-build git ca-certificates libgomp1 libnuma1 \
    lynx w3m elinks \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --python /usr/bin/python3.12

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --python /usr/bin/python3.12
RUN python -c "import torch, vllm; from gyokusai.normalizers import FixEncoding"

WORKDIR /work
CMD ["python"]
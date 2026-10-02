# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build — has cmake/gcc/openblas headers so `pip install dlib` can
# compile from source (dlib ships no prebuilt wheel on PyPI for any Python
# version, so this step is unavoidable and takes several minutes).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# --user installs into /root/.local so we can copy just that into the runtime
# stage without dragging the whole build toolchain along.
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — small image, only the shared libs actually needed at
# runtime (openblas for dlib's linear algebra, libgl/glib for opencv).
# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgl1 \
        libglib2.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data

WORKDIR /app
COPY . .

# Created here as a fallback; docker-compose.yml normally mounts a volume
# over /app/data so this data survives rebuilds/restarts.
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "run.py"]

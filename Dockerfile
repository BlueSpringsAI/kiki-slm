FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip curl git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install dependencies
RUN uv pip install --system -e ".[serve]"

# Default: launch the serving script
EXPOSE 8000
ENTRYPOINT ["python", "scripts/serve.py"]
CMD ["--config", "configs/serving/vllm_multi_lora.yaml"]

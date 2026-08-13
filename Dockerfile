ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS base

# Versions match the reference Space (prithivMLmods/Flux.2-Klein-Edit-Ultra-Fast), which is
# the configuration the model was actually validated against. Flux2KleinPipeline is new in
# diffusers 0.39 and pulls a transformers 5.x tokenizer stack, so these move together.
ARG DIFFUSERS_VERSION=0.39.0
ARG TRANSFORMERS_VERSION=5.14.1
ARG BUILD_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# torch is pinned to the CUDA channel FIRST so the diffusers install below sees it already
# satisfied and never resolves a PyPI build linking a different libcudart. The Space uses
# cu130; cu128 carries the same torch 2.11.0 and keeps the driver floor at 12.0 rather than
# 13.0, which would needlessly narrow the pool of hosts this can land on.
ARG TORCH_CUDA_CHANNEL=cu128
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
 && python -c "import torch; \
    __import__('sys').exit(f'torch {torch.__version__}') if torch.__version__.split('+')[-1] != '${TORCH_CUDA_CHANNEL}' else None"

RUN uv pip install \
    "diffusers==${DIFFUSERS_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    accelerate safetensors sentencepiece protobuf pillow \
    runpod requests

# Stage 2: weights baked in.
FROM base AS final

ENV HF_HOME=/opt/hf-cache
ENV KLEIN_WEIGHTS_DIR=/weights

# hf_xet's chunk parallelism and its in-flight buffers OOM-killed the model download on a
# cold GitHub runner in this repo's ComfyUI predecessor, and did the same to the sibling
# Qwen worker at 42 minutes in. Only the build downloads anything — the final stage asserts
# every weight is present and HF_HUB_OFFLINE is set below — so this is inert at run time.
ENV HF_XET_HIGH_PERFORMANCE=0
ENV HF_HUB_DISABLE_XET=1

COPY scripts/download_weights.py /usr/local/bin/download_weights.py
RUN python /usr/local/bin/download_weights.py --weights-dir "${KLEIN_WEIGHTS_DIR}" \
    && test -f "${KLEIN_WEIGHTS_DIR}/klein/model_index.json" \
    && test -d "${KLEIN_WEIGHTS_DIR}/klein/transformer" \
    && test -d "${KLEIN_WEIGHTS_DIR}/klein/text_encoder" \
    && test -f "${KLEIN_WEIGHTS_DIR}/small-decoder/config.json"

ENV HF_HUB_OFFLINE=1

RUN python -m compileall -q /opt/venv/lib/python3.12/site-packages || true

# nvidia/cuda ships forward-compat driver stubs here and the runtime prefers them over the
# host driver whenever the host is older. NVIDIA supports that only on data-center GPUs, so
# on a 4090/5090 it fails with CUDA error 804 after the container has already started.
RUN rm -rf /usr/local/cuda/compat

ARG REQUIRE_CUDA=12.0
ENV NVIDIA_REQUIRE_CUDA="cuda>=${REQUIRE_CUDA}"

ENV BUILD_VERSION=${BUILD_VERSION}

# Entrypoint and handler last, after the ~16 GB bake, so editing either reuses the cached
# weight layer and RunPod re-pulls only this tiny layer.
ADD src/start.sh test_input.json ./
RUN chmod +x /start.sh

COPY handler.py /handler.py

CMD ["/start.sh"]

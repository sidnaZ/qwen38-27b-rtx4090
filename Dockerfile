# Same stack as the README's venv install, frozen: Python 3.12 venv at /app/venv,
# vLLM 0.27.1 (torch 2.13 / cu130 / Triton 3.7.1), core patches plus the
# explicitly separated Codex/Responses compatibility layer applied, and
# verify.sh --install run at build time.
#
# The base image is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their own
# CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel with nvcc
# on first use (batch mode, CTX=long) and Triton needs a C compiler for its launchers.
# The compiled kernels and the torch.compile cache live in the /cache volume, so
# that only happens once.
#
# Use the profile-aware wrappers documented in README.md.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04@sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip==26.2.1
COPY docker/requirements.txt docker/quality-requirements.txt docker/constraints.txt docker/
RUN venv/bin/pip install -c docker/constraints.txt -r docker/requirements.txt
RUN venv/bin/pip install -c docker/constraints.txt -r docker/quality-requirements.txt

COPY . .
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    for p in patches/*.patch; do \
      echo "== $p"; patch -p1 -d "$SP" < "$p"; \
    done; \
    echo "== Codex/Responses compatibility (not core GPU patches)"; \
    for p in integrations/codex/patches/*.patch; do \
      echo "== $p"; patch -p1 -d "$SP" < "$p"; \
    done; \
    bash verify.sh --install

# HOME is a volume: torch.compile cache (~/.cache/vllm), Triton (~/.triton),
# FlashInfer JIT (~/.cache/flashinfer), HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
VOLUME ["/cache", "/app/models"]
EXPOSE 18020
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["single"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CI-only: layers a dynamo-<BACKEND>-sidecar binary onto an already-built
# backend runtime test image, so tests/serve/test_{vllm,sglang,trtllm}_sidecar.py
# can run lib/sidecar/<backend>/launch/agg.sh end-to-end. Not used for any
# shipped image — see .github/workflows/pr.yaml sidecar-*-test-image jobs.

ARG BASE_IMAGE
ARG SIDECAR_IMAGE

FROM ${SIDECAR_IMAGE} AS sidecar

FROM ${BASE_IMAGE}
ARG BACKEND
COPY --from=sidecar /usr/local/bin/dynamo-${BACKEND}-sidecar /usr/local/bin/dynamo-${BACKEND}-sidecar

# vllm-rs ships inside the vllm wheel, not on PATH — lib/sidecar/vllm/launch/agg.sh
# expects it on PATH, but lib/sidecar/vllm/deploy/agg.yaml resolves it the same
# way this does. A no-op for sglang/trtllm; each backend gets its own
# conditional RUN block below rather than a shared one, so this stays one
# Dockerfile for all three instead of three near-duplicate ones.
RUN if [ "$BACKEND" = "vllm" ]; then \
      ln -sf "$(python3 -c 'import os,vllm; print(os.path.join(os.path.dirname(vllm.__file__), "vllm-rs"))')" /usr/local/bin/vllm-rs; \
    fi

# smg-grpc-proto is TRT-LLM's optional `grpc-smg` extra; the 1.3.0rc25 runtime
# image does not bundle it. lib/sidecar/trtllm/launch/agg.sh pip-installs it at
# runtime if missing, which would otherwise be a live network fetch on every CI
# run. Pre-baking it here with the same version constraint agg.sh uses makes
# that check a no-op and keeps the test hermetic.
RUN if [ "$BACKEND" = "trtllm" ]; then \
      python3 -m pip install --no-cache-dir "smg-grpc-proto>=0.4.2"; \
    fi

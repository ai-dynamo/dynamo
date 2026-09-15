# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Layer the RIVA NIM Python client on top of a Dynamo image so the cascaded
# voice pipeline workers (ASR/TTS) can reach RIVA gRPC services. The base
# defaults to the vLLM runtime image (which carries the Dynamo Python bindings);
# override with --build-arg BASE_IMAGE=... to layer on a different Dynamo image.
ARG BASE_IMAGE=dynamo:latest-vllm-runtime
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/riva-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/riva-requirements.txt

# Bake the worker package in (on PYTHONPATH) so `python -m riva_nim.<worker>`
# runs without bind-mounting the source. A mounted checkout still overrides it.
COPY riva_nim/ /opt/riva-cascaded-pipeline/riva_nim/
ENV PYTHONPATH=/opt/riva-cascaded-pipeline

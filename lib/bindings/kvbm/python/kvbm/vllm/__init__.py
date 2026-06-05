# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM-aware launch wrappers for vLLM.

Sub-modules:

- :mod:`kvbm.vllm.prefill` — `python -m kvbm.vllm.prefill` standalone
  OpenAI-API server that auto-attaches a `PrefillRouterHandler` to its
  engine when the rendered kv-transfer-config carries a hub URL and a
  prefill role. Used as a lightweight alternative to the
  `dynamo.vllm` worker when running KVBM conditional disagg.
"""

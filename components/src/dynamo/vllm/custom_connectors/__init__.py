#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Custom vLLM KV-connector subclasses for Dynamo.

These connectors are loaded via vLLM's `kv_connector_module_path` mechanism
(see `vllm/distributed/kv_transfer/kv_connector/factory.py`). They subclass
the in-tree connectors to add Dynamo-specific instrumentation or behavior
without requiring upstream vLLM patches.

Target: vLLM 0.19.0 (pinned by Dynamo 1.1.x).
"""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perf model configs for the AIS perf model tests of the router and of replay.

The router section of an AISimulate config and the replay keywords take the
same value, so their tests write the same configs. This module needs AI
Simulate: test modules check for it with ``pytest.importorskip`` before
importing this one.
"""

from __future__ import annotations

from aisimulate_core.sdk import ForwardPassPerfModelConfig

CONFIG = {"backend": "vllm", "system": "h200_sxm", "model": "example/model"}
PREFILL_ENTRY = {**CONFIG, "worker_type": "prefill", "backend_version": "0.11.0"}
AGGREGATED_ENTRY = {**CONFIG, "worker_type": "aggregated"}


def normalized(config: dict) -> dict:
    """The config as the router's perf config holds it."""
    return ForwardPassPerfModelConfig(**config).to_dict()

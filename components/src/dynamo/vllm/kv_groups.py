# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache group predicates that work on both vLLM KV APIs.

Older vLLM gives each KV cache group one ``kv_cache_spec``. Newer vLLM gives it
per-layer ``kv_cache_specs`` (``distinct_specs`` for the whole group) and moves
retention into the group's manager policy.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec


def kv_group_specs(group: Any) -> list[Any]:
    """The distinct layer specs of one KV cache group, on either vLLM KV API."""
    spec = getattr(group, "kv_cache_spec", None)
    if spec is not None:
        return [spec]
    distinct = getattr(group, "distinct_specs", None)
    if distinct is not None:
        return list(distinct)
    return list(dict.fromkeys((getattr(group, "kv_cache_specs", None) or {}).values()))


def kv_spec_is_state(spec: Any) -> bool:
    """Recurrent (Mamba/KDA) state spec."""
    return isinstance(spec, MambaSpec) or "Mamba" in type(spec).__name__


def kv_group_is_state(group: Any) -> bool:
    """Recurrent (Mamba/KDA) state group."""
    return any(kv_spec_is_state(spec) for spec in kv_group_specs(group))

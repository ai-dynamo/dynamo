# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS+TP>1 autotuner-disable gate in llm_worker.

The gate must read effective TP from the merged ``arg_map`` (post
extra_engine_args / override_engine_args merge), not from the CLI default,
because production DGDs commonly set TP only via the JSON/YAML surfaces.
"""

from __future__ import annotations

import pytest


# Import the pure helper directly. It has no tensorrt_llm side effects of its
# own, but llm_worker.py imports tensorrt_llm at module scope, so skip cleanly
# if the stack is absent.
tensorrt_llm = pytest.importorskip("tensorrt_llm")

from dynamo.trtllm.workers.llm_worker import _apply_autotuner_gate  # noqa: E402


pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
]


def test_gate_fires_when_tp_set_via_extra_engine_args():
    """DGD pattern: CLI TP=1, extra_engine_args overrides to TP=8."""
    arg_map = {"tensor_parallel_size": 8}

    _apply_autotuner_gate(arg_map, load_format="gms")

    assert arg_map["enable_autotuner"] is False


def test_gate_fires_at_tp2():
    arg_map = {"tensor_parallel_size": 2}

    _apply_autotuner_gate(arg_map, load_format="gms")

    assert arg_map["enable_autotuner"] is False


def test_gate_does_not_fire_at_tp1():
    arg_map = {"tensor_parallel_size": 1}

    _apply_autotuner_gate(arg_map, load_format="gms")

    assert "enable_autotuner" not in arg_map


def test_gate_does_not_fire_without_gms():
    arg_map = {"tensor_parallel_size": 8}

    _apply_autotuner_gate(arg_map, load_format="auto")

    assert "enable_autotuner" not in arg_map


def test_gate_preserves_user_enable_autotuner_when_not_firing():
    arg_map = {"tensor_parallel_size": 1, "enable_autotuner": True}

    _apply_autotuner_gate(arg_map, load_format="gms")

    assert arg_map["enable_autotuner"] is True


def test_gate_defaults_to_tp1_when_missing():
    """tensor_parallel_size may be absent if neither CLI nor overrides set it."""
    arg_map: dict = {}

    _apply_autotuner_gate(arg_map, load_format="gms")

    assert "enable_autotuner" not in arg_map

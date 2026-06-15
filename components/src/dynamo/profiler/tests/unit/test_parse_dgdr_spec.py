# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the profiler ``--config`` parser (_parse_dgdr_spec)."""

import json

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

try:
    from dynamo.profiler.__main__ import _parse_dgdr_spec
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)


def test_parses_short_inline_json() -> None:
    spec = _parse_dgdr_spec(json.dumps({"model": "Qwen/Qwen3-32B"}))
    assert spec.model == "Qwen/Qwen3-32B"


def test_parses_large_inline_json_exceeding_path_limit() -> None:
    """A large inline JSON config must not be mistaken for a file path.

    Regression test for the case where ``Path(config_arg).is_file()`` raises
    ``OSError`` (ENAMETOOLONG) for an inline JSON string longer than the OS
    path limit. The profiler must fall through to JSON parsing instead of
    crashing before it ever reaches ``json.loads``.
    """
    spec_dict = {
        "model": "Qwen/Qwen3-32B",
        "overrides": {"dgd": {f"field_{i}": "x" * 64 for i in range(80)}},
    }
    config_arg = json.dumps(spec_dict)
    # Well beyond the typical 255-byte per-component path limit.
    assert len(config_arg) > 255

    spec = _parse_dgdr_spec(config_arg)
    assert spec.model == "Qwen/Qwen3-32B"


def test_invalid_inline_json_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        _parse_dgdr_spec("{not valid json")

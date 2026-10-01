# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler ``--config`` parsing in ``dynamo.profiler.__main__``."""

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
    import dynamo.profiler.__main__ as profiler_main
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)


class _DummySpec:
    @staticmethod
    def model_validate(data):
        return data


def test_parse_dgdr_spec_prefers_inline_json_before_path_probe(monkeypatch) -> None:
    """Valid inline JSON should be parsed before any filesystem probing."""
    monkeypatch.setattr(profiler_main, "DynamoGraphDeploymentRequestSpec", _DummySpec)

    def _unexpected_probe(_: object) -> bool:
        raise AssertionError("Path.is_file() should not be called for inline JSON")

    monkeypatch.setattr(profiler_main.Path, "is_file", _unexpected_probe)

    spec = {"model": "foo", "request": {"targetQps": 1}}
    parsed = profiler_main._parse_dgdr_spec(json.dumps(spec))

    assert parsed == spec


def test_parse_dgdr_spec_handles_oserror_during_path_probe(monkeypatch) -> None:
    """Path probing errors should not abort JSON parsing fallback."""
    monkeypatch.setattr(profiler_main, "DynamoGraphDeploymentRequestSpec", _DummySpec)

    def _path_too_long(_: object) -> bool:
        raise OSError(36, "File name too long")

    monkeypatch.setattr(profiler_main.Path, "is_file", _path_too_long)

    with pytest.raises(ValueError, match="neither a valid file path nor valid JSON"):
        profiler_main._parse_dgdr_spec("not-json")


def test_parse_dgdr_spec_reads_json_file(monkeypatch, tmp_path) -> None:
    """A file path still works for existing JSON config files."""
    monkeypatch.setattr(profiler_main, "DynamoGraphDeploymentRequestSpec", _DummySpec)

    spec = {"model": "foo", "request": {"targetQps": 1}}
    config_file = tmp_path / "spec.json"
    config_file.write_text(json.dumps(spec))

    parsed = profiler_main._parse_dgdr_spec(str(config_file))

    assert parsed == spec

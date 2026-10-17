# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

from benchmarks.router import common
from benchmarks.router.common import add_expected_osl, tag_requests_with_priority

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


@pytest.fixture
def common_without_synthesizer(monkeypatch):
    """Load the real module with synthesis unavailable, without changing other tests."""
    monkeypatch.setitem(sys.modules, "prefix_data_generator.synthesizer", None)
    spec = importlib.util.spec_from_file_location(
        "_router_common_without_synthesizer", common.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace_args(tmp_path):
    """Use CLI defaults and a local trace with an existing router hint."""
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "timestamp": 0,
                "input_length": 64,
                "output_length": 32,
                "extra": {"nvext": {"agent_hints": {"priority": 7}}},
            }
        )
        + "\n"
    )
    parser = argparse.ArgumentParser()
    common.add_common_args(parser)
    common.add_synthesis_args(parser)
    return parser.parse_args(["--input-dataset", str(source)])


@pytest.mark.parametrize("use_expected_osl", [False, True])
def test_trace_passthrough_without_synthesizer(
    common_without_synthesizer, trace_args, tmp_path, use_expected_osl
):
    """Import and prepare traces without optional dependencies in either passthrough mode."""
    trace_args.use_expected_osl = use_expected_osl
    source = Path(trace_args.input_dataset)
    original = source.read_text()

    requests, dataset_path = common_without_synthesizer.prepare_trace_dataset(
        trace_args, str(tmp_path), logging.getLogger(__name__)
    )

    expected = json.loads(original)
    if use_expected_osl:
        expected["extra"]["nvext"]["agent_hints"]["osl"] = 32
        assert dataset_path != str(source)
    else:
        assert dataset_path == str(source)
    assert requests == [expected]
    assert [
        json.loads(line) for line in Path(dataset_path).read_text().splitlines()
    ] == [expected]
    assert source.read_text() == original


def test_trace_synthesis_without_dependencies_raises(
    common_without_synthesizer, trace_args, tmp_path
):
    """Requesting synthesis reports the missing dependency before creating output."""
    trace_args.num_requests = 1

    with pytest.raises(
        ModuleNotFoundError,
        match="Trace dataset synthesis requires the prefix data generator dependencies",
    ):
        common_without_synthesizer.prepare_trace_dataset(
            trace_args, str(tmp_path), logging.getLogger(__name__)
        )

    assert not (tmp_path / "synthetic_trace.jsonl").exists()


def test_expected_osl_uses_mooncake_output_length_and_preserves_hints():
    """Prefer Mooncake output length without replacing existing request hints."""
    request = {
        "output_length": 32,
        "output_tokens": 64,
        "extra": {
            "metadata": "preserved",
            "nvext": {"agent_hints": {"priority": 7}},
        },
    }

    add_expected_osl(request)

    assert request["extra"] == {
        "metadata": "preserved",
        "nvext": {"agent_hints": {"priority": 7, "osl": 32}},
    }
    assert "nvext" not in request


def test_expected_osl_supports_legacy_output_tokens():
    """Use the legacy output token field when output length is absent."""
    request = {"output_tokens": 48}

    add_expected_osl(request)

    assert request["extra"]["nvext"]["agent_hints"]["osl"] == 48


def test_priority_tagging_does_not_mutate_source_request():
    """Add priority to a deep copy while preserving the source request."""
    request = {
        "extra": {
            "metadata": {"request_id": "request-1"},
            "nvext": {"agent_hints": {"osl": 32}},
        }
    }

    tagged_request = tag_requests_with_priority([request], priority=7)[0]

    assert request["extra"]["nvext"]["agent_hints"] == {"osl": 32}
    assert tagged_request["extra"] == {
        "metadata": {"request_id": "request-1"},
        "nvext": {"agent_hints": {"osl": 32, "priority": 7}},
    }
    assert tagged_request["extra"] is not request["extra"]

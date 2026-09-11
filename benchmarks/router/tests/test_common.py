# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.router.common import add_expected_osl, tag_requests_with_priority

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


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

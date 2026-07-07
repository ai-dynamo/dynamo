# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fail-fast vLLM/vLLM-Omni version compatibility guard."""

import importlib.metadata
from unittest.mock import patch

import pytest

from dynamo.vllm.omni.version_check import (
    OmniVersionMismatchError,
    _major_minor,
    check_vllm_omni_compatibility,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _fake_version(mapping):
    def _inner(name):
        return mapping[name]

    return _inner


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.23.0", (0, 23)),
        ("0.23.0rc1", (0, 23)),
        ("0.21.0rc1", (0, 21)),
        ("1.2.3.post1", (1, 2)),
        ("0", None),
        ("dev", None),
        ("a.b.c", None),
    ],
)
def test_major_minor_parsing(version, expected):
    assert _major_minor(version) == expected


def test_mismatch_raises():
    # Reproduces the 1.3.0-rc.2 image: omni 0.21.x against vLLM 0.23.0.
    with patch(
        "importlib.metadata.version",
        _fake_version({"vllm": "0.23.0", "vllm-omni": "0.21.0rc1"}),
    ):
        with pytest.raises(OmniVersionMismatchError) as exc_info:
            check_vllm_omni_compatibility()
    assert "0.23.0" in str(exc_info.value)
    assert "0.21.0rc1" in str(exc_info.value)


def test_aligned_versions_pass():
    with patch(
        "importlib.metadata.version",
        _fake_version({"vllm": "0.23.0", "vllm-omni": "0.23.0rc1"}),
    ):
        check_vllm_omni_compatibility()


def test_missing_metadata_is_skipped():
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    with patch("importlib.metadata.version", _raise):
        # Should not raise; downstream import handles the missing package.
        check_vllm_omni_compatibility()

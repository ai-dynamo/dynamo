# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``local_media_reference``.

The redirect-SSRF fix: a URL reference must be fetched through the policy-aware
client (which revalidates each redirect hop) into a temp file, so the caller
hands a downstream generator only a trusted local path — never a URL it would
fetch and redirect itself. Local references pass through unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dynamo.common.http.media_reference import local_media_reference
from dynamo.common.http.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


async def test_url_reference_is_fetched_to_a_temp_path(monkeypatch) -> None:
    seen = {}

    async def fake_fetch(url, timeout, *, policy=None):
        seen["url"] = url
        return b"PNGDATA"

    # patched on the package: local_media_reference imports it lazily as `fetch_bytes`
    monkeypatch.setattr("dynamo.common.http.fetch_bytes", fake_fetch)

    # allow_private_ips=True keeps validate_url off the network in the test.
    policy = UrlValidationPolicy(allow_http=True, allow_private_ips=True)
    url = "https://example.com/img.png"

    async with local_media_reference(url, policy) as path:
        assert path != url  # a local path, not the URL
        assert os.path.exists(path)
        assert Path(path).read_bytes() == b"PNGDATA"
        assert path.endswith(".png")  # suffix preserved for the generator
        tmp = path

    assert seen["url"] == url
    assert not os.path.exists(tmp)  # temp cleaned on exit


async def test_local_reference_passes_through(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", str(tmp_path))
    ref = tmp_path / "x.png"
    ref.write_bytes(b"x")

    async with local_media_reference(str(ref), UrlValidationPolicy.from_env()) as path:
        assert path == str(ref.resolve())  # yielded as-is, no temp file

    assert ref.exists()  # a local reference is never deleted

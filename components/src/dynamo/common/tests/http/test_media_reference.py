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

from dynamo.common.http.media_reference import MAX_MEDIA_BYTES, local_media_reference
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


async def test_url_reference_is_fetched_to_a_temp_path(monkeypatch) -> None:
    seen = {}

    async def fake_fetch(url, timeout, *, policy=None, max_bytes=None):
        seen["url"] = url
        seen["policy"] = policy
        seen["max_bytes"] = max_bytes
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
    # Without the policy, fetch_bytes takes the backend's own redirect path and
    # no hop is revalidated — the entire redirect-SSRF guard would be gone with
    # every other assertion here still passing.
    assert seen["policy"] is policy
    assert seen["max_bytes"] == MAX_MEDIA_BYTES


async def test_local_reference_passes_through(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", str(tmp_path))
    ref = tmp_path / "x.png"
    ref.write_bytes(b"x")

    async with local_media_reference(str(ref), UrlValidationPolicy.from_env()) as path:
        assert path == str(ref.resolve())  # yielded as-is, no temp file

    assert ref.exists()  # a local reference is never deleted


async def test_a_data_uri_is_rejected_not_yielded_as_a_path() -> None:
    """`data:` passes validate_url, but it is a URI, not a filesystem path.

    Yielding it unchanged hands the generator a string it will try to open as a
    file, and puts the whole inline payload — which can be megabytes — wherever
    that failure is reported.
    """
    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError, match="cannot be resolved to a local file"):
        async with local_media_reference("data:image/png;base64,iVBORw0KGgo=", policy):
            pass


async def test_data_uri_rejection_does_not_echo_the_payload() -> None:
    payload = "A" * 200_000

    with pytest.raises(UrlValidationError) as excinfo:
        async with local_media_reference(
            f"data:image/png;base64,{payload}", UrlValidationPolicy()
        ):
            pass

    assert payload not in str(excinfo.value)
    assert len(str(excinfo.value)) < 500


async def test_an_overlong_url_extension_does_not_become_a_temp_filename(
    monkeypatch,
) -> None:
    """The temp-file suffix comes from the client's URL path.

    Passed through, a 300-character extension makes mkstemp raise ENAMETOOLONG
    — an OSError the caller does not expect, carrying the server's temp
    directory back to the client.
    """

    async def fake_fetch(url, timeout, *, policy=None, max_bytes=None):
        return b"PNGDATA"

    monkeypatch.setattr("dynamo.common.http.fetch_bytes", fake_fetch)
    policy = UrlValidationPolicy(allow_http=True, allow_private_ips=True)

    async with local_media_reference(
        "https://example.com/img." + "a" * 300, policy
    ) as path:
        assert Path(path).read_bytes() == b"PNGDATA"
        assert len(Path(path).name) < 64


async def test_a_non_alphanumeric_url_extension_is_dropped(monkeypatch) -> None:
    async def fake_fetch(url, timeout, *, policy=None, max_bytes=None):
        return b"PNGDATA"

    monkeypatch.setattr("dynamo.common.http.fetch_bytes", fake_fetch)
    policy = UrlValidationPolicy(allow_http=True, allow_private_ips=True)

    async with local_media_reference(
        "https://example.com/img.%2F..%2F..%2Fetc%2Fpasswd", policy
    ) as path:
        assert Path(path).suffix == ""


async def test_the_download_cap_reaches_the_client(monkeypatch) -> None:
    """The caller's limit must be handed to fetch_bytes, which enforces it.

    Enforcement itself is collect_capped's job (test_http_facade.py); what can
    silently regress here is the limit never being passed at all.
    """
    seen = {}

    async def fake_fetch(url, timeout, *, policy=None, max_bytes=None):
        seen["max_bytes"] = max_bytes
        return b"x"

    monkeypatch.setattr("dynamo.common.http.fetch_bytes", fake_fetch)
    policy = UrlValidationPolicy(allow_http=True, allow_private_ips=True)

    async with local_media_reference(
        "https://example.com/big.png", policy, max_bytes=512
    ):
        pass

    assert seen["max_bytes"] == 512

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading file:// and data: media bytes.

These schemes exist so H.264/H.265 from a local file or data URI can reach
NVDEC. The codec-compliant images ship no software video decoder, so without
this path those schemes have no decoder at all. The security boundary is
``validate_local_path``: local reads stay refused unless DYN_MM_LOCAL_PATH is
set, and resolved paths must stay inside it.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal import media_source as media_source_module
from dynamo.common.multimodal.media_source import (
    decode_data_uri,
    describe_media_source,
    is_local_media_url,
    read_local_media_bytes,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

PAYLOAD = b"\x00\x00\x00\x18ftypisom....avc1 fake h264 bytes"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("file:///tmp/clip.mp4", True),
        ("data:video/mp4;base64,AAAA", True),
        ("http://example.com/clip.mp4", False),
        ("https://example.com/clip.mp4", False),
        ("s3://bucket/clip.mp4", False),
    ],
)
def test_is_local_media_url(url, expected):
    assert is_local_media_url(url) is expected


async def test_reads_file_url_inside_allowed_prefix(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(PAYLOAD)
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    assert await read_local_media_bytes(clip.as_uri(), policy) == PAYLOAD


async def test_file_url_refused_when_local_access_disabled(tmp_path):
    """Default policy has allowed_local_path unset -- reads must stay refused."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(PAYLOAD)

    with pytest.raises(UrlValidationError, match="not permitted"):
        await read_local_media_bytes(clip.as_uri(), UrlValidationPolicy())


async def test_file_url_cannot_escape_allowed_prefix(tmp_path):
    """Path traversal out of the sandbox must be refused."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(PAYLOAD)
    policy = UrlValidationPolicy(allowed_local_path=str(allowed))

    with pytest.raises(UrlValidationError):
        await read_local_media_bytes(outside.as_uri(), policy)


async def test_file_url_cannot_escape_via_symlink(tmp_path):
    """resolve() happens before the prefix check, so symlinks cannot escape."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(PAYLOAD)
    link = allowed / "innocent.mp4"
    link.symlink_to(outside)
    policy = UrlValidationPolicy(allowed_local_path=str(allowed))

    with pytest.raises(UrlValidationError):
        await read_local_media_bytes(link.as_uri(), policy)


async def test_reads_base64_data_uri():
    url = "data:video/mp4;base64," + base64.b64encode(PAYLOAD).decode()
    # data: carries its own bytes, so it needs no local-path permission.
    assert await read_local_media_bytes(url, UrlValidationPolicy()) == PAYLOAD


@pytest.mark.parametrize(
    "url,match",
    [
        ("data:video/mp4;base64", "missing"),  # no comma
        ("data:video/mp4,notbase64", "expected base64"),  # not base64-tagged
        ("data:video/mp4;base64,!!!not-base64!!!", "Malformed base64"),
    ],
)
async def test_malformed_data_uri_rejected(url, match):
    with pytest.raises(UrlValidationError, match=match):
        await read_local_media_bytes(url, UrlValidationPolicy())


@pytest.mark.parametrize("size", [3, 4, 5], ids=["no-padding", "two-pad", "one-pad"])
@pytest.mark.parametrize("escaped", [False, True])
def test_decode_data_uri_size_bound_is_exact(size, escaped):
    encoded = base64.b64encode(b"x" * size).decode()
    if escaped:
        encoded = "".join(f"%{ord(char):02X}" for char in encoded)
    url = "data:video/mp4;base64," + encoded
    assert decode_data_uri(url, max_bytes=size) == b"x" * size
    with pytest.raises(UrlValidationError, match="maximum allowed size"):
        decode_data_uri(url, max_bytes=size - 1)


def test_extra_padding_does_not_lower_decoded_size():
    # Extra padding is accepted only on some Python versions; the size
    # guard must reject this payload before strict decoding either way.
    url = "data:image/png;base64," + base64.b64encode(b"abc").decode() + "=="
    with pytest.raises(UrlValidationError, match="maximum allowed size"):
        decode_data_uri(url, max_bytes=2)


def test_decode_data_uri_size_bound_rejects_before_unquote(monkeypatch):
    """A payload that is entirely percent-escaped is 3x its unescaped length,
    so both the raw and unescaped forms already exceed max_bytes -- the
    outcome alone can't tell the pre-unquote guard apart from the
    post-unquote one. Replacing unquote() with a spy and asserting it is
    never called is what actually proves the guard runs first."""
    spy = MagicMock()
    monkeypatch.setattr(media_source_module, "unquote", spy)
    payload = "%41" * 10_000_000
    url = "data:video/mp4;base64," + payload
    with pytest.raises(UrlValidationError, match="maximum allowed size"):
        decode_data_uri(url, max_bytes=1024)
    spy.assert_not_called()


async def test_unsupported_scheme_rejected():
    with pytest.raises(UrlValidationError, match="Unsupported local media scheme"):
        await read_local_media_bytes("s3://bucket/clip.mp4", UrlValidationPolicy())


def test_describe_media_source_elides_a_data_uri_payload() -> None:
    """A data: URI is the whole media payload; describing one must never
    reproduce it, only its type and size."""
    payload = "A" * 100_000
    label = describe_media_source(f"data:video/mp4;base64,{payload}")

    assert payload not in label
    assert "data:video/mp4" in label
    assert "payload elided" in label
    assert len(label) < 100


def test_describe_media_source_keeps_an_ordinary_url_intact() -> None:
    url = "https://example.com/clip.mp4"
    assert describe_media_source(url) == url


def test_describe_media_source_bounds_an_overlong_url() -> None:
    url = "https://example.com/" + "a" * 500
    label = describe_media_source(url)

    assert label.startswith("https://example.com/")
    assert len(label) < len(url)
    assert str(len(url)) in label  # keeps the true size visible


def test_describe_media_source_survives_a_non_string() -> None:
    """The video loop labels whatever it was handed, including malformed
    items, before it has established the input is a string."""
    assert describe_media_source(None) == "<non-string media source>"  # type: ignore[arg-type]

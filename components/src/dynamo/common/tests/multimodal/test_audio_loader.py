# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

import dynamo.common.multimodal.audio_loader as audio_loader_module
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _permissive_http_policy() -> UrlValidationPolicy:
    """Policy that lets existing tests keep using https://example.com/... URLs.

    Private/loopback IPs and DNS checks are bypassed so tests don't depend on
    real DNS resolution of example.com.
    """
    return UrlValidationPolicy(
        allow_http=True,
        allow_private_ips=True,
    )


def test_normalize_audio_url_converts_local_paths(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    assert validate_media_url(str(audio_path), policy) == audio_path.resolve().as_uri()


def test_normalize_audio_url_preserves_data_urls():
    data_url = "data:audio/wav;base64,UklGRg=="
    policy = UrlValidationPolicy()
    assert validate_media_url(data_url, policy) == data_url


def test_normalize_audio_url_preserves_http_urls():
    url = "https://example.com/audio.wav"
    policy = _permissive_http_policy()
    assert validate_media_url(url, policy) == url


def test_normalize_audio_url_rejects_bare_path_by_default(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError, match="Local media paths are not permitted"):
        validate_media_url(str(audio_path), policy)


def test_normalize_audio_url_rejects_private_ip():
    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError):
        validate_media_url("https://169.254.169.254/audio.wav", policy)


def test_normalize_audio_url_accepts_file_uri_inside_prefix(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    file_uri = audio_path.resolve().as_uri()
    assert validate_media_url(file_uri, policy) == file_uri


def test_normalize_audio_url_rejects_file_uri_outside_prefix(tmp_path):
    allowed = tmp_path / "media"
    allowed.mkdir()
    other = tmp_path / "secret.wav"
    other.write_bytes(b"RIFF")
    policy = UrlValidationPolicy(allowed_local_path=str(allowed))

    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        validate_media_url(other.resolve().as_uri(), policy)


@pytest.mark.asyncio
async def test_load_audio_rejects_http_by_default():
    loader = AudioLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(ValueError, match="not allowed"):
        await loader.load_audio("http://example.com/x.wav")


@pytest.mark.asyncio
async def test_load_audio_blocks_redirect_to_private_ip():
    """A 302 to a blocked IP must be rejected per-hop, not only the initial URL."""
    loader = AudioLoader(url_policy=UrlValidationPolicy())

    redirect = MagicMock(spec=httpx.Response)
    redirect.status_code = 302
    redirect.is_redirect = True
    redirect.headers = {"location": "https://169.254.169.254/evil"}
    redirect.url = httpx.URL("https://8.8.8.8/a.wav")
    redirect.aclose = AsyncMock()

    client = MagicMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock(return_value=MagicMock(spec=httpx.Request))
    client.send = AsyncMock(return_value=redirect)

    with patch(
        "dynamo.common.multimodal.audio_loader.get_http_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="blocked range"):
            await loader.load_audio("https://8.8.8.8/a.wav")


@pytest.mark.asyncio
async def test_load_audio_uses_vllm_media_connector():
    loader = AudioLoader()
    loader._url_policy = UrlValidationPolicy()
    waveform = np.random.randn(16000).astype(np.float32)
    sr = 44100.0
    loader._load_audio_with_vllm = AsyncMock(  # type: ignore[method-assign]
        return_value=(waveform, sr)
    )

    loaded_waveform, loaded_sr = await loader.load_audio(
        "data:audio/wav;base64,UklGRg=="
    )

    np.testing.assert_array_equal(loaded_waveform, waveform)
    assert loaded_sr == sr


@pytest.mark.asyncio
async def test_load_audio_rejects_empty_waveform():
    loader = AudioLoader(url_policy=_permissive_http_policy())
    loader._load_audio_with_vllm = AsyncMock(  # type: ignore[method-assign]
        return_value=(np.array([], dtype=np.float32), 16000.0)
    )

    with pytest.raises(ValueError, match="empty"):
        await loader.load_audio("https://example.com/empty.wav")


@pytest.mark.asyncio
async def test_load_audio_batch_uses_url_loader():
    loader = AudioLoader()
    first = (np.zeros(8000, dtype=np.float32), 16000.0)
    second = (np.ones(8000, dtype=np.float32), 44100.0)
    loader.load_audio = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    audios = await loader.load_audio_batch(
        [
            {"Url": "https://example.com/one.wav"},
            {"Url": "https://example.com/two.wav"},
        ]
    )

    assert len(audios) == 2
    np.testing.assert_array_equal(audios[0][0], first[0])
    assert audios[0][1] == first[1]
    np.testing.assert_array_equal(audios[1][0], second[0])
    assert audios[1][1] == second[1]


@pytest.mark.asyncio
async def test_load_audio_batch_rejects_malformed_items():
    loader = AudioLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="Invalid audio multimodal item"):
        await loader.load_audio_batch([{"bad_key": "value"}])


@pytest.mark.asyncio
async def test_load_audio_batch_rejects_decoded_variant_without_frontend_decoding():
    loader = AudioLoader(enable_frontend_decoding=False)

    with pytest.raises(ValueError, match="enable_frontend_decoding=False"):
        await loader.load_audio_batch([{"Decoded": {"shape": [16000]}}])


@pytest.mark.asyncio
async def test_load_audio_batch_reads_decoded_variant(monkeypatch):
    # Construct with enable_frontend_decoding=False to skip real NIXL init,
    # then set the flags directly so the decoded path is exercised.
    loader = AudioLoader(enable_frontend_decoding=False)
    loader._enable_frontend_decoding = True
    loader._nixl_connector = object()

    decoded_item = {
        "shape": [16000],
        "metadata": {"sample_rate": 44100},
    }
    waveform = np.random.randn(16000).astype(np.float32)
    read_decoded = AsyncMock(return_value=(waveform, decoded_item["metadata"]))
    monkeypatch.setattr(
        audio_loader_module, "read_decoded_media_via_nixl", read_decoded
    )

    audios = await loader.load_audio_batch([{"Decoded": decoded_item}])

    assert len(audios) == 1
    np.testing.assert_array_equal(audios[0][0], waveform)
    assert audios[0][1] == 44100.0
    read_decoded.assert_awaited_once_with(
        loader._nixl_connector,
        decoded_item,
        return_metadata=True,
    )

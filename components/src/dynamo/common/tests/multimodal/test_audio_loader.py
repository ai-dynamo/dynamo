# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import numpy as np
import pytest

import dynamo.common.multimodal.audio_loader as audio_loader_module
from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

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


@pytest.mark.asyncio
async def test_load_audio_rejects_http_by_default():
    """Wiring smoke: AudioLoader plumbs ``url_policy`` to the validator.

    Validator behavior is covered in ``test_url_validator.py``;
    per-hop SSRF revalidation in ``http/test_http_backends.py``.
    """
    loader = AudioLoader(url_policy=UrlValidationPolicy())

    with pytest.raises(UrlValidationError, match="not allowed"):
        await loader.load_audio("http://example.com/x.wav")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.wav"),
    ],
)
async def test_load_audio_preserves_client_error(client_error):
    loader = AudioLoader()
    loader._load_audio_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=client_error
    )

    with pytest.raises(type(client_error)) as exc_info:
        await loader.load_audio("https://example.com/x.wav")

    assert exc_info.value is client_error


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
@pytest.mark.parametrize(
    "client_error",
    [
        UrlValidationError("blocked host"),
        HttpStatusError(415, "Unsupported Media Type", "https://example.com/x.wav"),
    ],
)
async def test_load_audio_batch_prioritizes_typed_client_error(client_error):
    loader = AudioLoader()
    loader.load_audio = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("decode failed"), client_error]
    )

    with pytest.raises(type(client_error)) as exc_info:
        await loader.load_audio_batch(
            [
                {"Url": "https://example.com/bad.wav"},
                {"Url": "https://localhost/private.wav"},
            ]
        )

    assert exc_info.value is client_error


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


@pytest.mark.asyncio
async def test_load_audio_missing_decoder_is_actionable():
    """vLLM's own hint here is `pip install vllm[audio]`, which drags in an
    unpinned stack; the wrap must point at the validated bounded install and
    say there is no hardware alternative for audio."""
    loader = AudioLoader()
    loader._load_audio_with_vllm = AsyncMock(  # type: ignore[method-assign]
        side_effect=ImportError("Please install vllm[audio] for audio support")
    )

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_audio("https://example.com/x.mp3")

    msg = str(exc_info.value)
    assert VALIDATED_SPECS["av"] in msg
    assert "install_media_decoders vllm" in msg
    assert "NVDEC does not decode audio" in msg


@pytest.mark.asyncio
async def test_load_audio_batch_preserves_missing_decoder_error():
    """The batch aggregate wraps failures in a generic Exception; the
    missing-decoder type must survive it (review finding)."""
    loader = AudioLoader()
    err = audio_loader_module.audio_decoder_missing("vllm")
    loader.load_audio = AsyncMock(side_effect=err)  # type: ignore[method-assign]

    with pytest.raises(MissingMediaDecoderError) as exc_info:
        await loader.load_audio_batch([{"Url": "https://example.com/x.mp3"}])

    assert exc_info.value is err

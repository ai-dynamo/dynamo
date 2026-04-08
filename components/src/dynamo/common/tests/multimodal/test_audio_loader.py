# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import numpy as np
import pytest

import dynamo.common.multimodal.audio_loader as audio_loader_module
from dynamo.common.multimodal.audio_loader import AudioLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_normalize_audio_url_converts_local_paths(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    assert (
        AudioLoader._normalize_audio_url(str(audio_path))
        == audio_path.resolve().as_uri()
    )


def test_normalize_audio_url_preserves_data_urls():
    data_url = "data:audio/wav;base64,UklGRg=="
    assert AudioLoader._normalize_audio_url(data_url) == data_url


def test_normalize_audio_url_preserves_http_urls():
    url = "https://example.com/audio.wav"
    assert AudioLoader._normalize_audio_url(url) == url


def test_normalize_audio_url_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="Error reading file"):
        AudioLoader._normalize_audio_url("/nonexistent/audio.wav")


@pytest.mark.asyncio
async def test_load_audio_uses_vllm_media_connector():
    loader = AudioLoader()
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
    loader = AudioLoader()
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

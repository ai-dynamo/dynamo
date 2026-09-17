# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VideoGenerationWorkerHandler's input_reference handling.

The video (I2V) handler had the same unvalidated ``input_reference`` ->
``image_path`` passthrough as image diffusion and shares the fix, but had no
test of its own — so the shared validation could be dropped from this handler
alone and every test would still pass.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from dynamo.common.http.url_validator import UrlValidationError
from dynamo.sglang.request_handlers.video_generation.video_generation_handler import (
    VideoGenerationWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,  # No GPU needed for unit tests
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


@pytest.fixture
def handler():
    config = MagicMock()
    config.dynamo_args.media_output_fs_url = "file:///tmp/videos"
    config.dynamo_args.media_output_http_url = "file:///tmp/videos"
    handler = VideoGenerationWorkerHandler(
        generator=MagicMock(), config=config, publisher=None, fs=MagicMock()
    )
    handler._frames_to_video = AsyncMock(return_value=b"mp4-bytes")
    return handler


def _generate_video(handler, **overrides):
    kwargs = dict(
        prompt="a cat",
        width=256,
        height=256,
        num_frames=8,
        fps=8,
        num_inference_steps=10,
        guidance_scale=5.0,
        seed=42,
        request_id="req-1",
    )
    kwargs.update(overrides)
    return handler._generate_video(**kwargs)


async def test_i2v_passes_the_resolved_path_to_the_generator(
    handler, tmp_path, monkeypatch
) -> None:
    """The generator gets the resolved path, not the client's reference."""
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", str(tmp_path))
    (tmp_path / "sub").mkdir()
    reference = tmp_path / "first_frame.png"
    reference.write_bytes(b"x")
    # Route through a "sub/.." segment so the resolved form differs from the
    # string the client sent; otherwise this passes even with no resolution.
    sent = str(tmp_path / "sub" / ".." / "first_frame.png")
    handler.generator.generate = Mock(return_value=SimpleNamespace(frames=[object()]))

    await _generate_video(handler, input_reference=sent)

    args = handler.generator.generate.call_args[1]["sampling_params_kwargs"]
    assert args["image_path"] != sent
    assert args["image_path"] == str(reference.resolve())


async def test_i2v_rejects_a_reference_outside_the_allowed_dir(
    handler, tmp_path, monkeypatch
) -> None:
    """Traversal is refused before the generator ever sees a path."""
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", str(tmp_path))
    handler.generator.generate = Mock()

    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        await _generate_video(handler, input_reference="/etc/passwd")

    handler.generator.generate.assert_not_called()


async def test_i2v_rejection_message_is_bounded(handler, tmp_path, monkeypatch) -> None:
    """input_reference is unbounded and this message reaches the response body."""
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", str(tmp_path))
    handler.generator.generate = Mock()

    with pytest.raises(UrlValidationError) as excinfo:
        await _generate_video(handler, input_reference="/nope/" + "A" * 200_000)

    assert len(str(excinfo.value)) < 500


async def test_t2v_passes_no_image_path(handler) -> None:
    """Control: a text-only request must not gain an image_path."""
    handler.generator.generate = Mock(return_value=SimpleNamespace(frames=[object()]))

    await _generate_video(handler, input_reference=None)

    args = handler.generator.generate.call_args[1]["sampling_params_kwargs"]
    assert "image_path" not in args

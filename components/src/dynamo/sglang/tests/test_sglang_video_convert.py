# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-B tests for the SGLang video path: to_canonical() + handler adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_HANDLER_MODULE = (
    "dynamo.sglang.request_handlers.video_generation.video_generation_handler"
)


class TestSglangVideoToCanonical:
    """to_canonical() maps DiffGenerator frames to canonical frames losslessly."""

    def test_roundtrip_from_pil_is_bit_exact(self):
        Image = pytest.importorskip("PIL.Image")
        from dynamo.sglang.request_handlers.video_generation.video_convert import (
            to_canonical,
        )

        # Distinctive per-pixel values catch axis / channel-order bugs.
        truth = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
        native = [Image.fromarray(truth[i]) for i in range(truth.shape[0])]

        out = to_canonical(native)

        assert out.dtype == np.uint8
        assert np.array_equal(out, truth)

    def test_roundtrip_from_numpy_is_bit_exact(self):
        from dynamo.sglang.request_handlers.video_generation.video_convert import (
            to_canonical,
        )

        truth = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
        native = [truth[i] for i in range(truth.shape[0])]

        out = to_canonical(native)

        assert np.array_equal(out, truth)


class TestSglangVideoHandlerAdapter:
    """_generate_video converts to canonical frames, then calls encode_video."""

    @pytest.mark.asyncio
    async def test_converts_then_encodes(self):
        from dynamo.sglang.request_handlers.video_generation.video_generation_handler import (  # noqa: E501
            VideoGenerationWorkerHandler,
        )

        frames = [MagicMock()]
        handler = object.__new__(VideoGenerationWorkerHandler)
        handler._generate_lock = asyncio.Lock()
        handler.generator = SimpleNamespace(
            generate=lambda **kw: SimpleNamespace(frames=frames)
        )
        canonical = np.zeros((2, 4, 4, 3), dtype=np.uint8)

        with patch(
            f"{_HANDLER_MODULE}.to_canonical", return_value=canonical
        ) as m_conv, patch(
            f"{_HANDLER_MODULE}.encode_video", return_value=b"bytes"
        ) as m_enc:
            out = await handler._generate_video(
                prompt="p",
                width=8,
                height=8,
                num_frames=2,
                fps=16,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=0,
                request_id="r",
            )

        assert out == b"bytes"
        m_conv.assert_called_once_with(frames)
        args, _ = m_enc.call_args
        assert args[0] is canonical
        assert args[1] == 16

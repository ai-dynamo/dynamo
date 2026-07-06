# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``EncodeWorkerHandler.__init__`` on non-Qwen vision models.

Regression guard for NvBug 5935944.

``EncodeWorkerHandler.__init__`` reads the embedding hidden dimension off the
loaded vision model. Qwen-VL loads via vLLM ``LLM(mm_encoder_only=True)`` and
returns a ViT module that exposes ``out_hidden_size`` directly; non-Qwen families
(e.g. LLaVA-1.5) load via ``AutoModel.from_pretrained`` and return a HuggingFace
model with **no** ``out_hidden_size`` attribute.

The original bug read ``self.vision_model.out_hidden_size`` unconditionally, so
every non-Qwen vision model crashed the encode worker on startup with
``AttributeError: 'LlavaModel' object has no attribute 'out_hidden_size'``. The
handler must construct successfully regardless of whether that attribute exists.

The single meaningful guard is therefore behavioral: ``__init__`` must not raise
for a vision model lacking ``out_hidden_size``. The resolved dimension itself is
only logged (never used downstream), so it is deliberately not asserted here —
that would couple the test to a debug-log string without testing any behavior.
No GPU is required: ``load_vision_model`` (and the other model-touching seams)
are mocked, so this exercises only the constructor path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dynamo.vllm.constants import EmbeddingTransferMode
from dynamo.vllm.multimodal_handlers.encode_worker_handler import EncodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_HANDLER_MOD = "dynamo.vllm.multimodal_handlers.encode_worker_handler"


def _consume_coro(coro):
    """Stand-in for ``asyncio.create_task``.

    ``__init__`` schedules the ``check_complete`` background task, which needs a
    running event loop we don't have here. Closing the coroutine avoids both the
    loop requirement and the "coroutine was never awaited" RuntimeWarning (the
    pytest config promotes warnings to errors).
    """
    coro.close()
    return MagicMock()


def _construct_handler(vision_model, model_name: str) -> EncodeWorkerHandler:
    """Build an ``EncodeWorkerHandler`` with every model-touching seam mocked.

    The seams mirror ``__init__``: the image processor download, the vision-model
    load (the object whose ``out_hidden_size`` shape is under test), and the
    encoder-component extraction. ``load_vision_model`` is the seam called out by
    the ticket; the others are mocked only so the constructor can run GPU-free.
    """
    engine_args = SimpleNamespace(model=model_name, enforce_eager=True)
    with patch(f"{_HANDLER_MOD}.AutoImageProcessor") as mock_processor, patch(
        f"{_HANDLER_MOD}.load_vision_model", return_value=vision_model
    ) as mock_load, patch(
        f"{_HANDLER_MOD}.get_encoder_components",
        return_value=(MagicMock(name="vision_encoder"), MagicMock(name="projector")),
    ), patch(
        f"{_HANDLER_MOD}.asyncio.create_task", side_effect=_consume_coro
    ):
        mock_processor.from_pretrained.return_value = MagicMock(name="image_processor")
        handler = EncodeWorkerHandler(engine_args, EmbeddingTransferMode.LOCAL)

    # The handler must have loaded the vision model with the engine's model id
    # and eager setting — i.e. the object under test came from load_vision_model.
    mock_load.assert_called_once_with(model_name, enforce_eager=True)
    assert handler.vision_model is vision_model
    return handler


@pytest.mark.parametrize(
    "vision_model",
    [
        # LLaVA-family shape: no out_hidden_size, hidden dim on .config only.
        pytest.param(
            SimpleNamespace(config=SimpleNamespace(hidden_size=4096)),
            id="llava-no-out_hidden_size",
        ),
        # Degenerate shape: neither out_hidden_size nor a config. The handler
        # falls back to "unknown" instead of crashing.
        pytest.param(SimpleNamespace(), id="no-out_hidden_size-no-config"),
    ],
)
def test_init_does_not_crash_without_out_hidden_size(vision_model):
    """``__init__`` must construct for a vision model lacking ``out_hidden_size``.

    Direct NvBug 5935944 guard: pre-fix, both shapes raised ``AttributeError`` at
    ``__init__`` (the unconditional ``self.vision_model.out_hidden_size`` read).
    """
    assert not hasattr(vision_model, "out_hidden_size")  # premise of the test

    handler = _construct_handler(vision_model, "some/non-qwen-vlm")

    assert handler is not None

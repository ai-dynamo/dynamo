# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``EncodeWorkerHandler.__init__`` vision-model hidden-dim resolution.

Regression guard for NvBug 5935944.

``EncodeWorkerHandler.__init__`` reads the embedding hidden dimension off the
loaded vision model. The two supported load paths return different object shapes
(see ``multimodal_utils/model.py::load_vision_model``):

* Qwen-VL loads via vLLM ``LLM(mm_encoder_only=True)`` and returns a vLLM ViT
  module that exposes ``out_hidden_size`` directly on the module.
* Non-Qwen families (e.g. LLaVA-1.5) load via ``AutoModel.from_pretrained`` and
  return a HuggingFace model that has **no** ``out_hidden_size`` attribute — the
  hidden dim lives on ``.config.hidden_size``.

The original bug unconditionally read ``self.vision_model.out_hidden_size``, so
every non-Qwen vision model crashed the encode worker on startup with
``AttributeError: 'LlavaModel' object has no attribute 'out_hidden_size'``. The
handler must instead fall back to ``config.hidden_size``. These tests pin that
fallback so the regression cannot silently return.

No GPU is required: ``load_vision_model`` (and the other model-touching seams)
are mocked, so this exercises only the hidden-dim resolution branch.
"""

import logging
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


def _qwen_like_vision_model(out_hidden_size: int, config_hidden_size: int):
    """A vLLM-ViT-like object: ``out_hidden_size`` present on the module itself."""
    return SimpleNamespace(
        out_hidden_size=out_hidden_size,
        config=SimpleNamespace(hidden_size=config_hidden_size),
    )


def _llava_like_vision_model(config_hidden_size: int):
    """A HuggingFace-AutoModel-like object: no ``out_hidden_size``, only config."""
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=config_hidden_size))
    # Guard the premise of this test: the fake must reproduce the real LLaVA
    # shape (no out_hidden_size), otherwise it wouldn't exercise the fallback.
    assert not hasattr(model, "out_hidden_size")
    return model


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


class TestEncodeWorkerInitHiddenDim:
    """``__init__`` must resolve the hidden dim across vision-model families."""

    def test_init_handles_models_without_out_hidden_size(self, caplog):
        """LLaVA path: no ``out_hidden_size`` must not raise; falls back to config.

        This is the direct NvBug 5935944 regression: pre-fix this raised
        ``AttributeError`` before the handler finished constructing.
        """
        vision_model = _llava_like_vision_model(config_hidden_size=4096)

        with caplog.at_level(logging.DEBUG, logger=_HANDLER_MOD):
            handler = _construct_handler(vision_model, "llava-hf/llava-1.5-7b-hf")

        assert handler is not None
        # Fallback hidden-dim (config.hidden_size) must be logged.
        assert "embedding hidden dim: 4096" in caplog.text

    @pytest.mark.parametrize(
        "vision_model, model_name, expected_dim",
        [
            # Non-Qwen families load via AutoModel.from_pretrained and expose the
            # hidden dim only on .config.hidden_size (the fallback path). LLaVA-1.5
            # is the family the encoder currently supports (resolve_model_family +
            # get_encoder_components); other AutoModel families that don't yet
            # resolve are rejected later in __init__, so they aren't asserted here.
            pytest.param(
                _llava_like_vision_model(4096),
                "llava-hf/llava-1.5-7b-hf",
                4096,
                id="llava-1.5",
            ),
            # Qwen-VL loads via vLLM and exposes out_hidden_size on the module;
            # it must be preferred over config.hidden_size when both are present.
            pytest.param(
                _qwen_like_vision_model(out_hidden_size=1536, config_hidden_size=1280),
                "Qwen/Qwen2.5-VL-3B-Instruct",
                1536,
                id="qwen-vl-prefers-out_hidden_size",
            ),
        ],
    )
    def test_init_logs_hidden_dim_across_vision_model_families(
        self, caplog, vision_model, model_name, expected_dim
    ):
        """Curated family sweep over the encoder's supported families {LLaVA-1.5,
        Qwen-VL}.

        Guards both branches of the resolution: the ``out_hidden_size`` module
        attribute (Qwen) and the ``config.hidden_size`` fallback (LLaVA family).
        """
        with caplog.at_level(logging.DEBUG, logger=_HANDLER_MOD):
            handler = _construct_handler(vision_model, model_name)

        assert handler is not None
        assert f"embedding hidden dim: {expected_dim}" in caplog.text

    def test_init_does_not_raise_when_hidden_dim_unresolvable(self, caplog):
        """Defensive: neither ``out_hidden_size`` nor a config must not crash init.

        The handler logs ``unknown`` rather than raising, so an unexpected
        vision-model shape degrades to a diagnostic instead of a startup crash.
        """
        vision_model = SimpleNamespace()  # no out_hidden_size, no config
        assert not hasattr(vision_model, "out_hidden_size")

        with caplog.at_level(logging.DEBUG, logger=_HANDLER_MOD):
            handler = _construct_handler(vision_model, "some/exotic-vlm")

        assert handler is not None
        assert "embedding hidden dim: unknown" in caplog.text

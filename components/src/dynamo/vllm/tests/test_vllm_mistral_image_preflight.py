# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup preflight for image input on the Mistral tokenizer path.

The runtime images deliberately ship no OpenCV, but ``mistral_common`` resizes
every still image with ``cv2``. Left alone, that gap surfaces from inside
vLLM's multimodal profiling run, after the weights have loaded, as an upstream
``ImportError`` pointing at an install these images do not support.
"""

import dataclasses
import importlib
import importlib.machinery
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.common.multimodal.codec_errors import MissingMediaDecoderError
from dynamo.common.utils.install_media_decoders import VALIDATED_SPECS

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),
    pytest.mark.pre_merge,
]


def _load_vllm_main() -> ModuleType:
    """Load the entrypoint lazily, as the other vLLM unit tests do.

    ``dynamo.vllm.main`` imports uvloop at module scope, which the lightweight
    pre-commit collection environment omits.
    """
    return importlib.import_module("dynamo.vllm.main")


def _vllm_config(*, multimodal: bool, tokenizer_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=multimodal, tokenizer_mode=tokenizer_mode
        )
    )


def _pin_cv2(monkeypatch, *, present: bool) -> None:
    """Pin what ``find_spec("cv2")`` reports, in both directions.

    ``importlib.util.find_spec`` consults ``sys.modules`` first: a ``None``
    entry makes it return ``None``, and a module carrying a spec makes it
    return that spec. Pinning both directions keeps the test meaningful on a
    host that has opted into the documented install -- otherwise the failing
    branch would never run there.
    """
    if present:
        stub = ModuleType("cv2")
        stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        monkeypatch.setitem(sys.modules, "cv2", stub)
    else:
        monkeypatch.setitem(sys.modules, "cv2", None)


def test_mistral_multimodal_without_cv2_fails_before_the_engine_starts(monkeypatch):
    vllm_main = _load_vllm_main()
    _pin_cv2(monkeypatch, present=False)

    with pytest.raises(MissingMediaDecoderError) as excinfo:
        vllm_main.check_mistral_image_decoder(
            _vllm_config(multimodal=True, tokenizer_mode="mistral")
        )

    msg = str(excinfo.value)
    assert "cv2" in msg
    # The bounded spec and the installer command are what make the failure
    # actionable; upstream's own suggestion is the wrong one for these images.
    assert VALIDATED_SPECS["opencv-python-headless"] in msg
    assert "install_media_decoders vllm" in msg
    assert "mistral-common[opencv]" not in msg


@pytest.mark.parametrize(
    "multimodal, tokenizer_mode",
    [
        # Text-only Mistral models tokenize no images.
        (False, "mistral"),
        # The Hugging Face processor path imports no OpenCV, so a multimodal
        # model on it must still start without the install.
        (True, "auto"),
        (False, "auto"),
    ],
)
def test_preflight_stays_silent_off_the_mistral_image_path(
    monkeypatch, multimodal, tokenizer_mode
):
    vllm_main = _load_vllm_main()
    _pin_cv2(monkeypatch, present=False)

    assert (
        vllm_main.check_mistral_image_decoder(
            _vllm_config(multimodal=multimodal, tokenizer_mode=tokenizer_mode)
        )
        is None
    )


def test_preflight_passes_once_the_decoder_is_installed(monkeypatch):
    """The documented remedy has to clear the check, or a worker could never
    start after running it."""
    vllm_main = _load_vllm_main()
    _pin_cv2(monkeypatch, present=True)

    assert (
        vllm_main.check_mistral_image_decoder(
            _vllm_config(multimodal=True, tokenizer_mode="mistral")
        )
        is None
    )


def test_preflight_tolerates_a_config_without_the_attributes(monkeypatch):
    """A guard on the startup path must not itself break startup.

    The preflight reads both attributes defensively, so a config object that
    predates them -- or a vLLM release that moves them -- degrades to "do not
    check" rather than raising AttributeError on every worker.
    """
    vllm_main = _load_vllm_main()
    _pin_cv2(monkeypatch, present=False)

    assert (
        vllm_main.check_mistral_image_decoder(
            SimpleNamespace(model_config=SimpleNamespace())
        )
        is None
    )


def test_vllm_model_config_still_exposes_the_attributes_the_preflight_reads():
    """The loud half of that defensiveness.

    Reading through ``getattr`` means a rename upstream would silently disable
    the check. Pin the two names here so the rename fails CI instead.
    """
    model_config_cls = pytest.importorskip("vllm.config").ModelConfig

    assert hasattr(model_config_cls, "is_multimodal_model")
    assert "tokenizer_mode" in {
        field.name for field in dataclasses.fields(model_config_cls)
    }


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is not None,
    reason="cv2 is installed here, so mistral_common's own guard cannot fire",
)
def test_upstream_still_image_encode_needs_cv2():
    """Ground the preflight against upstream rather than against a mock.

    ``mistral_common.tokens.tokenizers.image.transform_image`` is the single
    site that asserts OpenCV is present, and it is reached for a plain still
    image with no video anywhere. If upstream ever stops needing OpenCV here,
    this test fails and the preflight can go.
    """
    pil_image = pytest.importorskip("PIL.Image")
    image_tokenizer = pytest.importorskip("mistral_common.tokens.tokenizers.image")

    with pytest.raises(ImportError) as excinfo:
        image_tokenizer.transform_image(pil_image.new("RGB", (64, 64), "red"), (64, 64))

    assert "opencv" in str(excinfo.value).lower()

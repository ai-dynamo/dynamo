# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import importlib.util
import io
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _load_example(name):
    path = Path(__file__).parents[1] / "examples/diffusers/deploy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():
    return _load_example("flux_image_worker")


@pytest.fixture
async def backend(worker):
    args = argparse.Namespace(
        served_model_name=None,
        model="test-model",
        default_size="32x32",
        max_image_side=2048,
        default_num_inference_steps=4,
        max_num_inference_steps=50,
        default_guidance_scale=1.0,
    )
    service = worker.FluxImageBackend(args)
    service.pipe = Mock(
        return_value=SimpleNamespace(images=[Image.new("RGB", (32, 32))])
    )
    service.supports_edit = True
    try:
        yield service
    finally:
        await service.close()


async def _request(worker, backend, **fields):
    request = worker.NvCreateImageRequest(prompt="test", **fields)
    return [
        result
        async for result in worker.FluxImageBackend.create_image.__wrapped__(
            backend, request
        )
    ]


@pytest.mark.parametrize(
    "fields, parameter",
    [
        ({"nvext": {"negative_prompt": "blurry"}}, "negative_prompt"),
        ({"nvext": {"negative_prompt": ""}}, "negative_prompt"),
        ({"background": "transparent"}, "background"),
    ],
)
async def test_unsupported_controls_fail_before_loading(
    worker, backend, fields, parameter
):
    backend._image_loader.load_image = AsyncMock()
    with pytest.raises(worker.InvalidArgument, match=parameter):
        await _request(worker, backend, input_reference="unused", **fields)
    backend._image_loader.load_image.assert_not_called()
    backend.pipe.assert_not_called()


async def test_cancelled_generation_keeps_admission_and_shutdown_waits(worker, backend):
    entered = threading.Event()
    release = threading.Event()
    pipeline = backend.pipe

    def generate(**kwargs):
        entered.set()
        assert release.wait(10), "generation probe deadline"
        return pipeline(**kwargs)

    backend.pipe = generate
    backend._image_loader.load_image = AsyncMock()
    first = asyncio.create_task(_request(worker, backend))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        for _ in range(16):
            with pytest.raises(worker.WorkerOverloaded):
                await _request(worker, backend, input_reference="unused")
        backend._image_loader.load_image.assert_not_called()
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        if closing is not None:
            await closing
    pipeline.assert_called_once()


async def test_resize_runs_off_loop_and_cancellation_retains_reference_slot(
    worker, backend, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    resize_threads = []
    original_resize = Image.Image.resize

    def resize(image, *args, **kwargs):
        resize_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10), "resize probe deadline"
        return original_resize(image, *args, **kwargs)

    data = io.BytesIO()
    Image.new("RGB", (7, 11)).save(data, format="PNG")
    reference = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    monkeypatch.setattr(Image.Image, "resize", resize)
    backend._image_loader.load_image = AsyncMock(wraps=backend._image_loader.load_image)
    first = asyncio.create_task(_request(worker, backend, input_reference=reference))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(worker.WorkerOverloaded):
            await _request(worker, backend, input_reference=reference)
        backend._image_loader.load_image.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await backend.close()
    assert resize_threads and all(t != loop_thread for t in resize_threads)
    assert backend.pipe.call_args.kwargs["image"].size == (32, 32)


async def test_failed_request_releases_admission(worker, backend):
    pipeline = backend.pipe
    backend.pipe = Mock(side_effect=RuntimeError("generation failed"))
    with pytest.raises(RuntimeError, match="generation failed"):
        await _request(worker, backend)
    backend.pipe = pipeline
    (response,) = await _request(worker, backend)
    assert response["nvext"]["usage"]["output_images"] == 1


def test_inconsistent_default_step_count_fails_at_startup(worker, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["flux_image_worker.py", "--default-num-inference-steps", "51"],
    )
    with pytest.raises(SystemExit) as exc:
        worker._parse_args()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", ["flux_image_worker.py"])
    assert worker._parse_args().default_num_inference_steps == 4


@pytest.fixture
def checkpoint(monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    model_class = diffusers.Flux2Transformer2DModel
    model = model_class(
        in_channels=16,
        num_layers=0,
        num_single_layers=0,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    monkeypatch.setattr(
        model_class, "load_config", Mock(return_value=dict(model.config))
    )
    tensor_module = ModuleType("torchao.prototype.mx_formats.nvfp4_tensor")
    tensor_module.NVFP4Tensor = Mock(
        side_effect=AssertionError("reached GPU conversion")
    )
    tensor_module.QuantizeTensorToNVFP4Kwargs = Mock()
    monkeypatch.setitem(sys.modules, tensor_module.__name__, tensor_module)
    loader = _load_example("nvfp4_loader")
    state = model.state_dict()
    quant = {}
    monkeypatch.setattr(loader, "load_nvfp4_state", lambda *a, **kw: (state, quant))
    return loader, state, quant, tensor_module


@pytest.mark.parametrize("mismatch", ["missing", "unexpected"])
def test_checkpoint_key_mismatch_fails_startup(checkpoint, mismatch):
    loader, state, _, tensor_module = checkpoint
    if mismatch == "missing":
        del state["x_embedder.weight"]
    else:
        state["unknown.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Checkpoint/config mismatch"):
        loader.build_nvfp4_transformer("unused", "unused", device="cpu", verbose=False)
    tensor_module.NVFP4Tensor.assert_not_called()


@pytest.mark.parametrize(
    "shape, dimension", [((16, 7), "in_features"), ((15, 8), "out_features")]
)
def test_quantized_shape_mismatch_fails_before_gpu_conversion(
    checkpoint, shape, dimension
):
    loader, state, quant, tensor_module = checkpoint
    del state["x_embedder.weight"]
    quant["x_embedder"] = {"qdata": torch.zeros(shape, dtype=torch.uint8)}
    with pytest.raises(ValueError, match=dimension):
        loader.build_nvfp4_transformer("unused", "unused", device="cpu", verbose=False)
    tensor_module.NVFP4Tensor.assert_not_called()


def test_complete_checkpoint_preserves_weights(checkpoint):
    loader, state, _, _ = checkpoint
    loaded = loader.build_nvfp4_transformer(
        "unused", "unused", device="cpu", verbose=False
    )
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[name])

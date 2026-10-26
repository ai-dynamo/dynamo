# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install and validate the pinned OpenEngine media overlay for CI."""

import argparse
import ast
import asyncio
import importlib.util
import inspect
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def _replace_definition(destination, source, class_name, names):
    """Overlay selected Python definitions without upgrading the framework."""
    original = destination.read_text()
    updated = source.read_text()
    original_tree, updated_tree = ast.parse(original), ast.parse(updated)

    def definitions(tree):
        if class_name is not None:
            tree = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        return {node.name: node for node in tree.body if hasattr(node, "name")}

    existing, replacements = definitions(original_tree), definitions(updated_tree)
    lines, source_lines = original.splitlines(keepends=True), updated.splitlines(
        keepends=True
    )
    edits = []
    for name in names:
        node = replacements[name]
        start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
        replacement = source_lines[start : node.end_lineno]
        if name in existing:
            old = existing[name]
            begin = min([old.lineno] + [item.lineno for item in old.decorator_list]) - 1
            edits.append((begin, old.end_lineno, replacement))
        else:
            owner = next(
                node
                for node in original_tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            edits.append((owner.end_lineno, owner.end_lineno, ["\n"] + replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = replacement
    result = "".join(lines)
    ast.parse(result)
    destination.write_text(result)


def _overlay_server(server, source):
    """Keep the release launcher while installing the class and its imports."""
    _replace_definition(server, source, None, ["OpenEngineServer"])
    server.write_text(
        server.read_text()
        .replace(
            "from typing import Any",
            "from typing import Any\nfrom concurrent.futures import ThreadPoolExecutor\n"
            "from .coordinator import FrontendClient\n"
            "from .kv_events import KvEventsUnavailableError, events_config",
        )
        .replace(
            "from openengine.v1 import openengine_pb2_grpc",
            "from .bindings import openengine_pb2_grpc",
        )
    )


def install_overlay(source):
    """Install the pinned media implementation into a release runtime."""
    package = Path(
        next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations))
    )
    source = Path(source) / "tensorrt_llm"
    for entry in (source / "grpc/openengine").iterdir():
        if entry.name not in {"__init__.py", "server.py", "__pycache__"}:
            target = package / "grpc/openengine" / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
    for module in ("media_io.py", "model_config.py", "multimodal_data.py"):
        shutil.copy2(source / "inputs" / module, package / "inputs" / module)
    shutil.copy2(
        source / "_torch/shared_tensor/shared_tensor.py",
        package / "_torch/shared_tensor/shared_tensor.py",
    )
    # Retain release-coupled launch_server; use the pinned class's worker pools
    # and cleanup rather than silently running preprocessing on the default pool.
    server = package / "grpc/openengine/server.py"
    _overlay_server(server, source / "grpc/openengine/server.py")
    registry = package / "inputs/registry.py"
    _replace_definition(
        registry,
        source / "inputs/registry.py",
        "BaseMultimodalInputProcessor",
        ["requires_input_placeholders"],
    )
    registry.write_text(
        registry.read_text().replace(
            "from .content_format import ContentFormat",
            "from .content_format import ContentFormat\n"
            "from .model_config import resolve_top_level_model_type",
        )
    )
    _replace_definition(
        package / "inputs/multimodal.py",
        source / "inputs/multimodal.py",
        "MultimodalParams",
        ["_apply_tensor_operation", "release_handles"],
    )
    utils = package / "inputs/utils.py"
    utils.write_text(
        "".join(
            line
            for line in utils.read_text().splitlines(keepends=True)
            if line.strip() != "session = await _get_aiohttp_session()"
        ).replace(", session=session)", ")")
    )
    check_signatures()


def check_signatures():
    package = Path(
        next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations))
    )
    loader = ast.parse((package / "inputs/media_io.py").read_text())
    fetch = next(
        node
        for node in loader.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_safe_aiohttp_get"
    )
    accepted = {arg.arg for arg in fetch.args.args + fetch.args.kwonlyargs}
    callers = ast.parse((package / "inputs/utils.py").read_text())
    checked = set()
    for function in callers.body:
        if not isinstance(function, ast.AsyncFunctionDef):
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_safe_aiohttp_get"
            ):
                assert all(
                    keyword.arg in accepted for keyword in node.keywords
                ), function.name
                checked.add(function.name)
    assert {"async_load_image", "async_load_video", "async_load_audio"} <= checked
    print("TRT-LLM overlay: retained media helper signatures passed")


async def check_server_startup():
    """Exercise the installed startup method without loading model weights."""
    from tensorrt_llm.grpc.openengine.kv_events import KvEventsUnavailableError
    from tensorrt_llm.grpc.openengine.server import OpenEngineServer

    for configured, available in ((False, False), (True, True), (True, False)):
        events = SimpleNamespace(enable_kv_cache_events=configured, publisher="zmq")
        capacity = Mock(return_value={"kvEventsEnabled": available})
        server = OpenEngineServer.__new__(OpenEngineServer)
        server._llm = SimpleNamespace(
            args=SimpleNamespace(
                kv_cache_config=SimpleNamespace(kv_events_config=events)
            ),
            _executor=SimpleNamespace(get_kv_cache_capacity=capacity),
        )
        server._server = AsyncMock()
        server.host, server.port = "127.0.0.1", 0
        if configured and not available:
            try:
                await server.start()
            except KvEventsUnavailableError:
                pass
            else:
                raise AssertionError(
                    "Startup accepted an unavailable KV event publisher"
                )
            server._server.start.assert_not_awaited()
        else:
            await server.start()
            server._server.start.assert_awaited_once()
        assert capacity.call_count == int(configured)
    print("TRT-LLM overlay: server startup with and without KV events passed")


async def validate():
    import numpy as np
    import soundfile
    from PIL import Image
    from tensorrt_llm.inputs import media_io, utils

    await check_server_startup()
    image_bytes = BytesIO()
    Image.new("RGB", (32, 32), "red").save(image_bytes, format="PNG")
    audio_bytes = BytesIO()
    soundfile.write(audio_bytes, np.zeros(1600), 16000, format="WAV")
    payloads = {
        "image": image_bytes.getvalue(),
        "video": b"video",
        "audio": audio_bytes.getvalue(),
    }
    calls = []

    async def fetch(url, timeout_sec=30):
        # Enforce the installed loader's signature without fetching external media.
        inspect.signature(media_io._safe_aiohttp_get).bind(url, timeout_sec=timeout_sec)
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        return payloads[name]

    def decode_video(path, *args, **kwargs):
        # OpenCV is optional in the base image; this checks the URL caller, not decoding.
        assert Path(path).read_bytes() == payloads["video"]
        return "video-decoded"

    with (
        patch.object(utils, "_safe_aiohttp_get", fetch),
        patch.object(utils, "_load_video_by_cv2", decode_video),
    ):
        image = await utils.async_load_image("https://example.com/image", format="pil")
        assert image.size == (32, 32)
        video = await utils.async_load_video(
            "https://example.com/video", num_frames=2, format="pil"
        )
        assert video == "video-decoded"
        audio, sample_rate = await utils.async_load_audio("https://example.com/audio")
        assert len(audio) == 1600 and sample_rate == 16000
    assert calls == ["image", "video", "audio"]
    print("TRT-LLM overlay: image, video, and audio helpers passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-source", type=Path)
    parser.add_argument("--check-signatures-only", action="store_true")
    args, _ = parser.parse_known_args()
    if args.install_source:
        install_overlay(args.install_source)
    check_signatures()
    if not args.check_signatures_only:
        asyncio.run(validate())

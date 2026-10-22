# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained runtime helpers against the overlaid fetch signature."""

import argparse
import ast
import asyncio
import importlib.util
import inspect
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def install_preparation_support(source_root):
    package = Path(
        next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations))
    )
    source = Path(source_root) / "tensorrt_llm"
    for name in (
        "registry.py",
        "multimodal.py",
        "multimodal_data.py",
        "content_format.py",
    ):
        shutil.copy2(source / "inputs" / name, package / "inputs" / name)
    shutil.copytree(
        source / "_torch/shared_tensor",
        package / "_torch/shared_tensor",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # Keep the release launcher; only the serving class owns preparation pools.
    target = package / "grpc/openengine/server.py"
    current = target.read_text()
    replacement = (source / "grpc/openengine/server.py").read_text()
    old_class = next(
        n
        for n in ast.parse(current).body
        if isinstance(n, ast.ClassDef) and n.name == "OpenEngineServer"
    )
    new_class = next(
        n
        for n in ast.parse(replacement).body
        if isinstance(n, ast.ClassDef) and n.name == "OpenEngineServer"
    )
    lines = current.splitlines(keepends=True)
    lines[old_class.lineno - 1 : old_class.end_lineno] = [
        "from concurrent.futures import ThreadPoolExecutor\n",
        "from .coordinator import FrontendClient\n",
        "from .kv_events import KvEventsUnavailableError, events_config\n\n",
        ast.get_source_segment(replacement, new_class) + "\n",
    ]
    target.write_text("".join(lines))

    target = package / "inputs/utils.py"
    current = target.read_text()
    replacement = (source / "inputs/utils.py").read_text()
    names = {"load_base64_video", "load_video", "async_load_video"}
    definitions = {
        n.name: n
        for n in ast.parse(replacement).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    }
    lines = current.splitlines(keepends=True)
    for node in reversed(ast.parse(current).body):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            lines[node.lineno - 1 : node.end_lineno] = [
                ast.get_source_segment(replacement, definitions.pop(node.name)) + "\n"
            ]
    assert not definitions, f"release image lacks video helpers: {sorted(definitions)}"
    lines.append(
        "\nfrom tensorrt_llm.inputs.media_io import VideoMediaIO, _decode_base64_with_limit\n"
    )
    target.write_text("".join(lines))


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
                assert all(keyword.arg in accepted for keyword in node.keywords), (
                    function.name
                )
                checked.add(function.name)
    assert {"async_load_image", "async_load_video", "async_load_audio"} <= checked
    print("TRT-LLM overlay: retained media helper signatures passed")


async def validate():
    import numpy as np
    import soundfile
    from PIL import Image
    from tensorrt_llm.grpc.openengine.server import OpenEngineServer
    from tensorrt_llm.inputs import media_io, utils
    from tensorrt_llm.inputs.multimodal import MultimodalParams
    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

    assert callable(BaseMultimodalInputProcessor.requires_input_placeholders)
    assert callable(MultimodalParams.release_handles)
    llm = SimpleNamespace(args=SimpleNamespace(guided_decoding_backend=None))
    server = OpenEngineServer("127.0.0.1", 0, llm, "overlay-smoke")
    try:
        assert server._input_executor._max_workers == 4
        assert server._media_executor._max_workers == 8
    finally:
        await server.stop(grace=0)

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
        assert (Path(path).read_bytes() if isinstance(path, str) else path) == payloads[
            "video"
        ]
        return "video-decoded"

    with (
        patch.object(utils, "_safe_aiohttp_get", fetch),
        patch.object(media_io, "_load_video_by_cv2", decode_video),
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
    parser.add_argument("--install-preparation-support")
    parser.add_argument("--check-signatures-only", action="store_true")
    args = parser.parse_args()
    if args.install_preparation_support:
        install_preparation_support(args.install_preparation_support)
    check_signatures()
    if not args.check_signatures_only:
        asyncio.run(validate())

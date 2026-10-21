# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained runtime helpers against the overlaid fetch signature."""

import ast
import asyncio
import importlib.util
import inspect
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


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


async def validate():
    import numpy as np
    import soundfile
    from PIL import Image
    from tensorrt_llm.inputs import media_io, utils

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
    check_signatures()
    if "--check-signatures-only" not in sys.argv[1:]:
        asyncio.run(validate())

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ComfyUI nodes that call NVIDIA Dynamo's OpenAI-compatible endpoints.

Endpoints used:
  POST /v1/images/generations  (NvCreateImageRequest -> NvImagesResponse)
  POST /v1/images/edits        (NvCreateImageRequest with input_reference)
  POST /v1/videos              (NvCreateVideoRequest -> NvVideosResponse)
  GET  /v1/models

Pure stdlib + torch/numpy/PIL (already shipped by ComfyUI). No external deps.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import torch
from PIL import Image

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_S = 600
SIZE_PRESETS_IMAGE = ["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"]
SIZE_PRESETS_VIDEO = ["832x480", "480x832", "1280x720", "720x1280"]


# ---------------------------------------------------------------------------
# tensor / bytes helpers
# ---------------------------------------------------------------------------


def _bytes_to_image_tensor(buf: bytes) -> torch.Tensor:
    """PNG/JPEG bytes -> ComfyUI IMAGE tensor (1, H, W, C) float32 in [0, 1]."""
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _image_tensor_to_b64_png(t: torch.Tensor) -> str:
    """ComfyUI IMAGE tensor (B, H, W, C) -> base64 PNG (first frame)."""
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _http_post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    ctx = (
        f"model={payload.get('model')!r} prompt={(payload.get('prompt') or '')[:200]!r}"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dynamo {e.code} from {url}: {detail} ({ctx})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Dynamo unreachable at {url}: {e.reason} ({ctx})") from e


def _http_get_json(url: str, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str, headers: dict, timeout: int) -> bytes:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _auth_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _build_nvext(
    *,
    negative_prompt: str = "",
    num_inference_steps: int = 0,
    guidance_scale: float = -1.0,
    seed: int = -1,
    num_frames: int = 0,
    fps: int = 0,
    boundary_ratio: float = -1.0,
    guidance_scale_2: float = -1.0,
) -> dict:
    """Build nvext payload. Float fields use -1.0 as the "use model default" sentinel
    so a literal 0.0 (e.g. CFG-distilled FLUX-klein, DMD2) is forwarded verbatim."""
    nv: dict[str, Any] = {}
    if negative_prompt:
        nv["negative_prompt"] = negative_prompt
    if num_inference_steps > 0:
        nv["num_inference_steps"] = num_inference_steps
    if guidance_scale >= 0:
        nv["guidance_scale"] = guidance_scale
    if seed >= 0:
        nv["seed"] = seed
    if num_frames > 0:
        nv["num_frames"] = num_frames
    if fps > 0:
        nv["fps"] = fps
    if boundary_ratio >= 0:
        nv["boundary_ratio"] = boundary_ratio
    if guidance_scale_2 >= 0:
        nv["guidance_scale_2"] = guidance_scale_2
    return nv


def _decode_image_data(item: dict, base_url: str, headers: dict, timeout: int) -> bytes:
    """Pull raw PNG/JPEG bytes out of an NvImagesResponse data item."""
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = item.get("url")
    if not url:
        raise RuntimeError(f"Dynamo response item has neither b64_json nor url: {item}")
    if url.startswith(("http://", "https://")):
        return _http_get_bytes(url, headers, timeout)
    if url.startswith("file://"):
        raise RuntimeError(
            "Dynamo returned a file:// URL. Either request response_format=b64_json "
            "or launch the worker with --media-output-http-url and serve the dir over HTTP."
        )
    raise RuntimeError(f"Unsupported Dynamo URL scheme: {url}")


# ---------------------------------------------------------------------------
# Endpoint Config (passes a tuple downstream so users don't repeat themselves)
# ---------------------------------------------------------------------------


class DynamoEndpointConfig:
    """Bundles base_url + api_key + timeout into a single DYNAMO_ENDPOINT object."""

    CATEGORY = "Dynamo"
    RETURN_TYPES = ("DYNAMO_ENDPOINT",)
    RETURN_NAMES = ("endpoint",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": (
                    "INT",
                    {"default": DEFAULT_TIMEOUT_S, "min": 5, "max": 7200},
                ),
            }
        }

    def build(self, base_url: str, api_key: str, timeout_s: int):
        endpoint = {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key.strip(),
            "timeout_s": int(timeout_s),
        }
        return (endpoint,)


def _resolve_endpoint(
    endpoint: dict | None, base_url: str, api_key: str, timeout_s: int
) -> dict:
    if endpoint:
        return endpoint
    return {
        "base_url": (base_url or DEFAULT_BASE_URL).rstrip("/"),
        "api_key": api_key.strip() if api_key else "",
        "timeout_s": int(timeout_s),
    }


# ---------------------------------------------------------------------------
# Text-to-Image
# ---------------------------------------------------------------------------


class DynamoTextToImage:
    CATEGORY = "Dynamo"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "Qwen/Qwen-Image"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (SIZE_PRESETS_IMAGE, {"default": "1024x1024"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": (
                    "INT",
                    {"default": DEFAULT_TIMEOUT_S, "min": 5, "max": 7200},
                ),
            },
            "optional": {
                "endpoint": ("DYNAMO_ENDPOINT",),
            },
        }

    def generate(
        self,
        model,
        prompt,
        size,
        n,
        steps,
        guidance_scale,
        seed,
        negative_prompt,
        response_format,
        base_url,
        api_key,
        timeout_s,
        endpoint=None,
    ):
        ep = _resolve_endpoint(endpoint, base_url, api_key, timeout_s)
        headers = _auth_headers(ep["api_key"])
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": int(n),
            "response_format": response_format,
        }
        nv = _build_nvext(
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if nv:
            payload["nvext"] = nv

        url = f"{ep['base_url']}/v1/images/generations"
        resp = _http_post_json(url, payload, headers, ep["timeout_s"])
        items = resp.get("data") or []
        if not items:
            raise RuntimeError(f"Dynamo returned no images: {resp}")

        tensors = [
            _bytes_to_image_tensor(
                _decode_image_data(item, ep["base_url"], headers, ep["timeout_s"])
            )
            for item in items
        ]
        return (torch.cat(tensors, dim=0),)


# ---------------------------------------------------------------------------
# Image Edit (image-to-image)
# ---------------------------------------------------------------------------


class DynamoImageEdit:
    CATEGORY = "Dynamo"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "Qwen/Qwen-Image"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (SIZE_PRESETS_IMAGE, {"default": "1024x1024"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": (
                    "INT",
                    {"default": DEFAULT_TIMEOUT_S, "min": 5, "max": 7200},
                ),
            },
            "optional": {
                "endpoint": ("DYNAMO_ENDPOINT",),
            },
        }

    def edit(
        self,
        image,
        model,
        prompt,
        size,
        steps,
        guidance_scale,
        seed,
        negative_prompt,
        response_format,
        base_url,
        api_key,
        timeout_s,
        endpoint=None,
    ):
        ep = _resolve_endpoint(endpoint, base_url, api_key, timeout_s)
        headers = _auth_headers(ep["api_key"])
        b64_input = _image_tensor_to_b64_png(image)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": response_format,
            "input_reference": f"data:image/png;base64,{b64_input}",
        }
        nv = _build_nvext(
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if nv:
            payload["nvext"] = nv

        url = f"{ep['base_url']}/v1/images/edits"
        resp = _http_post_json(url, payload, headers, ep["timeout_s"])
        items = resp.get("data") or []
        if not items:
            raise RuntimeError(f"Dynamo returned no images: {resp}")
        tensors = [
            _bytes_to_image_tensor(
                _decode_image_data(item, ep["base_url"], headers, ep["timeout_s"])
            )
            for item in items
        ]
        return (torch.cat(tensors, dim=0),)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _decode_video_to_path(
    item: dict, base_url: str, headers: dict, timeout: int
) -> str:
    """Return a local filesystem path containing the MP4 bytes from a video response item."""
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        url = item["url"]
        if url.startswith(("http://", "https://")):
            raw = _http_get_bytes(url, headers, timeout)
        elif url.startswith("file://"):
            raise RuntimeError(
                "Dynamo returned a file:// URL. Either request response_format=b64_json "
                "or launch with --media-output-http-url and an HTTP file server."
            )
        else:
            raise RuntimeError(f"Unsupported URL scheme: {url}")
    else:
        raise RuntimeError(f"Video response item missing data: {item}")

    fd, path = tempfile.mkstemp(prefix="dynamo_", suffix=".mp4")
    with open(fd, "wb") as f:
        f.write(raw)
    return path


def _wrap_video_for_comfy(path: str):
    """Wrap an MP4 path in ComfyUI's native VIDEO type if available; else return the path string.

    Stock ComfyUI exposes ``comfy_api.latest.InputImpl.VideoFromFile``. Older builds may
    not have it, in which case downstream nodes can still consume a path string via
    VHS / SaveVideo.
    """
    try:
        from comfy_api.latest import InputImpl  # type: ignore

        return InputImpl.VideoFromFile(path)
    except Exception:
        return path


# ---------------------------------------------------------------------------
# Text-to-Video
# ---------------------------------------------------------------------------


class DynamoTextToVideo:
    CATEGORY = "Dynamo"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (SIZE_PRESETS_VIDEO, {"default": "832x480"}),
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 1024}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 1200, "min": 5, "max": 7200}),
            },
            "optional": {
                "endpoint": ("DYNAMO_ENDPOINT",),
            },
        }

    def generate(
        self,
        model,
        prompt,
        size,
        num_frames,
        fps,
        steps,
        guidance_scale,
        seed,
        negative_prompt,
        response_format,
        base_url,
        api_key,
        timeout_s,
        endpoint=None,
    ):
        ep = _resolve_endpoint(endpoint, base_url, api_key, timeout_s)
        headers = _auth_headers(ep["api_key"])
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
        }
        nv = _build_nvext(
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_frames=num_frames,
            fps=fps,
        )
        if nv:
            payload["nvext"] = nv

        url = f"{ep['base_url']}/v1/videos"
        resp = _http_post_json(url, payload, headers, ep["timeout_s"])
        if resp.get("status") and resp["status"] not in ("completed", "succeeded"):
            raise RuntimeError(f"Dynamo video generation failed: {resp}")
        items = resp.get("data") or []
        if not items:
            raise RuntimeError(f"Dynamo returned no videos: {resp}")
        path = _decode_video_to_path(items[0], ep["base_url"], headers, ep["timeout_s"])
        return (_wrap_video_for_comfy(path), path)


# ---------------------------------------------------------------------------
# Image-to-Video
# ---------------------------------------------------------------------------


class DynamoImageToVideo:
    CATEGORY = "Dynamo"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "Wan-AI/Wan2.1-I2V-14B-480P"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (SIZE_PRESETS_VIDEO, {"default": "832x480"}),
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 1024}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "boundary_ratio": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "guidance_scale_2": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 1200, "min": 5, "max": 7200}),
            },
            "optional": {
                "endpoint": ("DYNAMO_ENDPOINT",),
            },
        }

    def generate(
        self,
        image,
        model,
        prompt,
        size,
        num_frames,
        fps,
        steps,
        guidance_scale,
        boundary_ratio,
        guidance_scale_2,
        seed,
        negative_prompt,
        response_format,
        base_url,
        api_key,
        timeout_s,
        endpoint=None,
    ):
        ep = _resolve_endpoint(endpoint, base_url, api_key, timeout_s)
        headers = _auth_headers(ep["api_key"])
        b64_input = _image_tensor_to_b64_png(image)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
            "input_reference": f"data:image/png;base64,{b64_input}",
        }
        nv = _build_nvext(
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_frames=num_frames,
            fps=fps,
            boundary_ratio=boundary_ratio,
            guidance_scale_2=guidance_scale_2,
        )
        if nv:
            payload["nvext"] = nv

        url = f"{ep['base_url']}/v1/videos"
        resp = _http_post_json(url, payload, headers, ep["timeout_s"])
        if resp.get("status") and resp["status"] not in ("completed", "succeeded"):
            raise RuntimeError(f"Dynamo video generation failed: {resp}")
        items = resp.get("data") or []
        if not items:
            raise RuntimeError(f"Dynamo returned no videos: {resp}")
        path = _decode_video_to_path(items[0], ep["base_url"], headers, ep["timeout_s"])
        return (_wrap_video_for_comfy(path), path)


# ---------------------------------------------------------------------------
# List models (utility)
# ---------------------------------------------------------------------------


class DynamoListModels:
    CATEGORY = "Dynamo"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models_json",)
    FUNCTION = "list"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 30, "min": 1, "max": 600}),
            },
            "optional": {
                "endpoint": ("DYNAMO_ENDPOINT",),
            },
        }

    def list(self, base_url, api_key, timeout_s, endpoint=None):
        ep = _resolve_endpoint(endpoint, base_url, api_key, timeout_s)
        headers = _auth_headers(ep["api_key"])
        resp = _http_get_json(f"{ep['base_url']}/v1/models", headers, ep["timeout_s"])
        return (json.dumps(resp, indent=2),)

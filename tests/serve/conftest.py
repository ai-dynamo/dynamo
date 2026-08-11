# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import pytest
from pytest_httpserver import HTTPServer

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.serve.lora_utils import MinioLoraConfig, MinioService
from tests.utils.port_utils import allocate_port, deallocate_port

# Shared constants for multimodal testing
IMAGE_SERVER_PORT = allocate_port(8765)
MULTIMODAL_IMG_URL = f"http://localhost:{IMAGE_SERVER_PORT}/llm-graphic.png"
# A yellow triangle sweeping left to right on a near-black background, 320x240,
# 10 frames. Deliberately one unmistakable object rather than a test pattern: the
# assertion below needs a word a vision model reliably reaches for.
#
# The three encodings come from ONE set of raw frames (see
# lib/llm/tests/data/media/make_triangle_fixture.py), so VP9, H.264 and H.265 carry identical
# content. That means every codec path can assert the same word, and a difference
# in output points at decode rather than at phrasing. Measured round-trip against
# the source: 1.59 / 1.68 / 1.63 mean absolute per-pixel difference -- within 0.1
# of each other, so the decoded frames are directly comparable across codecs. The
# previous fixtures differed by 17.77 between codecs, which made that impossible.
#
# The triangle moves, but do NOT read the motion as coverage of temporal
# sampling. Asked to describe the clip, Qwen2.5-VL-3B returns "A yellow triangle
# appears in the center of the screen. The triangle is static and does not move."
# -- it reports the object reliably and the movement not at all. A regression
# that collapsed frame selection to a single frame would therefore not show up in
# the response text. Cross-codec frame comparison is the check to build on for
# that, not the model's wording.
MULTIMODAL_VIDEO_PATH = os.path.join(
    WORKSPACE_DIR, "lib/llm/tests/data/media/triangle_240p_10.mp4"
)
# The H.264/H.265 clips exercise the NVDEC hardware-decode path, served over http
# by the image_server fixture.
#
# An earlier revision of this note claimed file:// video is not hardware-decoded.
# That is wrong for vLLM and SGLang: both read local media through
# read_local_media_bytes and then route it to NVDEC exactly as they do http(s)
# (common/multimodal/video_loader.py, sglang encode_worker_handler.py), because
# otherwise a local H.264 file would reach only the software decoder these images
# do not ship. TensorRT-LLM is the exception -- it has its own
# allowed_local_media_path handling and no NVDEC routing for local paths.
# MULTIMODAL_VIDEO_H264_FILE_URI below covers the local path.
_MEDIA_DIR = os.path.join(WORKSPACE_DIR, "lib/llm/tests/data/media")
_HTTP_SERVED_VIDEOS = (
    "triangle_240p_10.mp4",
    "triangle_240p_10_h264.mp4",
    "triangle_240p_10_h265.mp4",
)
MULTIMODAL_VIDEO_H264_URL = (
    f"http://localhost:{IMAGE_SERVER_PORT}/triangle_240p_10_h264.mp4"
)
MULTIMODAL_VIDEO_H265_URL = (
    f"http://localhost:{IMAGE_SERVER_PORT}/triangle_240p_10_h265.mp4"
)
# VP9 source clip over http, for backends whose video path needs a URL rather
# than a local file. Prefer this over any third-party URL: a remote host is an
# availability dependency the test does not control.
MULTIMODAL_VIDEO_URL = f"http://localhost:{IMAGE_SERVER_PORT}/triangle_240p_10.mp4"
# The same H.264 clip addressed as a local file. Reading it is gated by
# DYN_MM_LOCAL_PATH, which must name a directory containing the clip --
# MULTIMODAL_MEDIA_DIR below. Worth covering separately from http: the local
# branch has its own read path (read_local_media_bytes rather than a fetch)
# before it reaches the same NVDEC routing, so an http-only suite leaves the
# whole local read, its policy gate, and its hand-off to the decoder untested.
MULTIMODAL_VIDEO_H264_FILE_URI = (
    f"file://{os.path.join(_MEDIA_DIR, 'triangle_240p_10_h264.mp4')}"
)
# Value for DYN_MM_LOCAL_PATH in tests that use the file:// URI above.
MULTIMODAL_MEDIA_DIR = _MEDIA_DIR
# What the clip depicts, for expected_response assertions. The subject is chosen
# so the answer is unambiguous: a model that decoded the frames says "triangle",
# and one that did not cannot say it by luck.
#
# Confirmed against the real deployment on both H.264 and H.265, which returned
# the same sentence for each: "The video begins with a black screen. A yellow
# triangle appears in the center of the screen." One noun, no hedging -- a much
# wider margin than a word list over an abstract pattern.
#
# Verified to be capable of failing, too: forcing an impossible expected value
# makes the test fail, so a pass here is not vacuous.
#
# This replaces an earlier ["red", "static", "still"] against testsrc2 colour
# bars, which described content those clips never contained and passed only when
# the model happened to use one of the words. Matching is case-insensitive
# substring with OR logic (tests/utils/payloads.py).
MULTIMODAL_VIDEO_EXPECTED = ["triangle"]


def get_multimodal_test_image_bytes() -> bytes:
    """Return a deterministic PNG with an obvious green square."""

    # Lazy import so conftest loads in environments that don't have Pillow (e.g. pre-commit).
    from PIL import Image, ImageDraw

    buf = BytesIO()
    # Keep this synthetic so CI never depends on Git LFS media. The white
    # background plus large centered square gives VLMs a stronger signal than
    # an edge-to-edge flat color.
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((96, 96, 416, 416), fill=(0, 180, 0), outline=(0, 90, 0), width=8)
    draw.text((214, 444), "GREEN", fill=(0, 90, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def httpserver_listen_address():
    yield ("127.0.0.1", IMAGE_SERVER_PORT)
    deallocate_port(IMAGE_SERVER_PORT)


@pytest.fixture(scope="function")
def image_server(httpserver: HTTPServer):
    """
    Provide an HTTP server that serves test images for multimodal inference.

    This function-scoped fixture configures pytest-httpserver to serve
    a deterministic synthetic image. It's designed for testing multimodal
    inference capabilities where models need to fetch images via HTTP.

    Currently serves:
        - /llm-graphic.png - synthetic green-square PNG used by multimodal serve tests

    The handler honors `Range: bytes=A-B` and returns 206 Partial Content.
    The MM-routing dim-fetch path (`fetch_image_dims_uncached`) strictly
    requires 206 on Range probes so it never accidentally downloads a
    full image into memory; a bare `respond_with_data` would return 200
    and silently disable MM routing in the test.

    Usage:
        def test_multimodal(image_server):
            # Use MULTIMODAL_IMG_URL from this module
            # ... use url in your test payload
    """
    from werkzeug.wrappers import Request, Response

    image_data = get_multimodal_test_image_bytes()

    def _handler(request: Request) -> Response:
        range_hdr = request.headers.get("Range", "")
        if range_hdr.startswith("bytes="):
            spec = range_hdr[len("bytes=") :]
            lo_s, _, hi_s = spec.partition("-")
            try:
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else len(image_data) - 1
            except ValueError:
                return Response(status=416)
            hi = min(hi, len(image_data) - 1)
            lo = max(lo, 0)
            if lo > hi:
                return Response(status=416)
            chunk = image_data[lo : hi + 1]
            resp = Response(chunk, status=206, content_type="image/png")
            resp.headers["Content-Range"] = f"bytes {lo}-{hi}/{len(image_data)}"
            resp.headers["Accept-Ranges"] = "bytes"
            return resp
        return Response(image_data, status=200, content_type="image/png")

    httpserver.expect_request("/llm-graphic.png").respond_with_handler(_handler)

    # Serve the video fixtures over http (VP9 + the H.264/H.265 NVDEC clips) for
    # multimodal video tests. Guard against unpulled Git-LFS pointer files.
    for _fname in _HTTP_SERVED_VIDEOS:
        _fpath = os.path.join(_MEDIA_DIR, _fname)
        if not os.path.isfile(_fpath):
            continue
        with open(_fpath, "rb") as vf:
            video_data = vf.read()
        if video_data.startswith(b"version "):  # unresolved LFS pointer
            continue
        httpserver.expect_request(f"/{_fname}").respond_with_data(
            video_data, content_type="video/mp4"
        )

    return httpserver


@pytest.fixture(scope="function")
def minio_lora_service():
    """
    Provide a MinIO service with a pre-uploaded LoRA adapter for testing.

    This fixture:
    1. Connects to existing MinIO or starts a Docker container
    2. Creates the required S3 bucket
    3. Downloads the LoRA adapter from Hugging Face Hub
    4. Uploads it to MinIO
    5. Yields the MinioLoraConfig with connection details
    6. Cleans up after the test (only stops container if we started it)

    Usage:
        def test_lora(minio_lora_service):
            config = minio_lora_service
            # Use config.get_env_vars() for environment setup
            # Use config.get_s3_uri() to get the S3 URI for loading LoRA
    """
    config = MinioLoraConfig()
    service = MinioService(config)

    try:
        # Start or connect to MinIO
        service.start()

        # Create bucket and upload LoRA
        service.create_bucket()
        local_path = service.download_lora()
        service.upload_lora(local_path)

        # Clean up downloaded files (keep MinIO data intact)
        service.cleanup_download()

        yield config

    finally:
        # Stop MinIO only if we started it, clean up temp dirs
        service.stop()
        service.cleanup_temp()

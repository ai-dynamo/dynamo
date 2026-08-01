# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the paired custom-encoder HTTP benchmark."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

from examples.custom_encoder.benchmark import run_paired_http_benchmark

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _jpeg(dimensions: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    encoded = io.BytesIO()
    Image.new("RGB", dimensions, color=color).save(encoded, format="JPEG", quality=85)
    target = 7 * 1024 if dimensions == (300, 300) else 35 * 1024
    payload = encoded.getvalue()
    padding = bytearray(target - len(payload))
    padding[-1] = color[0]
    return payload + padding


def _write_workload(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for index in range(count):
        image_path = image_dir / f"image-{index}.jpg"
        image_path.write_bytes(_jpeg((300, 300), (index + 1, 2, 3)))
        rows.append(
            {
                "session_id": f"request-{index}",
                "image": str(image_path.relative_to(tmp_path)),
                "text": f"prompt-{index}",
            }
        )
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return input_file


async def _read_multipart_image(request: web.Request) -> bytes:
    reader = await request.multipart()
    part = await reader.next()
    assert part is not None and part.name == "image"
    image = bytes(await part.read())
    assert await reader.next() is None
    return image


def _decode_aggregated_image(payload: dict) -> bytes:
    data_uri = payload["messages"][0]["content"][0]["image_url"]["url"]
    return base64.b64decode(data_uri.partition(",")[2], validate=True)


def _aggregated_response() -> web.Response:
    return web.json_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"completion_tokens": 7},
        }
    )


async def test_lane_advances_only_after_both_responses_and_reuses_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = _write_workload(tmp_path, 2)
    encoder_images: list[bytes] = []
    aggregated_images: list[bytes] = []
    encoder_started = asyncio.Event()
    aggregated_started = asyncio.Event()
    encoder_response_completed = asyncio.Event()
    release_first_aggregated = asyncio.Event()

    original_post_encoder = run_paired_http_benchmark._post_encoder

    async def tracked_post_encoder(*args, **kwargs) -> float:
        latency_ms = await original_post_encoder(*args, **kwargs)
        encoder_response_completed.set()
        return latency_ms

    monkeypatch.setattr(
        run_paired_http_benchmark, "_post_encoder", tracked_post_encoder
    )

    async def encode(request: web.Request) -> web.Response:
        encoder_images.append(await _read_multipart_image(request))
        encoder_started.set()
        return web.Response(body=b"encoder-ok")

    async def aggregate(request: web.Request) -> web.Response:
        aggregated_images.append(_decode_aggregated_image(await request.json()))
        if len(aggregated_images) == 1:
            aggregated_started.set()
            await release_first_aggregated.wait()
        return _aggregated_response()

    encoder_app = web.Application()
    encoder_app.router.add_post("/encode", encode)
    aggregated_app = web.Application()
    aggregated_app.router.add_post("/v1/chat/completions", aggregate)
    async with (
        TestServer(encoder_app) as encoder_server,
        TestServer(aggregated_app) as aggregated_server,
    ):
        benchmark = asyncio.create_task(
            run_paired_http_benchmark.run_benchmark(
                input_file,
                1,
                encoder_url=str(encoder_server.make_url("/encode")),
                aggregated_url=str(aggregated_server.make_url("/v1/chat/completions")),
            )
        )
        await asyncio.wait_for(encoder_started.wait(), timeout=2)
        await asyncio.wait_for(encoder_response_completed.wait(), timeout=2)
        await asyncio.wait_for(aggregated_started.wait(), timeout=2)
        assert len(encoder_images) == len(aggregated_images) == 1

        release_first_aggregated.set()
        summary = await asyncio.wait_for(benchmark, timeout=2)

    assert summary["valid"]
    assert summary["completed_pairs"] == 2
    assert len(encoder_images) == len(aggregated_images) == 2
    assert sorted(encoder_images) == sorted(aggregated_images)


class _ConcurrencyTracker:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.active = 0
        self.maximum = 0
        self.release = asyncio.Event()

    async def enter(self) -> None:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.active == self.expected:
            self.release.set()
        await self.release.wait()

    def exit(self) -> None:
        self.active -= 1


async def test_concurrency_is_per_endpoint_and_throughput_counts_pairs(
    tmp_path: Path,
) -> None:
    input_file = _write_workload(tmp_path, 4)
    encoder_tracker = _ConcurrencyTracker(2)
    aggregated_tracker = _ConcurrencyTracker(2)

    async def encode(request: web.Request) -> web.Response:
        await _read_multipart_image(request)
        await encoder_tracker.enter()
        encoder_tracker.exit()
        return web.Response(body=b"encoder-ok")

    async def aggregate(request: web.Request) -> web.Response:
        await request.read()
        await aggregated_tracker.enter()
        aggregated_tracker.exit()
        return _aggregated_response()

    encoder_app = web.Application()
    encoder_app.router.add_post("/encode", encode)
    aggregated_app = web.Application()
    aggregated_app.router.add_post("/v1/chat/completions", aggregate)
    async with (
        TestServer(encoder_app) as encoder_server,
        TestServer(aggregated_app) as aggregated_server,
    ):
        summary = await run_paired_http_benchmark.run_benchmark(
            input_file,
            2,
            encoder_url=str(encoder_server.make_url("/encode")),
            aggregated_url=str(aggregated_server.make_url("/v1/chat/completions")),
        )

    assert summary["valid"]
    assert encoder_tracker.maximum == aggregated_tracker.maximum == 2
    assert summary["max_outstanding_http_requests"] == 4
    assert summary["paired_images_per_second"] == pytest.approx(
        4 / summary["wall_time_s"]
    )
    assert "total_http_requests_per_second" not in summary


async def test_failed_pair_invalidates_run_without_retry(tmp_path: Path) -> None:
    input_file = _write_workload(tmp_path, 1)
    encoder_calls = 0
    aggregated_calls = 0

    async def encode(request: web.Request) -> web.Response:
        nonlocal encoder_calls
        encoder_calls += 1
        await request.read()
        return web.Response(status=500, text="failed")

    async def aggregate(request: web.Request) -> web.Response:
        nonlocal aggregated_calls
        aggregated_calls += 1
        await request.read()
        return _aggregated_response()

    encoder_app = web.Application()
    encoder_app.router.add_post("/encode", encode)
    aggregated_app = web.Application()
    aggregated_app.router.add_post("/v1/chat/completions", aggregate)
    async with (
        TestServer(encoder_app) as encoder_server,
        TestServer(aggregated_app) as aggregated_server,
    ):
        summary = await run_paired_http_benchmark.run_benchmark(
            input_file,
            1,
            encoder_url=str(encoder_server.make_url("/encode")),
            aggregated_url=str(aggregated_server.make_url("/v1/chat/completions")),
        )

    assert not summary["valid"]
    assert summary["failed_pairs"] == 1
    assert summary["paired_images_per_second"] is None
    assert encoder_calls == aggregated_calls == 1

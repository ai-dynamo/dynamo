# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark paired encoder-only and aggregated HTTP requests."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData, TCPConnector
from PIL import Image

DEFAULT_ENCODER_URL = "http://localhost:8001/encode"
DEFAULT_AGGREGATED_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_TOKENS = 7
DUMMY_RESPONSE = b"encoder-ok"
JPEG_SIZE_BOUNDS = {
    (300, 300): (7 * 1024 - 256, 7 * 1024 + 256),
    (500, 500): (35 * 1024 - 256, 35 * 1024 + 256),
}


@dataclass(frozen=True)
class WorkloadItem:
    index: int
    session_id: str
    prompt: str
    image_path: Path
    image: bytes
    data_uri: str
    image_sha256: str
    dimensions: tuple[int, int]


class PairRequestError(RuntimeError):
    """One side of a paired request failed."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resolve_image_path(input_file: Path, value: str) -> Path:
    path = Path(value)
    return (
        path.resolve() if path.is_absolute() else (input_file.parent / path).resolve()
    )


def load_workload(input_file: Path) -> tuple[list[WorkloadItem], dict[str, Any]]:
    input_file = input_file.resolve()
    encoded_input = input_file.read_bytes()
    items: list[WorkloadItem] = []
    session_ids: set[str] = set()
    image_hashes: set[str] = set()
    size_counts: Counter[str] = Counter()
    size_bytes: dict[str, list[int]] = {}
    for line_number, line in enumerate(encoded_input.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise TypeError(f"line {line_number} must contain a JSON object")
        session_id = row.get("session_id")
        image_value = row.get("image")
        prompt = row.get("text")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"line {line_number} has an invalid session_id")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"line {line_number} has an invalid image path")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"line {line_number} has invalid text")
        if session_id in session_ids:
            raise ValueError(f"duplicate session_id {session_id!r}")

        image_path = _resolve_image_path(input_file, image_value)
        image = image_path.read_bytes()
        with Image.open(io.BytesIO(image)) as decoded:
            decoded.load()
            if decoded.format != "JPEG":
                raise ValueError(f"{image_path} is not a JPEG")
            dimensions = decoded.size
        bounds = JPEG_SIZE_BOUNDS.get(dimensions)
        if bounds is None:
            raise ValueError(f"unsupported image dimensions {dimensions}: {image_path}")
        if not bounds[0] <= len(image) <= bounds[1]:
            raise ValueError(
                f"{image_path} has {len(image)} bytes; expected "
                f"{bounds[0]}..{bounds[1]}"
            )
        image_sha256 = hashlib.sha256(image).hexdigest()
        if image_sha256 in image_hashes:
            raise ValueError(f"duplicate image content: {image_path}")

        size_key = f"{dimensions[0]}x{dimensions[1]}"
        size_counts[size_key] += 1
        size_bytes.setdefault(size_key, []).append(len(image))
        session_ids.add(session_id)
        image_hashes.add(image_sha256)
        items.append(
            WorkloadItem(
                index=len(items),
                session_id=session_id,
                prompt=prompt,
                image_path=image_path,
                image=image,
                data_uri=(
                    "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
                ),
                image_sha256=image_sha256,
                dimensions=dimensions,
            )
        )
    if not items:
        raise ValueError("input JSONL must contain at least one request")

    audit = {
        "input_sha256": hashlib.sha256(encoded_input).hexdigest(),
        "requests": len(items),
        "unique_sessions": len(session_ids),
        "unique_images": len(image_hashes),
        "images_by_size": dict(sorted(size_counts.items())),
        "image_bytes_by_size": {
            size: {
                "min": min(values),
                "mean": statistics.mean(values),
                "max": max(values),
            }
            for size, values in sorted(size_bytes.items())
        },
    }
    return items, audit


def _response_excerpt(body: bytes) -> str:
    return body[:200].decode("utf-8", errors="replace")


async def _post_encoder(session: ClientSession, url: str, item: WorkloadItem) -> float:
    form = FormData()
    form.add_field(
        "image",
        item.image,
        filename=f"{item.session_id}.jpg",
        content_type="image/jpeg",
    )
    started = time.perf_counter()
    async with session.post(url, data=form) as response:
        body = await response.read()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status != 200:
        raise PairRequestError(
            f"encoder-only returned HTTP {response.status}: {_response_excerpt(body)}"
        )
    if body != DUMMY_RESPONSE:
        raise PairRequestError(
            f"encoder-only returned {_response_excerpt(body)!r}; expected "
            f"{DUMMY_RESPONSE.decode()!r}"
        )
    return elapsed_ms


def _aggregated_payload(item: WorkloadItem, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": item.data_uri}},
                    {"type": "text", "text": item.prompt},
                ],
            }
        ],
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": False,
    }


async def _post_aggregated(
    session: ClientSession, url: str, item: WorkloadItem, model: str
) -> float:
    started = time.perf_counter()
    async with session.post(url, json=_aggregated_payload(item, model)) as response:
        body = await response.read()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status != 200:
        raise PairRequestError(
            f"aggregated returned HTTP {response.status}: {_response_excerpt(body)}"
        )
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PairRequestError("aggregated returned invalid JSON") from error
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise PairRequestError("aggregated response has no choices")
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") != OUTPUT_TOKENS:
        raise PairRequestError(
            "aggregated response did not report exactly "
            f"{OUTPUT_TOKENS} completion tokens"
        )
    return elapsed_ms


async def _run_pair(
    encoder_session: ClientSession,
    aggregated_session: ClientSession,
    encoder_url: str,
    aggregated_url: str,
    model: str,
    item: WorkloadItem,
) -> dict[str, Any]:
    started = time.perf_counter()
    encoder_result, aggregated_result = await asyncio.gather(
        _post_encoder(encoder_session, encoder_url, item),
        _post_aggregated(aggregated_session, aggregated_url, item, model),
        return_exceptions=True,
    )
    pair_latency_ms = (time.perf_counter() - started) * 1000
    errors = [
        f"{role}: {result}"
        for role, result in (
            ("encoder_only", encoder_result),
            ("aggregated", aggregated_result),
        )
        if isinstance(result, BaseException)
    ]
    if errors:
        raise PairRequestError("; ".join(errors))
    assert isinstance(encoder_result, float)
    assert isinstance(aggregated_result, float)
    return {
        "index": item.index,
        "session_id": item.session_id,
        "image_sha256": item.image_sha256,
        "encoder_only_latency_ms": encoder_result,
        "aggregated_latency_ms": aggregated_result,
        "pair_latency_ms": pair_latency_ms,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


async def run_benchmark(
    input_file: Path,
    concurrency: int,
    encoder_url: str = DEFAULT_ENCODER_URL,
    aggregated_url: str = DEFAULT_AGGREGATED_URL,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    if concurrency < 1 or timeout_seconds <= 0:
        raise ValueError("concurrency and timeout_seconds must be positive")
    items, workload_audit = load_workload(input_file)
    queue: asyncio.Queue[WorkloadItem] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    timeout = ClientTimeout(total=timeout_seconds)
    async with (
        ClientSession(
            timeout=timeout,
            connector=TCPConnector(limit=concurrency, limit_per_host=concurrency),
        ) as encoder_session,
        ClientSession(
            timeout=timeout,
            connector=TCPConnector(limit=concurrency, limit_per_host=concurrency),
        ) as aggregated_session,
    ):

        async def pair_lane() -> None:
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    results.append(
                        await _run_pair(
                            encoder_session,
                            aggregated_session,
                            encoder_url,
                            aggregated_url,
                            model,
                            item,
                        )
                    )
                except (ClientError, PairRequestError, TimeoutError) as error:
                    failures.append(
                        {
                            "index": item.index,
                            "session_id": item.session_id,
                            "error": str(error),
                        }
                    )
                finally:
                    queue.task_done()

        started = time.perf_counter()
        await asyncio.gather(*(pair_lane() for _ in range(concurrency)))
        wall_time_s = time.perf_counter() - started

    results.sort(key=lambda result: int(result["index"]))
    failures.sort(key=lambda failure: int(failure["index"]))
    valid = not failures and len(results) == len(items)
    pairs_per_second = len(items) / wall_time_s if valid else None
    return {
        "valid": valid,
        "input_file": str(input_file.resolve()),
        "model": model,
        "encoder_url": encoder_url,
        "aggregated_url": aggregated_url,
        "concurrency_per_endpoint": concurrency,
        "max_outstanding_http_requests": 2 * concurrency,
        "pairing_policy": (
            "each lane starts one request per endpoint and advances only after "
            "both responses complete"
        ),
        "requested_pairs": len(items),
        "completed_pairs": len(results),
        "failed_pairs": len(failures),
        "wall_time_s": wall_time_s,
        "paired_images_per_second": pairs_per_second,
        "encoder_only_requests_per_second": pairs_per_second,
        "aggregated_requests_per_second": pairs_per_second,
        "latency_ms": {
            "encoder_only": _latency_summary(
                [float(result["encoder_only_latency_ms"]) for result in results]
            ),
            "aggregated": _latency_summary(
                [float(result["aggregated_latency_ms"]) for result in results]
            ),
            "pair": _latency_summary(
                [float(result["pair_latency_ms"]) for result in results]
            ),
        },
        "workload": workload_audit,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=_positive_int, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--encoder-url", default=DEFAULT_ENCODER_URL)
    parser.add_argument("--aggregated-url", default=DEFAULT_AGGREGATED_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(
        run_benchmark(
            args.input_file,
            args.concurrency,
            encoder_url=args.encoder_url,
            aggregated_url=args.aggregated_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small OpenAI Responses gateway using the standalone KV routing services."""

import argparse
import asyncio
import hashlib
import json
import logging
import struct
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer

LOGGER = logging.getLogger("raw_engines_routing")
TENANT_ID = "default"
CONTROL_TIMEOUT = aiohttp.ClientTimeout(total=10)
UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10)

ROUTING_UNSAFE_FIELDS = {
    "cache_salt",
    "chat_template",
    "chat_template_kwargs",
    "conversation",
    "mm_processor_kwargs",
    "previous_input_messages",
    "previous_response_id",
    "reasoning",
    "tool_choice",
    "tools",
}


@dataclass(frozen=True, order=True)
class Worker:
    worker_id: int
    dp_rank: int
    url: str


@dataclass(frozen=True)
class Config:
    model: str
    block_size: int
    indexer_url: str
    slot_tracker_url: str
    prefill_load_scale: float
    workers: dict[tuple[int, int], Worker]
    host: str
    port: int
    log_level: str


@dataclass(frozen=True)
class Candidate:
    worker: Worker
    gpu_overlap_tokens: int
    potential_prefill_tokens: int
    potential_decode_blocks: int
    score: float


class GatewayError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def log_event(level: int, event: str, **fields: Any) -> None:
    if not LOGGER.isEnabledFor(level):
        return
    details = " ".join(
        f"{key}={json.dumps(value, sort_keys=True)}" for key, value in fields.items()
    )
    LOGGER.log(level, "%s%s", event, f" {details}" if details else "")


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def active_field(body: dict[str, Any], field: str) -> bool:
    value = body.get(field)
    return value not in (None, False, [], {})


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise GatewayError(400, "message content must be text")

    parts = []
    for part in content:
        if (
            not isinstance(part, dict)
            or part.get("type") != "input_text"
            or not isinstance(part.get("text"), str)
        ):
            raise GatewayError(
                400, "message content must contain only input_text items"
            )
        parts.append(part["text"])
    return "".join(parts)


def normalize_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    model = body.get("model")
    if not isinstance(model, str):
        raise GatewayError(400, "model must be a string")

    for field in ROUTING_UNSAFE_FIELDS:
        if active_field(body, field):
            raise GatewayError(
                400, f"{field} is not supported by this stateless example"
            )
    if active_field(body, "store"):
        raise GatewayError(400, "store=true is not supported by this stateless example")
    if active_field(body, "background"):
        raise GatewayError(400, "background=true is not supported by this example")
    if not isinstance(body.get("stream", False), bool):
        raise GatewayError(400, "stream must be a boolean")

    messages = []
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise GatewayError(400, "instructions must be a string")
        messages.append({"role": "system", "content": instructions})

    request_input = body.get("input")
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
        return messages
    if not isinstance(request_input, list) or not request_input:
        raise GatewayError(400, "input must be a string or a non-empty message list")

    for item in request_input:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            raise GatewayError(400, "input must contain only text message items")
        if item.get("status") not in (None, "completed"):
            raise GatewayError(400, "partial assistant continuation is not supported")

        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise GatewayError(400, "message role must be system, user, or assistant")
        messages.append({"role": role, "content": text_content(item.get("content"))})

    if messages[-1]["role"] != "user":
        raise GatewayError(400, "the final input message must have role=user")
    return messages


def tokenize(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise GatewayError(500, "tokenizer returned an unsupported token sequence")
    return token_ids


def slot_hashes(token_ids: list[int], block_size: int) -> list[int]:
    hashes = []
    parent = b""
    full_blocks = len(token_ids) // block_size
    for block_idx in range(full_blocks):
        block = token_ids[block_idx * block_size : (block_idx + 1) * block_size]
        encoded = b"".join(struct.pack("<I", token_id) for token_id in block)
        parent = hashlib.blake2b(parent + encoded, digest_size=8).digest()
        value = int.from_bytes(parent, "little")
        hashes.append(value if value < 2**63 else value - 2**64)
    return hashes


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> tuple[int, Any]:
    try:
        async with session.post(url, json=payload, timeout=CONTROL_TIMEOUT) as response:
            body = await response.text()
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        raise GatewayError(
            502, f"control-plane request failed for {url}: {error}"
        ) from error


def response_error(service: str, status: int, body: Any) -> GatewayError:
    return GatewayError(502, f"{service} returned HTTP {status}: {body}")


async def select_and_add(
    config: Config,
    session: aiohttp.ClientSession,
    request_id: str,
    scores: dict[str, Any],
    token_ids: list[int],
    sequence_hashes: list[int],
) -> Candidate:
    for attempt in range(2):
        loads_status, loads_body = await post_json(
            session,
            f"{config.slot_tracker_url}/potential_loads",
            {
                "model_name": config.model,
                "tenant_id": TENANT_ID,
                "sequence_hashes": sequence_hashes,
                "new_isl_tokens": len(token_ids),
            },
        )
        if loads_status != 200 or not isinstance(loads_body, list):
            raise response_error(
                "slot tracker /potential_loads", loads_status, loads_body
            )
        log_event(logging.DEBUG, "potential_loads", loads=loads_body)

        candidates = []
        for load in loads_body:
            if not isinstance(load, dict):
                raise GatewayError(
                    502, "slot tracker /potential_loads returned malformed load"
                )
            key = (load.get("worker_id"), load.get("dp_rank"))
            worker = config.workers.get(key)
            if worker is None:
                log_event(logging.DEBUG, "ignoring_unconfigured_worker", worker=key)
                continue

            worker_scores = scores.get(str(worker.worker_id), {})
            if not isinstance(worker_scores, dict):
                raise GatewayError(
                    502, "indexer /query returned malformed worker scores"
                )
            gpu_overlap_tokens = worker_scores.get(str(worker.dp_rank), 0)
            potential_prefill_tokens = load.get("potential_prefill_tokens")
            potential_decode_blocks = load.get("potential_decode_blocks")
            if (
                not isinstance(gpu_overlap_tokens, int)
                or not isinstance(potential_prefill_tokens, int)
                or not isinstance(potential_decode_blocks, int)
            ):
                raise GatewayError(
                    502, "routing services returned malformed load values"
                )

            adjusted_prefill_blocks = max(
                (potential_prefill_tokens - gpu_overlap_tokens) / config.block_size,
                0,
            )
            score = (
                config.prefill_load_scale * adjusted_prefill_blocks
                + potential_decode_blocks
            )
            candidate = Candidate(
                worker=worker,
                gpu_overlap_tokens=gpu_overlap_tokens,
                potential_prefill_tokens=potential_prefill_tokens,
                potential_decode_blocks=potential_decode_blocks,
                score=score,
            )
            candidates.append(candidate)
            log_event(
                logging.DEBUG,
                "candidate_score",
                worker_id=worker.worker_id,
                dp_rank=worker.dp_rank,
                gpu_overlap_tokens=gpu_overlap_tokens,
                potential_prefill_tokens=potential_prefill_tokens,
                potential_decode_blocks=potential_decode_blocks,
                score=score,
            )

        if not candidates:
            raise GatewayError(503, "no configured slot-tracker workers are available")
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.worker.worker_id,
                candidate.worker.dp_rank,
            ),
        )
        new_isl_tokens = max(len(token_ids) - selected.gpu_overlap_tokens, 0)
        add_status, add_body = await post_json(
            session,
            f"{config.slot_tracker_url}/add",
            {
                "model_name": config.model,
                "tenant_id": TENANT_ID,
                "request_id": request_id,
                "worker_id": selected.worker.worker_id,
                "dp_rank": selected.worker.dp_rank,
                "sequence_hashes": sequence_hashes,
                "new_isl_tokens": new_isl_tokens,
            },
        )
        log_event(logging.DEBUG, "slot_add", request_id=request_id, status=add_status)
        if add_status == 201:
            log_event(
                logging.INFO,
                "selected_worker",
                request_id=request_id,
                worker_id=selected.worker.worker_id,
                dp_rank=selected.worker.dp_rank,
                gpu_overlap_tokens=selected.gpu_overlap_tokens,
                potential_prefill_tokens=selected.potential_prefill_tokens,
                potential_decode_blocks=selected.potential_decode_blocks,
                score=selected.score,
            )
            return selected
        if add_status != 404 or attempt == 1:
            raise response_error("slot tracker /add", add_status, add_body)
        log_event(logging.INFO, "retrying_stale_selection", request_id=request_id)

    raise GatewayError(503, "unable to select an available worker")


async def lifecycle_write(
    config: Config,
    session: aiohttp.ClientSession,
    endpoint: str,
    request_id: str,
) -> None:
    try:
        status, body = await post_json(
            session,
            f"{config.slot_tracker_url}/{endpoint}",
            {
                "model_name": config.model,
                "tenant_id": TENANT_ID,
                "request_id": request_id,
            },
        )
        log_event(
            logging.DEBUG,
            f"slot_{endpoint}",
            request_id=request_id,
            status=status,
            body=body,
        )
    except GatewayError as error:
        log_event(
            logging.WARNING,
            f"slot_{endpoint}_failed",
            request_id=request_id,
            error=error.message,
        )


async def shielded_lifecycle_write(
    config: Config,
    session: aiohttp.ClientSession,
    endpoint: str,
    request_id: str,
) -> None:
    task = asyncio.create_task(lifecycle_write(config, session, endpoint, request_id))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        log_event(
            logging.DEBUG,
            f"slot_{endpoint}_continuing_after_disconnect",
            request_id=request_id,
        )


def pop_sse_frame(buffer: bytes) -> tuple[bytes | None, bytes]:
    boundaries = [
        (buffer.find(b"\r\n\r\n"), 4),
        (buffer.find(b"\n\n"), 2),
    ]
    boundaries = [boundary for boundary in boundaries if boundary[0] >= 0]
    if not boundaries:
        return None, buffer
    index, separator_len = min(boundaries)
    end = index + separator_len
    return buffer[:end], buffer[end:]


def is_generated_delta(frame: bytes) -> bool:
    data_lines = []
    for line in frame.splitlines():
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return False

    data = b"\n".join(data_lines)
    if data == b"[DONE]":
        return False
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(event, dict)
        and isinstance(event.get("type"), str)
        and event["type"].endswith(".delta")
        and bool(event.get("delta"))
    )


async def stream_upstream(
    config: Config,
    session: aiohttp.ClientSession,
    upstream: aiohttp.ClientResponse,
    request_id: str,
) -> AsyncIterator[bytes]:
    buffer = b""
    prefill_completed = False
    try:
        async for chunk in upstream.content.iter_any():
            buffer += chunk
            while True:
                frame, buffer = pop_sse_frame(buffer)
                if frame is None:
                    break
                if not prefill_completed and is_generated_delta(frame):
                    await lifecycle_write(
                        config, session, "prefill_complete", request_id
                    )
                    prefill_completed = True
                yield frame
        if buffer:
            yield buffer
    finally:
        upstream.close()
        await shielded_lifecycle_write(config, session, "free", request_id)


def upstream_response(
    status: int,
    body: bytes,
    content_type: str | None,
) -> Response:
    headers = {"content-type": content_type} if content_type else None
    return Response(content=body, status_code=status, headers=headers)


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.session = aiohttp.ClientSession()
        app.state.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, config.model
        )
        app.state.admission_lock = asyncio.Lock()
        yield
        await app.state.session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise GatewayError(400, "request body must be a JSON object")
            if body.get("model") != config.model:
                raise GatewayError(400, f"model must be {config.model}")

            messages = normalize_messages(body)
            token_ids = await asyncio.to_thread(
                tokenize, request.app.state.tokenizer, messages
            )
            sequence_hashes = await asyncio.to_thread(
                slot_hashes, token_ids, config.block_size
            )
            request_id = str(uuid.uuid4())
            log_event(
                logging.DEBUG,
                "normalized_request",
                request_id=request_id,
                token_count=len(token_ids),
                slot_hash_count=len(sequence_hashes),
            )
            indexer_status, indexer_body = await post_json(
                request.app.state.session,
                f"{config.indexer_url}/query",
                {
                    "model_name": config.model,
                    "tenant_id": TENANT_ID,
                    "token_ids": token_ids,
                },
            )
            if indexer_status != 200 or not isinstance(indexer_body, dict):
                raise response_error("indexer /query", indexer_status, indexer_body)
            scores = indexer_body.get("scores", {})
            if not isinstance(scores, dict):
                raise GatewayError(502, "indexer /query returned malformed scores")
            log_event(logging.DEBUG, "indexer_scores", scores=scores)
            log_event(logging.DEBUG, "waiting_for_admission", request_id=request_id)
            async with request.app.state.admission_lock:
                log_event(logging.DEBUG, "admission_started", request_id=request_id)
                selected = await select_and_add(
                    config,
                    request.app.state.session,
                    request_id,
                    scores,
                    token_ids,
                    sequence_hashes,
                )
            log_event(logging.DEBUG, "admission_completed", request_id=request_id)

            proxy_body = dict(body)
            proxy_body["store"] = False
            try:
                upstream = await request.app.state.session.post(
                    f"{selected.worker.url}/v1/responses",
                    json=proxy_body,
                    timeout=UPSTREAM_TIMEOUT,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                await shielded_lifecycle_write(
                    config, request.app.state.session, "free", request_id
                )
                raise GatewayError(
                    502, f"raw worker request failed: {error}"
                ) from error
            except asyncio.CancelledError:
                await shielded_lifecycle_write(
                    config, request.app.state.session, "free", request_id
                )
                raise

            content_type = upstream.headers.get("content-type")
            if upstream.status >= 400:
                try:
                    upstream_body = await upstream.read()
                    return upstream_response(
                        upstream.status, upstream_body, content_type
                    )
                finally:
                    upstream.close()
                    await shielded_lifecycle_write(
                        config, request.app.state.session, "free", request_id
                    )

            if body.get("stream") is True:
                headers = {"content-type": content_type} if content_type else None
                return StreamingResponse(
                    stream_upstream(
                        config,
                        request.app.state.session,
                        upstream,
                        request_id,
                    ),
                    status_code=upstream.status,
                    headers=headers,
                )

            try:
                upstream_body = await upstream.read()
                await lifecycle_write(
                    config,
                    request.app.state.session,
                    "prefill_complete",
                    request_id,
                )
                return upstream_response(upstream.status, upstream_body, content_type)
            finally:
                upstream.close()
                await shielded_lifecycle_write(
                    config, request.app.state.session, "free", request_id
                )
        except json.JSONDecodeError:
            return error_response(400, "request body must contain valid JSON")
        except GatewayError as error:
            return error_response(error.status_code, error.message)

    return app


def parse_worker(value: str) -> Worker:
    identity, separator, url = value.partition("=")
    if not separator or not url:
        raise argparse.ArgumentTypeError("worker must use WORKER_ID:DP_RANK=URL")
    worker_id, separator, dp_rank = identity.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("worker must use WORKER_ID:DP_RANK=URL")
    try:
        return Worker(
            worker_id=int(worker_id), dp_rank=int(dp_rank), url=url.rstrip("/")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "worker ID and DP rank must be integers"
        ) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--indexer-url", default="http://127.0.0.1:8090")
    parser.add_argument("--slot-tracker-url", default="http://127.0.0.1:8091")
    parser.add_argument("--prefill-load-scale", type=float, default=1.0)
    parser.add_argument("--worker", action="append", type=parse_worker, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    workers = {(worker.worker_id, worker.dp_rank): worker for worker in args.worker}
    if len(workers) != len(args.worker):
        raise SystemExit("duplicate --worker identity")
    if args.block_size <= 0:
        raise SystemExit("--block-size must be greater than zero")
    if args.prefill_load_scale < 0:
        raise SystemExit("--prefill-load-scale must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.setLevel(args.log_level.upper())
    config = Config(
        model=args.model,
        block_size=args.block_size,
        indexer_url=args.indexer_url.rstrip("/"),
        slot_tracker_url=args.slot_tracker_url.rstrip("/"),
        prefill_load_scale=args.prefill_load_scale,
        workers=workers,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()

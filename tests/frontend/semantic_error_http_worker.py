# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fault injection through native HTTP/TCP and optional SGLang preprocessing."""

import argparse
import asyncio
import contextlib
import json
import re
import uuid
from pathlib import Path

import httpx

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    HttpError,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    classified_error,
    make_engine,
    register_model,
    run_input,
)
from dynamo.runtime import DistributedRuntime
from tests.utils.port_utils import allocate_port, deallocate_port

# SGLang is optional for the Rust-preprocessor case.
try:
    from dynamo.frontend.sglang_processor import SglangEngineFactory
except ImportError:
    SglangEngineFactory = None

CASES = (
    ("CapacityExhausted", "capacity.worker_overloaded", 429, None),
    ("InvalidRequest", "request.invalid_argument", 422, "Invalid request options"),
    ("UnsupportedMedia", "request.unsupported_media", 409, "Text input is required"),
    ("Internal", "runtime.internal", 502, None),
    ("RateLimited", "request.rate_limited", 429, None),
)
PRIVATE = "private-engine-secret https://internal.invalid/?credential=hidden"
TERMINAL_TYPES = {
    "CapacityExhausted": "service_unavailable",
    "InvalidRequest": "invalid_request_error",
    "UnsupportedMedia": "invalid_request_error",
    "Internal": "internal_server_error",
    "RateLimited": "rate_limit_error",
}


class StatusLikeError(Exception):
    def __init__(self):
        self.status = 415
        self.message = PRIVATE
        super().__init__(PRIVATE)


def metric_value(text, error_class, reason):
    for line in text.splitlines():
        if line.startswith("#") or "_failures_total{" not in line:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if labels.get("class") == error_class and labels.get("reason") == reason:
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def generate(request, context):
    selector = request["stop_conditions"]["max_tokens"] - 16
    if selector == 1000:
        raise HttpError(415, PRIVATE)
    if selector == 1001:
        raise StatusLikeError()
    if selector == 99:
        yield {"token_ids": [1, 1], "finish_reason": "stop"}
        return
    case_index, stage_index = divmod(selector, 3)
    error_class, reason, _, message = CASES[case_index]
    if stage_index == 2:
        # Two actual chunks and a delay cross the bounded pre-commit peek window.
        yield {"token_ids": [1, 1, 1, 1], "finish_reason": None}
        yield {"token_ids": [1, 1, 1, 1], "finish_reason": None}
        await asyncio.sleep(0.3)
    details = {"type": "message", "message": message} if message is not None else None
    # A diagnostic shaped like an HTTP envelope must not override canonical Internal.
    diagnostic = (
        json.dumps({"code": 415, "message": PRIVATE})
        if error_class == "Internal"
        else PRIVATE
    )
    raise classified_error(
        error_class, reason=reason, diagnostic=diagnostic, public_details=details
    )


async def wait_ready(client, model):
    for _ in range(300):
        try:
            response = await client.get("/v1/models")
        except httpx.ConnectError:
            await asyncio.sleep(0.1)
            continue
        if response.status_code == 200 and any(
            entry["id"] == model for entry in response.json().get("data", [])
        ):
            return
        await asyncio.sleep(0.1)
    raise AssertionError("Native frontend did not expose the registered model")


def chat_factory(processor):
    if processor == "dynamo":
        return None
    if SglangEngineFactory is None:
        raise ImportError("The SGLang processor dependencies are required")
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(
        parser.parse_args(
            [
                "--dyn-chat-processor",
                "sglang",
                "--dyn-preprocess-workers",
                "0",
                "--discovery-backend",
                "mem",
                "--request-plane",
                "tcp",
            ]
        )
    )
    config.validate()
    return SglangEngineFactory(config).chat_engine_factory


def payload(model, selector, streaming=False):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16 + selector,
        "stream": streaming,
    }


def assert_response(response, stage_index, error_class, expected):
    body = response.text
    assert (
        PRIVATE not in body
        and "credential" not in body
        and "internal.invalid" not in body
    )
    if stage_index != 2:
        assert response.status_code == expected, body
        assert response.json()["code"] == expected, body
        return
    assert response.status_code == 200, body
    frames = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    errors = [frame["error"] for frame in frames if "error" in frame]
    assert len(errors) == 1, body
    assert errors[0]["code"] == expected, body
    assert errors[0]["type"] == TERMINAL_TYPES[error_class], body
    assert any("choices" in frame for frame in frames), body
    assert not any(
        choice.get("finish_reason") is not None
        for frame in frames
        for choice in frame.get("choices", [])
    ), body
    # [DONE] terminates SSE after an error too; it is not a successful finish_reason.
    assert body.rstrip().endswith("data: [DONE]"), body


async def check_case(client, model, case_index, stage_index, processor):
    error_class, reason, expected, message = CASES[case_index]
    before = metric_value((await client.get("/metrics")).text, error_class, reason)
    response = await client.post(
        "/v1/chat/completions",
        json=payload(model, case_index * 3 + stage_index, stage_index != 0),
    )
    assert_response(response, stage_index, error_class, expected)
    if message is not None:
        assert message in response.text, response.text
    after = metric_value((await client.get("/metrics")).text, error_class, reason)
    assert after - before == 1, (error_class, reason, before, after)
    return {
        "processor": processor,
        "class": error_class,
        "reason": reason,
        "stage": ("nonstream", "precommit", "postcommit")[stage_index],
        "status": response.status_code,
        "body": response.text,
        "metric_before": before,
        "metric_after": after,
        "metric_delta": after - before,
    }


async def check_controls(client, model, default_policy):
    rows = ()
    for selector, kind in ((1000, "HttpError"), (1001, "status_like")):
        response = await client.post(
            "/v1/chat/completions", json=payload(model, selector)
        )
        expected = 415 if default_policy else 409
        assert response.status_code == expected, response.text
        assert response.json() == {
            "message": "Unsupported media type",
            "type": "Unsupported Media Type",
            "code": expected,
        }, response.text
        assert PRIVATE not in response.text, response.text
        rows = (
            *rows,
            {"kind": kind, "status": response.status_code, "body": response.text},
        )
    response = await client.post("/v1/chat/completions", json=payload(model, 99))
    assert response.status_code == 200, response.text
    return (*rows, {"kind": "success", "status": response.status_code})


async def check_anthropic(client, model, case_index, stage_index, processor):
    error_class, reason, expected, message = CASES[case_index]
    expected_type = (
        "rate_limit_error" if error_class == "RateLimited" else "invalid_request_error"
    )
    before = metric_value((await client.get("/metrics")).text, error_class, reason)
    response = await client.post(
        "/v1/messages",
        json=payload(model, case_index * 3 + stage_index, stage_index != 0),
        headers={"anthropic-version": "2023-06-01"},
    )
    body = response.text
    assert PRIVATE not in body and "credential" not in body, body
    if stage_index != 2:
        assert response.status_code == expected, body
        assert response.json()["error"]["type"] == expected_type, body
    else:
        assert response.status_code == 200, body
        frames = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        errors = [frame for frame in frames if frame.get("type") == "error"]
        assert len(errors) == 1, body
        assert errors[0]["error"]["type"] == expected_type, body
        assert any(frame.get("type") == "content_block_delta" for frame in frames), body
        assert not any(frame.get("type") == "message_stop" for frame in frames), body
    if message is not None:
        assert message in body, body
    after = metric_value((await client.get("/metrics")).text, error_class, reason)
    assert after - before == 1, (error_class, reason, before, after)
    return {
        "processor": processor,
        "protocol": "anthropic",
        "class": error_class,
        "reason": reason,
        "stage": ("nonstream", "precommit", "postcommit")[stage_index],
        "status": response.status_code,
        "body": body,
        "metric_before": before,
        "metric_after": after,
        "metric_delta": after - before,
    }


async def check_requests(
    client, model, processor, output, controls_only, default_policy
):
    await wait_ready(client, model)
    rows = ()
    if not controls_only:
        for case_index in range(len(CASES)):
            for stage_index in range(3):
                row = await check_case(
                    client, model, case_index, stage_index, processor
                )
                rows = (*rows, row)
                output.write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        for case_index, stage_index in ((1, 0), (1, 2), (2, 0), (2, 1), (4, 0), (4, 1)):
            row = await check_anthropic(
                client, model, case_index, stage_index, processor
            )
            rows = (*rows, row)
            output.write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    controls = await check_controls(client, model, default_policy)
    output.write_text(json.dumps({"errors": rows, "controls": controls}, indent=2))
    print("CAMPAIGN_PASS", processor, len(rows), "error cases", controls, flush=True)


async def campaign(
    processor, model_path, output, controls_only=False, default_policy=False
):
    namespace = "err" + uuid.uuid4().hex[:12]
    model = namespace + "-model"
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(namespace + ".backend.generate")
    worker = asyncio.ensure_future(endpoint.serve_endpoint(generate))
    frontend = None
    port = allocate_port(21000)
    try:
        runtime_config = ModelRuntimeConfig()
        runtime_config.context_length = 4096
        await register_model(
            ModelInput.Tokens,
            ModelType.Chat,
            endpoint,
            model_path,
            model_name=model,
            worker_type=WorkerType.Aggregated,
            runtime_config=runtime_config,
            ignore_weights=True,
            self_host_metadata=False,
        )
        args = EntrypointArgs(
            EngineType.Dynamic,
            http_host="127.0.0.1",
            http_port=port,
            namespace=namespace,
            chat_engine_factory=chat_factory(processor),
            migration_limit=0,
            enable_anthropic_api=True,
        )
        engine = await make_engine(runtime, args)
        frontend = asyncio.ensure_future(run_input(runtime, "http", engine))
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=20
        ) as client:
            await check_requests(
                client, model, processor, output, controls_only, default_policy
            )
    finally:
        runtime.shutdown()
        for task in (worker, frontend):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        deallocate_port(port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", choices=["dynamo", "sglang"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-controls-only", action="store_true")
    parser.add_argument("--default-policy", action="store_true")
    options = parser.parse_args()
    asyncio.run(
        campaign(
            options.processor,
            options.model_path,
            options.output,
            options.legacy_controls_only,
            options.default_policy,
        )
    )


if __name__ == "__main__":
    main()

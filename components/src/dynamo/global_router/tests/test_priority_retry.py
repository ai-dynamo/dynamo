#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for priority retry forwarding in the global router handler."""

import json
from pathlib import Path
from typing import Any

import pytest

from dynamo.common.global_router_protocol import (
    GLOBAL_ROUTER_CONTROL_FIELD,
    GLOBAL_ROUTER_RETRY_ATTEMPT_KEY,
)
from dynamo.global_router.handler import GlobalRouterHandler

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


class FakeClient:
    def __init__(
        self,
        name: str,
        outputs: list[dict[str, Any]] | None = None,
        fail_on_generate: bool = False,
        fail_before_output: bool = False,
        fail_after_output: bool = False,
    ):
        self.name = name
        self.outputs = outputs or []
        self.fail_on_generate = fail_on_generate
        self.fail_before_output = fail_before_output
        self.fail_after_output = fail_after_output
        self.calls = 0
        self.forward_calls = 0
        self.forward_requests: list[dict[str, Any]] = []

    async def generate(self, request: dict[str, Any]):
        self.calls += 1
        if self.fail_on_generate:
            raise RuntimeError(f"{self.name} generate failed")

        async def stream():
            if self.fail_before_output:
                raise RuntimeError(f"{self.name} stream failed")
            for output in self.outputs:
                yield output
                if self.fail_after_output:
                    raise RuntimeError(f"{self.name} stream failed after output")

        return stream()

    async def forward(
        self,
        request: dict[str, Any],
        response_connection_info: dict[str, Any],
        context: Any | None = None,
    ):
        self.forward_calls += 1
        self.forward_requests.append(request)
        if self.fail_on_generate:
            raise RuntimeError(f"{self.name} forward failed")


async def _collect_outputs(generator) -> list[dict[str, Any]]:
    return [output async for output in generator]


class FakeContext:
    def __init__(self):
        self._connection_info = {
            "transport": "tcp",
            "info": json.dumps({"subject": "frontend-response-subject"}),
        }

    def id(self) -> str:
        return "request-123"

    def connection_info(self) -> dict[str, Any]:
        return self._connection_info


def _write_config(tmp_dir: Path, config_data: dict[str, Any]) -> Path:
    config_path = tmp_dir / "config.json"
    config_path.write_text(json.dumps(config_data))
    return config_path


def _disagg_config(
    enable_priority_retry: bool = True,
    prefill_pool_priorities: list[int] | None = None,
    decode_pool_priorities: list[int] | None = None,
) -> dict[str, Any]:
    config = {
        "mode": "disagg",
        "enable_priority_retry": enable_priority_retry,
        "num_prefill_pools": 3,
        "num_decode_pools": 3,
        "prefill_pool_dynamo_namespaces": [
            "prefill-fast",
            "prefill-mid",
            "prefill-slow",
        ],
        "decode_pool_dynamo_namespaces": [
            "decode-fast",
            "decode-mid",
            "decode-slow",
        ],
        "prefill_pool_selection_strategy": {
            "isl_min": 0,
            "isl_max": 32000,
            "isl_resolution": 1,
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 1,
            "prefill_pool_mapping": [[2]],
        },
        "decode_pool_selection_strategy": {
            "context_length_min": 0,
            "context_length_max": 32000,
            "context_length_resolution": 1,
            "itl_min": 10,
            "itl_max": 500,
            "itl_resolution": 1,
            "decode_pool_mapping": [[2]],
        },
    }
    if prefill_pool_priorities is not None:
        config["prefill_pool_priorities"] = prefill_pool_priorities
    if decode_pool_priorities is not None:
        config["decode_pool_priorities"] = decode_pool_priorities
    return config


def _agg_config() -> dict[str, Any]:
    return {
        "mode": "agg",
        "enable_priority_retry": True,
        "num_agg_pools": 3,
        "agg_pool_dynamo_namespaces": ["agg-slow", "agg-fast", "agg-mid"],
        "agg_pool_priorities": [10, 0, 5],
        "agg_pool_selection_strategy": {
            "ttft_min": 10,
            "ttft_max": 3000,
            "ttft_resolution": 1,
            "itl_min": 5,
            "itl_max": 200,
            "itl_resolution": 1,
            "agg_pool_mapping": [[0]],
        },
    }


def _handler(
    config_path: Path,
    enable_delegated_response_stream: bool = True,
) -> GlobalRouterHandler:
    return GlobalRouterHandler(
        runtime=object(),
        config_path=str(config_path),
        model_name="test-model",
        enable_delegated_response_stream=enable_delegated_response_stream,
    )


@pytest.mark.asyncio
async def test_prefill_retries_faster_pools_until_success(tmp_path):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    fast = FakeClient("prefill-fast", outputs=[{"pool": "prefill-fast"}])
    mid = FakeClient("prefill-mid", fail_before_output=True)
    slow = FakeClient("prefill-slow", fail_on_generate=True)
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": mid,
        "prefill-slow": slow,
    }

    outputs = await _collect_outputs(handler.handle_prefill({"token_ids": [1, 2, 3]}))

    assert outputs == [{"pool": "prefill-fast"}]
    assert slow.calls == 1
    assert mid.calls == 1
    assert fast.calls == 1


@pytest.mark.asyncio
async def test_decode_retries_using_custom_pool_priorities(tmp_path):
    config = _disagg_config(
        decode_pool_priorities=[5, 0, 10],
    )
    handler = _handler(_write_config(tmp_path, config))
    fast = FakeClient("decode-fast", outputs=[{"pool": "decode-fast"}])
    mid = FakeClient("decode-mid", outputs=[{"pool": "decode-mid"}])
    slow = FakeClient("decode-slow", fail_on_generate=True)
    handler.decode_clients = {
        "decode-fast": fast,
        "decode-mid": mid,
        "decode-slow": slow,
    }

    outputs = await _collect_outputs(handler.handle_decode({"token_ids": [1, 2, 3]}))

    assert outputs == [{"pool": "decode-fast"}]
    assert slow.calls == 1
    assert fast.calls == 1
    assert mid.calls == 0


@pytest.mark.asyncio
async def test_priority_retry_disabled_raises_first_failure(tmp_path):
    handler = _handler(
        _write_config(tmp_path, _disagg_config(enable_priority_retry=False))
    )
    fast = FakeClient("prefill-fast", outputs=[{"pool": "prefill-fast"}])
    slow = FakeClient("prefill-slow", fail_on_generate=True)
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": slow,
    }

    with pytest.raises(RuntimeError, match="prefill-slow generate failed"):
        await _collect_outputs(handler.handle_prefill({"token_ids": [1, 2, 3]}))

    assert slow.calls == 1
    assert fast.calls == 0


@pytest.mark.asyncio
async def test_delegation_disabled_uses_relay_when_priority_retry_has_fallback(
    tmp_path,
):
    handler = _handler(
        _write_config(tmp_path, _disagg_config()),
        enable_delegated_response_stream=False,
    )
    fast = FakeClient("prefill-fast", outputs=[{"pool": "prefill-fast"}])
    mid = FakeClient("prefill-mid", fail_before_output=True)
    slow = FakeClient("prefill-slow", fail_on_generate=True)
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": mid,
        "prefill-slow": slow,
    }

    outputs = await _collect_outputs(
        handler.handle_prefill({"token_ids": [1, 2, 3]}, context=FakeContext())
    )

    assert outputs == [{"pool": "prefill-fast"}]
    assert slow.calls == 1
    assert mid.calls == 1
    assert fast.calls == 1
    assert slow.forward_calls == 0
    assert mid.forward_calls == 0
    assert fast.forward_calls == 0


@pytest.mark.asyncio
async def test_delegated_priority_retry_requires_frontend_adapter(tmp_path):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    handler.prefill_clients = {
        "prefill-fast": FakeClient("prefill-fast"),
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": FakeClient("prefill-slow"),
    }

    with pytest.raises(RuntimeError, match="dyn-routed-engine-adapter=global-router"):
        await _collect_outputs(
            handler.handle_prefill({"token_ids": [1, 2, 3]}, context=FakeContext())
        )


@pytest.mark.asyncio
async def test_delegated_retry_metadata_is_ignored_when_delegation_disabled(tmp_path):
    handler = _handler(
        _write_config(tmp_path, _disagg_config()),
        enable_delegated_response_stream=False,
    )
    slow = FakeClient("prefill-slow", outputs=[{"pool": "prefill-slow"}])
    handler.prefill_clients = {
        "prefill-fast": FakeClient("prefill-fast"),
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": slow,
    }
    request = {
        "token_ids": [1, 2, 3],
        "routing": {GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 1},
    }

    outputs = await _collect_outputs(
        handler.handle_prefill(request, context=FakeContext())
    )

    assert outputs == [{"pool": "prefill-slow"}]
    assert slow.calls == 1
    assert slow.forward_calls == 0


@pytest.mark.asyncio
async def test_delegated_response_uses_forward_when_priority_retry_has_no_fallback(
    tmp_path,
):
    handler = _handler(
        _write_config(tmp_path, _disagg_config(enable_priority_retry=False))
    )
    slow = FakeClient("prefill-slow", outputs=[{"pool": "prefill-slow"}])
    handler.prefill_clients = {
        "prefill-fast": FakeClient("prefill-fast"),
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": slow,
    }

    outputs = await _collect_outputs(
        handler.handle_prefill({"token_ids": [1, 2, 3]}, context=FakeContext())
    )

    assert outputs == []
    assert slow.forward_calls == 1
    assert slow.calls == 0


@pytest.mark.asyncio
async def test_delegated_retry_attempt_uses_selected_priority_pool(tmp_path):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    fast = FakeClient("prefill-fast")
    mid = FakeClient("prefill-mid")
    slow = FakeClient("prefill-slow")
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": mid,
        "prefill-slow": slow,
    }
    request = {
        "token_ids": [1, 2, 3],
        "routing": {GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 1},
    }

    outputs = await _collect_outputs(
        handler.handle_prefill(request, context=FakeContext())
    )

    assert outputs == []
    assert slow.forward_calls == 0
    assert mid.forward_calls == 1
    assert fast.forward_calls == 0
    assert mid.forward_requests == [request]


@pytest.mark.asyncio
async def test_delegated_retry_attempt_returns_retry_control_on_forward_failure(
    tmp_path,
):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    fast = FakeClient("prefill-fast")
    mid = FakeClient("prefill-mid")
    slow = FakeClient("prefill-slow", fail_on_generate=True)
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": mid,
        "prefill-slow": slow,
    }
    request = {
        "token_ids": [1, 2, 3],
        "routing": {GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 0},
    }

    outputs = await _collect_outputs(
        handler.handle_prefill(request, context=FakeContext())
    )

    assert len(outputs) == 1
    control = outputs[0][GLOBAL_ROUTER_CONTROL_FIELD]
    assert control["action"] == "retry"
    assert control["retry_attempt"] == 0
    assert control["next_retry_attempt"] == 1
    assert control["failed_namespace"] == "prefill-slow"
    assert control["next_namespace"] == "prefill-mid"
    assert slow.forward_calls == 1
    assert mid.forward_calls == 0
    assert fast.forward_calls == 0


@pytest.mark.asyncio
async def test_delegated_retry_attempt_returns_exhausted_control(tmp_path):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    fast = FakeClient("prefill-fast", fail_on_generate=True)
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": FakeClient("prefill-slow"),
    }
    request = {
        "token_ids": [1, 2, 3],
        "routing": {GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 2},
    }

    outputs = await _collect_outputs(
        handler.handle_prefill(request, context=FakeContext())
    )

    assert len(outputs) == 1
    control = outputs[0][GLOBAL_ROUTER_CONTROL_FIELD]
    assert control["action"] == "exhausted"
    assert control["retry_attempt"] == 2
    assert control["failed_namespace"] == "prefill-fast"
    assert "no priority retry pools remain" in control["error"]
    assert fast.forward_calls == 1


@pytest.mark.asyncio
async def test_delegated_retry_attempt_returns_exhausted_control_when_attempt_exceeds_order(
    tmp_path,
):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    handler.prefill_clients = {
        "prefill-fast": FakeClient("prefill-fast"),
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": FakeClient("prefill-slow"),
    }
    request = {
        "token_ids": [1, 2, 3],
        "routing": {GLOBAL_ROUTER_RETRY_ATTEMPT_KEY: 3},
    }

    outputs = await _collect_outputs(
        handler.handle_prefill(request, context=FakeContext())
    )

    assert len(outputs) == 1
    control = outputs[0][GLOBAL_ROUTER_CONTROL_FIELD]
    assert control["action"] == "exhausted"
    assert control["retry_attempt"] == 3
    assert "exceeds retry order" in control["error"]


@pytest.mark.asyncio
async def test_failure_after_streaming_starts_is_not_retried(tmp_path):
    handler = _handler(_write_config(tmp_path, _disagg_config()))
    fast = FakeClient("prefill-fast", outputs=[{"pool": "prefill-fast"}])
    slow = FakeClient(
        "prefill-slow",
        outputs=[{"pool": "prefill-slow"}],
        fail_after_output=True,
    )
    handler.prefill_clients = {
        "prefill-fast": fast,
        "prefill-mid": FakeClient("prefill-mid"),
        "prefill-slow": slow,
    }

    with pytest.raises(RuntimeError, match="stream failed after output"):
        await _collect_outputs(handler.handle_prefill({"token_ids": [1, 2, 3]}))

    assert slow.calls == 1
    assert fast.calls == 0


@pytest.mark.asyncio
async def test_agg_retries_with_custom_pool_priorities(tmp_path):
    handler = _handler(_write_config(tmp_path, _agg_config()))
    slow = FakeClient("agg-slow", fail_on_generate=True)
    fast = FakeClient("agg-fast", outputs=[{"pool": "agg-fast"}])
    mid = FakeClient("agg-mid", outputs=[{"pool": "agg-mid"}])
    handler.agg_clients = {
        "agg-slow": slow,
        "agg-fast": fast,
        "agg-mid": mid,
    }

    outputs = await _collect_outputs(handler.handle_generate({"token_ids": [1]}))

    assert outputs == [{"pool": "agg-mid"}]
    assert slow.calls == 1
    assert mid.calls == 1
    assert fast.calls == 0

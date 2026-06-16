#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for priority retry forwarding in the global router handler."""

import json
from pathlib import Path
from typing import Any

import pytest

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


async def _collect_outputs(generator) -> list[dict[str, Any]]:
    return [output async for output in generator]


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


def _handler(config_path: Path) -> GlobalRouterHandler:
    return GlobalRouterHandler(
        runtime=object(),
        config_path=str(config_path),
        model_name="test-model",
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

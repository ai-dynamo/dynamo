# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter contract tests; native TokenSpeed/GPU processes are replaced at the boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed import args, disagg, llm_engine
from dynamo.tokenspeed.disagg import BOOTSTRAP_HOST_ENV
from dynamo.tokenspeed.llm_engine import TokenspeedLLMEngine

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def server_args(**overrides):
    values = dict(
        model="test-model",
        served_model_name="test-model",
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        disaggregation_bootstrap_port=9000,
        host="127.0.0.1",
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
        prefix_granularity=64,
        max_model_len=4096,
        max_total_tokens=4096,
        enable_prefix_caching=True,
        kv_events_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def native_engine(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    native = Mock()
    native.scheduler_info = {
        "max_model_len": 8192,
        "max_total_num_tokens": 8192,
        "max_num_seqs": 32,
    }
    constructor = Mock(return_value=native)
    monkeypatch.setattr(llm_engine, "_tokenspeed_engine_cls", lambda: constructor)
    monkeypatch.setattr(llm_engine, "_generate_req_input_cls", lambda: SimpleNamespace)
    return native, constructor


@pytest.mark.parametrize(
    "mode,component,expected",
    [
        ("null", "backend", DisaggregationMode.AGGREGATED),
        ("prefill", "prefill", DisaggregationMode.PREFILL),
        ("decode", "backend", DisaggregationMode.DECODE),
    ],
)
@pytest.mark.parametrize("explicit_endpoint", [False, True])
async def test_cli_registers_worker_role(
    monkeypatch, mode, component, expected, explicit_endpoint
):
    class NativeArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model")
            parser.add_argument("--disaggregation-mode", default="null")

        @staticmethod
        def from_cli_args(parsed):
            return server_args(**vars(parsed))

    monkeypatch.setattr(args, "_server_args_cls", lambda: NativeArgs)
    cli = [
        "--model",
        "test-model",
        "--disaggregation-mode",
        mode,
        "--namespace",
        "test",
    ]
    if explicit_endpoint:
        cli += ["--endpoint", "dyn://custom.worker.generate"]
    _, config = await TokenspeedLLMEngine.from_args(cli)
    assert config.disaggregation_mode == expected
    assert config.component == ("worker" if explicit_endpoint else component)
    assert config.namespace == ("custom" if explicit_endpoint else "test")
    assert config.endpoint == "generate"


@pytest.mark.parametrize("mode", ["prefill", "decode", "null"])
@pytest.mark.parametrize("legacy_block_size", [False, True])
async def test_registration_advertises_prefill_bootstrap_only(
    monkeypatch, native_engine, mode, legacy_block_size
):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "prefill.example")
    native_args = server_args(disaggregation_mode=mode)
    if legacy_block_size:
        native_args.block_size = native_args.prefix_granularity
        del native_args.prefix_granularity
    engine = TokenspeedLLMEngine(native_args)
    try:
        registration = await engine.start(worker_id=1)
        assert registration.llm.context_length == 8192
        assert registration.llm.total_kv_blocks == 128
        assert registration.llm.kv_cache_block_size == 64
        assert registration.llm.bootstrap_host == (
            "prefill.example" if mode == "prefill" else None
        )
        assert registration.llm.bootstrap_port == (9000 if mode == "prefill" else None)
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
async def test_generate_forwards_handoff_and_preserves_decode_budget(
    native_engine, mode
):
    native, _ = native_engine
    inputs = []

    async def generate(obj):
        inputs.append(obj)
        yield {
            "output_ids": [10, 11, 20],
            "meta_info": {"completion_tokens": 1, "finish_reason": {"type": "length"}},
        }

    native.tokenizer_manager.generate_request = generate
    engine = TokenspeedLLMEngine(
        server_args(disaggregation_mode=mode, host="prefill.example")
    )
    await engine.start(worker_id=1)
    request = {
        "token_ids": [10, 11],
        "stop_conditions": {"max_tokens": 30, "min_tokens": 10},
        "bootstrap_info": {
            "bootstrap_host": "selected-prefill.example",
            "bootstrap_port": 9010,
            "bootstrap_room": 0,
            "handoff_id": "1cfe6c6e-d8c8-4bea-a0a6-ccde2bd97c54",
        },
    }
    context = SimpleNamespace(id=lambda: "request-1")
    try:
        chunks = [chunk async for chunk in engine.generate(request, context)]
        if mode == "prefill":
            assert chunks[0] == {
                "token_ids": [],
                "disaggregated_params": request["bootstrap_info"],
            }
            assert chunks[1]["token_ids"] == [20]
        else:
            assert chunks[0]["token_ids"] == [20]
        assert inputs[0].rid == "request-1"
        assert inputs[0].bootstrap_host == "selected-prefill.example"
        assert inputs[0].bootstrap_port == 9010
        assert inputs[0].bootstrap_room == 0
        assert not hasattr(inputs[0], "handoff_id")
        if mode == "prefill":
            assert inputs[0].sampling_params["max_new_tokens"] == 1
            assert "min_new_tokens" not in inputs[0].sampling_params
        else:
            assert inputs[0].sampling_params["max_new_tokens"] == 30
            assert inputs[0].sampling_params["min_new_tokens"] == 10
        assert engine._active_rids_by_context == {}
    finally:
        await engine.cleanup()


@pytest.mark.parametrize(
    "bootstrap",
    [
        None,
        {},
        {"bootstrap_host": "", "bootstrap_port": 9000, "bootstrap_room": 1},
        {"bootstrap_host": "p", "bootstrap_port": 65536, "bootstrap_room": 1},
        {"bootstrap_host": "p", "bootstrap_port": 9000, "bootstrap_room": -1},
        {"bootstrap_host": "p", "bootstrap_port": 9000, "bootstrap_room": True},
        {"bootstrap_host": "p", "bootstrap_port": 9000, "bootstrap_room": 2**64},
    ],
)
async def test_bad_handoff_rejected_before_native_request(native_engine, bootstrap):
    native, _ = native_engine
    engine = TokenspeedLLMEngine(server_args(disaggregation_mode="decode"))
    await engine.start(worker_id=1)
    try:
        with pytest.raises(InvalidArgument, match="bootstrap"):
            async for _ in engine.generate(
                {"token_ids": [1], "bootstrap_info": bootstrap},
                SimpleNamespace(id=lambda: "bad"),
            ):
                pass
        native.tokenizer_manager.generate_request.assert_not_called()
    finally:
        await engine.cleanup()


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {"mapping": SimpleNamespace(attn=SimpleNamespace(dp_size=2))},
            "attention DP=1",
        ),
        ({"disaggregation_transfer_backend": "fake"}, "Mooncake"),
        ({"prefix_granularity": 0}, "positive --prefix-granularity"),
        ({"prefix_granularity": -1}, "positive --prefix-granularity"),
    ],
)
async def test_unsupported_disagg_fails_before_start(native_engine, overrides, message):
    _, constructor = native_engine
    engine = TokenspeedLLMEngine(server_args(disaggregation_mode="decode", **overrides))
    with pytest.raises(ValueError, match=message):
        await engine.start(worker_id=1)
    constructor.assert_not_called()
    await engine.cleanup()


async def test_kv_sources_use_unique_ipc_endpoints_and_cleanup(native_engine):
    engines = [
        TokenspeedLLMEngine(
            server_args(kv_events_config='{"enable_kv_cache_events":true}')
        )
        for _ in range(2)
    ]
    paths = []
    try:
        for engine in engines:
            await engine.start(worker_id=1)
            [source] = await engine.kv_event_sources()
            native_config = json.loads(engine.server_args.kv_events_config)
            assert source.endpoint == native_config["endpoint"]
            assert source.dp_rank == 0
            assert source.endpoint.startswith("ipc://")
            paths.append(Path(source.endpoint.removeprefix("ipc://")).parent)
            assert paths[-1].is_dir()
        assert paths[0] != paths[1]
    finally:
        for engine in engines:
            await engine.cleanup()
            await engine.cleanup()
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize(
    "native_config,expected",
    [
        ({"endpoint": "tcp://*:19000"}, "tcp://127.0.0.1:19000"),
        ({"endpoint": "ipc:///test-events", "publisher": None}, "ipc:///test-events"),
    ],
)
async def test_kv_source_matches_native_bind_and_topic(
    native_engine, native_config, expected
):
    native_config.update(enable_kv_cache_events=True, topic="kv")
    engine = TokenspeedLLMEngine(
        server_args(kv_events_config=json.dumps(native_config))
    )
    try:
        await engine.start(worker_id=1)
        [source] = await engine.kv_event_sources()
        assert source.endpoint == expected
        assert source.topic == "kv"
        assert json.loads(engine.server_args.kv_events_config) == native_config
    finally:
        await engine.cleanup()


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_events_config": None},
        {"kv_events_config": '{"enable_kv_cache_events":false}'},
        {"kv_events_config": '{"enable_kv_cache_events":true,"publisher":"null"}'},
        {
            "kv_events_config": '{"enable_kv_cache_events":true}',
            "enable_prefix_caching": False,
        },
    ],
)
async def test_disabled_kv_events_have_no_source(native_engine, overrides, caplog):
    native, constructor = native_engine
    native_configs = []

    def construct(*, server_args):
        native_configs.append(json.loads(server_args.kv_events_config or "{}"))
        return native

    constructor.side_effect = construct
    engine = TokenspeedLLMEngine(server_args(**overrides))
    try:
        await engine.start(worker_id=1)
        assert await engine.kv_event_sources() == []
        if overrides.get("enable_prefix_caching") is False:
            assert native_configs[0]["enable_kv_cache_events"] is False
            assert native_configs[0]["publisher"] == "null"
            assert "enable_prefix_caching=False" in caplog.text
            assert "KV events" in caplog.text
        else:
            assert caplog.text == ""
    finally:
        await engine.cleanup()


async def test_connecting_native_publisher_is_rejected(native_engine):
    _, constructor = native_engine
    engine = TokenspeedLLMEngine(
        server_args(
            kv_events_config=json.dumps(
                {
                    "enable_kv_cache_events": True,
                    "endpoint": "tcp://127.0.0.1:19000",
                }
            )
        )
    )
    with pytest.raises(ValueError, match="binding endpoint"):
        await engine.start(worker_id=1)
    constructor.assert_not_called()
    await engine.cleanup()


async def test_failed_start_cleans_up_event_endpoint(native_engine):
    _, constructor = native_engine
    constructor.side_effect = RuntimeError("native startup failed")
    engine = TokenspeedLLMEngine(
        server_args(kv_events_config='{"enable_kv_cache_events":true}')
    )
    with pytest.raises(RuntimeError, match="native startup failed"):
        await engine.start(worker_id=1)
    path = Path(
        json.loads(engine.server_args.kv_events_config)["endpoint"].removeprefix(
            "ipc://"
        )
    ).parent
    await engine.cleanup()
    assert not path.exists()
    assert await engine.kv_event_sources() == []


@pytest.mark.parametrize("enabled", ["false", "true", None, 1])
async def test_non_boolean_event_flag_rejected_before_start(native_engine, enabled):
    _, constructor = native_engine
    engine = TokenspeedLLMEngine(
        server_args(
            kv_events_config=json.dumps(
                {
                    "enable_kv_cache_events": enabled,
                }
            )
        )
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        await engine.start(worker_id=1)
    constructor.assert_not_called()
    await engine.cleanup()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "::1", "localhost"])
@pytest.mark.parametrize("address", ["127.0.1.1", "0.0.0.0"])
async def test_prefill_rejects_local_only_derived_advertisement(
    monkeypatch, native_engine, host, address
):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    monkeypatch.setattr(disagg.socket, "gethostbyname", lambda _: address)
    _, constructor = native_engine
    engine = TokenspeedLLMEngine(server_args(disaggregation_mode="prefill", host=host))
    with pytest.raises(ValueError, match=BOOTSTRAP_HOST_ENV):
        await engine.start(worker_id=1)
    constructor.assert_not_called()
    await engine.cleanup()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "::1", "localhost"])
async def test_prefill_derives_non_loopback_advertisement(
    monkeypatch, native_engine, host
):
    monkeypatch.setattr(disagg.socket, "gethostbyname", lambda _: "192.0.2.10")
    engine = TokenspeedLLMEngine(server_args(disaggregation_mode="prefill", host=host))
    try:
        config = await engine.start(worker_id=1)
        assert config.llm.bootstrap_host == "192.0.2.10"
    finally:
        await engine.cleanup()


async def test_explicit_loopback_prefill_allowed_for_single_host(
    monkeypatch, native_engine
):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "127.0.0.1")
    engine = TokenspeedLLMEngine(server_args(disaggregation_mode="prefill"))
    try:
        config = await engine.start(worker_id=1)
        assert config.llm.bootstrap_host == "127.0.0.1"
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("explicit_null", [False, True])
async def test_prefill_creates_handoff_when_router_has_no_bootstrap_endpoint(
    monkeypatch, native_engine, explicit_null
):
    """The synchronous path supplies the same fresh room to prefill and decode."""
    native, _ = native_engine
    inputs = []
    rooms = Mock(side_effect=[17, 23])
    monkeypatch.setattr(
        disagg, "secrets", SimpleNamespace(randbits=rooms), raising=False
    )

    async def generate(obj):
        inputs.append(obj)
        yield {"output_ids": [20], "meta_info": {"completion_tokens": 1}}

    native.tokenizer_manager.generate_request = generate
    engine = TokenspeedLLMEngine(
        server_args(disaggregation_mode="prefill", host="prefill.example")
    )
    registration = await engine.start(worker_id=1)
    try:
        for index, room in enumerate([17, 23]):
            request = {"token_ids": [10, 11], "stop_conditions": {"max_tokens": 30}}
            if explicit_null:
                request["bootstrap_info"] = None
            stream = engine.generate(request, SimpleNamespace(id=lambda: "fallback"))
            try:
                first = await anext(stream)
                assert first["token_ids"] == []
                handoff = first["disaggregated_params"]
                assert handoff == {
                    "bootstrap_host": registration.llm.bootstrap_host,
                    "bootstrap_port": registration.llm.bootstrap_port,
                    "bootstrap_room": room,
                }
                assert len(inputs) == index
                chunks = [chunk async for chunk in stream]
                assert chunks[0]["token_ids"] == [20]
                for key, value in handoff.items():
                    assert getattr(inputs[index], key) == value
                assert (
                    disagg.bootstrap_kwargs(
                        {"bootstrap_info": handoff}, DisaggregationMode.DECODE
                    )
                    == handoff
                )
                assert engine._active_rids_by_context == {}
            finally:
                await stream.aclose()
        assert rooms.call_count == 2
        rooms.assert_called_with(63)
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("router_bootstrap", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_prefill_cancellation_after_handoff_prevents_submission(
    native_engine, router_bootstrap, cancel
):
    """An abort before native admission survives resuming the handoff stream."""
    native, _ = native_engine
    submitted = []

    async def generate(obj):
        submitted.append(obj.rid)
        yield {"output_ids": [20], "meta_info": {"finish_reason": {"type": "length"}}}

    native.tokenizer_manager.generate_request = generate
    engine = TokenspeedLLMEngine(
        server_args(disaggregation_mode="prefill", host="prefill.example")
    )
    registration = await engine.start(worker_id=1)
    request = {"token_ids": [10, 11]}
    if router_bootstrap:
        request["bootstrap_info"] = {
            "bootstrap_host": registration.llm.bootstrap_host,
            "bootstrap_port": registration.llm.bootstrap_port,
            "bootstrap_room": 1,
        }
    context = SimpleNamespace(id=lambda: "handoff-cancel")
    stream = engine.generate(request, context)
    try:
        first = await anext(stream)
        assert first["disaggregated_params"]
        assert submitted == []
        if cancel:
            await engine.abort(context)
            await engine.abort(context)
        chunks = [chunk async for chunk in stream]
        assert submitted == ([] if cancel else ["handoff-cancel"])
        assert chunks[-1]["finish_reason"] == ("cancelled" if cancel else "length")
        assert engine._active_rids_by_context == {}
        # Reusing the context must not inherit cancellation from the old stream.
        chunks = [chunk async for chunk in engine.generate(request, context)]
        assert chunks[-1]["finish_reason"] == "length"
        assert submitted[-1] == "handoff-cancel"
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_cancelled_native_stream_preserves_cancelled_finish(native_engine):
    native, _ = native_engine
    aborted = set()
    native.tokenizer_manager.abort_request.side_effect = aborted.add

    async def generate(obj):
        yield {"output_ids": [20]}
        assert obj.rid in aborted
        yield {
            "output_ids": [],
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": "AbortReq from client",
                    "err_type": 522,
                }
            },
        }

    native.tokenizer_manager.generate_request = generate
    engine = TokenspeedLLMEngine(server_args())
    await engine.start(worker_id=1)
    context = SimpleNamespace(id=lambda: "stream-cancel")
    stream = engine.generate({"token_ids": [10]}, context)
    try:
        assert (await anext(stream))["token_ids"] == [20]
        await engine.abort(context)
        chunks = [chunk async for chunk in stream]
        assert chunks[-1]["finish_reason"] == "cancelled"
        assert engine._active_rids_by_context == {}
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_kv_replay_rejected_before_native_start(native_engine, tmp_path):
    _, constructor = native_engine
    engine = TokenspeedLLMEngine(
        server_args(
            kv_events_config=json.dumps(
                {
                    "enable_kv_cache_events": True,
                    "endpoint": f"ipc://{tmp_path}/events",
                    "replay_endpoint": f"ipc://{tmp_path}/replay",
                }
            )
        )
    )
    try:
        with pytest.raises(ValueError, match="replay_endpoint"):
            await engine.start(worker_id=1)
        constructor.assert_not_called()
    finally:
        await engine.cleanup()

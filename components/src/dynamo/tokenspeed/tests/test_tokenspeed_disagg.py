# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter contract tests. The native TokenSpeed pieces (tokenspeed_engine's
frontend, the msgpack encoder and the headless engine launcher) are replaced
at the boundary by fakes that follow the wire's semantics."""

import asyncio
import itertools
import json
import signal
import threading
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

HANDSHAKE_PORT = "45678"


def server_args(**overrides):
    values = dict(
        model="test-model",
        served_model_name="test-model",
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        disaggregation_bootstrap_port=9000,
        host="127.0.0.1",
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=30500,
        zmq_engine_index=0,
        nnodes=1,
        node_rank=0,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1), nnodes=1),
        max_num_seqs=32,
        chunked_prefill_size=8192,
        enable_prefix_caching=True,
        kv_events_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def wire_output(output_ids, finish_reason=None, finish_details=None, **fields):
    chunk = {
        "output_ids": output_ids,
        "finish_reason": finish_reason,
        "finish_details": finish_details,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }
    chunk.update(fields)
    return chunk


def finished(output_ids, **fields):
    return wire_output(output_ids, "length", {"type": "length"}, **fields)


class FakeProcess:
    def __init__(self, argv, returncode=None):
        self.argv = argv
        self.pid = -1
        self.returncode = returncode
        self._exited = threading.Event()
        if returncode is not None:
            self._exited.set()

    def poll(self):
        return self.returncode

    def wait(self):
        self._exited.wait()
        return self.returncode

    def exit(self, code):
        self.returncode = code
        self._exited.set()


class FakeFrontend:
    """tokenspeed_engine.Frontend, connected to scripted engines."""

    _CLOSED = object()

    def __init__(self, native):
        self.native = native
        self.outputs = asyncio.Queue()
        self.unfinished = set()

    def engines(self):
        return [dict(engine) for engine in self.native.engines]

    async def generate(self, frames):
        request = self.native.decode(frames)
        if self.native.submit_gate is not None:
            await self.native.submit_gate.wait()
        self.native.submitted.append(request)
        self.unfinished.add(request.rid)
        for item in self.native.respond(request):
            self.deliver(request.rid, item)

    def deliver(self, rid, item):
        if isinstance(item, dict) and item["finish_reason"] is not None:
            self.unfinished.discard(rid)
        self.outputs.put_nowait((rid, item))

    def in_flight(self):
        return [len(self.unfinished)]

    def close(self):
        self.outputs.put_nowait(self._CLOSED)

    async def next_outputs(self):
        batch = [await self.outputs.get()]
        while not self.outputs.empty():
            batch.append(self.outputs.get_nowait())
        if self._CLOSED in batch:
            raise RuntimeError("the frontend is closed")
        return batch

    async def abort(self, rid):
        self.native.aborted.append(rid)
        if rid not in self.unfinished:
            return False
        self.deliver(
            rid,
            wire_output(
                [],
                "abort",
                {"type": "abort", "message": "AbortReq from client", "err_type": 522},
            ),
        )
        return True


class FakeNative:
    """The native side: tokenspeed_engine, io_struct and the launcher."""

    def __init__(self):
        self.engines = [
            {
                "index": 0,
                "max_model_len": 8192,
                "data_parallel_rank": 0,
                "data_parallel_size": 1,
                "kv_cache_size_tokens": 8192,
                "request_types": ["generate"],
                "disaggregation_mode": None,
                "bootstrap_host": None,
                "bootstrap_port": None,
            }
        ]
        self.respond = lambda request: [finished([20], completion_tokens=1)]
        self.binds = []
        self.launches = []
        self.launcher_exit_code = None
        self.submitted = []
        self.aborted = []
        self.submit_gate = None
        self.frontend = None
        self._registered = threading.Event()
        self._registered.set()
        self._encoded = {}
        self._ids = itertools.count()
        self.module = SimpleNamespace(FrontendListener=SimpleNamespace(bind=self.bind))
        self.io_struct = SimpleNamespace(
            MsgpackEncoder=lambda: SimpleNamespace(encode=self.encode),
            TokenizedGenerateReqInput=SimpleNamespace,
        )

    def never_register(self):
        self._registered.clear()

    def bind(self, **kwargs):
        self.binds.append(kwargs)
        address = kwargs["handshake_address"].replace(":0", f":{HANDSHAKE_PORT}")

        def wait():
            self._registered.wait()
            self.frontend = FakeFrontend(self)
            return self.frontend

        pending = SimpleNamespace(wait=wait)
        return SimpleNamespace(handshake_address=address, accept=lambda: pending)

    def encode(self, request):
        key = next(self._ids)
        self._encoded[key] = request
        return [str(key).encode()]

    def decode(self, frames):
        return self._encoded[int(frames[0])]

    def start_launcher(self, argv):
        process = FakeProcess(argv, self.launcher_exit_code)
        self.launches.append(process)
        return process

    @staticmethod
    def stop_launcher(process):
        process.exit(0)

    def release(self):
        self._registered.set()
        for process in self.launches:
            process._exited.set()


@pytest.fixture
def native(monkeypatch):
    monkeypatch.delenv(BOOTSTRAP_HOST_ENV, raising=False)
    fake = FakeNative()
    monkeypatch.setattr(llm_engine, "_tokenspeed_engine_module", lambda: fake.module)
    monkeypatch.setattr(llm_engine, "_io_struct_module", lambda: fake.io_struct)
    monkeypatch.setattr(llm_engine, "_sampling_params_cls", lambda: dict)
    monkeypatch.setattr(llm_engine, "_start_launcher", fake.start_launcher)
    monkeypatch.setattr(llm_engine, "_stop_launcher", fake.stop_launcher)
    yield fake
    fake.release()


def make_engine(kv_block_size=None, **overrides):
    return TokenspeedLLMEngine(
        server_args(**overrides),
        engine_argv=["--model", "test-model"],
        kv_block_size=kv_block_size,
    )


def context(rid):
    return SimpleNamespace(id=lambda: rid)


async def drain(stream):
    return [chunk async for chunk in stream]


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
        "--kv-block-size",
        "64",
    ]
    if explicit_endpoint:
        cli += ["--endpoint", "dyn://custom.worker.generate"]
    engine, config = await TokenspeedLLMEngine.from_args(cli)
    assert config.disaggregation_mode == expected
    assert config.component == ("worker" if explicit_endpoint else component)
    assert config.namespace == ("custom" if explicit_endpoint else "test")
    assert config.endpoint == "generate"
    # The launcher gets TokenSpeed's own flags only.
    assert engine.engine_argv == [
        "--model",
        "test-model",
        "--disaggregation-mode",
        mode,
    ]
    assert engine.kv_block_size == 64


@pytest.mark.parametrize("block_size", ["0", "-16"])
def test_cli_rejects_non_positive_kv_block_size(monkeypatch, block_size):
    class NativeArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model")

        @staticmethod
        def from_cli_args(parsed):
            return server_args(**vars(parsed))

    monkeypatch.setattr(args, "_server_args_cls", lambda: NativeArgs)
    with pytest.raises(ValueError, match="--kv-block-size"):
        args.parse_args(["--model", "m", "--kv-block-size", block_size])


@pytest.mark.parametrize(
    "mode,routing",
    [
        ("null", "least_in_flight"),
        ("prefill", "bootstrap_room"),
        ("decode", "bootstrap_room"),
    ],
)
async def test_start_binds_frontend_and_launches_headless_engine(
    monkeypatch, native, mode, routing
):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "prefill.example")
    engine = make_engine(kv_block_size=64, disaggregation_mode=mode)
    try:
        registration = await engine.start(worker_id=1)
        [bind] = native.binds
        # A single node binds free ports, so workers can share a host.
        assert bind["handshake_address"] == "tcp://127.0.0.1:0"
        assert bind["input_address"] == "tcp://127.0.0.1:0"
        assert bind["output_address"] == "tcp://127.0.0.1:0"
        assert bind["num_engines"] == 1
        assert bind["first_engine_index"] == 0
        assert bind["routing"] == routing
        [launch] = native.launches
        assert launch.argv == [
            "--model",
            "test-model",
            "--data-parallel-address",
            "127.0.0.1",
            "--data-parallel-rpc-port",
            HANDSHAKE_PORT,
        ]
        assert registration.llm.context_length == 8192
        assert registration.llm.kv_cache_block_size == 64
        assert registration.llm.total_kv_blocks == 128
        assert registration.llm.max_num_seqs == 32
        assert registration.llm.max_num_batched_tokens == 8192
        assert registration.llm.bootstrap_host == (
            "prefill.example" if mode == "prefill" else None
        )
        assert registration.llm.bootstrap_port == (9000 if mode == "prefill" else None)
    finally:
        await engine.cleanup()
    assert launch.returncode == 0


async def test_multi_node_binds_the_configured_handshake_port(native):
    engine = make_engine(
        data_parallel_address="10.0.0.5",
        data_parallel_rpc_port=31000,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1), nnodes=2),
    )
    try:
        await engine.start(worker_id=1)
        assert native.binds[0]["handshake_address"] == "tcp://10.0.0.5:31000"
        assert native.binds[0]["input_address"] == "tcp://10.0.0.5:0"
    finally:
        await engine.cleanup()


async def test_follower_node_is_rejected_before_launch(native):
    engine = make_engine(node_rank=1)
    with pytest.raises(ValueError, match="first node"):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


async def test_registration_without_block_size_leaves_kv_geometry_unset(native):
    engine = make_engine()
    try:
        registration = await engine.start(worker_id=1)
        assert registration.llm.kv_cache_block_size is None
        assert registration.llm.total_kv_blocks is None
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("flag", [None, 128])
async def test_engine_reported_prefix_granularity_is_preferred(native, flag):
    native.engines[0]["prefix_granularity"] = 128
    engine = make_engine(kv_block_size=flag)
    try:
        registration = await engine.start(worker_id=1)
        assert registration.llm.kv_cache_block_size == 128
        assert registration.llm.total_kv_blocks == 64
    finally:
        await engine.cleanup()


async def test_block_size_flag_must_match_engine_report(native):
    native.engines[0]["prefix_granularity"] = 128
    engine = make_engine(kv_block_size=64)
    try:
        with pytest.raises(ValueError, match="does not match"):
            await engine.start(worker_id=1)
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
async def test_generate_forwards_handoff_and_preserves_decode_budget(native, mode):
    native.respond = lambda request: [wire_output([20]), finished([21])]
    engine = make_engine(disaggregation_mode=mode, host="prefill.example")
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
    try:
        chunks = await drain(engine.generate(request, context("request-1")))
        tokens = [chunk["token_ids"] for chunk in chunks]
        if mode == "prefill":
            assert chunks[0] == {
                "index": 0,
                "token_ids": [],
                "disaggregated_params": request["bootstrap_info"],
            }
            tokens = tokens[1:]
        assert tokens == [[20], [21]]
        assert chunks[-1]["finish_reason"] == "length"
        [submitted] = native.submitted
        assert submitted.rid == "request-1"
        assert submitted.input_ids == [10, 11]
        assert submitted.stream is True
        assert submitted.bootstrap_host == "selected-prefill.example"
        assert submitted.bootstrap_port == 9010
        assert submitted.bootstrap_room == 0
        assert not hasattr(submitted, "handoff_id")
        if mode == "prefill":
            assert submitted.sampling_params["max_new_tokens"] == 1
            assert "min_new_tokens" not in submitted.sampling_params
        else:
            assert submitted.sampling_params["max_new_tokens"] == 30
            assert submitted.sampling_params["min_new_tokens"] == 10
        assert engine._streams == {}
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
async def test_bad_handoff_rejected_before_native_request(native, bootstrap):
    engine = make_engine(disaggregation_mode="decode")
    await engine.start(worker_id=1)
    try:
        with pytest.raises(InvalidArgument, match="bootstrap"):
            await drain(
                engine.generate(
                    {"token_ids": [1], "bootstrap_info": bootstrap}, context("bad")
                )
            )
        assert native.submitted == []
        assert engine._streams == {}
    finally:
        await engine.cleanup()


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            {"mapping": SimpleNamespace(attn=SimpleNamespace(dp_size=2), nnodes=1)},
            "attention DP=1",
        ),
        ({"disaggregation_transfer_backend": "fake"}, "Mooncake"),
    ],
)
async def test_unsupported_disagg_fails_before_start(native, overrides, message):
    engine = make_engine(disaggregation_mode="decode", **overrides)
    with pytest.raises(ValueError, match=message):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


async def test_kv_events_require_block_size(native):
    engine = make_engine(kv_events_config='{"enable_kv_cache_events":true}')
    with pytest.raises(ValueError, match="--kv-block-size"):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


async def test_kv_sources_use_unique_ipc_endpoints_and_cleanup(native):
    engines = [
        make_engine(
            kv_block_size=64, kv_events_config='{"enable_kv_cache_events":true}'
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
        # The launcher publishes to the endpoint the worker subscribes to.
        for launch, engine in zip(native.launches, engines):
            flag = launch.argv.index("--kv-events-config")
            assert launch.argv[flag + 1] == engine.server_args.kv_events_config
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
async def test_kv_source_matches_native_bind_and_topic(native, native_config, expected):
    native_config.update(enable_kv_cache_events=True, topic="kv")
    engine = make_engine(kv_block_size=64, kv_events_config=json.dumps(native_config))
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
async def test_disabled_kv_events_have_no_source(native, overrides, caplog):
    engine = make_engine(**overrides)
    try:
        await engine.start(worker_id=1)
        assert await engine.kv_event_sources() == []
        [launch] = native.launches
        if overrides.get("enable_prefix_caching") is False:
            flag = launch.argv.index("--kv-events-config")
            native_config = json.loads(launch.argv[flag + 1])
            assert native_config["enable_kv_cache_events"] is False
            assert native_config["publisher"] == "null"
            assert "enable_prefix_caching=False" in caplog.text
        else:
            assert "--kv-events-config" not in launch.argv
    finally:
        await engine.cleanup()


async def test_connecting_native_publisher_is_rejected(native):
    engine = make_engine(
        kv_block_size=64,
        kv_events_config=json.dumps(
            {"enable_kv_cache_events": True, "endpoint": "tcp://127.0.0.1:19000"}
        ),
    )
    with pytest.raises(ValueError, match="binding endpoint"):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


async def test_launcher_exit_before_registration_fails_start_and_cleans_up(native):
    native.never_register()
    native.launcher_exit_code = 3
    engine = make_engine(
        kv_block_size=64, kv_events_config='{"enable_kv_cache_events":true}'
    )
    with pytest.raises(RuntimeError, match="exited with code 3"):
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
async def test_non_boolean_event_flag_rejected_before_start(native, enabled):
    engine = make_engine(
        kv_block_size=64,
        kv_events_config=json.dumps({"enable_kv_cache_events": enabled}),
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


async def test_kv_replay_rejected_before_native_start(native, tmp_path):
    engine = make_engine(
        kv_block_size=64,
        kv_events_config=json.dumps(
            {
                "enable_kv_cache_events": True,
                "endpoint": f"ipc://{tmp_path}/events",
                "replay_endpoint": f"ipc://{tmp_path}/replay",
            }
        ),
    )
    try:
        with pytest.raises(ValueError, match="replay_endpoint"):
            await engine.start(worker_id=1)
        assert native.launches == []
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "::1", "localhost"])
@pytest.mark.parametrize("address", ["127.0.1.1", "0.0.0.0"])
async def test_prefill_rejects_local_only_derived_advertisement(
    monkeypatch, native, host, address
):
    monkeypatch.setattr(disagg.socket, "gethostbyname", lambda _: address)
    engine = make_engine(disaggregation_mode="prefill", host=host)
    with pytest.raises(ValueError, match=BOOTSTRAP_HOST_ENV):
        await engine.start(worker_id=1)
    assert native.launches == []
    await engine.cleanup()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "::1", "localhost"])
async def test_prefill_derives_non_loopback_advertisement(monkeypatch, native, host):
    monkeypatch.setattr(disagg.socket, "gethostbyname", lambda _: "192.0.2.10")
    engine = make_engine(disaggregation_mode="prefill", host=host)
    try:
        config = await engine.start(worker_id=1)
        assert config.llm.bootstrap_host == "192.0.2.10"
    finally:
        await engine.cleanup()


async def test_explicit_loopback_prefill_allowed_for_single_host(monkeypatch, native):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "127.0.0.1")
    engine = make_engine(disaggregation_mode="prefill")
    try:
        config = await engine.start(worker_id=1)
        assert config.llm.bootstrap_host == "127.0.0.1"
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("explicit_null", [False, True])
async def test_prefill_creates_handoff_when_router_has_no_bootstrap_endpoint(
    monkeypatch, native, explicit_null
):
    """The synchronous path supplies the same fresh room to prefill and decode."""
    rooms = Mock(side_effect=[17, 23])
    monkeypatch.setattr(
        disagg, "secrets", SimpleNamespace(randbits=rooms), raising=False
    )
    engine = make_engine(disaggregation_mode="prefill", host="prefill.example")
    registration = await engine.start(worker_id=1)
    try:
        for index, room in enumerate([17, 23]):
            request = {"token_ids": [10, 11], "stop_conditions": {"max_tokens": 30}}
            if explicit_null:
                request["bootstrap_info"] = None
            stream = engine.generate(request, context("fallback"))
            try:
                first = await anext(stream)
                assert first["token_ids"] == []
                handoff = first["disaggregated_params"]
                assert handoff == {
                    "bootstrap_host": registration.llm.bootstrap_host,
                    "bootstrap_port": registration.llm.bootstrap_port,
                    "bootstrap_room": room,
                }
                assert len(native.submitted) == index
                chunks = await drain(stream)
                assert chunks[0]["token_ids"] == [20]
                for key, value in handoff.items():
                    assert getattr(native.submitted[index], key) == value
                assert (
                    disagg.bootstrap_kwargs(
                        {"bootstrap_info": handoff}, DisaggregationMode.DECODE
                    )
                    == handoff
                )
                assert engine._streams == {}
            finally:
                await stream.aclose()
        assert rooms.call_count == 2
        rooms.assert_called_with(63)
    finally:
        await engine.cleanup()


@pytest.mark.parametrize("router_bootstrap", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_prefill_cancellation_after_handoff_prevents_submission(
    native, router_bootstrap, cancel
):
    """An abort before native admission survives resuming the handoff stream."""
    engine = make_engine(disaggregation_mode="prefill", host="prefill.example")
    registration = await engine.start(worker_id=1)
    request = {"token_ids": [10, 11]}
    if router_bootstrap:
        request["bootstrap_info"] = {
            "bootstrap_host": registration.llm.bootstrap_host,
            "bootstrap_port": registration.llm.bootstrap_port,
            "bootstrap_room": 1,
        }
    ctx = context("handoff-cancel")
    stream = engine.generate(request, ctx)
    try:
        first = await anext(stream)
        assert first["disaggregated_params"]
        assert native.submitted == []
        if cancel:
            await engine.abort(ctx)
            await engine.abort(ctx)
        chunks = await drain(stream)
        assert [r.rid for r in native.submitted] == (
            [] if cancel else ["handoff-cancel"]
        )
        assert chunks[-1]["finish_reason"] == ("cancelled" if cancel else "length")
        assert engine._streams == {}
        # Reusing the context must not inherit cancellation from the old stream.
        chunks = await drain(engine.generate(request, ctx))
        assert chunks[-1]["finish_reason"] == "length"
        assert native.submitted[-1].rid == "handoff-cancel"
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_abort_mid_stream_ends_with_cancelled(native):
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine()
    await engine.start(worker_id=1)
    ctx = context("stream-cancel")
    stream = engine.generate({"token_ids": [10]}, ctx)
    try:
        assert (await anext(stream))["token_ids"] == [20]
        await engine.abort(ctx)
        chunks = await drain(stream)
        assert chunks[-1]["finish_reason"] == "cancelled"
        assert native.aborted == ["stream-cancel"]
        assert engine._streams == {}
        assert engine._cancelled == set()
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_abort_racing_submission_still_aborts_native_request(native):
    native.respond = lambda request: [wire_output([20])]
    native.submit_gate = asyncio.Event()
    engine = make_engine()
    await engine.start(worker_id=1)
    ctx = context("racing-cancel")
    stream = engine.generate({"token_ids": [10]}, ctx)
    first = asyncio.ensure_future(anext(stream))
    try:
        await asyncio.sleep(0)
        # The engine does not hold the request yet: its abort is a no-op.
        await engine.abort(ctx)
        assert native.submitted == []
        native.submit_gate.set()
        chunks = [await first] + await drain(stream)
        assert chunks[-1]["finish_reason"] == "cancelled"
        assert native.aborted == ["racing-cancel", "racing-cancel"]
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_closing_stream_early_aborts_native_request(native):
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine()
    await engine.start(worker_id=1)
    stream = engine.generate({"token_ids": [10]}, context("abandoned"))
    try:
        assert (await anext(stream))["token_ids"] == [20]
        await stream.aclose()
        for _ in range(3):
            await asyncio.sleep(0)
        assert native.aborted == ["abandoned"]
        assert engine._streams == {}
    finally:
        await engine.cleanup()


async def test_abort_of_dropped_stream_is_safe_outside_the_event_loop(native):
    """A generator the caller drops is finalized by the garbage collector,
    possibly on another thread with no running loop: its abort must still
    reach the engine, through the loop the engine started on."""
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine()
    await engine.start(worker_id=1)
    stream = engine.generate({"token_ids": [10]}, context("finalized"))
    try:
        assert (await anext(stream))["token_ids"] == [20]
        await asyncio.to_thread(engine._schedule_abort, "finalized")
        for _ in range(3):
            await asyncio.sleep(0)
        assert native.aborted == ["finalized"]
    finally:
        await stream.aclose()
        await engine.cleanup()


@pytest.mark.parametrize("mode", ["null", "decode"])
async def test_only_prefill_reports_quiescence(native, mode):
    engine = make_engine(disaggregation_mode=mode)
    await engine.start(worker_id=1)
    try:
        assert await engine.is_quiescent() is None
    finally:
        await engine.cleanup()


async def test_prefill_is_quiescent_once_its_handoffs_resolve(monkeypatch, native):
    monkeypatch.setenv(BOOTSTRAP_HOST_ENV, "prefill.example")
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine(disaggregation_mode="prefill")
    await engine.start(worker_id=1)
    stream = engine.generate({"token_ids": [10]}, context("transferring"))
    try:
        assert await engine.is_quiescent() is True
        await anext(stream)  # the handoff
        await anext(stream)  # submitted; its KV is still in flight
        assert await engine.is_quiescent() is False
        native.frontend.deliver("transferring", finished([]))
        await drain(stream)
        assert await engine.is_quiescent() is True
    finally:
        await stream.aclose()
        await engine.cleanup()
    assert await engine.is_quiescent() is True


async def test_stream_ended_by_engines_raises(native):
    native.respond = lambda request: [
        wire_output([20]),
        RuntimeError("the engines ended the stream"),
    ]
    engine = make_engine()
    await engine.start(worker_id=1)
    try:
        with pytest.raises(RuntimeError, match="ended the stream"):
            await drain(engine.generate({"token_ids": [10]}, context("dropped")))
        assert engine._streams == {}
    finally:
        await engine.cleanup()


async def test_closed_frontend_fails_in_flight_and_new_requests(native):
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine()
    await engine.start(worker_id=1)
    stream = engine.generate({"token_ids": [10]}, context("in-flight"))
    try:
        assert (await anext(stream))["token_ids"] == [20]
        native.frontend.close()
        with pytest.raises(RuntimeError, match="frontend is closed"):
            await drain(stream)
        with pytest.raises(RuntimeError, match="frontend is closed"):
            await drain(engine.generate({"token_ids": [10]}, context("late")))
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_launcher_exit_fails_requests_and_stops_worker(monkeypatch, native):
    kill = Mock()
    monkeypatch.setattr(llm_engine.os, "kill", kill)
    native.respond = lambda request: [wire_output([20])]
    engine = make_engine()
    await engine.start(worker_id=1)
    stream = engine.generate({"token_ids": [10]}, context("orphaned"))
    try:
        assert (await anext(stream))["token_ids"] == [20]
        native.launches[0].exit(-9)
        with pytest.raises(RuntimeError, match="exited with code -9"):
            await drain(stream)
        kill.assert_called_once_with(llm_engine.os.getpid(), signal.SIGTERM)
    finally:
        await stream.aclose()
        await engine.cleanup()


async def test_cleanup_does_not_stop_worker(monkeypatch, native):
    kill = Mock()
    monkeypatch.setattr(llm_engine.os, "kill", kill)
    engine = make_engine()
    await engine.start(worker_id=1)
    await engine.cleanup()
    await asyncio.sleep(0.05)
    kill.assert_not_called()
    assert native.launches[0].returncode == 0


async def test_generate_unpacks_packed_token_ids(native):
    engine = make_engine()
    await engine.start(worker_id=1)
    request = {"token_ids": b"".join(i.to_bytes(4, "little") for i in (10, 11))}
    try:
        chunks = await drain(engine.generate(request, context("packed-1")))
        assert chunks[-1]["token_ids"] == [20]
        assert native.submitted[0].input_ids == [10, 11]
    finally:
        await engine.cleanup()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claim, lease and budget logic of the GPU-resident hidden-state connector."""

import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector"
)

from dynamo.artifacts.vllm import gpu_hidden_states as gpu  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.vllm,
]


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done


class _Operation:
    def __init__(self):
        self.released = False

    def __exit__(self, *_):
        self.released = True


def _connector(max_held_bytes=1 << 20):
    connector = object.__new__(gpu.DynamoGpuHiddenStatesConnector)
    connector._max_held_bytes = max_held_bytes
    connector._claim_timeout_s = 30.0
    connector._max_lease_s = 300
    connector._held = {}
    connector._held_bytes = 0
    connector._lock = threading.Lock()
    connector._loop = None
    connector._nixl_connector = None
    connector.scheduled = []
    connector.operations = []

    def run(coroutine, timeout=30.0):
        coroutine.close()
        operation = _Operation()
        connector.operations.append(operation)
        return {"token": len(connector.operations)}

    connector._run = run
    connector._call_soon = lambda fn, *args: connector.scheduled.append((fn, args))
    connector._call_later = lambda delay, fn, *args: connector.scheduled.append(
        (fn, args)
    )
    return connector


def _hold(connector, req_id, nbytes=64, done=True, error=None):
    tensor = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    gathered = gpu._Gathered(
        None if error else tensor, [1, 2], _Event(done), nbytes, error=error
    )
    connector._held_bytes += nbytes if not error else 0
    connector._held[req_id] = gathered
    return gathered


def test_unknown_handle_returns_none_so_other_workers_answer():
    assert _connector().claim("missing", 60) is None


def test_claim_is_pending_until_the_gather_completes():
    connector = _connector()
    gathered = _hold(connector, "r", done=False)
    assert connector.claim("r", 60) == {"state": "pending"}
    gathered.event.done = True
    result = connector.claim("r", 60)
    assert result["state"] == "ready"
    assert result["dtype"] == "bf16"
    assert result["shape"] == [2, 3, 4]
    assert result["token_ids"] == [1, 2]
    assert result["rdma_metadata"] == {"token": 1}
    assert result["expires_at"].endswith("Z")


def test_a_handle_can_be_claimed_once():
    connector = _connector()
    _hold(connector, "r")
    assert connector.claim("r", 60)["state"] == "ready"
    assert connector.claim("r", 60) == {"state": "failed", "error": "already claimed"}


@pytest.mark.parametrize("lease", [0, 301, True, "60"])
def test_invalid_leases_are_refused(lease):
    connector = _connector()
    _hold(connector, "r")
    assert connector.claim("r", lease)["state"] == "failed"
    assert connector._held["r"].claimed is False


def test_dropped_gather_reports_its_error_once():
    connector = _connector()
    _hold(connector, "r", error="gpu budget exhausted")
    assert connector.claim("r", 60) == {
        "state": "failed",
        "error": "gpu budget exhausted",
    }
    assert connector.claim("r", 60) is None


def test_release_frees_budget_and_deregisters():
    connector = _connector()
    _hold(connector, "r", nbytes=64)
    connector.claim("r", 60)
    connector._held["r"].operation = connector.operations[0]
    connector._release("r")
    assert connector._held_bytes == 0
    assert "r" not in connector._held
    assert connector.operations[0].released
    connector._release("r")
    assert connector._held_bytes == 0


def test_unclaimed_gather_expires_but_a_claimed_one_does_not():
    connector = _connector()
    _hold(connector, "a", nbytes=10)
    _hold(connector, "b", nbytes=20)
    connector.claim("b", 60)
    connector._expire_unclaimed("a")
    connector._expire_unclaimed("b")
    assert "a" not in connector._held
    assert "b" in connector._held
    assert connector._held_bytes == 20


@pytest.mark.asyncio
async def test_poll_helper_waits_through_pending_and_ignores_other_workers():
    answers = [[None, {"state": "pending"}], [None, {"state": "ready", "x": 1}]]
    calls = []

    async def rpc(method, args):
        calls.append((method, args))
        return answers.pop(0)

    result = await gpu.claim_hidden_states(rpc, "h", 60, timeout_seconds=5)
    assert result == {"state": "ready", "x": 1}
    assert calls == [("dynamo_claim_hidden_states", ("h", 60))] * 2


@pytest.mark.asyncio
async def test_poll_helper_times_out_and_rejects_duplicate_owners():
    async def pending(method, args):
        return [{"state": "pending"}]

    with pytest.raises(TimeoutError):
        await gpu.claim_hidden_states(pending, "h", 60, timeout_seconds=0.01)

    async def twice(method, args):
        return [{"state": "ready"}, {"state": "ready"}]

    with pytest.raises(RuntimeError, match="several workers"):
        await gpu.claim_hidden_states(twice, "h", 60)


def test_worker_start_warms_nixl_off_the_engine_thread(monkeypatch):
    connector = _connector()
    connector._is_tp_rank_zero = True
    monkeypatch.setattr(
        gpu.ExampleHiddenStatesConnector,
        "register_kv_caches",
        lambda self, caches: None,
    )
    connector.register_kv_caches({})
    assert [fn.__name__ for fn, _ in connector.scheduled] == ["_warm_nixl"]

    connector.scheduled.clear()
    connector._is_tp_rank_zero = False
    connector.register_kv_caches({})
    assert connector.scheduled == []


@pytest.mark.asyncio
async def test_all_engines_rpc_sees_handles_on_every_data_parallel_engine():
    calls = []

    class Client:
        async def call_utility_all_async(self, name, method, timeout, args, kwargs):
            calls.append((name, method, args))
            # Engine 0 doesn't know the handle; engine 1 has two TP ranks.
            return [[None], [None, {"state": "ready"}]]

    class Engine:
        engine_core = Client()

        async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return [None]  # what vLLM returns under DP: engine 0 only

    rpc = gpu.all_engines_rpc(Engine())
    assert await rpc("dynamo_claim_hidden_states", args=("h", 60)) == [
        None,
        None,
        {"state": "ready"},
    ]
    assert calls == [("collective_rpc", "dynamo_claim_hidden_states", ("h", 60))]
    claim = await gpu.claim_hidden_states(rpc, "h", 60)
    assert claim == {"state": "ready"}


def test_all_engines_rpc_falls_back_without_data_parallel_client():
    class Engine:
        engine_core = object()

        async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return []

    engine = Engine()
    assert gpu.all_engines_rpc(engine) == engine.collective_rpc


def test_engines_that_do_not_own_a_handle_stay_silent_on_bad_leases():
    assert _connector().claim("missing", 0) is None

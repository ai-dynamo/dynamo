# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Random recurrent state must not corrupt aliased attention/prefix storage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from dynamo.vllm.benchmark_points import RANDOM_KDA_BOUND, RANDOM_KDA_REQUEST_PREFIX
from dynamo.vllm.benchmark_worker import BenchmarkWorker, Worker, fill_recurrent_states

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_random_state_is_finite_reproducible_and_preserves_rng_and_other_blocks():
    states = (
        torch.full((4, 3, 8), 7, dtype=torch.bfloat16),
        torch.full((4, 2, 4, 4), 7.0),
    )
    before = [tensor.clone() for tensor in states]
    rng = torch.random.get_rng_state()
    fill_recurrent_states(states, [2], request_id="request", layer_name="layer", rank=0)
    for state, original in zip(states, before):
        assert torch.equal(state[[0, 1, 3]], original[[0, 1, 3]])
        assert torch.isfinite(state[2]).all()
        assert (state[2].abs() <= RANDOM_KDA_BOUND).all()
        assert torch.count_nonzero(state[2]) > 0
    assert torch.equal(rng, torch.random.get_rng_state())
    first = [tensor.clone() for tensor in states]
    fill_recurrent_states(states, [2], request_id="request", layer_name="layer", rank=0)
    assert all(torch.equal(a, b) for a, b in zip(states, first))
    fill_recurrent_states(states, [2], request_id="request", layer_name="layer", rank=1)
    assert not torch.equal(states[1][2], first[1][2])


@pytest.mark.parametrize("block_id", [0, -1, 4])
def test_random_state_rejects_null_or_invalid_blocks(block_id):
    with pytest.raises(ValueError, match="Invalid synthetic"):
        fill_recurrent_states(
            [torch.zeros(4, 2)], [block_id], request_id="r", layer_name="l", rank=0
        )


@pytest.mark.parametrize("runner_api", ["v1", "v2", "v2_lazy"])
def test_worker_initializes_only_private_kda_after_zeroing_and_disables_before_serving(
    monkeypatch,
    runner_api,
):
    # Different typed views share a pool, as in the hybrid allocator. Block 1
    # belongs to attention; block 2 to the source chain; only block 3 is KDA.
    pool = torch.full((6, 32), 7.0)
    states = (pool[:, :8].view(6, 2, 4), pool[:, 8:].view(6, 2, 3, 4))
    worker = BenchmarkWorker.__new__(BenchmarkWorker)
    worker.rank = 0
    worker.device = torch.device("cpu")
    events = []

    def zero(block_ids):
        events.append("zero")
        pool[block_ids] = 0

    if runner_api == "v1":
        worker.model_runner = SimpleNamespace(_zero_block_ids=zero)
    else:
        zeroer = SimpleNamespace(zero_block_ids=zero)
        worker.model_runner = SimpleNamespace(
            kv_block_zeroer=None if runner_api == "v2_lazy" else zeroer,
            _init_kv_zero_meta=Mock(
                side_effect=lambda: setattr(
                    worker.model_runner, "kv_block_zeroer", zeroer
                )
            ),
        )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"kda": SimpleNamespace(kv_cache=states)}
        )
    )
    spec = MambaSpec(block_size=16, shapes=(), dtypes=(), mamba_cache_mode="align")
    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object(), layer_names=["mla"]),
            SimpleNamespace(kv_cache_spec=spec, layer_names=["kda"]),
        ]
    )
    monkeypatch.setattr(Worker, "initialize_from_config", lambda *_args: None)
    worker.initialize_from_config(cache_config)
    if runner_api != "v1":
        assert worker.model_runner._init_kv_zero_meta.call_count == (
            runner_api == "v2_lazy"
        )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(synchronize=lambda: events.append("sync")),
    )
    request = SimpleNamespace(
        req_id=RANDOM_KDA_REQUEST_PREFIX + "0", block_ids=([1], [0, 3])
    )
    output = SimpleNamespace(scheduled_new_reqs=[request], new_block_ids_to_zero=[3])

    def execute(_worker, passed):
        events.append("execute")
        assert passed.new_block_ids_to_zero is None
        assert torch.count_nonzero(pool[3]) > 0
        return "result"

    monkeypatch.setattr(Worker, "execute_model", execute)
    assert worker.execute_model(output) == "result"
    assert events == ["zero", "sync", "execute"]
    assert output.new_block_ids_to_zero == [3]  # original scheduler record preserved
    assert torch.all(pool[[0, 1, 2, 4, 5]] == 7)

    native = Mock(return_value="steady")
    monkeypatch.setattr(Worker, "execute_model", native)
    before = pool.clone()
    steady = SimpleNamespace(scheduled_new_reqs=[])
    assert worker.execute_model(steady) == "steady"
    assert torch.equal(pool, before)
    worker.finish_benchmark_kda_state()
    worker.execute_model(output)
    assert torch.equal(pool, before)
    assert worker._benchmark_kda_layers == []


@pytest.mark.parametrize("has_zeroing_api", [True, False])
def test_worker_binds_only_recurrent_layers_from_runtime_cache_groups(
    monkeypatch, has_zeroing_api
):
    worker = BenchmarkWorker.__new__(BenchmarkWorker)
    state = (torch.zeros(4, 2, 4), torch.zeros(4, 2, 3, 4))
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "kda": SimpleNamespace(kv_cache=state),
                "mla": object(),
            }
        )
    )
    worker.model_runner = SimpleNamespace(
        _zero_block_ids=(lambda _ids: None) if has_zeroing_api else None
    )
    spec = MambaSpec(block_size=16, shapes=(), dtypes=(), mamba_cache_mode="align")
    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object(), layer_names=["mla"]),
            SimpleNamespace(kv_cache_spec=spec, layer_names=["kda"]),
        ]
    )
    monkeypatch.setattr(Worker, "initialize_from_config", lambda *_args: None)
    if not has_zeroing_api:
        with pytest.raises(ValueError, match="KV-zeroing API"):
            worker.initialize_from_config(cache_config)
        assert not worker._benchmark_kda_active
        return
    worker.initialize_from_config(cache_config)
    assert worker._benchmark_kda_layers == [(1, "kda", state)]
    assert worker._benchmark_kda_active

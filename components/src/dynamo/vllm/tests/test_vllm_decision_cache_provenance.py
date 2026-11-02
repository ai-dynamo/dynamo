# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU qualification of pinned scheduler statistics, not just output integer types."""

import ast
import builtins
import inspect
import json
import textwrap
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.vllm.decision_cache_provenance import qualified_cache_runtime
from dynamo.vllm.decision_execution import is_exact_vllm_release
from dynamo.vllm.tests.decision_cache_contract import CONTRACT_PATH, project_release

# Load the backend before pytest traverses the sibling Dynamo package name.
pytest.importorskip("vllm")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def scheduler_api():
    pytest.importorskip("vllm")
    if not is_exact_vllm_release(version("vllm"), "0.33.0"):
        pytest.skip("Optional pinned vLLM 0.33.0 CPU runtime is not installed")
    assert qualified_cache_runtime(version("vllm"))
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    assert AsyncScheduler.update_from_output is Scheduler.update_from_output
    return Scheduler


def test_causal_scheduler_qualification_ignores_native_only_source_mismatch(
    monkeypatch, request
):
    tree = ast.parse(Path(__file__).read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert all(
        "decision_native" not in name and "diffusion_scheduler" not in name
        for name in imports
        if name
    )
    original_read = Path.read_bytes
    original_import = builtins.__import__
    native_reads = []

    def read(path):
        if path.as_posix().endswith("vllm/model_executor/models/d3.py"):
            native_reads.append(path)
            return b"native-only source mismatch"
        return original_read(path)

    def import_without_native(name, *args, **kwargs):
        if "decision_native" in name or "diffusion_scheduler" in name:
            raise AssertionError("causal qualification must not import native owners")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(builtins, "__import__", import_without_native)
    scheduler = request.getfixturevalue("scheduler_api")
    assert scheduler.__name__ == "Scheduler"
    assert native_reads == []


def test_installed_runtime_matches_the_qualified_cache_source_projection(scheduler_api):
    import vllm

    root = Path(vllm.__file__).resolve().parent.parent
    assert qualified_cache_runtime(version("vllm"))
    assert not qualified_cache_runtime("0.32.0")  # Cannot mislabel 0.33 source as 0.32.
    actual = project_release(lambda path: (root / path).read_text())
    contract = json.loads(CONTRACT_PATH.read_text())
    assert actual == contract["releases"]["0.33.0"]["projection"]


def capture_first_prefill(scheduler, request, cached):
    """Execute the exact hash-verified first-prefill capture block, not a copy."""
    source = ast.parse(textwrap.dedent(inspect.getsource(scheduler.schedule)))
    capture = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "request.prefill_stats and request.num_preemptions <= 0"
    ]
    assert len(capture) == 1
    state = {
        "request": request,
        "num_computed_tokens": cached,
        "num_new_local_computed_tokens": cached,
        "num_external_computed_tokens": 0,
    }
    exec(
        compile(
            ast.Module(body=capture, type_ignores=[]), "pinned-prefill-capture", "exec"
        ),
        state,
    )


def scheduler_state(request):
    from vllm.v1.request import RequestStatus

    def stop(current, tokens, *, is_stale):
        assert not is_stale
        current.status = RequestStatus.FINISHED_STOPPED
        return tokens, True

    return SimpleNamespace(
        requests={request.request_id: request},
        defer_block_free=False,
        perf_metrics=None,
        num_spec_tokens=0,
        num_sampled_tokens_per_step=1,
        is_mm_encoder_only=False,
        structured_output_manager=SimpleNamespace(accept_tokens=lambda *_: True),
        aux_output_connector=None,
        kv_cache_manager=SimpleNamespace(
            estimate_cached_tokens=lambda _: 2, take_events=lambda: None
        ),
        return_sampling_mask=False,
        _update_request_with_output=stop,
        _handle_stopped_request=lambda _: True,
        _free_request=lambda _: (None, None),
        running=[request],
        grammar_compile_error_reqs={},
        encoder_cache_mismatch_reqs=set(),
        ec_connector=None,
        connector=None,
        finished_req_ids_dict={},
        make_stats=lambda *_, **__: None,
    )


@pytest.mark.parametrize("pooling", [False, True])
@pytest.mark.parametrize("cached", [0, 2])
def test_first_successful_scheduler_output_preserves_prefill_evidence(
    scheduler_api, pooling, cached
):
    import torch
    from vllm import PoolingParams, SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.request import Request, RequestStatus

    from dynamo.vllm.handlers import BaseWorkerHandler

    sampling = None if pooling else SamplingParams(max_tokens=1)
    params = PoolingParams(task="classify") if pooling else None
    request = Request("cache-provenance", [1, 2, 3], sampling, params)
    capture_first_prefill(scheduler_api, request, cached)
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 1
    request.num_computed_tokens = 3
    scheduled = SimpleNamespace(
        num_scheduled_tokens={request.request_id: 1}, scheduled_spec_decode_tokens={}
    )
    runner = SimpleNamespace(
        sampled_token_ids=[] if pooling else [[9]],
        logprobs=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
        pooler_output=[torch.tensor([0.2, 0.8])] if pooling else None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={request.request_id: 0},
    )
    output = scheduler_api.update_from_output(
        scheduler_state(request), scheduled, runner
    )
    frames = output[request.client_index].outputs
    assert len(frames) == 1
    assert frames[0].prefill_stats is not None
    assert frames[0].prefill_stats.num_cached_tokens == cached
    assert request.take_prefill_stats() is None
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    core_request = EngineCoreRequest(
        request_id=request.request_id,
        external_req_id=request.request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=sampling,
        pooling_params=params,
        arrival_time=0,
        lora_request=None,
        cache_salt="isolated",
        data_parallel_rank=None,
    )
    processor.add_request(core_request, prompt=None)
    final = processor.process_outputs(frames).request_outputs
    assert len(final) == 1
    assert final[0].num_cached_tokens == cached
    if not pooling:
        usage = BaseWorkerHandler._build_completion_usage(final[0])
        assert usage["prompt_tokens_details"]["cached_tokens"] == cached
        assert usage["prompt_tokens"] == 3
        assert usage["completion_tokens"] == 1

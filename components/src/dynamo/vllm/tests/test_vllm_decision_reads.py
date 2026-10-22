# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native generative decision contracts, using hash-verified CPU strategies."""

import asyncio
import copy
import math
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm import decision_reads as reads
from dynamo.vllm.tests.test_vllm_decision_reads_native import Engine

pytest_plugins = ["dynamo.vllm.tests.test_vllm_decision_reads_native"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@asynccontextmanager
async def monitor(context, request_id):
    context.append(request_id)
    yield


def question(ordinal=0, kind="Choice", values=("red", "blue")):
    return {
        "ordinal": ordinal,
        "kind": kind,
        "instructions": "Choose.",
        "candidates": [{"value": value} for value in values],
    }


def wire(profile=reads.LIQUID_PROFILE, questions=None):
    return {
        "decision_read": {
            "protocol_version": 1,
            "profile": profile,
            "model": "test-model",
            "input": "state",
            "questions": questions or [question()],
            "temperature": 1,
            "cache_salt": "request-isolation",
            "max_input_tokens": 16384,
            "max_tokenization_work": 32768,
            "max_encoding_bytes": 65536,
        }
    }


async def adapter(engine, profile=reads.LIQUID_PROFILE):
    return await reads.ReadExecutionAdapter.create(
        engine,
        profile,
        source_commit=reads.PINNED_VLLM_COMMIT,
        abort_monitor=monitor,
        max_concurrency=2,
    )


async def execute(instance, request):
    context = []
    response = await instance.execute(
        request,
        context,
        request_id="native-read",
        cancelled=lambda: False,
        trace_headers={"traceparent": "test"},
    )
    return response, context


@pytest.mark.asyncio
async def test_liquid_uses_trained_prompt_and_native_group_max_reducer(native_reads):
    engine = Engine()
    instance = await adapter(engine)
    request = wire(questions=[question(kind="Predicate", values=(True, False))])
    response, context = await execute(instance, request)

    assert engine.rendered[0][0]["content"].startswith("state\n\n\nQUESTION:\n")
    assert "Reply with yes or no only." in engine.rendered[0][0]["content"]
    call = engine.calls[0]
    assert len(call[1].logprob_token_ids) == 6
    assert response["outcomes"][0]["probabilities"] == pytest.approx(
        [
            math.exp(-1) / (math.exp(-1) + math.exp(-4)),
            math.exp(-4) / (math.exp(-1) + math.exp(-4)),
        ]
    )
    assert response["outcomes"][0]["label_mass"] == pytest.approx(
        math.exp(-1) + math.exp(-2) + 4 * math.exp(-4)
    )
    assert response["usage"] == {
        "input_tokens": len(call[0]["prompt_token_ids"]),
        "output_tokens": 1,
        "cached_tokens": None,
    }
    assert context == ["native-read-0"]
    assert call[0]["cache_salt"] == "request-isolation"
    assert call[3]["trace_headers"] == {"traceparent": "test"}


@pytest.mark.asyncio
async def test_diffusion_keeps_thought_outside_deterministic_read_only_canvas(
    native_reads,
):
    engine = Engine(diffusion=True)
    instance = await adapter(engine, reads.DIFFUSION_PROFILE)
    request = wire(reads.DIFFUSION_PROFILE)
    original = copy.deepcopy(request)
    prepared = await instance.prepare(request)
    repeated = await instance.prepare(request)
    assert (
        prepared.reads[0].sampling_params.extra_args
        == repeated.reads[0].sampling_params.extra_args
    )
    response, _ = await execute(instance, request)
    prompt, params, _, _ = engine.calls[0]
    assert prompt["prompt_token_ids"][-2:] == [8001, 8002]
    assert params.max_tokens == 2
    assert params.extra_args == {
        "diffusion_seed_canvas": params.extra_args["diffusion_seed_canvas"],
        "diffusion_canvas_length": 16,
        "diffusion_max_steps": 1,
        "diffusion_read_only": True,
    }
    assert len(params.extra_args["diffusion_seed_canvas"]) == 16
    assert params.extra_args["diffusion_seed_canvas"][1] == 8003
    assert response["usage"]["output_tokens"] == 0
    assert request == original


@pytest.mark.asyncio
async def test_every_question_preflights_before_admission(native_reads):
    engine = Engine()
    instance = await adapter(engine)
    request = wire(questions=[question(), question(1)])
    request["decision_read"]["questions"][1]["instructions"] = (
        "x" * engine.model_config.max_model_len
    )
    with pytest.raises(reads.ReadValidationError):
        await execute(instance, request)
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_diffusion_capacity_reserves_thought_and_entire_canvas(native_reads):
    engine = Engine(diffusion=True)
    instance = await adapter(engine, reads.DIFFUSION_PROFILE)
    prepared = await instance.prepare(wire(reads.DIFFUSION_PROFILE))
    engine.model_config.max_model_len = (
        len(prepared.reads[0].prompt["prompt_token_ids"]) + 15
    )
    instance.strategy.max_model_len = engine.model_config.max_model_len
    with pytest.raises(reads.ReadValidationError):
        await execute(instance, wire(reads.DIFFUSION_PROFILE))
    assert engine.calls == []


@pytest.mark.asyncio
async def test_prefix_is_monitored_and_included_in_actual_usage(native_reads):
    engine = Engine()
    engine.model_config.is_hybrid = True
    engine.vllm_config.cache_config.enable_prefix_caching = True
    instance = await adapter(engine)
    request = wire(questions=[question(), question(1)])
    request["decision_read"]["input"] = "s" * 200
    engine.cached_tokens = 0
    response, context = await execute(instance, request)
    assert context[0] == "native-read-prefix"
    assert len(engine.calls) == 3
    assert all(call[0]["cache_salt"] == "request-isolation" for call in engine.calls)
    assert response["usage"]["input_tokens"] == sum(
        len(call[0]["prompt_token_ids"]) for call in engine.calls
    )
    assert response["usage"]["output_tokens"] == 3
    assert response["usage"]["cached_tokens"] == 0


@pytest.mark.asyncio
async def test_raw_missing_cache_cannot_be_replaced_by_native_default_zero(
    native_reads,
):
    engine = Engine()
    response, _ = await execute(
        await adapter(engine), wire(questions=[question(), question(1)])
    )
    assert response["usage"]["cached_tokens"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0),
        ("max_input_tokens", 1),
        ("max_tokenization_work", 1),
        ("max_encoding_bytes", 1),
        ("profile", "unknown"),
    ],
)
async def test_invalid_request_has_no_generation(native_reads, field, value):
    engine = Engine()
    request = wire()
    request["decision_read"][field] = value
    with pytest.raises(reads.ReadValidationError):
        await execute(await adapter(engine), request)
    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_config", object()),
        ("speculative_config", object()),
        ("kv_transfer_config", object()),
        ("ec_transfer_config", object()),
    ],
)
async def test_unsupported_effective_runtime_rejected(native_reads, field, value):
    engine = Engine()
    setattr(engine.vllm_config, field, value)
    with pytest.raises(reads.ReadRuntimeError):
        await adapter(engine)


@pytest.mark.asyncio
async def test_source_commit_mismatch_fails_closed():
    with pytest.raises(reads.ReadRuntimeError):
        await reads.ReadExecutionAdapter.create(
            Engine(), reads.LIQUID_PROFILE, source_commit="wrong", abort_monitor=monitor
        )


@pytest.mark.asyncio
async def test_global_engine_concurrency_is_bounded_and_cancellation_drains(
    native_reads,
):
    engine = Engine()
    engine.block = asyncio.Event()
    instance = await adapter(engine)
    pending = asyncio.create_task(
        execute(instance, wire(questions=[question(i) for i in range(6)]))
    )
    await engine.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert engine.active == 0
    assert engine.peak_active <= 2
    assert set(engine.aborted) == set(engine.closed)
    assert engine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["missing_logprobs", "nonfinite", "unfinished", "wrong_id"]
)
async def test_bad_engine_evidence_is_failure_not_refusal(native_reads, failure):
    engine = Engine()
    engine.failure = failure
    with pytest.raises(reads.ReadExecutionError):
        await execute(await adapter(engine), wire())
    assert engine.active == 0


@pytest.mark.asyncio
async def test_prefix_work_is_included_in_the_admission_budget(native_reads):
    engine = Engine()
    engine.model_config.is_hybrid = True
    engine.vllm_config.cache_config.enable_prefix_caching = True
    instance = await adapter(engine)
    request = wire(questions=[question(), question(1)])
    request["decision_read"]["input"] = "s" * 200
    plan = await instance.prepare(request)
    request["decision_read"]["max_input_tokens"] = sum(
        len(read.prompt["prompt_token_ids"]) for read in plan.reads
    )
    with pytest.raises(reads.ReadValidationError):
        await execute(instance, request)
    assert engine.admissions == []
    assert engine.calls == []


@pytest.mark.asyncio
async def test_score_and_custom_choice_labels_use_native_trained_prompts(native_reads):
    engine = Engine()
    instance = await adapter(engine)
    await execute(
        instance,
        wire(questions=[question(values=("Q", "B")), question(1, "Score", ("0", "1"))]),
    )
    assert "Q Q\nB B" in engine.rendered[0][0]["content"]
    assert "Reply with a single digit 0-1 only." in engine.rendered[1][0]["content"]
    assert engine.calls[0][1].logprob_token_ids[:2] == [
        1000 + ord("Q"),
        2000 + ord("Q"),
    ]
    assert engine.calls[1][1].logprob_token_ids == [1000 + ord("0"), 1000 + ord("1")]


@pytest.mark.asyncio
async def test_cancellation_callback_blocks_preparation_and_admission(native_reads):
    engine = Engine()
    instance = await adapter(engine)
    with pytest.raises(asyncio.CancelledError):
        await instance.execute(
            wire(), [], request_id="cancelled", cancelled=lambda: True
        )
    assert engine.admissions == []
    assert engine.rendered == []


@pytest.mark.asyncio
async def test_failure_cancels_and_drains_other_actual_engine_children(native_reads):
    class FailingEngine(Engine):
        async def generate(self, prompt, params, request_id, **kwargs):
            if request_id.endswith("-0"):
                try:
                    await self.started.wait()
                    raise reads.ReadExecutionError("failed engine child")
                finally:
                    self.closed.append(request_id)
            async for output in super().generate(prompt, params, request_id, **kwargs):
                yield output

    engine = FailingEngine()
    engine.block = asyncio.Event()
    with pytest.raises(reads.ReadExecutionError, match="failed engine child"):
        await execute(
            await adapter(engine), wire(questions=[question(i) for i in range(4)])
        )
    assert engine.active == 0
    assert set(engine.aborted) == set(engine.closed)
    assert len(engine.aborted) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["architecture", "runner", "dp", "logprobs", "canvas"]
)
async def test_incompatible_model_runtime_fails_initialization(native_reads, change):
    engine = Engine(diffusion=change == "canvas")
    profile = reads.DIFFUSION_PROFILE if change == "canvas" else reads.LIQUID_PROFILE
    if change == "architecture":
        engine.model_config.architecture = "Qwen3ForCausalLM"
    elif change == "runner":
        engine.model_config.runner_type = "pooling"
    elif change == "dp":
        engine.vllm_config.parallel_config.data_parallel_size = 2
    elif change == "logprobs":
        engine.model_config.logprobs_mode = "raw_logits"
    else:
        engine.vllm_config.diffusion_config.canvas_length = 15
    with pytest.raises(reads.ReadRuntimeError):
        await adapter(engine, profile)
    assert engine.calls == []


def test_installed_source_bytes_are_verified_before_importing_optional_strategy(
    tmp_path, monkeypatch
):
    package = tmp_path / "vllm"
    package.mkdir()
    installed = package / "source.py"
    installed.write_text("foreign source", encoding="utf-8")
    monkeypatch.setattr(reads, "_PINNED_SOURCE_BLOBS", (("vllm/source.py", "wrong"),))
    calls = []

    def import_package(name):
        calls.append(name)
        assert name == "vllm"
        return type("Package", (), {"__file__": str(package / "__init__.py")})

    monkeypatch.setattr(reads, "import_module", import_package)
    with pytest.raises(reads.ReadRuntimeError, match="differs from pinned"):
        reads._load_native(reads.PINNED_VLLM_COMMIT)
    assert calls == ["vllm"]


@pytest.mark.asyncio
async def test_zero_vocabulary_mass_is_preserved_not_invented(native_reads):
    class UnderflowEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for output in super().generate(*args, **kwargs):
                for logprob in output.outputs[0].logprobs[0].values():
                    logprob.logprob = -1000.0
                yield output

    response, _ = await execute(await adapter(UnderflowEngine()), wire())
    assert response["outcomes"][0]["label_mass"] == 0
    assert response["outcomes"][0]["probabilities"] == [0.5, 0.5]


@pytest.mark.asyncio
async def test_excess_vocabulary_mass_is_rejected_not_clamped(native_reads):
    class ExcessMassEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for output in super().generate(*args, **kwargs):
                values = output.outputs[0].logprobs[0].values()
                for logprob in values:
                    logprob.logprob = math.log((1 + 1e-7) / len(values))
                yield output

    with pytest.raises(reads.ReadExecutionError, match="label mass"):
        await execute(await adapter(ExcessMassEngine()), wire())


@pytest.mark.asyncio
async def test_native_renderer_validation_remains_a_pre_admission_client_error(
    native_reads, monkeypatch
):
    engine = Engine()
    instance = await adapter(engine)
    monkeypatch.setattr(
        instance.strategy.context.online_renderer,
        "preprocess_chat",
        AsyncMock(side_effect=native_reads.validation_error("prompt exceeds context")),
    )
    with pytest.raises(reads.ReadValidationError):
        await execute(instance, wire())
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_native_reduction_does_not_repeat_tokenization_after_engine_admission(
    native_reads, monkeypatch
):
    engine = Engine()
    instance = await adapter(engine)
    prepared = await instance.prepare(wire())
    monkeypatch.setattr(instance, "prepare", AsyncMock(return_value=prepared))
    tokenizer = engine.renderer.get_tokenizer()
    encode = Mock(wraps=tokenizer.encode)
    monkeypatch.setattr(tokenizer, "encode", encode)

    await execute(instance, wire())

    encode.assert_not_called()


@pytest.mark.asyncio
async def test_native_zero_probability_labels_keep_negative_infinity_logprobs(
    native_reads,
):
    class ZeroLabelEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for output in super().generate(*args, **kwargs):
                for logprob in list(output.outputs[0].logprobs[0].values())[:2]:
                    logprob.logprob = -math.inf
                yield output

    response, _ = await execute(await adapter(ZeroLabelEngine()), wire())
    assert response["outcomes"][0]["probabilities"] == [0, 1]
    assert math.isfinite(response["outcomes"][0]["label_mass"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("model_impl", "transformers"),
        ("model_class_overrides", {"Lfm2VlForConditionalGeneration": "custom.Model"}),
        ("convert_type", "classify"),
        ("is_diffusion", True),
    ],
)
async def test_unqualified_native_implementation_is_not_advertiseable(
    native_reads, field, value
):
    engine = Engine()
    setattr(engine.model_config, field, value)
    with pytest.raises(reads.ReadRuntimeError):
        await adapter(engine)

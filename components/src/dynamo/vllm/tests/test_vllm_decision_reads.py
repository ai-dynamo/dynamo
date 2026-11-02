# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native generative decision contracts, using hash-verified CPU strategies."""

import asyncio
import copy
import math
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.llm.exceptions import HttpError
from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_reads as reads
from dynamo.vllm.tests.decision_read_contract_support import (
    adapter,
    execute,
    question,
    wire,
)
from dynamo.vllm.tests.decision_read_test_support import Engine

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.mark.asyncio
async def test_sampling_parameter_type_error_is_a_backend_failure(
    native_reads, monkeypatch
):
    instance = await adapter(Engine())

    def broken_sampling_params(*args):
        raise TypeError("private sampling failure")

    monkeypatch.setattr(instance.strategy, "_sampling_params", broken_sampling_params)
    with pytest.raises(HttpError) as failure:
        await execute(instance, wire())
    assert failure.value.code == 500
    assert isinstance(failure.value.__cause__, TypeError)
    assert instance.engine.admissions == [] and instance.engine.calls == []


@pytest.mark.asyncio
async def test_expanded_question_bytes_are_charged_once_per_actual_render(
    native_reads, monkeypatch
):
    instances = []

    class TrackingBudget(reads.BudgetTokenizer):
        def __init__(self, *args):
            super().__init__(*args)
            self.encodings = []
            instances.append(self)

        def encode(self, text, **kwargs):
            self.encodings.append(text)
            return super().encode(text, **kwargs)

    monkeypatch.setattr(reads, "BudgetTokenizer", TrackingBudget)
    instance = await adapter(Engine())
    request = wire(questions=[question(index) for index in range(4)])
    request["decision_read"]["input"] = "context π " * 100
    prepared = await instance._prepare(request)
    state = instance.strategy.render_state(request["decision_read"]["input"])
    content_bytes = sum(
        len(instance.strategy._content(state, item).encode("utf-8"))
        for item in prepared.questions
    )
    label_bytes = sum(len(text.encode("utf-8")) for text in instances[0].encodings)
    expected = content_bytes + label_bytes
    assert (
        request["decision_read"]["max_encoding_bytes"] - instances[0].bytes == expected
    )
    request["decision_read"]["max_encoding_bytes"] = expected
    assert len((await instance._prepare(request)).reads) == 4
    request["decision_read"]["max_encoding_bytes"] = expected - 1
    with pytest.raises(reads.NativeValidationError):
        await instance._prepare(request)
    assert not instance.engine.calls


@pytest.mark.asyncio
async def test_rendered_prompt_token_work_is_counted_once(native_reads, monkeypatch):
    async def render(self, request, messages, **kwargs):
        ids = [
            9000,
            *self.renderer.get_tokenizer().encode(messages[0]["content"]),
            9001,
        ]
        return None, (
            {
                "type": "token",
                "prompt_token_ids": ids,
                "cache_salt": request.cache_salt,
            },
        )

    monkeypatch.setattr(native_reads.online_renderer, "preprocess_chat", render)
    instance = await adapter(Engine())
    request = wire()
    request["decision_read"]["input"] = "x" * 400
    baseline = await instance._prepare(request)
    request["decision_read"]["max_tokenization_work"] = (
        len(baseline.reads[0].rendered_ids) + 100
    )
    prepared = await instance._prepare(request)
    assert prepared.reads[0].rendered_ids == baseline.reads[0].rendered_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize(
    "fault", ["missing_salt", "changed_salt", "nontext", "invalid_tokens"]
)
async def test_renderer_contract_faults_are_backend_errors_before_admission(
    native_reads, monkeypatch, diffusion, fault
):
    engine = Engine(diffusion=diffusion)
    profile = (
        native_registry.DIFFUSION_PROFILE
        if diffusion
        else native_registry.LIQUID_PROFILE
    )
    instance = await adapter(engine, profile)
    renderer = instance.strategy.context.online_renderer
    preprocess = renderer.preprocess_chat

    async def broken_render(*args, **kwargs):
        conversation, inputs = await preprocess(*args, **kwargs)
        (prompt,) = inputs
        if fault == "missing_salt":
            prompt = {
                key: value for key, value in prompt.items() if key != "cache_salt"
            }
        elif fault == "changed_salt":
            prompt = {**prompt, "cache_salt": "different-isolation-domain"}
        elif fault == "nontext":
            prompt = {**prompt, "type": "embeds"}
        else:
            prompt = {**prompt, "prompt_token_ids": [True]}
        return conversation, [prompt]

    monkeypatch.setattr(renderer, "preprocess_chat", broken_render)
    with pytest.raises(reads.NativeExecutionError) as failure:
        await execute(instance, wire(profile))
    assert failure.value.code == 502
    assert engine.admissions == [] and engine.calls == []


@pytest.mark.asyncio
async def test_diffusion_hybrid_prefix_path_is_not_qualified(native_reads):
    engine = Engine(diffusion=True)
    engine.model_config.is_hybrid = True
    with pytest.raises(reads.NativeRuntimeError, match="runtime"):
        await adapter(engine, native_registry.DIFFUSION_PROFILE)


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
        "cached_tokens": 0,
    }
    assert context == ["native-read-0"]
    assert call[0]["cache_salt"] == "request-isolation"
    assert call[3]["trace_headers"] == {"traceparent": "test"}


@pytest.mark.asyncio
@pytest.mark.parametrize("diffusion", [False, True])
async def test_stateless_reducer_matches_actual_pinned_strategy(
    native_reads, diffusion
):
    engine = Engine(diffusion)
    profile = (
        native_registry.DIFFUSION_PROFILE
        if diffusion
        else native_registry.LIQUID_PROFILE
    )
    instance = await adapter(engine, profile)
    prepared = await instance._prepare(
        wire(profile, [question(), question(1, "Predicate", (True, False))])
    )
    results = []
    for index, read in enumerate(prepared.reads):
        async for result in engine.generate(
            read.prompt, read.sampling_params, f"parent-{index}"
        ):
            results.append(result)
    actual = instance._reduce(prepared, results)
    expected = await instance.strategy.read(
        list(prepared.questions),
        None,
        instance.strategy.render_state("state"),
        request_id="independent-native",
        chat_template_kwargs=None,
        lora_request=None,
        priority=0,
        cache_salt=prepared.cache_salt,
        trace_headers=None,
    )
    assert [(item.probs, item.label_mass) for item in actual] == [
        (item.probs, item.label_mass) for item in expected
    ]


@pytest.mark.asyncio
async def test_diffusion_keeps_thought_outside_deterministic_read_only_canvas(
    native_reads,
):
    engine = Engine(diffusion=True)
    instance = await adapter(engine, native_registry.DIFFUSION_PROFILE)
    assert instance.strategy.joint_reads is False
    request = wire(native_registry.DIFFUSION_PROFILE)
    original = copy.deepcopy(request)
    prepared = await instance._prepare(request)
    repeated = await instance._prepare(request)
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
    with pytest.raises(reads.NativeValidationError):
        await execute(instance, request)
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_diffusion_capacity_reserves_thought_and_entire_canvas(native_reads):
    engine = Engine(diffusion=True)
    instance = await adapter(engine, native_registry.DIFFUSION_PROFILE)
    prepared = await instance._prepare(wire(native_registry.DIFFUSION_PROFILE))
    engine.model_config.max_model_len = (
        len(prepared.reads[0].prompt["prompt_token_ids"]) + 15
    )
    instance.strategy.max_model_len = engine.model_config.max_model_len
    with pytest.raises(reads.NativeValidationError):
        await execute(instance, wire(native_registry.DIFFUSION_PROFILE))
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
    assert engine.admissions == [3]
    assert all(call[0]["cache_salt"] == "request-isolation" for call in engine.calls)
    assert response["usage"]["input_tokens"] == sum(
        len(call[0]["prompt_token_ids"]) for call in engine.calls
    )
    assert response["usage"]["output_tokens"] == 3
    assert response["usage"]["cached_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [False, True])
async def test_read_usage_keeps_validated_cache_measurement(
    native_reads, monkeypatch, prefix
):
    engine = Engine()
    engine.cached_tokens = 3
    engine.model_config.is_hybrid = prefix
    engine.vllm_config.cache_config.enable_prefix_caching = prefix
    outputs = []
    generate = engine.generate

    async def capture_outputs(*args, **kwargs):
        async for output in generate(*args, **kwargs):
            outputs.append(output)
            yield output

    monkeypatch.setattr(engine, "generate", capture_outputs)
    instance = await adapter(engine)
    reduce = instance._reduce

    def reduce_then_change_raw_cache(prepared, children):
        reduced = reduce(prepared, children)
        for output in outputs:
            output.num_cached_tokens = None
        return reduced

    monkeypatch.setattr(instance, "_reduce", reduce_then_change_raw_cache)
    request = wire(questions=[question(), question(1)])
    request["decision_read"]["input"] = "s" * 200
    response, _ = await execute(instance, request)
    assert len(outputs) == 2 + int(prefix)
    assert response["usage"]["cached_tokens"] == 3 * len(outputs)
    assert response["usage"]["input_tokens"] == sum(
        len(output.prompt_token_ids) for output in outputs
    )
    assert response["usage"]["output_tokens"] == sum(
        len(output.outputs[0].token_ids) for output in outputs
    )


@pytest.mark.asyncio
async def test_raw_missing_cache_cannot_be_replaced_by_native_default_zero(
    native_reads,
):
    engine = Engine()
    engine.cached_tokens = None
    with pytest.raises(reads.NativeExecutionError) as failure:
        await execute(await adapter(engine), wire(questions=[question(), question(1)]))
    assert failure.value.code == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [None, True, -1, 1_000_000])
@pytest.mark.parametrize("diffusion", [False, True])
async def test_native_read_cache_failure_drains_admitted_children(
    native_reads, cached, diffusion
):
    engine = Engine(diffusion=diffusion)
    engine.cached_tokens = cached
    profile = (
        native_registry.DIFFUSION_PROFILE
        if diffusion
        else native_registry.LIQUID_PROFILE
    )
    with pytest.raises(reads.NativeExecutionError) as failure:
        await execute(await adapter(engine, profile), wire(profile))
    assert failure.value.code == 502
    assert engine.active == 0
    assert engine.calls
    assert set(engine.aborted) == set(engine.closed)


@pytest.mark.asyncio
async def test_changed_diffusion_canvas_is_a_typed_backend_failure_before_admission(
    native_reads, monkeypatch
):
    engine = Engine(diffusion=True)
    instance = await adapter(engine, native_registry.DIFFUSION_PROFILE)
    sampling_params = instance.strategy._sampling_params

    def changed_sampling_params(*args):
        params = sampling_params(*args)
        params.extra_args = {**params.extra_args, "diffusion_read_only": False}
        return params

    monkeypatch.setattr(instance.strategy, "_sampling_params", changed_sampling_params)
    with pytest.raises(reads.NativeExecutionError) as failure:
        await execute(instance, wire(native_registry.DIFFUSION_PROFILE))
    assert failure.value.code == 502
    assert engine.calls == []
    assert engine.admissions == []


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
    error = (
        reads.NativeValidationError
        if field.startswith("max_")
        else reads.NativeExecutionError
    )
    with pytest.raises(error):
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
    with pytest.raises(reads.NativeRuntimeError):
        await adapter(engine)


def test_native_read_create_has_no_self_asserted_source_parameter():
    import inspect

    assert (
        "source_commit"
        not in inspect.signature(reads.ReadExecutionAdapter.create).parameters
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
    with pytest.raises(reads.NativeExecutionError):
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
    plan = await instance._prepare(request)
    request["decision_read"]["max_input_tokens"] = sum(
        len(read.prompt["prompt_token_ids"]) for read in plan.reads
    )
    with pytest.raises(reads.NativeValidationError):
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
                    raise reads.NativeExecutionError("failed engine child")
                finally:
                    self.closed.append(request_id)
            async for output in super().generate(prompt, params, request_id, **kwargs):
                yield output

    engine = FailingEngine()
    engine.block = asyncio.Event()
    with pytest.raises(reads.NativeExecutionError, match="failed engine child"):
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
    profile = (
        native_registry.DIFFUSION_PROFILE
        if change == "canvas"
        else native_registry.LIQUID_PROFILE
    )
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
    with pytest.raises(reads.NativeRuntimeError):
        await adapter(engine, profile)
    assert engine.calls == []


def test_installed_source_bytes_are_verified_before_importing_optional_strategy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(native_loader, "version", lambda _: "0.33.0")
    package = tmp_path / "vllm"
    package.mkdir()
    installed = package / "source.py"
    installed.write_text("foreign source", encoding="utf-8")
    monkeypatch.setattr(
        native_loader, "PINNED_SOURCE_BLOBS", (("vllm/source.py", "wrong"),)
    )
    calls = []

    def import_package(name):
        calls.append(name)
        assert name == "vllm"
        return type("Package", (), {"__file__": str(package / "__init__.py")})

    monkeypatch.setattr(cache, "import_module", import_package)
    with pytest.raises(reads.NativeRuntimeError, match="differs from pinned"):
        native_loader.load_native()
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
async def test_float32_vocabulary_mass_roundoff_is_bounded(native_reads):
    class ConfidentEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for output in super().generate(*args, **kwargs):
                values = list(output.outputs[0].logprobs[0].values())
                values[0].logprob = 0.0
                for value in values[1:]:
                    value.logprob = -20.0
                yield output

    response, _ = await execute(await adapter(ConfidentEngine()), wire())
    outcome = response["outcomes"][0]
    assert outcome["label_mass"] == 1.0
    assert sum(outcome["probabilities"]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_excess_vocabulary_mass_is_rejected_not_clamped(native_reads):
    class ExcessMassEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for output in super().generate(*args, **kwargs):
                values = output.outputs[0].logprobs[0].values()
                for logprob in values:
                    logprob.logprob = math.log(1.01 / len(values))
                yield output

    with pytest.raises(reads.NativeExecutionError, match="label mass"):
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
    with pytest.raises(reads.NativeValidationError):
        await execute(instance, wire())
    assert engine.calls == []
    assert engine.admissions == []


@pytest.mark.asyncio
async def test_native_reduction_does_not_repeat_tokenization_after_engine_admission(
    native_reads, monkeypatch
):
    engine = Engine()
    instance = await adapter(engine)
    prepared = await instance._prepare(wire())
    monkeypatch.setattr(instance, "_prepare", AsyncMock(return_value=prepared))
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
    with pytest.raises(reads.NativeRuntimeError):
        await adapter(engine)

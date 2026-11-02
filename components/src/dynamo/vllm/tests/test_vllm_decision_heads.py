# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the optional, unregistered native head executor."""

import asyncio
import copy
import importlib.util
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_heads as heads
from dynamo.vllm import decision_native_loader as native_loader
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm import decision_native_wire as native_wire
from dynamo.vllm.tests.decision_head_test_support import (
    DecisionContext,
    Engine,
    NativeApi,
    Tokenizer,
    adapter,
    execute,
    question,
    wire,
)

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_optional_head_runtime_targets_release_candidate():
    assert native_loader.PINNED_VLLM_TAG == "v0.33.0"
    assert (
        native_loader.PINNED_VLLM_COMMIT == "4ba787ec9959e566d3b587a796dff07673324da5"
    )


def test_native_create_has_no_self_asserted_source_parameter():
    import inspect

    assert (
        "source_commit"
        not in inspect.signature(heads.HeadExecutionAdapter.create).parameters
    )


def test_missing_native_backend_is_a_runtime_error(monkeypatch):
    monkeypatch.setattr(native_loader, "version", lambda _: "0.33.0")

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cache, "import_module", missing)
    with pytest.raises(heads.NativeRuntimeError, match="unavailable"):
        native_loader.load_native()


def test_modified_installed_source_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(native_loader, "version", lambda _: "0.33.0")
    changed = tmp_path / "changed.py"
    changed.write_text("modified source")
    monkeypatch.setattr(
        native_loader, "PINNED_SOURCE_BLOBS", (("changed.py", "wrong"),)
    )
    monkeypatch.setattr(
        cache,
        "import_module",
        lambda name: SimpleNamespace(__file__=tmp_path / "vllm" / "__init__.py"),
    )
    with pytest.raises(heads.NativeRuntimeError, match="differs"):
        native_loader.load_native()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model_config", "runner_type", "generate"),
        ("model_config", "architecture", "Qwen3ForCausalLM"),
        ("parallel_config", "data_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
    ],
)
async def test_runtime_gate_rejects_unqualified_config(native, section, field, value):
    engine = Engine()
    setattr(getattr(engine.vllm_config, section), field, value)
    with pytest.raises(heads.NativeRuntimeError):
        await adapter(engine)
    assert not engine.calls


@pytest.mark.asyncio
async def test_d3_native_temperature_and_predicate_inverse(native):
    engine = Engine()
    instance = await adapter(engine)
    response = await execute(
        instance, wire(questions=[question(kind="Predicate", values=(True, False))])
    )
    expected = math.e / (1 + math.e)
    assert response["outcomes"] == [
        {"ordinal": 0, "probabilities": pytest.approx([expected, 1 - expected])}
    ]
    assert "label_mass" not in response["outcomes"][0]
    call = engine.calls[0]
    assert call["pooling_params"].task == "classify"
    assert call["pooling_params"].use_activation is False
    assert response["usage"] == {
        "input_tokens": len(call["prompt"]["prompt_token_ids"]),
        "output_tokens": 0,
        "cached_tokens": 1,
    }
    assert engine.admissions == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile", [native_registry.CLEF_FLASH_PROFILE, native_registry.CLEF_OMNI_PROFILE]
)
async def test_clef_joint_read_restores_each_order_and_counts_usage_once(
    native, profile
):
    architecture = (
        "ClefFlashForJointSchema"
        if profile == native_registry.CLEF_FLASH_PROFILE
        else "ClefOmniForJointSchema"
    )
    engine = Engine(architecture)
    engine.values = [0.7, 0.3, 0.4, 0.6]
    instance = await adapter(engine, profile)
    response = await execute(
        instance,
        wire(profile, [question(), question(1, "Predicate", (True, False))]),
    )
    assert len(engine.calls) == 1
    params = engine.calls[0]["pooling_params"]
    assert params.task == "token_classify"
    assert params.extra_kwargs["joint_schema"]["option_counts"] == [2, 2]
    assert response["outcomes"][0]["probabilities"] == [0.3, 0.7]
    assert response["outcomes"][1]["probabilities"] == [0.4, 0.6]
    assert response["usage"]["input_tokens"] == len(
        engine.calls[0]["prompt"]["prompt_token_ids"]
    )
    assert engine.admissions == [1]


@pytest.mark.asyncio
async def test_typed_choices_and_caller_names_are_preserved_without_mutating_request(
    native,
):
    instance = await adapter(Engine())
    request = wire(questions=[question(values=(True, "true"))])
    original = copy.deepcopy(request)
    prepared = await instance._prepare(request)
    prompt = bytes(prepared.reads[0].prompt_ids).decode()
    assert '\ntrue\n"true"' in prompt
    assert "client-name" not in prompt
    assert request == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_version", True),
        ("protocol_version", 2),
        ("profile", "vllm_causal_decision_v1"),
        ("temperature", 0.5),
        ("cache_salt", ""),
        ("max_input_tokens", 1),
        ("max_tokenization_work", 1),
        ("max_encoding_bytes", 1),
        ("unknown", True),
    ],
)
async def test_invalid_wire_is_rejected_before_any_engine_work(native, field, value):
    engine = Engine()
    instance = await adapter(engine)
    request = wire()
    request["decision_head"][field] = value
    error = (
        heads.NativeValidationError
        if field.startswith("max_")
        else heads.NativeExecutionError
    )
    with pytest.raises(error):
        await execute(instance, request)
    assert not engine.calls and not engine.admissions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,error",
    [
        (lambda q: {**q, "ordinal": 1}, heads.NativeExecutionError),
        (lambda q: {**q, "kind": "Unknown"}, heads.NativeExecutionError),
        (lambda q: {**q, "kind": []}, heads.NativeExecutionError),
        (lambda q: {**q, "candidates": []}, heads.NativeValidationError),
        (lambda q: {**q, "candidates": [{"value": 1}]}, heads.NativeValidationError),
        (
            lambda q: {**q, "candidates": [{"value": "a"}, {"value": "a"}]},
            heads.NativeValidationError,
        ),
    ],
)
async def test_invalid_canonical_questions_fail_preflight(native, change, error):
    instance = await adapter(Engine())
    with pytest.raises(error):
        await instance._prepare(wire(questions=[change(question())]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("finished", False),
        ("request_id", "other-request"),
        ("num_cached_tokens", -1),
        ("num_cached_tokens", None),
        ("num_cached_tokens", True),
        ("num_cached_tokens", 99999),
        ("prompt_token_ids", [999]),
        ("error", SimpleNamespace(message="sensitive payload", code="failure")),
    ],
)
async def test_bad_terminal_never_becomes_success(native, field, value):
    engine = Engine()
    engine.mutate_output = lambda output: SimpleNamespace(
        **{**vars(output), field: value}
    )
    instance = await adapter(engine)
    with pytest.raises(heads.NativeExecutionError) as failure:
        await execute(instance, wire())
    assert "sensitive payload" not in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[], [1.0], [0.0, float("nan"), 2.0]])
async def test_bad_d3_vectors_fail(native, values):
    engine = Engine()
    engine.values = values
    with pytest.raises(heads.NativeExecutionError):
        await execute(await adapter(engine), wire())


@pytest.mark.asyncio
async def test_data_after_terminal_is_rejected_and_aborted(native):
    engine = Engine()
    original_encode = engine.encode

    async def duplicate(**kwargs):
        async for output in original_encode(**kwargs):
            yield output
            yield output

    engine.encode = duplicate
    with pytest.raises(heads.NativeExecutionError, match="after its terminal"):
        await execute(await adapter(engine), wire())
    assert engine.aborted == ["parent-head-0"]


@pytest.mark.asyncio
async def test_empty_engine_stream_is_not_success(native):
    engine = Engine()

    async def empty(**kwargs):
        for output in ():
            yield output

    engine.encode = empty
    with pytest.raises(heads.NativeExecutionError, match="finished"):
        await execute(await adapter(engine), wire())
    assert engine.aborted == ["parent-head-0"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values", [[0.5], [0.4, 0.4], [-0.1, 1.1], [0.0, 0.0], [0.500002, 0.5]]
)
async def test_bad_clef_distributions_fail(native, values):
    engine = Engine("ClefFlashForJointSchema")
    engine.values = values
    instance = await adapter(engine, native_registry.CLEF_FLASH_PROFILE)
    with pytest.raises(heads.NativeExecutionError):
        await execute(instance, wire(native_registry.CLEF_FLASH_PROFILE))


@pytest.mark.asyncio
async def test_unknown_cache_usage_is_a_typed_executor_failure(native):
    engine = Engine()
    engine.mutate_output = lambda output: SimpleNamespace(
        **{**vars(output), "num_cached_tokens": None}
    )
    with pytest.raises(heads.NativeExecutionError) as failure:
        await execute(await adapter(engine), wire())
    assert failure.value.code == 502


@pytest.mark.asyncio
async def test_measured_zero_cache_usage_remains_success(native):
    engine = Engine()
    engine.mutate_output = lambda output: SimpleNamespace(
        **{**vars(output), "num_cached_tokens": 0}
    )
    response = await execute(await adapter(engine), wire())
    assert response["usage"]["cached_tokens"] == 0


@pytest.mark.asyncio
async def test_d3_decoding_calls_the_pinned_softmax_once(native):
    instance = await adapter(Engine())
    softmax = Mock(return_value=[0.25, 0.75])
    instance.native.questions = SimpleNamespace(
        **{**vars(instance.native.questions), "label_softmax": softmax}
    )
    result = await execute(instance, wire())
    softmax.assert_called_once_with([0.0, 1.0])
    assert result["outcomes"][0]["probabilities"] == [0.25, 0.75]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [0.9741412401199341, 0.02585877664387226],
        [0.896975576877594, 0.1030244305729866],
        [0.051193997263908386, 0.2584656774997711, 0.6903403401374817],
        [0.4999999701976776, 0.5],
        [0.3333333432674408] * 3,
        [0.0, 0.20000000298023224, 0.800000011920929],
    ],
)
async def test_joint_head_canonicalizes_native_roundoff(native, values):
    engine = Engine("ClefFlashForJointSchema")
    engine.values = values
    instance = await adapter(engine, native_registry.CLEF_FLASH_PROFILE)
    request = wire(
        native_registry.CLEF_FLASH_PROFILE,
        [question(values=tuple("zyx"[: len(values)]))],
    )
    order = (await instance._prepare(request)).reads[0].orders[0]
    result = await execute(instance, request)
    probabilities = result["outcomes"][0]["probabilities"]
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-15)
    for index, original in zip(order, values):
        assert probabilities[index] == original / math.fsum(values)
        assert abs(probabilities[index] - original) < 1e-6


@pytest.mark.asyncio
async def test_all_preflight_finishes_before_first_d3_read(native):
    engine = Engine()
    engine.model_config.max_model_len = 60
    instance = await adapter(engine)
    questions = [question(), {**question(1), "instructions": "x" * 100}]
    with pytest.raises(heads.NativeValidationError):
        await execute(instance, wire(questions=questions))
    assert not engine.calls


@pytest.mark.asyncio
async def test_d3_concurrency_is_bounded_and_cancellation_drains_children(native):
    engine = Engine()
    engine.block = asyncio.Event()
    instance = await adapter(engine, max_concurrency=2)
    request = wire(questions=[question(i) for i in range(6)])
    task = asyncio.create_task(execute(instance, request))
    await asyncio.wait_for(engine.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.peak_active <= 2
    assert engine.active == 0
    assert set(engine.aborted) == {call["request_id"] for call in engine.calls}


@pytest.mark.asyncio
async def test_cancelled_request_never_starts_engine_work(native):
    engine = Engine()
    instance = await adapter(engine)
    with pytest.raises(asyncio.CancelledError):
        await instance.execute(
            wire(), DecisionContext(), request_id="cancelled", cancelled=lambda: True
        )
    assert not engine.calls


@pytest.mark.asyncio
async def test_installed_pinned_native_preparation():
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("Pinned optional vLLM runtime is not installed")
    try:
        native_loader.load_native()
    except heads.NativeRuntimeError:
        pytest.skip("Installed vLLM sources do not match the pinned head runtime")
    engine = Engine("ClefFlashForJointSchema")
    engine.model_config = SimpleNamespace(
        **vars(engine.model_config),
        model="test-clef-flash",
        hf_config=SimpleNamespace(model_type="qwen3_5"),
    )
    engine.vllm_config = SimpleNamespace(
        **{**vars(engine.vllm_config), "model_config": engine.model_config}
    )
    tokenizer = Tokenizer()
    engine.renderer = SimpleNamespace(
        tokenizer=tokenizer, get_tokenizer=lambda: tokenizer
    )
    instance = await adapter(engine, native_registry.CLEF_FLASH_PROFILE)
    prepared = await instance._prepare(wire(native_registry.CLEF_FLASH_PROFILE))
    assert len(prepared.reads) == 1
    params = prepared.reads[0].pooling_params
    assert params.task == "token_classify"
    assert params.extra_kwargs["joint_schema"]["option_counts"] == [2]


@pytest.mark.parametrize(
    "evidence,expected",
    [
        ({"description": "Low"}, "Low"),
        ({"label": "Low"}, "Low"),
        ({"label": "Low", "description": "Small impact"}, "Low - Small impact"),
        ({"description": {"impact": ["Low", "minor"]}}, '{"impact": ["Low", "minor"]}'),
    ],
)
def test_installed_native_score_evidence_is_rendered_once(evidence, expected):
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("Pinned optional vLLM runtime is not installed")
    try:
        native = native_loader.load_native()
    except heads.NativeRuntimeError:
        pytest.skip("Installed vLLM sources do not match the pinned head runtime")
    raw = question(kind="Score", values=("0", "1"))
    raw["instructions"] = "Rate the impact."
    raw["candidates"][0].update(evidence)
    raw["candidates"][1].update(label="High")
    result = native_wire._question(raw, 0, native, 10)
    labeled = native.questions.Question(
        result.id,
        result.type,
        result.instructions,
        result.options,
        result.type.label_set[: len(result.options)],
    )
    prompt = (
        f"Question: Rate the impact.\n0: {expected}\n1: High\n"
        "Answer with the number of one level only."
    )
    assert result.type.prompt(labeled) == prompt
    assert (
        native.strategies.DiffusionGemmaCanvasStrategy._content(
            None, "Customer requests a refund.", labeled
        )
        == f"Customer requests a refund.\n\n{prompt}"
    )
    assert [(o.name, o.description) for o in result.options] == [
        (expected, None),
        ("High", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [None, "4", True, 0, -1])
async def test_head_rejects_scheduler_capacity_before_arithmetic(native, capacity):
    engine = Engine()
    engine.vllm_config.scheduler_config.max_num_seqs = capacity
    with pytest.raises(heads.NativeRuntimeError, match="concurrency"):
        await adapter(engine)


@pytest.mark.parametrize(
    "kind,values", [("Choice", ("billing", "technical")), ("Predicate", (True, False))]
)
def test_non_score_option_descriptions_are_preserved(kind, values):
    raw = question(kind=kind, values=values)
    raw["candidates"][0]["description"] = {"evidence": ["Payments"]}
    result = native_wire._question(raw, 0, NativeApi(), 10)
    assert result.options[0].description == '{"evidence": ["Payments"]}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("model_impl", "transformers"),
        ("model_class_overrides", {"D3ForSequenceClassification": "custom.Model"}),
        ("convert_type", "classify"),
        ("is_diffusion", True),
        ("is_encoder_decoder", True),
    ],
)
async def test_unqualified_native_head_implementation_is_rejected(native, field, value):
    engine = Engine()
    setattr(engine.model_config, field, value)
    with pytest.raises(heads.NativeRuntimeError):
        await adapter(engine)

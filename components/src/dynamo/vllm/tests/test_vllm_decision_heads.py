# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the optional, unregistered native head executor."""

import asyncio
import copy
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dynamo.vllm import decision_heads as heads

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@dataclass(frozen=True)
class Option:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: object
    instructions: str
    options: tuple
    labels: tuple


class Tensor:
    def __init__(self, values):
        self.values = values
        self.ndim = 1

    def tolist(self):
        return list(self.values)


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class D3Strategy:
    def __init__(self, context):
        self.context = context
        self.codes = ["A", "B", "C"]
        self.temperature = 2.0
        self.tokenizer = context.online_renderer.renderer.get_tokenizer()
        self.max_model_len = context.engine_client.model_config.max_model_len

    def limits(self):
        return SimpleNamespace(max_questions=64, max_options=len(self.codes))

    def render_state(self, state):
        return "(empty)" if state is None or state == "" else str(state)

    def _options(self, question):
        if question.type.name == "noul":
            return ["No / false", "Yes / true"], [1, 0]
        return [option.name for option in question.options], list(
            range(len(question.options))
        )

    def _prompt(self, state, question, media):
        assert media == []
        names, _ = self._options(question)
        return "\n".join([state, question.instructions, *names])

    def _tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


class JointStrategy(D3Strategy):
    def limits(self):
        return SimpleNamespace(max_questions=64, max_options=26)

    def encode(self, questions, state):
        orders = [
            sorted(range(len(q.options)), key=lambda i: q.options[i].name)
            if q.type.name == "choice"
            else list(range(len(q.options)))
            for q in questions
        ]
        prompt = self._tokens(state + "|" + "|".join(q.id for q in questions))
        schema = {
            "question_types": [0 if q.type.name == "noul" else 1 for q in questions],
            "question_spans": [[-2, -1]] * len(questions),
            "option_spans": [[-2, -1]] * sum(len(q.options) for q in questions),
            "option_counts": [len(q.options) for q in questions],
        }
        return prompt, schema, orders


class NativeApi:
    overload_errors = ()
    strategies = SimpleNamespace(
        D3Strategy=D3Strategy,
        ClefFlashStrategy=JointStrategy,
        JointSchemaStrategy=JointStrategy,
        ReadContext=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    questions = SimpleNamespace(
        Option=Option,
        Question=Question,
        StructuredDecisionError=ValueError,
        get_question_type=lambda name: SimpleNamespace(name=name),
    )

    @staticmethod
    def pooling_params(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def tokens_input(token_ids, *, cache_salt):
        return {"prompt_token_ids": list(token_ids), "cache_salt": cache_salt}

    @staticmethod
    def online_renderer(**kwargs):
        return SimpleNamespace(**kwargs)


class Engine:
    def __init__(self, architecture="D3ForSequenceClassification"):
        self.model_config = SimpleNamespace(
            architecture=architecture,
            runner_type="pooling",
            max_model_len=8192,
            is_encoder_decoder=False,
            is_diffusion=False,
            model_impl="auto",
            model_class_overrides={},
            convert_type="none",
        )
        self.vllm_config = SimpleNamespace(
            model_config=self.model_config,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False, max_num_seqs=8
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
        )
        self.renderer = SimpleNamespace(get_tokenizer=lambda: Tokenizer())
        self.calls = []
        self.aborted = []
        self.admissions = []
        self.values = [0.0, 2.0, 99.0]
        self.mutate_output = lambda output: output
        self.block = None
        self.started = asyncio.Event()
        self.active = 0
        self.peak_active = 0

    async def get_supported_tasks(self):
        return ("classify", "token_classify")

    def check_admission(self, count):
        self.admissions.append(count)

    async def abort(self, request_id):
        self.aborted.append(request_id)

    async def encode(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            if self.block is not None:
                await self.block.wait()
            output = SimpleNamespace(
                request_id=kwargs["request_id"],
                prompt_token_ids=kwargs["prompt"]["prompt_token_ids"],
                num_cached_tokens=1,
                finished=True,
                error=None,
                outputs=SimpleNamespace(data=Tensor(self.values)),
            )
            yield self.mutate_output(output)
        finally:
            self.active -= 1


@asynccontextmanager
async def monitor(context, request_id):
    context.append(request_id)
    yield


def question(ordinal=0, kind="Choice", values=("z", "a")):
    return {
        "ordinal": ordinal,
        "id": "client-name-not-instructions",
        "kind": kind,
        "instructions": "Choose.",
        "candidates": [
            {"value": value, "description": None, "label": None} for value in values
        ],
    }


def wire(profile=None, questions=None):
    return {
        "decision_head": {
            "protocol_version": 1,
            "profile": profile or heads.D3_PROFILE,
            "model": "test/model",
            "input": "state",
            "questions": questions or [question()],
            "temperature": 1.0,
            "cache_salt": "isolated",
            "max_input_tokens": 16384,
            "max_tokenization_work": 65536,
            "max_encoding_bytes": 65536,
        }
    }


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(heads, "_load_native", lambda commit: NativeApi())


async def adapter(engine, profile=None, **kwargs):
    return await heads.HeadExecutionAdapter.create(
        engine,
        profile or heads.D3_PROFILE,
        source_commit=heads.PINNED_VLLM_COMMIT,
        abort_monitor=monitor,
        **kwargs,
    )


async def execute(instance, request, **kwargs):
    return await instance.execute(
        request, [], request_id="parent", cancelled=lambda: False, **kwargs
    )


def test_wrong_source_pin_fails_before_backend_import(monkeypatch):
    load = SimpleNamespace(called=False)

    def reject_import(name):
        load.called = True
        raise AssertionError(name)

    monkeypatch.setattr(heads, "import_module", reject_import)
    with pytest.raises(heads.HeadRuntimeError, match="source"):
        heads._load_native("unverified")
    assert load.called is False


def test_missing_native_backend_is_a_runtime_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(heads, "import_module", missing)
    with pytest.raises(heads.HeadRuntimeError, match="unavailable"):
        heads._load_native(heads.PINNED_VLLM_COMMIT)


def test_modified_installed_source_is_rejected(monkeypatch, tmp_path):
    changed = tmp_path / "changed.py"
    changed.write_text("modified source")
    monkeypatch.setattr(
        heads, "import_module", lambda name: SimpleNamespace(__file__=changed)
    )
    with pytest.raises(heads.HeadRuntimeError, match="differs"):
        heads._load_native(heads.PINNED_VLLM_COMMIT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model_config", "runner_type", "generate"),
        ("model_config", "architecture", "Qwen3ForCausalLM"),
        ("parallel_config", "data_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "enable_chunked_prefill", True),
    ],
)
async def test_runtime_gate_rejects_unqualified_config(native, section, field, value):
    engine = Engine()
    setattr(getattr(engine.vllm_config, section), field, value)
    with pytest.raises(heads.HeadRuntimeError):
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
@pytest.mark.parametrize("profile", [heads.CLEF_FLASH_PROFILE, heads.CLEF_OMNI_PROFILE])
async def test_clef_joint_read_restores_each_order_and_counts_usage_once(
    native, profile
):
    architecture = (
        "ClefFlashForJointSchema"
        if profile == heads.CLEF_FLASH_PROFILE
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
    prepared = instance.prepare(request)
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
    with pytest.raises(heads.HeadValidationError):
        await execute(instance, request)
    assert not engine.calls and not engine.admissions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        lambda q: {**q, "ordinal": 1},
        lambda q: {**q, "kind": "Unknown"},
        lambda q: {**q, "kind": []},
        lambda q: {**q, "candidates": []},
        lambda q: {**q, "candidates": [{"value": 1}]},
        lambda q: {**q, "candidates": [{"value": "a"}, {"value": "a"}]},
    ],
)
async def test_invalid_canonical_questions_fail_preflight(native, change):
    instance = await adapter(Engine())
    with pytest.raises(heads.HeadValidationError):
        instance.prepare(wire(questions=[change(question())]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("finished", False),
        ("request_id", "other-request"),
        ("num_cached_tokens", -1),
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
    with pytest.raises(heads.HeadExecutionError) as failure:
        await execute(instance, wire())
    assert "sensitive payload" not in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[], [1.0], [0.0, float("nan"), 2.0]])
async def test_bad_d3_vectors_fail(native, values):
    engine = Engine()
    engine.values = values
    with pytest.raises(heads.HeadExecutionError):
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
    with pytest.raises(heads.HeadExecutionError, match="after its terminal"):
        await execute(await adapter(engine), wire())
    assert engine.aborted == ["parent-head-0"]


@pytest.mark.asyncio
async def test_empty_engine_stream_is_not_success(native):
    engine = Engine()

    async def empty(**kwargs):
        for output in ():
            yield output

    engine.encode = empty
    with pytest.raises(heads.HeadExecutionError, match="finished"):
        await execute(await adapter(engine), wire())
    assert engine.aborted == ["parent-head-0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[0.5], [0.4, 0.4], [-0.1, 1.1]])
async def test_bad_clef_distributions_fail(native, values):
    engine = Engine("ClefFlashForJointSchema")
    engine.values = values
    instance = await adapter(engine, heads.CLEF_FLASH_PROFILE)
    with pytest.raises(heads.HeadExecutionError):
        await execute(instance, wire(heads.CLEF_FLASH_PROFILE))


@pytest.mark.asyncio
async def test_unknown_cache_usage_stays_unknown(native):
    engine = Engine()
    engine.mutate_output = lambda output: SimpleNamespace(
        **{**vars(output), "num_cached_tokens": None}
    )
    response = await execute(await adapter(engine), wire())
    assert response["usage"]["cached_tokens"] is None


@pytest.mark.asyncio
async def test_all_preflight_finishes_before_first_d3_read(native):
    engine = Engine()
    engine.model_config.max_model_len = 60
    instance = await adapter(engine)
    questions = [question(), {**question(1), "instructions": "x" * 100}]
    with pytest.raises(heads.HeadValidationError):
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
            wire(), [], request_id="cancelled", cancelled=lambda: True
        )
    assert not engine.calls

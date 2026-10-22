# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the optional, unregistered native head executor."""

import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from uuid import uuid4

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


@pytest.mark.asyncio
async def test_installed_pinned_native_preparation():
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("Pinned optional vLLM runtime is not installed")
    try:
        heads._load_native(heads.PINNED_VLLM_COMMIT)
    except heads.HeadRuntimeError:
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
    instance = await adapter(engine, heads.CLEF_FLASH_PROFILE)
    prepared = instance.prepare(wire(heads.CLEF_FLASH_PROFILE))
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
        native = heads._load_native(heads.PINNED_VLLM_COMMIT)
    except heads.HeadRuntimeError:
        pytest.skip("Installed vLLM sources do not match the pinned head runtime")
    raw = question(kind="Score", values=("0", "1"))
    raw["instructions"] = "Rate the impact."
    raw["candidates"][0].update(evidence)
    raw["candidates"][1].update(label="High")
    result = heads._question(raw, 0, native, 10)
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


@pytest.mark.parametrize(
    "kind,values", [("Choice", ("billing", "technical")), ("Predicate", (True, False))]
)
def test_non_score_option_descriptions_are_preserved(kind, values):
    raw = question(kind=kind, values=values)
    raw["candidates"][0]["description"] = {"evidence": ["Payments"]}
    result = heads._question(raw, 0, NativeApi(), 10)
    assert result.options[0].description == '{"evidence": ["Payments"]}'


@pytest.fixture
def pinned_preparation(monkeypatch):
    source_repo = os.environ.get("DYNAMO_VLLM_SOURCE_REPO")
    if source_repo is None:
        pytest.skip(
            "Set DYNAMO_VLLM_SOURCE_REPO for exact-source CPU preparation fixtures"
        )

    def source(module_name):
        result = subprocess.run(
            [
                "git",
                "-c",
                "gc.auto=0",
                "-C",
                source_repo,
                "show",
                f"{heads.PINNED_VLLM_COMMIT}:{module_name.replace('.', '/')}.py",
            ],
            check=True,
            capture_output=True,
            timeout=10,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        expected = dict(heads._PINNED_SOURCES)[module_name]
        assert hashlib.sha256(result.stdout).hexdigest() == expected
        return result.stdout.decode()

    question_name = f"_pinned_head_questions_{uuid4().hex}"
    question_module = ModuleType(question_name)
    monkeypatch.setitem(sys.modules, question_name, question_module)
    exec(  # noqa: S102 -- Executes hash-verified upstream fixtures, never request data.
        compile(
            source("vllm.entrypoints.openai.decisions.question_types"),
            question_name,
            "exec",
        ),
        question_module.__dict__,
    )
    tree = ast.parse(source("vllm.entrypoints.openai.decisions.strategies"))
    selected = {
        "state_text",
        "DecisionLimits",
        "D3Strategy",
        "JointSchemaStrategy",
        "ClefFlashStrategy",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected
    ]

    class ReadStrategy:
        def __init__(self, context):
            self.context = context

    strategy_module = ModuleType(f"_pinned_head_strategies_{uuid4().hex}")
    monkeypatch.setitem(sys.modules, strategy_module.__name__, strategy_module)
    namespace = strategy_module.__dict__
    namespace.update(
        {
            "ReadStrategy": ReadStrategy,
            "json": json,
            "dataclass": dataclass,
            "LABELS": question_module.LABELS,
            "StructuredDecisionError": question_module.StructuredDecisionError,
        }
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Executes hash-verified upstream fixtures, never request data.
        compile(ast.fix_missing_locations(module), "pinned-head-strategies", "exec"),
        namespace,
    )
    api = NativeApi()
    api.questions = question_module
    api.strategies = SimpleNamespace(
        **{name: namespace[name] for name in selected if name != "state_text"},
        ReadContext=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(heads, "_load_native", lambda commit: api)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_exact_pinned_d3_prompt_and_order(pinned_preparation):
    engine = Engine()
    engine.model_config.hf_config = SimpleNamespace(
        decision_config={"codes": ["A", "B", "C"], "temperature": 2.0}
    )
    instance = await adapter(engine)
    prepared = instance.prepare(
        wire(questions=[question(kind="Predicate", values=(True, False))])
    )
    prompt = bytes(prepared.reads[0].prompt_ids).decode()
    assert (
        "State:\nstate\n\nQuestion:\nChoose.\n\nOptions:\nA: No / false\nB: Yes / true"
        in prompt
    )
    assert prompt.endswith("Reply with only the code of the best option.")
    assert prepared.reads[0].orders == ((1, 0),)
    assert "client-name" not in prompt


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "profile,architecture",
    [
        (heads.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema"),
        (heads.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema"),
    ],
)
async def test_exact_pinned_clef_joint_schema_and_state_truncation(
    pinned_preparation, profile, architecture
):
    engine = Engine(architecture)
    engine.model_config.max_model_len = 1200
    instance = await adapter(engine, profile)
    questions = [question(), question(1, "Predicate", (True, False))]
    request = wire(profile, questions)
    request["decision_head"]["input"] = "x" * 3000
    prepared = instance.prepare(request)
    read = prepared.reads[0]
    schema = read.pooling_params.extra_kwargs["joint_schema"]
    assert len(read.prompt_ids) == 1200
    assert schema["question_types"] == [1, 0]
    assert schema["option_counts"] == [2, 2]
    assert read.orders == ((1, 0), (0, 1))
    prompt = bytes(read.prompt_ids).decode()
    assert "JOINT SCHEMA DECISIONS:" in prompt
    assert "ID: question-0" in prompt and "ID: question-1" in prompt
    for start, end in [*schema["question_spans"], *schema["option_spans"]]:
        assert -len(read.prompt_ids) <= start < end <= 0


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
    with pytest.raises(heads.HeadRuntimeError):
        await adapter(engine)

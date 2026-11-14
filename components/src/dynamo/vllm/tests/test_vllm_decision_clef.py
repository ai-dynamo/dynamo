# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused native joint-head tests without a model checkpoint."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.llm.exceptions import HttpError, InvalidArgument
from dynamo.vllm import decision_clef as clef

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class Strategy:
    prefix = [1]
    suffix = [2]
    max_length = 2048
    tokenizer = Tokenizer()

    def render_state(self, value):
        return value

    def encode(self, questions, state):
        tokens = self.tokenizer.encode(state)
        orders = [list(reversed(range(len(q.options)))) for q in questions]
        return (
            self.prefix + tokens[: self.max_length - 2] + self.suffix,
            {"option_counts": [len(q.options) for q in questions]},
            orders,
        )


class Context:
    metadata = {"dynamo.internal.decision": "head"}

    def __init__(self):
        self.cancelled = asyncio.Event()

    def id(self):
        return "native-request"

    def is_stopped(self):
        return self.cancelled.is_set()

    def is_killed(self):
        return self.cancelled.is_set()

    async def async_killed_or_stopped(self):
        await self.cancelled.wait()

    def trace_headers(self):
        return {"traceparent": "test-trace"}


@pytest.fixture
def wire():
    return {
        "decision_head": {
            "protocol_version": 1,
            "profile": clef.CLEF_FLASH_PROFILE,
            "model": "clef-alias",
            "input": "state",
            "temperature": 1,
            "cache_salt": "request-local-salt",
            "max_input_tokens": 1024,
            "max_tokenization_work": 2048,
            "max_encoding_bytes": 4096,
            "questions": [
                {
                    "ordinal": 0,
                    "kind": "Choice",
                    "instructions": "department",
                    "candidates": [{"value": True}, {"value": "true"}],
                },
                {
                    "ordinal": 1,
                    "kind": "Predicate",
                    "instructions": "approved",
                    "candidates": [{"value": True}, {"value": False}],
                },
                {
                    "ordinal": 2,
                    "kind": "Score",
                    "instructions": "severity",
                    "candidates": [
                        {"value": "0", "label": "low"},
                        {"value": "1", "label": "high"},
                    ],
                },
            ],
        }
    }


@pytest.fixture
def executor():
    @asynccontextmanager
    async def abort_monitor(context, request_id):
        yield

    async def encode(**kwargs):
        output = SimpleNamespace(
            prompt_token_ids=kwargs["prompt"]["prompt_token_ids"],
            num_cached_tokens=0,
            outputs=SimpleNamespace(
                data=SimpleNamespace(
                    ndim=1,
                    tolist=lambda: [0.2, 0.8]
                    * len(
                        kwargs["pooling_params"]["extra_kwargs"]["joint_schema"][
                            "option_counts"
                        ]
                    ),
                )
            ),
            error=None,
            finished=True,
        )
        yield output

    engine = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        encode=Mock(side_effect=encode),
        abort=AsyncMock(),
    )
    questions = SimpleNamespace(
        Option=lambda name, description=None: SimpleNamespace(
            name=name, description=description
        ),
        Question=lambda **kwargs: SimpleNamespace(**kwargs),
        get_question_type=lambda name: SimpleNamespace(name=name),
        StructuredDecisionError=ValueError,
    )
    native = SimpleNamespace(
        questions=questions,
        pooling_params=lambda **kwargs: kwargs,
        tokens_input=lambda ids, **kwargs: {"prompt_token_ids": ids, **kwargs},
        overload_errors=(OverflowError,),
        retryable_errors=(ConnectionError,),
        engine_dead_error=BrokenPipeError,
    )
    handler = SimpleNamespace(
        engine_client=engine,
        _abort_monitor=abort_monitor,
        shutdown_event=asyncio.Event(),
    )
    return clef.ClefDecisionExecutor(handler, native, Strategy(), max_concurrency=2)


@pytest.mark.asyncio
async def test_joint_execution_preserves_order_typed_options_and_accounting(
    executor, wire
):
    result = await executor.execute(wire, Context())
    assert result["profile"] == clef.CLEF_FLASH_PROFILE
    assert result["finish_reason"] == "stop"
    assert [row["ordinal"] for row in result["outcomes"]] == [0, 1, 2]
    assert result["outcomes"][0]["probabilities"] == pytest.approx([0.8, 0.2])
    assert result["usage"] == {
        "input_tokens": 7,
        "output_tokens": 0,
        "cached_tokens": 0,
    }
    call = executor.handler.engine_client.encode.call_args.kwargs
    assert call["prompt"]["cache_salt"] == "request-local-salt"
    assert call["pooling_params"]["extra_kwargs"]["joint_schema"]["option_counts"] == [
        2,
        2,
        2,
    ]
    assert call["trace_headers"] == {"traceparent": "test-trace"}
    choice = executor._question(wire["decision_head"]["questions"][0], 0)
    assert choice.options[0].name != choice.options[1].name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", True),
        ("profile", "unknown"),
        ("temperature", 0.7),
        ("cache_salt", ""),
        ("max_input_tokens", True),
        ("max_tokenization_work", 0),
        ("max_encoding_bytes", -1),
    ],
)
def test_invalid_wire_rejects_before_engine_execution(executor, wire, field, value):
    wire["decision_head"][field] = value
    with pytest.raises((HttpError, InvalidArgument)):
        executor.prepare(wire)
    executor.handler.engine_client.encode.assert_not_called()


@pytest.mark.parametrize(
    "shape",
    ["ordinal", "duplicate", "count", "predicate", "score", "unicode", "unknown"],
)
def test_invalid_questions_reject_before_inference(executor, wire, shape):
    question = wire["decision_head"]["questions"][0]
    if shape == "ordinal":
        question["ordinal"] = True
    elif shape == "duplicate":
        question["candidates"][1]["value"] = True
    elif shape == "count":
        question["candidates"] *= 14
    elif shape == "predicate":
        question["kind"] = "Predicate"
    elif shape == "score":
        question["kind"] = "Score"
    elif shape == "unicode":
        question["instructions"] = "\ud800"
    else:
        question["kind"] = "unknown"
    with pytest.raises((HttpError, InvalidArgument)):
        executor.prepare(wire)


@pytest.mark.parametrize("limit", ["input", "tokens", "bytes"])
def test_no_budget_overrun(executor, wire, limit):
    wire["decision_head"]["input"] = "x" * 3000
    if limit == "input":
        wire["decision_head"]["max_input_tokens"] = 4
    elif limit == "tokens":
        wire["decision_head"]["max_tokenization_work"] = 4
    elif limit == "bytes":
        wire["decision_head"]["max_encoding_bytes"] = 4
    with pytest.raises((HttpError, InvalidArgument)):
        executor.prepare(wire)


def test_reference_state_truncation_is_preserved(executor, wire):
    wire["decision_head"]["input"] = "x" * 3000
    wire["decision_head"]["max_input_tokens"] = 4096
    wire["decision_head"]["max_tokenization_work"] = 8192
    wire["decision_head"]["max_encoding_bytes"] = 16384
    assert len(executor.prepare(wire).prompt_ids) == 2048


@pytest.mark.parametrize(
    "values", [[0.1], [float("nan")] * 6, [0.2] * 6, [-0.2, 1.2] * 3]
)
def test_malformed_distributions_fail(executor, wire, values):
    prepared = executor.prepare(wire)
    output = SimpleNamespace(
        prompt_token_ids=list(prepared.prompt_ids),
        num_cached_tokens=0,
        error=None,
        finished=True,
        outputs=SimpleNamespace(data=SimpleNamespace(ndim=1, tolist=lambda: values)),
    )
    with pytest.raises(HttpError):
        executor.decode(prepared, output)


@pytest.mark.asyncio
async def test_cancelled_context_never_enters_engine(executor, wire):
    context = Context()
    context.cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await executor.execute(wire, context)
    executor.handler.engine_client.encode.assert_not_called()


@pytest.mark.asyncio
async def test_context_cancellation_aborts_and_drains_owned_engine_request(
    executor, wire
):
    entered = asyncio.Event()

    async def blocked(**kwargs):
        entered.set()
        await asyncio.Future()
        yield

    executor.handler.engine_client.encode.side_effect = blocked
    context = Context()
    task = asyncio.create_task(executor.execute(wire, context))
    await asyncio.wait_for(entered.wait(), timeout=1)
    context.cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    executor.handler.engine_client.abort.assert_awaited_once()


@pytest.mark.parametrize(
    "field", ["num_cached_tokens", "prompt_token_ids", "error", "finished"]
)
def test_invalid_engine_metadata_is_not_a_normal_answer(executor, wire, field):
    prepared = executor.prepare(wire)
    output = SimpleNamespace(
        prompt_token_ids=list(prepared.prompt_ids),
        num_cached_tokens=0,
        error=None,
        finished=True,
        outputs=SimpleNamespace(
            data=SimpleNamespace(ndim=1, tolist=lambda: [0.2, 0.8] * 3)
        ),
    )
    setattr(
        output,
        field,
        {
            "num_cached_tokens": None,
            "prompt_token_ids": [999],
            "error": object(),
            "finished": False,
        }[field],
    )
    with pytest.raises(HttpError):
        executor.decode(prepared, output)


@pytest.fixture
def startup(executor, monkeypatch):
    monkeypatch.setattr(clef, "_released_runtime", lambda: True)
    monkeypatch.setattr(clef, "verify_installed_sources", lambda _: True)
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="ClefFlashForJointSchema",
            runner_type="pooling",
            is_encoder_decoder=False,
            is_diffusion=False,
            convert_type="none",
            model_impl="auto",
            model_class_overrides={},
            max_model_len=2048,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            async_scheduling=False,
            scheduler_cls=None,
            enable_chunked_prefill=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
    )
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_multimodal=False,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        realtime=False,
    )
    executor.handler.engine_client.renderer = object()
    executor.handler.engine_client.get_supported_tasks = AsyncMock(
        return_value=("token_classify",)
    )
    modules = {
        "vllm.entrypoints.openai.decisions.strategies": SimpleNamespace(
            ReadContext=lambda **kwargs: SimpleNamespace(**kwargs),
            ClefFlashStrategy=lambda _: Strategy(),
        ),
        "vllm.renderers.online_renderer": SimpleNamespace(
            OnlineRenderer=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        "vllm.entrypoints.openai.decisions.question_types": executor.native.questions,
        "vllm.pooling_params": SimpleNamespace(
            PoolingParams=executor.native.pooling_params
        ),
        "vllm.inputs": SimpleNamespace(tokens_input=executor.native.tokens_input),
        "vllm.exceptions": SimpleNamespace(
            QueueOverflowError=OverflowError,
            MaxQueuedTokensError=BufferError,
            RetryableRequestError=ConnectionError,
        ),
        "vllm.v1.engine.exceptions": SimpleNamespace(
            EngineGenerateError=ChildProcessError, EngineDeadError=BrokenPipeError
        ),
    }
    monkeypatch.setattr(clef, "import_module", modules.__getitem__)
    return executor.handler, runtime, config


@pytest.mark.asyncio
async def test_native_canary_runs_before_successful_initialization(startup):
    handler, runtime, config = startup
    initialized = await clef.initialize_clef(handler, runtime, config)
    assert isinstance(initialized, clef.ClefDecisionExecutor)
    handler.engine_client.encode.assert_called_once()


@pytest.mark.asyncio
async def test_other_architectures_do_not_import_optional_native_runtime(
    startup, monkeypatch
):
    handler, runtime, config = startup
    runtime.model_config.architecture = "BertForSequenceClassification"
    loader = Mock(side_effect=AssertionError("unrelated model loaded Clef APIs"))
    monkeypatch.setattr(clef, "import_module", loader)
    assert await clef.initialize_clef(handler, runtime, config) is None
    loader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "version",
        "source",
        "multimodal",
        "runner",
        "tasks",
        "canary",
        "prefix_cache",
        "chunked_prefill",
        "capacity",
    ],
)
async def test_recognized_clef_fails_startup_when_not_qualified(
    startup, monkeypatch, change
):
    handler, runtime, config = startup
    if change == "version":
        monkeypatch.setattr(clef, "_released_runtime", lambda: False)
    elif change == "source":
        monkeypatch.setattr(clef, "verify_installed_sources", lambda _: False)
    elif change == "multimodal":
        config.enable_multimodal = True
    elif change == "runner":
        runtime.model_config.runner_type = "generate"
    elif change == "tasks":
        handler.engine_client.get_supported_tasks.return_value = ("classify",)
    elif change == "prefix_cache":
        runtime.cache_config.enable_prefix_caching = True
    elif change == "chunked_prefill":
        runtime.scheduler_config.enable_chunked_prefill = True
    elif change == "capacity":
        runtime.scheduler_config.max_num_seqs = 5
    else:
        handler.engine_client.encode.side_effect = ValueError("invalid canary")
    with pytest.raises((RuntimeError, ValueError)):
        await clef.initialize_clef(handler, runtime, config)


@pytest.mark.parametrize("eligible", [False, True])
def test_only_initialized_pooling_workers_publish_the_capability(eligible):
    runtime = SimpleNamespace(set_engine_specific=Mock())
    clef.publish_clef_capability(
        runtime,
        eligible,
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
        [],
    )
    assert runtime.set_engine_specific.call_count == (2 if eligible else 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_input", ModelInput.Tokens),
        ("worker_type", WorkerType.Decode),
        ("needs", [[WorkerType.Prefill]]),
        ("model_type", ModelType.Embedding),
    ],
)
def test_invalid_worker_surface_is_not_advertised(field, value):
    arguments = dict(
        runtime_config=SimpleNamespace(set_engine_specific=Mock()),
        enabled=True,
        model_input=ModelInput.Text,
        model_type=ModelType.Classify | ModelType.Pooling,
        worker_type=WorkerType.Aggregated,
        needs=[],
    )
    arguments[field] = value
    clef.publish_clef_capability(**arguments)
    arguments["runtime_config"].set_engine_specific.assert_not_called()


def test_health_payload_uses_native_pooling_and_preserves_override(monkeypatch):
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    payload = clef.ClefHealthCheckPayload("model-alias").to_dict()
    assert payload["native_pooling_health"] is True
    assert payload["model"] == "model-alias"
    assert payload[clef.HEALTH_CHECK_KEY] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OverflowError, ConnectionError, BrokenPipeError])
async def test_native_errors_retain_overload_and_shutdown_meaning(
    executor, wire, error
):
    executor.handler.engine_client.encode.side_effect = error("engine failed")
    expected = BrokenPipeError if error is BrokenPipeError else Exception
    with pytest.raises(expected):
        await executor.execute(wire, Context())
    assert executor.handler.shutdown_event.is_set() is (error is BrokenPipeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [False, True])
async def test_classify_handler_dispatch_requires_internal_metadata(
    executor, wire, trusted
):
    from dynamo.vllm.pooling_handlers import ClassifyWorkerHandler

    handler = ClassifyWorkerHandler.__new__(ClassifyWorkerHandler)
    handler.decision_clef = executor
    context = Context()
    context.metadata = {"dynamo.internal.decision": "head"} if trusted else {}
    if trusted:
        responses = [row async for row in handler.generate(wire, context)]
        assert len(responses) == 1
        assert len(responses[0]["outcomes"]) == 3
    else:
        with pytest.raises(InvalidArgument):
            [row async for row in handler.generate(wire, context)]


def test_installed_strategy_preserves_question_types_and_option_order(executor, wire):
    # This uses the installed release directly, with no copied upstream sources.
    strategies = pytest.importorskip("vllm.entrypoints.openai.decisions.strategies")
    assert clef.verify_installed_sources(clef._SOURCE_BLOBS)
    questions = pytest.importorskip("vllm.entrypoints.openai.decisions.question_types")
    tokenizer = Tokenizer()
    context = SimpleNamespace(
        engine_client=executor.handler.engine_client,
        online_renderer=SimpleNamespace(
            renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
        ),
    )
    executor.strategy = strategies.ClefFlashStrategy(context)
    executor.native.questions = questions
    wire["decision_head"]["max_input_tokens"] = 2048
    wire["decision_head"]["max_tokenization_work"] = 8192
    wire["decision_head"]["max_encoding_bytes"] = 16384
    prepared = executor.prepare(wire)
    assert prepared.schema["question_types"] == [1, 0, 2]
    assert prepared.schema["option_counts"] == [2, 2, 2]
    assert prepared.orders[0] == (1, 0)
    assert prepared.orders[1:] == ((0, 1), (0, 1))


@pytest.mark.asyncio
async def test_invalid_output_closes_iterator_and_preserves_primary_error(
    executor, wire
):
    closed = asyncio.Event()

    async def invalid(**kwargs):
        try:
            yield SimpleNamespace(error=object())
        finally:
            closed.set()

    executor.handler.engine_client.encode.side_effect = invalid
    executor.handler.engine_client.abort.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(HttpError, match="invalid terminal metadata"):
        await executor.execute(wire, Context())
    assert closed.is_set()
    executor.handler.engine_client.abort.assert_awaited_once()
    assert executor.handler.shutdown_event.is_set()
    calls = executor.handler.engine_client.encode.call_count
    with pytest.raises(HttpError, match="cleanup did not settle"):
        await executor.execute(wire, Context())
    assert executor.handler.engine_client.encode.call_count == calls


@pytest.mark.asyncio
async def test_iterator_close_failure_does_not_hide_an_engine_error(executor, wire):
    class InvalidIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionError("primary engine failure")

        async def aclose(self):
            raise RuntimeError("secondary close failure")

    executor.handler.engine_client.encode.side_effect = lambda **_: InvalidIterator()
    with pytest.raises(HttpError, match="temporarily unavailable") as raised:
        await executor.execute(wire, Context())
    assert isinstance(raised.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_yielded_retryable_error_retains_service_unavailable(executor, wire):
    from vllm.exceptions import RetryableRequestError
    from vllm.outputs import RequestError

    executor.native.retryable_errors = (RetryableRequestError,)

    async def failed(**kwargs):
        yield SimpleNamespace(
            error=RequestError("capacity", "retry later", retryable=True)
        )

    executor.handler.engine_client.encode.side_effect = failed
    with pytest.raises(HttpError, match="temporarily unavailable") as raised:
        await executor.execute(wire, Context())
    assert isinstance(raised.value.__cause__, RetryableRequestError)


@pytest.mark.asyncio
async def test_cleanup_timeout_stops_further_admission(executor, wire, monkeypatch):
    original_wait_for = asyncio.wait_for

    async def bounded(awaitable, *, timeout):
        return await original_wait_for(awaitable, timeout=0.01)

    async def stalled_abort(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(clef.asyncio, "wait_for", bounded)
    executor.handler.engine_client.abort.side_effect = stalled_abort
    await executor._settle(None, "cancelled-request", False)
    assert executor.handler.shutdown_event.is_set()
    with pytest.raises(HttpError, match="cleanup did not settle"):
        await executor.execute(wire, Context())
    executor.handler.engine_client.encode.assert_not_called()

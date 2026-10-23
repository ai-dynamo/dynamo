#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Frontend TTFT boundaries with lightweight backend and native-binding stubs."""

import asyncio
import importlib
import pickle
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_sglang_processor_metrics_unit as processor_tests
from test_sglang_processor_metrics_unit import _load_processor_module

# Register the shared fixture for pytest discovery.
module_stubs = processor_tests.module_stubs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.core,
]


class RecordingContext:
    ttft_enabled = True

    def __init__(self):
        self.events = []
        self.intervals = []
        self.quality = []

    @contextmanager
    def start_ttft_stage(self, stage):
        self.events.append(("start", stage))
        try:
            yield
        finally:
            self.events.append(("end", stage))

    def record_ttft_batch_ago(self, intervals, coarse_reasons=()):
        self.intervals.extend(intervals)
        self.quality.extend(("coarse", reason) for reason in coarse_reasons)

    def mark_ttft_quality(self, quality, reason):
        self.quality.append((quality, reason))


class Tokenizer:
    chat_template = "{{ messages }}"
    bos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, text):
        return [4, 5]


def chat_request():
    return {"model": "test", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def frontend_stubs(request):
    install, remove = request.getfixturevalue("module_stubs")
    # Load only the pure Python utilities used here; common.utils.__init__
    # otherwise imports unrelated runtime-backed endpoint helpers.
    install(
        "dynamo.common.utils",
        __path__=[str(Path(__file__).parents[2] / "common" / "utils")],
    )
    install("dynamo._internal", ModelDeploymentCard=object)
    # Deliberately synthetic IDs prove the Python transport uses the binding
    # registry, rather than copying production enum discriminants.
    names = (
        "preprocess",
        "preprocess_queue",
        "template_tokenize",
        "template_render",
        "tokenize",
        "structural_tag_build",
        "multimodal_prepare",
        "mm_routing_build",
    )
    install("dynamo._core", ttft_stage_registry=lambda: list(enumerate(names, 100)))
    from dynamo.frontend import ttft

    ttft.stage_registry.cache_clear()
    install("dynamo.frontend.frontend_args", FrontendConfig=object)
    install(
        "dynamo.llm",
        ModelCardInstanceId=object,
        PythonAsyncEngine=object,
        RoutedEngine=object,
    )
    install(
        "dynamo.llm.exceptions",
        HttpError=type("HttpError", (Exception,), {}),
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
        WorkerUnavailable=type("WorkerUnavailable", (Exception,), {}),
    )
    # Restore all frontend modules imported against these stubs after each test.
    for name in (
        "utils",
        "sglang_prepost",
        "sglang_processor",
        "prepost",
        "vllm_processor",
    ):
        remove("dynamo.frontend." + name)
    for name in ("dynamo.common.utils.guided_json", "dynamo.vllm.errors"):
        remove(name)
    return install, remove


@pytest.fixture
def sglang_modules(frontend_stubs, monkeypatch):
    processor = _load_processor_module(frontend_stubs)
    _, remove = frontend_stubs
    importlib.import_module("dynamo.llm").HttpError = importlib.import_module(
        "dynamo.llm.exceptions"
    ).HttpError
    remove("dynamo.frontend.sglang_prepost")
    prepost = importlib.import_module("dynamo.frontend.sglang_prepost")
    monkeypatch.setattr(
        processor, "preprocess_chat_request", prepost.preprocess_chat_request
    )
    monkeypatch.setattr(
        processor, "SglangStreamingPostProcessor", lambda **kwargs: object()
    )

    async def stream(*args, **kwargs):
        yield {"ok": True}

    monkeypatch.setattr(processor.SglangProcessor, "_generate_and_stream", stream)
    return processor, prepost


def make_sglang_processor(module, **kwargs):
    return module.SglangProcessor(
        tokenizer=Tokenizer(),
        routed_engine=None,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
        **kwargs,
    )


def test_sglang_inline_preprocess_and_legacy_context(sglang_modules):
    module, _ = sglang_modules
    processor = make_sglang_processor(module)
    context = RecordingContext()

    async def collect(context):
        return [
            item async for item in processor.generator(chat_request(), context=context)
        ]

    assert asyncio.run(collect(context)) == [{"ok": True}]
    assert context.events == [
        ("start", "preprocess"),
        ("start", "template_tokenize"),
        ("end", "template_tokenize"),
        ("start", "structural_tag_build"),
        ("end", "structural_tag_build"),
        ("end", "preprocess"),
    ]
    assert asyncio.run(collect(None)) == [{"ok": True}]
    assert asyncio.run(collect(object())) == [{"ok": True}]


def test_sglang_template_error_closes_spans(sglang_modules):
    from jinja2.exceptions import TemplateError

    module, _ = sglang_modules
    context = RecordingContext()

    class BrokenTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise TemplateError("bad template")

    processor = make_sglang_processor(module)
    processor.tokenizer = BrokenTokenizer()

    async def collect():
        return [
            item async for item in processor.generator(chat_request(), context=context)
        ]

    with pytest.raises(module.InvalidArgument, match="bad template"):
        asyncio.run(collect())
    assert context.events[-2:] == [
        ("end", "template_tokenize"),
        ("end", "preprocess"),
    ]


def test_sglang_deepseek_custom_encoder_is_timed(
    sglang_modules, frontend_stubs, monkeypatch
):
    _, prepost = sglang_modules
    install, _ = frontend_stubs
    context = RecordingContext()

    def encode_messages(*args, **kwargs):
        assert context.events[-1] == ("start", "template_render")
        return "rendered"

    install(
        "sglang.srt.entrypoints.openai.encoding_dsv4", encode_messages=encode_messages
    )
    monkeypatch.setattr(
        prepost, "_should_use_deepseek_v4_encoding", lambda *args, **kwargs: True
    )
    result = prepost.preprocess_chat_request(
        chat_request(),
        tokenizer=Tokenizer(),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        context=context,
    )
    assert result.prompt_token_ids == [4, 5]
    assert context.events == [
        ("start", "template_render"),
        ("end", "template_render"),
        ("start", "tokenize"),
        ("end", "tokenize"),
        ("start", "structural_tag_build"),
        ("end", "structural_tag_build"),
    ]


def test_sglang_worker_returns_picklable_monotonic_intervals(
    sglang_modules, monkeypatch
):
    module, _ = sglang_modules
    monkeypatch.setattr(module, "_w_tokenizer", Tokenizer())
    from dynamo.frontend import ttft

    clock = iter([10_000_000_000, 10_200_000_000, 10_300_000_000, 10_400_000_000])
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: next(clock))
    result = module._preprocess_worker(chat_request(), "test", None, True)
    result = pickle.loads(pickle.dumps(result))
    assert result.timing_batch.intervals_ns == (
        ttft.TimingInterval(
            ttft.stage_registry()["template_tokenize"], 10_000_000_000, 10_200_000_000
        ),
        ttft.TimingInterval(
            ttft.stage_registry()["structural_tag_build"],
            10_300_000_000,
            10_400_000_000,
        ),
    )
    assert result.timing_batch.coarse_reasons == ("opaque_template",)
    assert result.prompt_token_ids == [1, 2, 3]
    disabled = module._preprocess_worker(chat_request(), "test", None)
    assert not hasattr(disabled, "timing_batch")
    assert b"timing_batch" not in pickle.dumps(disabled)


def test_sglang_worker_rejects_unsupported_ttft_stage(sglang_modules):
    module, _ = sglang_modules
    timing = module._WorkerPreprocessTiming()
    with pytest.raises(ValueError, match="Unsupported TTFT stage: arbitrary"):
        with timing.start_ttft_stage("arbitrary"):
            pass
    assert timing.intervals_ns == []


def test_sglang_pool_reattaches_worker_intervals(sglang_modules, monkeypatch):
    module, _ = sglang_modules
    context = RecordingContext()
    from dynamo.frontend import ttft

    result = module.SglangTimedPreprocessWorkerResult(
        prompt_token_ids=[1],
        dynamo_preproc={"sampling_options": {"guided_decoding": None}},
        request=chat_request(),
        timing_batch=ttft.TimingBatch(
            1,
            ttft.clock_domain(),
            (
                ttft.TimingInterval(
                    ttft.stage_registry()["template_render"],
                    10_000_000_000,
                    10_200_000_000,
                ),
                ttft.TimingInterval(
                    ttft.stage_registry()["tokenize"], 10_200_000_000, 10_400_000_000
                ),
            ),
        ),
    )

    class Pool:
        def submit(self, fn, *args):
            assert args[-1] is True
            assert context not in args
            assert context.events[-1] == ("end", "preprocess_queue")
            future = Future()
            future.set_result(result)
            return future

    clock = iter([9_000_000_000, 11_000_000_000])
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: next(clock))
    processor = make_sglang_processor(module, preprocess_pool=Pool())

    async def collect():
        return [
            item async for item in processor.generator(chat_request(), context=context)
        ]

    assert asyncio.run(collect()) == [{"ok": True}]
    assert context.intervals == [
        (ttft.stage_registry()["template_render"], 1_000_000_000, 800_000_000),
        (ttft.stage_registry()["tokenize"], 800_000_000, 600_000_000),
    ]
    assert context.events == [
        ("start", "preprocess"),
        ("start", "preprocess_queue"),
        ("end", "preprocess_queue"),
        ("end", "preprocess"),
    ]
    assert processor._worker_semaphore._value == 2


@pytest.mark.parametrize("queued", [False, True])
def test_sglang_pool_cancellation_preserves_semaphore(sglang_modules, queued):
    module, _ = sglang_modules
    context = RecordingContext()

    class Pool:
        def submit(self, *args):
            assert not queued
            return Future()

    processor = make_sglang_processor(module, preprocess_pool=Pool())
    initial_capacity = 0 if queued else 2
    processor._worker_semaphore = asyncio.Semaphore(initial_capacity)

    async def cancel():
        task = asyncio.create_task(
            anext(processor.generator(chat_request(), context=context))
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert context.events[-1] == ("end", "preprocess")
    assert processor._worker_semaphore._value == initial_capacity


@pytest.fixture
def vllm_modules(frontend_stubs):
    install, _ = frontend_stubs
    dummy = type("BackendType", (), {})
    install("vllm")
    install(
        "vllm.config",
        CacheConfig=dummy,
        LoadConfig=dummy,
        ModelConfig=dummy,
        VllmConfig=dummy,
    )
    install(
        "vllm.entrypoints.chat_utils",
        make_tool_call_id=lambda: "call",
        load_chat_template=lambda value: value,
    )
    install(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionNamedFunction=dummy,
        ChatCompletionNamedToolChoiceParam=dummy,
        ChatCompletionRequest=dummy,
    )
    install("vllm.reasoning", ReasoningParser=dummy, ReasoningParserManager=dummy)
    install("vllm.renderers", ChatParams=dummy, merge_kwargs=lambda *args: {})
    install(
        "vllm.sampling_params",
        SamplingParams=SimpleNamespace,
        RequestOutputKind=SimpleNamespace(DELTA=0),
    )
    install("vllm.tokenizers", TokenizerLike=dummy)
    install("vllm.tool_parsers", ToolParser=dummy, ToolParserManager=dummy)
    install("vllm.tool_parsers.utils", get_json_schema_from_tools=lambda *args: None)
    install("vllm.utils.async_utils", make_async=lambda fn, **kwargs: fn)
    install("vllm.tasks", GENERATION_TASKS=("generate",))
    install(
        "vllm.v1.engine",
        EngineCoreOutput=dummy,
        EngineCoreRequest=dummy,
        FinishReason=type("FinishReason", (), {"STOP": 0, "LENGTH": 1, "ABORT": 2}),
    )
    install("vllm.v1.engine.input_processor", InputProcessor=dummy)
    install(
        "vllm.v1.engine.output_processor",
        OutputProcessor=dummy,
        OutputProcessorOutput=dummy,
    )
    install("vllm.v1.engine.parallel_sampling", ParentRequest=dummy)
    install(
        "vllm.exceptions",
        VLLMClientError=type("VLLMClientError", (Exception,), {}),
        VLLMNotFoundError=type("VLLMNotFoundError", (Exception,), {}),
        VLLMUnprocessableEntityError=type(
            "VLLMUnprocessableEntityError", (Exception,), {}
        ),
    )
    install("msgspec.structs", replace=lambda value, **kwargs: value)
    install(
        "dynamo.frontend.vllm_protocol",
        DeltaFunctionCall=dummy,
        DeltaMessage=dummy,
        DeltaToolCall=dummy,
        FunctionDefinition=dummy,
    )
    install(
        "dynamo.common.multimodal.mm_kwargs_transfer",
        MmKwargsSender=dummy,
        MmKwargsShmSender=dummy,
        MmKwargsNixlSender=dummy,
    )
    install(
        "dynamo.common.multimodal.routing_utils",
        build_mm_routing_info_from_features=lambda *args, **kwargs: {
            "block_mm_infos": [[]]
        },
    )
    importlib.import_module("dynamo.llm").HttpError = importlib.import_module(
        "dynamo.llm.exceptions"
    ).HttpError
    prepost = importlib.import_module("dynamo.frontend.prepost")
    processor = importlib.import_module("dynamo.frontend.vllm_processor")
    return processor, prepost


@pytest.mark.parametrize(
    "pretokenized,fail_render", [(False, False), (True, False), (False, True)]
)
def test_vllm_template_tokenizer_preserves_tokens_and_closes_on_error(
    vllm_modules, monkeypatch, pretokenized, fail_render
):
    _, prepost = vllm_modules
    context = RecordingContext()
    request = SimpleNamespace(
        tool_choice="auto",
        structured_outputs=None,
        add_special_tokens=True,
        extract_structured_outputs=lambda: None,
    )
    monkeypatch.setattr(
        prepost, "_validate_chat_completion_request", lambda value: request
    )
    monkeypatch.setattr(prepost, "_build_assistant_guided_decoding", lambda value: None)
    monkeypatch.setattr(prepost, "_has_explicit_output_constraint", lambda value: False)
    monkeypatch.setattr(
        prepost,
        "_prepare_request",
        lambda *args, **kwargs: (request, None, {}, [], None),
    )
    monkeypatch.setattr(
        prepost, "build_tool_call_guided_decoding", lambda *args, **kwargs: None
    )

    class Renderer:
        async def render_messages_async(self, messages, params):
            if fail_render:
                raise ValueError("render failed")
            return [], {"prompt_token_ids": [1, 2]} if pretokenized else {
                "prompt": "hello"
            }

    async def tokenize(prompt, **kwargs):
        assert not pretokenized
        assert context.events[-1] == ("start", "tokenize")
        return SimpleNamespace(input_ids=[1, 2])

    monkeypatch.setattr(prepost, "_get_async_tokenizer", lambda tokenizer: tokenize)
    coro = prepost.preprocess_chat_request(
        chat_request(),
        tokenizer=Tokenizer(),
        renderer=Renderer(),
        tool_parser_class=None,
        context=context,
    )
    if fail_render:
        with pytest.raises(ValueError, match="render failed"):
            asyncio.run(coro)
    else:
        assert asyncio.run(coro).prompt_token_ids == [1, 2]
    from dynamo.frontend import ttft

    label = "template_tokenize" if pretokenized else "template_render"
    assert context.intervals[0][0] == ttft.stage_registry()[label]
    assert context.events == (
        []
        if fail_render
        else (
            ([] if pretokenized else [("start", "tokenize"), ("end", "tokenize")])
            + [("start", "structural_tag_build"), ("end", "structural_tag_build")]
        )
    )
    assert context.quality == ([("coarse", "opaque_template")] if pretokenized else [])


@pytest.mark.parametrize("has_media", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_vllm_generator_forwards_timeline_and_times_input_preparation(
    vllm_modules, monkeypatch, has_media, enabled
):
    module, _ = vllm_modules
    context = RecordingContext()
    context.ttft_enabled = enabled

    class RequestForSampling(SimpleNamespace):
        model_fields = frozenset()

    request_for_sampling = RequestForSampling(
        max_completion_tokens=None,
        max_tokens=1,
        logprobs=False,
        top_logprobs=None,
        cache_salt=None,
        mm_processor_kwargs=None,
    )

    async def preprocess(request, **kwargs):
        if enabled:
            assert kwargs["context"].start_ttft_stage == context.start_ttft_stage
            assert context.events[-1] == ("start", "preprocess")
        else:
            assert kwargs["context"] is None
            assert context.events == []
        return SimpleNamespace(
            request_for_sampling=request_for_sampling,
            tool_parser=None,
            chat_template_kwargs={},
            engine_prompt={},
            prompt_token_ids=[1],
            guided_decoding=None,
        )

    monkeypatch.setattr(module, "preprocess_chat_request", preprocess)
    monkeypatch.setattr(
        module,
        "extract_mm_urls",
        lambda messages: ({"image_url": ["image"]} if has_media else None, None),
    )
    expected_stage = "multimodal_prepare" if has_media else "preprocess"

    class ReachedInputPreparation(Exception):
        pass

    async def build_inputs(*args, **kwargs):
        if enabled:
            assert context.events[-1] == ("start", expected_stage)
        else:
            assert context.events == []
        raise ReachedInputPreparation()

    monkeypatch.setattr(module, "_build_engine_inputs", build_inputs)
    processor = module.VllmProcessor(
        tokenizer=Tokenizer(),
        input_processor=SimpleNamespace(
            renderer=None, model_config=None, generation_config_fields={}
        ),
        output_processor=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        routed_engine=None,
    )

    async def collect():
        return [
            item async for item in processor.generator(chat_request(), context=context)
        ]

    with pytest.raises(ReachedInputPreparation):
        asyncio.run(collect())
    assert context.events == (
        [
            ("start", "preprocess"),
            ("end", "preprocess"),
            ("start", expected_stage),
            ("end", expected_stage),
        ]
        if enabled
        else []
    )


@pytest.mark.parametrize("fail_transfer", [False, True])
def test_vllm_mm_routing_and_transfer_have_separate_intervals(
    vllm_modules, monkeypatch, fail_transfer
):
    module, _ = vllm_modules
    context = RecordingContext()

    def build_metadata(*args, **kwargs):
        assert context.events[-1] == ("start", "mm_routing_build")
        return {"block_mm_infos": [[]]}

    monkeypatch.setattr(module, "build_mm_routing_info_from_features", build_metadata)
    feature = SimpleNamespace(
        modality="image",
        mm_hash="hash",
        mm_position=SimpleNamespace(offset=1, length=2),
    )
    preprocessed = SimpleNamespace(mm_features=[feature], prompt_token_ids=[1, 2])

    class Sender:
        async def prepare(self, *args, **kwargs):
            assert context.events[-1] == ("start", "multimodal_prepare")
            if fail_transfer:
                raise ValueError("transfer failed")
            return {"transport": "shm"}, ["cleanup"]

    processor = module.VllmProcessor(
        tokenizer=Tokenizer(),
        input_processor=None,
        output_processor=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        routed_engine=None,
    )
    processor.nixl_mm_enabled = True
    processor._sender = Sender()
    request = {}
    routing, cleanup, transferred = asyncio.run(
        processor._prepare_mm_routing(
            preprocessed,
            request,
            context=context,
        )
    )
    assert routing == {"block_mm_infos": [[]]}
    assert transferred is not fail_transfer
    assert cleanup == ([] if fail_transfer else ["cleanup"])
    assert request["extra_args"]["expanded_token_ids"] == [1, 2]
    assert context.events == [
        ("start", "mm_routing_build"),
        ("end", "mm_routing_build"),
        ("start", "multimodal_prepare"),
        ("end", "multimodal_prepare"),
    ]


def test_vllm_text_and_uuid_only_paths_skip_mm_stages(vllm_modules):
    module, _ = vllm_modules
    processor = module.VllmProcessor(
        tokenizer=Tokenizer(),
        input_processor=None,
        output_processor=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        routed_engine=None,
    )
    context = RecordingContext()
    for request in ({}, {"multi_modal_uuids": {"image": ["cached"]}}):
        assert asyncio.run(
            processor._prepare_mm_routing(
                SimpleNamespace(mm_features=[]),
                request,
                context=context,
            )
        ) == (None, [], False)
    assert context.events == []


@pytest.mark.parametrize("pool", [False, True])
def test_disabled_context_skips_native_calls_and_pool_clocks(
    sglang_modules, monkeypatch, pool
):
    module, _ = sglang_modules

    class DisabledContext:
        ttft_enabled = False

        def start_ttft_stage(self, *args):
            pytest.fail("OFF must not call PyO3 stage entry")

        def record_ttft_stage_ago(self, *args):
            pytest.fail("OFF must not import pool timing")

    def unexpected_clock():
        pytest.fail("OFF must not read the pool timing clock")

    from dynamo.frontend import ttft

    monkeypatch.setattr(ttft.time, "monotonic_ns", unexpected_clock)
    monkeypatch.setattr(module, "_w_tokenizer", Tokenizer())

    class Pool:
        def submit(self, fn, *args):
            assert len(args) == 3
            result = fn(*args)
            assert not hasattr(result, "timing_batch")
            assert b"timing_batch" not in pickle.dumps(result)
            future = Future()
            future.set_result(result)
            return future

    processor = make_sglang_processor(module, preprocess_pool=Pool() if pool else None)

    async def collect():
        return [
            item
            async for item in processor.generator(
                chat_request(), context=DisabledContext()
            )
        ]

    assert asyncio.run(collect()) == [{"ok": True}]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"version": 2}, "invalid_version"),
        ({"version": True}, "invalid_version"),
        ({"clock_domain": "foreign-clock"}, "clock_mismatch"),
        ({"truncated": True}, "interval_limit"),
        ({"invalid": True}, "invalid_batch"),
        ({"intervals_ns": None}, "invalid_batch"),
        ({"coarse_reasons": ("unknown",)}, "invalid_batch"),
        ({"coarse_reasons": ([],)}, "invalid_batch"),
    ],
)
def test_worker_batch_rejects_invalid_metadata(
    frontend_stubs, monkeypatch, change, reason
):
    from dataclasses import replace

    from dynamo.frontend import ttft

    context = RecordingContext()
    batch = ttft.TimingBatch(1, ttft.clock_domain(), ())
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: 100)
    ttft.import_worker_batch(context, replace(batch, **change), 10)
    assert context.intervals == []
    assert context.quality == [("degraded", reason)]


@pytest.mark.parametrize("field", ["intervals_ns", "coarse_reasons"])
def test_worker_batch_rejects_tuple_subclasses_before_iteration(
    frontend_stubs, monkeypatch, field
):
    from dataclasses import replace

    from dynamo.frontend import ttft

    calls = []

    class LyingTuple(tuple):
        def __len__(self):
            calls.append("len")
            return 0

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("malformed timing payload must not be iterated")

    context = RecordingContext()
    batch = ttft.TimingBatch(1, ttft.clock_domain(), ())
    payload = LyingTuple(range(ttft.MAX_TIMING_INTERVALS + 1))
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: 100)
    ttft.import_worker_batch(context, replace(batch, **{field: payload}), 10)
    assert calls == []
    assert context.intervals == []
    assert context.quality == [("degraded", "invalid_batch")]


@pytest.mark.parametrize("kind", ["batch", "interval"])
def test_worker_batch_rejects_dataclass_subclasses_before_field_access(
    frontend_stubs, monkeypatch, kind
):
    from dynamo.frontend import ttft

    calls = []

    class BatchSubclass(ttft.TimingBatch):
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("malformed timing batch must not be inspected")

    class IntervalSubclass(ttft.TimingInterval):
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("malformed timing interval must not be inspected")

    batch = (
        BatchSubclass(1, ttft.clock_domain(), ())
        if kind == "batch"
        else ttft.TimingBatch(1, ttft.clock_domain(), (IntervalSubclass(103, 20, 30),))
    )
    context = RecordingContext()
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: 100)
    ttft.import_worker_batch(context, batch, 10)
    assert calls == []
    assert context.intervals == []
    assert context.quality == [("degraded", "invalid_batch")]


@pytest.mark.parametrize(
    "stage,start,end,reason",
    [
        (-1, 20, 30, "invalid_stage"),
        (True, 20, 30, "invalid_stage"),
        (103, 30, 20, "invalid_interval"),
        (103, 9, 20, "invalid_interval"),
        (103, 20, 101, "invalid_interval"),
        (103, -1, 30, "timestamp_overflow"),
        (103, 20, 1 << 65, "timestamp_overflow"),
        (103, 20.5, 30, "timestamp_overflow"),
    ],
)
def test_worker_batch_rejects_invalid_intervals(
    frontend_stubs, monkeypatch, stage, start, end, reason
):
    from dynamo.frontend import ttft

    context = RecordingContext()
    batch = ttft.TimingBatch(
        1, ttft.clock_domain(), (ttft.TimingInterval(stage, start, end),)
    )
    monkeypatch.setattr(ttft.time, "monotonic_ns", lambda: 100)
    ttft.import_worker_batch(context, batch, 10)
    assert context.intervals == []
    assert context.quality == [("degraded", reason)]


def test_worker_batch_bounds_repeats_and_unsupported_clock(frontend_stubs, monkeypatch):
    from dataclasses import replace

    from dynamo.frontend import ttft

    context = RecordingContext()
    interval = ttft.TimingInterval(ttft.stage_registry()["tokenize"], 20, 30)
    batch = ttft.TimingBatch(1, ttft.clock_domain(), (interval, interval))
    clock_reads = []

    def now():
        clock_reads.append(True)
        return 100

    monkeypatch.setattr(ttft.time, "monotonic_ns", now)
    ttft.import_worker_batch(context, batch, 10)
    assert len(clock_reads) == 1
    assert context.intervals == [(interval.stage_id, 80, 70)] * 2
    ttft.import_worker_batch(context, replace(batch, intervals_ns=(interval,) * 65), 10)
    assert context.quality[-1] == ("degraded", "interval_limit")
    ttft.import_worker_batch(context, replace(batch, clock_domain="unsupported"), 10)
    assert context.quality[-1] == ("coarse", "unsupported_clock")
    assert len(context.intervals) == 2


def test_worker_recorder_is_bounded_before_clock_reads(frontend_stubs, monkeypatch):
    from contextlib import ExitStack

    from dynamo.frontend import ttft

    clock_reads = []
    monkeypatch.setattr(
        ttft.time, "monotonic_ns", lambda: clock_reads.append(1) or len(clock_reads)
    )
    worker = ttft.WorkerTiming()
    with ExitStack() as stack:
        for _ in range(70):
            stack.enter_context(worker.start_ttft_stage("template_render"))
    assert len(worker.batch().intervals_ns) == 64
    assert worker.batch().truncated is True
    assert len(clock_reads) == 128


def test_worker_invalid_stage_check_survives_python_optimized(frontend_stubs):
    import subprocess
    import sys

    from dynamo.frontend import ttft

    code = """
import importlib.util
import sys
import types
path = sys.argv[1]
spec = importlib.util.spec_from_file_location('timing_test_module', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.stage_registry = lambda: {'template_render': 19}
try:
    with module.WorkerTiming().start_ttft_stage('unknown'):
        pass
except ValueError:
    pass
else:
    raise RuntimeError('stage validation disabled by -O')
"""
    subprocess.run([sys.executable, "-O", "-c", code, ttft.__file__], check=True)


def _same_host_pool_batch(registry):
    """Actual spawned worker; only the unavailable native registry is stubbed."""
    import sys
    import types

    from dynamo.frontend import ttft

    sys.modules["dynamo._core"] = types.SimpleNamespace(
        ttft_stage_registry=lambda: [
            (stage_id, name) for name, stage_id in registry.items()
        ]
    )
    ttft.stage_registry.cache_clear()
    timing = ttft.WorkerTiming()
    with timing.start_ttft_stage("template_render"):
        pass
    with timing.start_ttft_stage("tokenize"):
        pass
    return timing.batch()


def test_actual_same_host_process_pool_batch(frontend_stubs):
    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor

    from dynamo.frontend import ttft

    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        submitted = time.monotonic_ns()
        batch = pool.submit(_same_host_pool_batch, ttft.stage_registry()).result(
            timeout=30
        )
    context = RecordingContext()
    ttft.import_worker_batch(context, batch, submitted)
    assert context.quality == []
    assert [interval[0] for interval in context.intervals] == [
        ttft.stage_registry()["template_render"],
        ttft.stage_registry()["tokenize"],
    ]
    assert all(start >= end >= 0 for _, start, end in context.intervals)

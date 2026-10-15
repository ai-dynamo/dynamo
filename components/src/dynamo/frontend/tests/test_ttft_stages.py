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
from test_sglang_processor_metrics_unit import _load_processor_module
from test_sglang_processor_metrics_unit import module_stubs as module_stubs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.core,
]


class RecordingContext:
    def __init__(self):
        self.events = []
        self.intervals = []

    @contextmanager
    def start_ttft_stage(self, stage):
        self.events.append(("start", stage))
        try:
            yield
        finally:
            self.events.append(("end", stage))

    def record_ttft_stage_ago(self, stage, start, end):
        self.intervals.append((stage, start, end))


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
        ("start", "python_preprocess"),
        ("start", "template_tokenize"),
        ("end", "template_tokenize"),
        ("end", "python_preprocess"),
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
        ("end", "python_preprocess"),
    ]


def test_sglang_deepseek_custom_encoder_is_timed(
    sglang_modules, frontend_stubs, monkeypatch
):
    _, prepost = sglang_modules
    install, _ = frontend_stubs
    context = RecordingContext()

    def encode_messages(*args, **kwargs):
        assert context.events[-1] == ("start", "template_tokenize")
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
        ("start", "template_tokenize"),
        ("end", "template_tokenize"),
    ]


def test_sglang_worker_returns_picklable_monotonic_intervals(
    sglang_modules, monkeypatch
):
    module, _ = sglang_modules
    monkeypatch.setattr(module, "_w_tokenizer", Tokenizer())
    clock = iter([10_000_000_000, 10_200_000_000])
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic_ns=lambda: next(clock))
    )
    result = module._preprocess_worker(chat_request(), "test", None, True)
    result = pickle.loads(pickle.dumps(result))
    assert result.template_tokenize_intervals_ns == [(10_000_000_000, 10_200_000_000)]
    assert result.prompt_token_ids == [1, 2, 3]
    assert (
        module._preprocess_worker(
            chat_request(), "test", None
        ).template_tokenize_intervals_ns
        is None
    )


def test_sglang_worker_rejects_unsupported_ttft_stage(sglang_modules):
    module, _ = sglang_modules
    timing = module._WorkerPreprocessTiming()
    with pytest.raises(ValueError, match="Unsupported TTFT stage: python_preprocess"):
        with timing.start_ttft_stage("python_preprocess"):
            pass
    assert timing.intervals_ns == []


def test_sglang_pool_reattaches_worker_intervals(sglang_modules, monkeypatch):
    module, _ = sglang_modules
    context = RecordingContext()
    result = module.SglangPreprocessWorkerResult(
        prompt_token_ids=[1],
        dynamo_preproc={"sampling_options": {"guided_decoding": None}},
        request=chat_request(),
        template_tokenize_intervals_ns=[(10_000_000_000, 10_200_000_000)],
    )

    class Pool:
        def submit(self, fn, *args):
            assert args[-1] is True
            assert context not in args
            assert context.events[-1] == ("end", "preprocess_queue")
            future = Future()
            future.set_result(result)
            return future

    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic_ns=lambda: 11_000_000_000)
    )
    processor = make_sglang_processor(module, preprocess_pool=Pool())

    async def collect():
        return [
            item async for item in processor.generator(chat_request(), context=context)
        ]

    assert asyncio.run(collect()) == [{"ok": True}]
    assert context.intervals == [("template_tokenize", 1.0, 0.8)]
    assert context.events == [
        ("start", "python_preprocess"),
        ("start", "preprocess_queue"),
        ("end", "preprocess_queue"),
        ("end", "python_preprocess"),
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
    assert context.events[-1] == ("end", "python_preprocess")
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
            assert context.events[-1] == ("start", "template_tokenize")
            if fail_render:
                raise ValueError("render failed")
            return [], {"prompt_token_ids": [1, 2]} if pretokenized else {
                "prompt": "hello"
            }

    async def tokenize(prompt, **kwargs):
        assert not pretokenized
        assert context.events[-1] == ("start", "template_tokenize")
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
    assert context.events == [
        ("start", "template_tokenize"),
        ("end", "template_tokenize"),
    ]


@pytest.mark.parametrize("has_media", [False, True])
def test_vllm_generator_forwards_timeline_and_times_input_preparation(
    vllm_modules, monkeypatch, has_media
):
    module, _ = vllm_modules
    context = RecordingContext()

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
        assert kwargs["context"] is context
        assert context.events[-1] == ("start", "python_preprocess")
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
    expected_stage = "multimodal_prepare" if has_media else "python_preprocess"

    class ReachedInputPreparation(Exception):
        pass

    async def build_inputs(*args, **kwargs):
        assert context.events[-1] == ("start", expected_stage)
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
    assert context.events == [
        ("start", "python_preprocess"),
        ("end", "python_preprocess"),
        ("start", expected_stage),
        ("end", expected_stage),
    ]


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

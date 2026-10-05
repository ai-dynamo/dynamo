# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang worker checks client token IDs on every request path."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import HttpError
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.embedding.embedding_handler import (
    EmbeddingWorkerHandler,
)
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.diffusion_handler import DiffusionWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    MultimodalWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

# Qwen3-0.6B has vocab_size 151936.
MAX_TOKEN_ID = 151935
OUT_OF_RANGE = 2**32 - 1
# An image placeholder one past a 32000-entry text vocabulary.
IMAGE_TOKEN_ID = 32000


class _Reached(Exception):
    """Raised by a fake engine to show that a request reached it."""


def _context():
    return SimpleNamespace(
        id=lambda: "request-id",
        trace_id="trace-id",
        trace_headers=lambda: {},
        is_stopped=lambda: False,
        notify_first_token=lambda: None,
    )


def _with_token_input(
    handler, *, max_token_id=MAX_TOKEN_ID, engine=None, tokenizer=None
):
    handler.use_sglang_tokenizer = tokenizer is not None
    handler.input_param_manager = InputParamManager(tokenizer)
    handler._max_input_token_id = max_token_id
    handler.engine = engine if engine is not None else SimpleNamespace()
    return handler


def _decode_handler(**kwargs):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.serving_mode = DisaggregationMode.AGGREGATED
    handler._first_token_source = None
    return _with_token_input(handler, **kwargs)


def _decode_handler_for_generate(engine):
    """A decode handler whose generate() runs up to the engine call."""
    handler = _decode_handler(engine=engine)
    handler.shutdown_event = None
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(
            served_model_name="test-model", skip_tokenizer_init=False
        ),
        dynamo_args=SimpleNamespace(enable_rl=False),
    )
    handler._enable_frontend_decoding = False
    handler._mm_hashes_supported = False
    handler._engine_supports_priority = False
    handler._routed_experts_kwargs = {}
    handler.enable_trace = False
    handler._resolve_lora = lambda request: None
    return handler


def _token_request(token_ids):
    return {
        "token_ids": token_ids,
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 4},
    }


def _image_placeholder_engine():
    return SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            mm_processor=SimpleNamespace(),
            model_config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        )
    )


def _native_request(token_ids):
    return {
        "token_ids": token_ids,
        "extra_args": {"sglang_tito": {"sampling_params": {"max_new_tokens": 1}}},
    }


def test_token_array_prompt_without_token_in_rejects_out_of_range_id():
    handler = _decode_handler()

    with pytest.raises(
        HttpError, match=r"Token id 4294967295 is out of vocabulary at token_ids\[1\]$"
    ) as error:
        handler._get_input_param({"token_ids": [1, OUT_OF_RANGE]})

    assert error.value.code == 400


def test_token_array_prompt_with_in_range_ids_passes():
    handler = _decode_handler()

    request = {"token_ids": [0, 1, MAX_TOKEN_ID]}

    assert handler._get_input_param(request) == {"input_ids": [0, 1, MAX_TOKEN_ID]}


@pytest.mark.asyncio
async def test_decode_passes_in_range_ids_to_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _decode_handler_for_generate(engine)

    with pytest.raises(_Reached):
        await anext(handler.generate(_token_request([1, MAX_TOKEN_ID]), _context()))

    assert engine.async_generate.await_args.kwargs["input_ids"] == [1, MAX_TOKEN_ID]


@pytest.mark.asyncio
async def test_decode_rejects_out_of_range_id_before_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _decode_handler_for_generate(engine)

    with pytest.raises(HttpError) as error:
        await anext(handler.generate(_token_request([1, OUT_OF_RANGE]), _context()))

    assert error.value.code == 400
    engine.async_generate.assert_not_awaited()


def test_media_request_passes_with_its_placeholder_id():
    handler = _decode_handler(max_token_id=31999, engine=_image_placeholder_engine())
    request = {
        "token_ids": [1, IMAGE_TOKEN_ID, 2],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/image.png"}]},
    }

    assert handler._get_input_param(request) == {"input_ids": [1, IMAGE_TOKEN_ID, 2]}


def test_placeholder_id_without_media_is_rejected():
    handler = _decode_handler(max_token_id=31999, engine=_image_placeholder_engine())

    with pytest.raises(
        HttpError, match=r"Token id 32000 is out of vocabulary at token_ids\[1\]$"
    ):
        handler._get_input_param({"token_ids": [1, IMAGE_TOKEN_ID, 2]})


class _EmbeddingsProcessor:
    """Stands in for the NIXL read of the encode worker's embeddings."""

    def __init__(self, rows):
        self.rows = rows

    async def process_embeddings(self, request):
        return torch.zeros(self.rows, 8), 7

    def create_multimodal_image_item(self, embeddings, grids):
        return {"modality": "IMAGE", "rows": embeddings.shape[0], "grids": grids}

    def release_embeddings(self, tensor_id):
        pass


@pytest.mark.asyncio
async def test_request_forwarded_by_encode_worker_passes_with_placeholder_ids():
    image_token_id = 151655
    num_mm_tokens = 4
    # The encode worker expands one placeholder into one ID per embedding row and
    # clears the media URL before it forwards the request.
    token_ids = [1, 2] + [image_token_id] * num_mm_tokens + [3]
    payload = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=token_ids,
            stop_conditions=StopConditions(max_tokens=4),
            sampling_options=SamplingOptions(),
        ),
        multimodal_inputs=[
            MultiModalGroup(
                multimodal_input=MultiModalInput(image_url=None),
                image_grid_thw=[1, 4, 4],
                num_mm_tokens=num_mm_tokens,
            )
        ],
        embeddings_shape=(num_mm_tokens, 8),
    ).model_dump_json()

    handler = MultimodalWorkerHandler.__new__(MultimodalWorkerHandler)
    handler.serving_mode = DisaggregationMode.AGGREGATED
    handler.enable_trace = False
    # The placeholder is above the bound, as for a model whose placeholder
    # sits past the text vocabulary.
    handler._max_input_token_id = image_token_id - 1
    handler.embeddings_processor = _EmbeddingsProcessor(rows=num_mm_tokens)

    async def async_generate(**kwargs):
        async def stream():
            yield {
                "output_ids": [42],
                "meta_info": {"finish_reason": {"type": "length"}},
            }

        return stream()

    handler.engine = SimpleNamespace(
        async_generate=AsyncMock(side_effect=async_generate)
    )

    outputs = [json.loads(o) async for o in handler.generate(payload, _context())]

    assert [o.get("error") for o in outputs] == [None]
    assert outputs[0]["token_ids"] == [42]
    kwargs = handler.engine.async_generate.await_args.kwargs
    assert kwargs["input_ids"] == token_ids
    assert kwargs["image_data"] == [
        {"modality": "IMAGE", "rows": num_mm_tokens, "grids": [[1, 4, 4]]}
    ]


def test_native_batch_with_in_range_ids_passes():
    handler = _decode_handler()
    batch = [[1, 2], [3, MAX_TOKEN_ID]]

    assert handler._get_input_param(_native_request(batch)) == {"input_ids": batch}


def test_native_batch_with_one_out_of_range_id_is_rejected():
    handler = _decode_handler()

    with pytest.raises(
        HttpError,
        match=r"Token id 4294967295 is out of vocabulary at input_ids\[1\]\[1\]$",
    ) as error:
        handler._get_input_param(_native_request([[1, 2], [3, OUT_OF_RANGE]]))

    assert error.value.code == 400


def test_native_request_rejects_out_of_range_id():
    handler = _decode_handler()

    with pytest.raises(
        HttpError, match=r"Token id 4294967295 is out of vocabulary at input_ids\[1\]$"
    ):
        handler._get_input_param(_native_request([1, OUT_OF_RANGE]))


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ([[1], 2], r"input_ids\[1\] must be a token ID list$"),
        ([[1, True]], r"input_ids\[0\]\[1\] must be an integer token ID$"),
        ([[[1]]], r"input_ids\[0\]\[0\] must be an integer token ID$"),
    ],
)
def test_native_batch_rejects_malformed_inner_lists(token_ids, message):
    handler = _decode_handler()

    with pytest.raises(HttpError, match=message):
        handler._get_input_param(_native_request(token_ids))


class _ChatTokenizer:
    chat_template = ""

    def apply_chat_template(self, messages, **kwargs):
        return "<|user|>hi<|assistant|>"


def test_text_from_the_sglang_tokenizer_passes_unchanged():
    handler = _decode_handler(tokenizer=_ChatTokenizer())
    request = {"messages": [{"role": "user", "content": "hi"}]}

    assert handler._get_input_param(request) == {"prompt": "<|user|>hi<|assistant|>"}


class _EmbeddingEngine:
    def __init__(self):
        self.requests = []
        self.tokenizer_manager = SimpleNamespace(generate_request=self._generate)

    async def _generate(self, request, context):
        self.requests.append(request)
        yield {"embedding": [0.1, 0.2], "meta_info": {"prompt_tokens": 2}}


def _embedding_handler():
    handler = EmbeddingWorkerHandler.__new__(EmbeddingWorkerHandler)
    handler.engine = _EmbeddingEngine()
    handler.enable_trace = False
    handler._max_input_token_id = MAX_TOKEN_ID
    return handler


def _embedding_context():
    return SimpleNamespace(trace_id="embedding-trace", trace_headers=lambda: {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding_input", "position"),
    [
        ([1, OUT_OF_RANGE], r"input\[1\]"),
        ([[1, 2], [3, OUT_OF_RANGE]], r"input\[1\]\[1\]"),
    ],
)
async def test_embedding_token_input_with_out_of_range_id_is_rejected(
    embedding_input, position
):
    handler = _embedding_handler()

    with pytest.raises(
        HttpError, match=rf"Token id 4294967295 is out of vocabulary at {position}$"
    ) as error:
        await anext(
            handler.generate(
                {"model": "embedding-model", "input": embedding_input},
                _embedding_context(),
            )
        )

    assert error.value.code == 400
    assert handler.engine.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_input", [[1, MAX_TOKEN_ID], [[1], [MAX_TOKEN_ID]]])
async def test_embedding_token_input_with_in_range_ids_passes(embedding_input):
    handler = _embedding_handler()

    outputs = [
        output
        async for output in handler.generate(
            {"model": "embedding-model", "input": embedding_input},
            _embedding_context(),
        )
    ]

    assert len(outputs) == 1
    [request] = handler.engine.requests
    assert request.input_ids == embedding_input


def _prefill_handler(engine):
    """A prefill handler whose generate() runs up to the engine call."""
    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 1234
    handler._generate_bootstrap_room = lambda: 17
    handler.enable_trace = False
    handler._resolve_lora = lambda request: None
    handler._priority_kwargs = lambda priority: {}
    return _with_token_input(handler, engine=engine)


@pytest.mark.asyncio
async def test_prefill_passes_in_range_ids_to_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _prefill_handler(engine)

    with pytest.raises(_Reached):
        await anext(handler.generate(_token_request([1, MAX_TOKEN_ID]), _context()))

    assert engine.async_generate.await_args.kwargs["input_ids"] == [1, MAX_TOKEN_ID]


@pytest.mark.asyncio
async def test_prefill_rejects_out_of_range_id_before_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _prefill_handler(engine)

    with pytest.raises(
        HttpError, match=r"Token id 4294967295 is out of vocabulary at token_ids\[1\]$"
    ):
        await anext(handler.generate(_token_request([1, OUT_OF_RANGE]), _context()))

    engine.async_generate.assert_not_awaited()


def _diffusion_handler(engine):
    handler = DiffusionWorkerHandler.__new__(DiffusionWorkerHandler)
    handler.enable_trace = False
    return _with_token_input(handler, engine=engine)


@pytest.mark.asyncio
async def test_diffusion_passes_in_range_ids_to_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _diffusion_handler(engine)

    with pytest.raises(_Reached):
        await anext(handler.generate(_token_request([1, MAX_TOKEN_ID]), _context()))

    assert engine.async_generate.await_args.kwargs["input_ids"] == [1, MAX_TOKEN_ID]


@pytest.mark.asyncio
async def test_diffusion_rejects_out_of_range_id_before_the_engine():
    engine = SimpleNamespace(async_generate=AsyncMock(side_effect=_Reached))
    handler = _diffusion_handler(engine)

    with pytest.raises(HttpError) as error:
        await anext(handler.generate(_token_request([1, OUT_OF_RANGE]), _context()))

    assert error.value.code == 400
    engine.async_generate.assert_not_awaited()

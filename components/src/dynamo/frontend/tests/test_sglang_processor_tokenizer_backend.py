#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""`--tokenizer` / `DYN_TOKENIZER` in the SGLang chat processor (`--dyn-chat-processor sglang`)."""

import json
import types

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import dynamo.frontend.sglang_processor as sglang_processor_module
from dynamo.frontend.sglang_prepost import preprocess_chat_request
from dynamo.frontend.sglang_processor import (
    SglangEngineFactory,
    _init_worker,
    _load_tokenizer_with_fallback,
    _preprocess_worker,
)

fastokens = pytest.importorskip("fastokens")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.model("Qwen/Qwen3-VL-2B-Instruct"),
    pytest.mark.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

MODEL = "Qwen/Qwen3-0.6B"


def _backend_module(tokenizer) -> str:
    return type(tokenizer.backend_tokenizer).__module__.split(".")[0]


@pytest.fixture(autouse=True)
def _restore_process_state(monkeypatch):
    """Undo what these tests change process-wide: the preprocess-worker globals
    that _init_worker() sets, and fastokens' patch of transformers."""
    for name in [n for n in vars(sglang_processor_module) if n.startswith("_w_")]:
        monkeypatch.setattr(
            sglang_processor_module, name, getattr(sglang_processor_module, name)
        )
    yield
    fastokens.unpatch_transformers()


@pytest.fixture(scope="module")
def hf_tokenizer():
    """HuggingFace reference tokenizers, loaded once per model for this module."""
    cache = {}

    def load(model):
        if model not in cache:
            tokenizer, backend = _load_tokenizer_with_fallback(
                model, False, "huggingface", True
            )
            assert backend == "huggingface"
            assert _backend_module(tokenizer) == "tokenizers"
            cache[model] = tokenizer
        return cache[model]

    return load


@pytest.fixture
def unsupported_tokenizer_dir(tmp_path):
    """A valid HuggingFace tokenizer that fastokens cannot load (WordPiece)."""
    tok = Tokenizer(
        models.WordPiece({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    return str(tmp_path)


def test_default_backend_uses_huggingface(hf_tokenizer):
    hf_tokenizer(MODEL)  # asserts the HuggingFace backend


def test_unsupported_tokenizer_falls_back_to_huggingface(unsupported_tokenizer_dir):
    tokenizer, backend = _load_tokenizer_with_fallback(
        unsupported_tokenizer_dir, False, "fastokens", True
    )
    assert backend == "huggingface"
    assert _backend_module(tokenizer) == "tokenizers"
    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]


def test_unsupported_tokenizer_raises_without_fallback(unsupported_tokenizer_dir):
    with pytest.raises(RuntimeError, match="fastokens failed to load"):
        _load_tokenizer_with_fallback(
            unsupported_tokenizer_dir, False, "fastokens", False
        )


def test_fastokens_after_another_models_fallback(unsupported_tokenizer_dir):
    """A frontend serving several models: one model's fallback must not
    silently downgrade a later model that also requests fastokens."""
    _load_tokenizer_with_fallback(MODEL, False, "fastokens", True)
    _load_tokenizer_with_fallback(unsupported_tokenizer_dir, False, "fastokens", True)
    tokenizer, backend = _load_tokenizer_with_fallback(MODEL, False, "fastokens", True)
    assert backend == "fastokens"
    assert _backend_module(tokenizer) == "fastokens"


@pytest.mark.parametrize(
    "frontend_backend, expected",
    [
        ("default", "huggingface"),
        ("fastokens", "fastokens"),
        ("basetenkenizer", "huggingface"),
    ],
)
def test_factory_maps_frontend_tokenizer_flag(frontend_backend, expected):
    config = types.SimpleNamespace(
        trust_remote_code=False,
        tokenizer_backend=frontend_backend,
        tokenizer_fallback=False,
    )
    factory = SglangEngineFactory(config)
    assert factory.tokenizer_backend == expected
    assert factory.tokenizer_fallback is False


# Token IDs must not change when --tokenizer fastokens is selected. Each model
# is one the frontend tests already use. TinyLlama's byte-fallback tokenizer is
# one fastokens cannot load, so it checks the fallback path on a real model.
TOKEN_ID_MODELS = [
    pytest.param(MODEL, "fastokens", True, id="qwen3"),
    pytest.param("Qwen/Qwen3-VL-2B-Instruct", "fastokens", True, id="qwen3-vl"),
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "huggingface", False, id="tinyllama"
    ),
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

EDGE_TEXTS = [
    "Hello, world!",
    "naïve café — 東京 🚀 Здравствуйте مرحبا नमस्ते 안녕하세요",
    "👩\u200d👩\u200d👧\u200d👦 🇯🇵 👍🏽 e\u0301 \u200b\u200d\ufeff",
    "def f(x):\n\treturn {'a': [1, 2]}  # comment\n\n\n    trailing   ",
    "1234567890 3.14159 1e-9 0xDEADBEEF ١٢٣",
    "<think> literal </think> <|im_start|> <|endoftext|> [INST] </s> <s>",
    "\r\n CRLF \x0b \x0c",
    "ab" * 2000,
]


def _requests(with_tools: bool) -> list[dict]:
    long_text = " ".join(f"token{i} 東京 🚀" for i in range(6000))
    reqs = [
        {"messages": [{"role": "user", "content": text}]} for text in EDGE_TEXTS
    ] + [
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": EDGE_TEXTS[1]},
                {"role": "assistant", "content": EDGE_TEXTS[3]},
                {"role": "user", "content": EDGE_TEXTS[5]},
            ]
        },
        {"messages": [{"role": "user", "content": long_text}]},
    ]
    if with_tools:
        reqs += [
            {
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [WEATHER_TOOL],
            },
            {
                "messages": [
                    {"role": "user", "content": "Weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_0", "content": "19 °C"},
                ],
                "tools": [WEATHER_TOOL],
            },
        ]
    return reqs


def _prompt_ids(tokenizer, request: dict, model: str) -> list[int]:
    return preprocess_chat_request(
        {**request, "model": model},
        tokenizer=tokenizer,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    ).prompt_token_ids


@pytest.mark.parametrize("model, expected_backend, with_tools", TOKEN_ID_MODELS)
def test_prompt_token_ids_unchanged(hf_tokenizer, model, expected_backend, with_tools):
    """The request path's prompt token IDs are identical under both backends."""
    hf = hf_tokenizer(model)
    fast, backend = _load_tokenizer_with_fallback(model, False, "fastokens", True)
    assert backend == expected_backend
    for request in _requests(with_tools):
        assert _prompt_ids(fast, request, model) == _prompt_ids(hf, request, model)


@pytest.mark.parametrize(
    "model", [MODEL, "Qwen/Qwen3-VL-2B-Instruct"], ids=["qwen3", "qwen3-vl"]
)
def test_encode_and_decode_unchanged(hf_tokenizer, model):
    hf = hf_tokenizer(model)
    fast, backend = _load_tokenizer_with_fallback(model, False, "fastokens", True)
    assert backend == "fastokens"
    for text in EDGE_TEXTS:
        for add_special_tokens in (False, True):
            ids = hf.encode(text, add_special_tokens=add_special_tokens)
            assert fast.encode(text, add_special_tokens=add_special_tokens) == ids
            assert fast.decode(ids) == hf.decode(ids)


def test_preprocess_worker_token_ids_unchanged(hf_tokenizer):
    """A preprocess worker on fastokens yields the main process's HuggingFace IDs."""
    hf = hf_tokenizer(MODEL)
    _init_worker(MODEL, None, None, tokenizer_backend="fastokens")
    assert _backend_module(sglang_processor_module._w_tokenizer) == "fastokens"
    for request in _requests(with_tools=True):
        result = _preprocess_worker({**request, "model": MODEL}, MODEL, None)
        assert result.prompt_token_ids == _prompt_ids(hf, request, MODEL)

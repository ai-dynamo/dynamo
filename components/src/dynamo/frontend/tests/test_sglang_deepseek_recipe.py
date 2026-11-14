# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe encoding against SGLang's real adapter, without loading a model."""

import asyncio
import copy
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

if importlib.util.find_spec("sglang") is None:
    pytest.skip("SGLang is not installed", allow_module_level=True)
from _routed_engine_fakes import FakeRoutedEngine  # noqa: E402

from dynamo.frontend import sglang_deepseek  # noqa: E402 -- optional SGLang guard
from dynamo.frontend import sglang_processor as module  # noqa: E402
from dynamo.frontend.sglang_prepost import create_parsers  # noqa: E402
from dynamo.frontend.sglang_processor import SglangProcessor  # noqa: E402
from dynamo.frontend.sglang_prompt_renderer import DeepSeekPromptRenderer  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def tokenizer():
    # Marker-only collection stubs SGLang without installing recipe dependencies.
    import deepseek_recipe as recipe
    import tokenizers

    assert (
        sglang_deepseek.recipe_renderer is not None
    ), "The supported SGLang installation must provide the recipe renderer"
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.add_special_tokens(
        [
            getattr(recipe.SpecialToken, name).default_str()
            for name in dir(recipe.SpecialToken)
            if name[0].isupper()
        ]
        + ["<｜deepseek_image｜>"]
    )

    class ServedTokenizer:
        backend_tokenizer = backend

        def decode(self, ids, *, skip_special_tokens=False):
            return backend.decode(ids, skip_special_tokens=skip_special_tokens)

    return ServedTokenizer()


def render(request, tokenizer, *, spec="dsv4", images=None):
    return sglang_deepseek.render_recipe_prompt(
        request,
        messages=request["messages"],
        tokenizer=tokenizer,
        tools=request.get("tools"),
        spec=spec,
        image_urls=images or [],
    )


def output_processor(tokenizer, request, spec="dsv4", default_thinking_mode=None):
    suffix = "v41" if spec == "dsv41" else "v4"
    tool_name, reasoning_name = f"deepseek{suffix}", f"deepseek-{suffix}"
    tool_parser, reasoning_parser = create_parsers(
        request,
        tool_call_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        tokenizer=tokenizer,
    )
    processor = SglangProcessor(
        tokenizer,
        None,
        tool_name,
        reasoning_name,
        [tokenizer.backend_tokenizer.token_to_id("<｜end▁of▁sentence｜>")],
        deepseek_spec=spec,
        default_thinking_mode=default_thinking_mode,
    )
    return processor._build_postprocessor(
        request,
        {"sampling_options": {"guided_decoding": None}, "require_reasoning": False},
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        named_zero_arg_tool=None,
        prompt_token_ids=[],
    )


@pytest.mark.parametrize("spec", ["dsv4", "dsv41"])
@pytest.mark.parametrize("thinking,chunk_size", [(False, 1), (True, 4096)])
def test_recipe_output_preserves_tool_arguments(tokenizer, spec, thinking, chunk_size):
    request = {
        "tools": [{"type": "function", "function": {"name": "echo"}}],
        "chat_template_kwargs": {"thinking": thinking},
    }
    text = (
        ("check</think>" if thinking else "")
        + '<｜DSML｜tool_calls><｜DSML｜invoke name="echo">'
        '<｜DSML｜parameter name="text" string="true">  <think>literal</think>  </｜DSML｜parameter>'
        '<｜DSML｜parameter name="large" string="false">9007199254740993.0</｜DSML｜parameter>'
        '<｜DSML｜parameter name="tiny" string="false">1e-400</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜tool_calls>"
    )
    post = output_processor(tokenizer, request, spec)
    ids = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
    arguments, reasoning, names = [], [], []
    for start in range(0, len(ids), chunk_size):
        choice = post.process_output(
            {
                "token_ids": ids[start : start + chunk_size],
                "finish_reason": "stop" if start + chunk_size >= len(ids) else None,
            }
        )
        if choice is None:
            continue
        delta = choice["delta"]
        reasoning.append(delta.get("reasoning_content", ""))
        for call in delta.get("tool_calls", []):
            arguments.append(call["function"]["arguments"])
            if "name" in call["function"]:
                names.append(call["function"]["name"])
    assert names == ["echo"]
    assert "".join(arguments) == (
        '{"text": "  <think>literal</think>  ", "large": 9007199254740993.0, "tiny": 1e-400}'
    )
    assert "".join(reasoning) == ("check" if thinking else "")
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("chunk_size", [1, 4096])
def test_recipe_output_local_stop_and_logprobs(tokenizer, chunk_size):
    post = output_processor(tokenizer, {"stop": "STOP"})
    ids = tokenizer.backend_tokenizer.encode("answerSTOPignored").ids
    choices = []
    for start in range(0, len(ids), chunk_size):
        batch = ids[start : start + chunk_size]
        choice = post.process_output(
            {"token_ids": batch, "log_probs": [-0.1] * len(batch)}
        )
        if choice is not None:
            choices.append(choice)
        if post.locally_finished:
            break
    assert "".join(c["delta"].get("content", "") for c in choices) == "answer"
    assert choices[-1]["finish_reason"] == "stop"
    assert post.local_stop_reason == "STOP"
    assert post.post_stop_token_count() == (len("ignored") if chunk_size > 1 else 0)
    assert (
        "".join(
            entry["token"]
            for c in choices
            if c["logprobs"]
            for entry in c["logprobs"]["content"]
        )
        == "answer"
    )


@pytest.mark.parametrize("stop_field", ["stop", "stop_token_ids"])
def test_recipe_output_token_stop(tokenizer, stop_field):
    ids = tokenizer.backend_tokenizer.encode("answer!ignored").ids
    stop_id = tokenizer.backend_tokenizer.encode("!").ids[0]
    post = output_processor(tokenizer, {stop_field: [stop_id]})
    output = post.process_output({"token_ids": ids})
    assert output["delta"]["content"] == "answer"
    assert output["finish_reason"] == "stop"
    assert post.local_stop_reason == stop_id


def test_recipe_output_waits_for_utf8_before_local_stop(tokenizer):
    post = output_processor(tokenizer, {"stop": "STOP"})
    ids = tokenizer.backend_tokenizer.encode("answerSTOP🐈ignored").ids
    split = len(tokenizer.backend_tokenizer.encode("answerSTOP").ids) + 1
    assert post.process_output({"token_ids": ids[:split]}) is None
    output = post.process_output({"token_ids": ids[split:]})
    assert output["delta"]["content"] == "answer"
    assert output["finish_reason"] == "stop"
    assert post.post_stop_token_count() == len(ids) - len("answerSTOP")


def test_recipe_output_json_and_eos_without_tool_parsing(tokenizer):
    post = output_processor(
        tokenizer,
        {
            "tools": [{"type": "function", "function": {"name": "echo"}}],
            "tool_choice": "none",
            "response_format": {"type": "json_object"},
        },
    )
    ids = tokenizer.backend_tokenizer.encode('{"value":1}<｜end▁of▁sentence｜>').ids
    output = post.process_output({"token_ids": ids, "finish_reason": "stop"})
    assert output["delta"]["content"] == '{"value":1}'
    assert "tool_calls" not in output["delta"]
    assert output["finish_reason"] == "stop"


@pytest.mark.parametrize("use_pool", [False, True])
def test_recipe_output_generator_inline_and_pool(tokenizer, monkeypatch, use_pool):
    renderer = DeepSeekPromptRenderer(
        tokenizer,
        "dsv4",
        tool_call_parser_name="deepseekv4",
        reasoning_parser_name="deepseek-v4",
    )
    for key, value in {
        "_w_tokenizer": tokenizer,
        "_w_prompt_renderer": renderer,
        "_w_tool_call_parser_name": "deepseekv4",
        "_w_reasoning_parser_name": "deepseek-v4",
        "_w_default_thinking_mode": "enabled",
    }.items():
        monkeypatch.setattr(module, key, value)
    engine = FakeRoutedEngine(
        items=[
            {
                "token_ids": tokenizer.backend_tokenizer.encode(
                    "check</think>answer"
                ).ids,
                "finish_reason": "stop",
            }
        ]
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        processor = module.SglangProcessor(
            tokenizer,
            engine,
            "deepseekv4",
            "deepseek-v4",
            [],
            deepseek_spec="dsv4",
            prompt_renderer=renderer,
            default_thinking_mode="enabled",
            preprocess_pool=pool if use_pool else None,
            preprocess_workers=1,
        )

        async def collect():
            return [
                item
                async for item in processor.generator(
                    {
                        "model": "deepseek",
                        "messages": [{"role": "user", "content": "Hello"}],
                    }
                )
            ]

        choices = [
            choice
            for item in asyncio.run(collect())
            for choice in item["data"]["choices"]
        ]
    assert "".join(c["delta"].get("reasoning_content", "") for c in choices) == "check"
    assert "".join(c["delta"].get("content", "") for c in choices) == "answer"
    assert choices[-1]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "separate,stream", [(True, True), (False, True), (True, False)]
)
def test_recipe_output_uses_deployment_thinking_and_reasoning_controls(
    tokenizer, separate, stream
):
    post = output_processor(
        tokenizer,
        {
            "separate_reasoning": separate,
            "stream_reasoning": stream,
        },
        default_thinking_mode="enabled",
    )
    ids = tokenizer.backend_tokenizer.encode("check").ids
    first = post.process_output({"token_ids": ids})
    answer = post.process_output(
        {"token_ids": tokenizer.backend_tokenizer.encode("</think>answer").ids}
    )
    final = post.process_output({"token_ids": [], "finish_reason": "length"})
    deltas = [c["delta"] for c in (first, answer, final) if c is not None]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == (
        "check" if separate else ""
    )
    assert "".join(d.get("content", "") for d in deltas) == (
        "answer" if separate else "checkanswer"
    )
    assert final["finish_reason"] == "length"
    if separate and not stream:
        assert first is None or "reasoning_content" not in first["delta"]


def test_recipe_output_keeps_forced_json_array_parser(tokenizer):
    post = output_processor(
        tokenizer,
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )
    ids = tokenizer.backend_tokenizer.encode(
        '[{"name":"echo","parameters":{"value":"ok"}}]'
    ).ids
    output = post.process_output({"token_ids": ids, "finish_reason": "stop"})
    assert output["delta"]["tool_calls"][0]["function"] == {
        "name": "echo",
        "arguments": '{"value": "ok"}',
    }


@pytest.mark.parametrize("spec,effort", [("dsv4", "max"), ("dsv41", 37)])
@pytest.mark.parametrize("thinking", [False, True])
def test_recipe_uses_served_tokenizer_and_request_effort(
    tokenizer, spec, effort, thinking
):
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": effort,
        "chat_template_kwargs": {"thinking": thinking, "reasoning_effort": "low"},
        "temperature": 0,
        "max_tokens": 19,
    }
    original = copy.deepcopy(request)
    ids = render(request, tokenizer, spec=spec)
    prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
    assert "Hello" in prompt
    assert prompt.endswith("<think>" if thinking else "</think>")
    if thinking:
        assert "Reasoning Effort: 37" in prompt if spec == "dsv41" else "max" in prompt
    assert ids[0] == tokenizer.backend_tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
    assert request == original


def test_filtered_tools_leave_tool_choice_to_host_grammar(tokenizer):
    request = {
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
    }
    ids = render(request, tokenizer)
    prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
    assert "weather" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_v41_image_slots_match_worker_order(tokenizer):
    images = ["https://example.com/first.png", "https://example.com/second.png"]
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}} for url in images
                ],
            }
        ]
    }
    ids = render(request, tokenizer, spec="dsv41", images=images)
    image_id = tokenizer.backend_tokenizer.token_to_id("<｜deepseek_image｜>")
    assert ids.count(image_id) == 2
    with pytest.raises(sglang_deepseek.RecipeRequestError, match="image order"):
        render(request, tokenizer, spec="dsv41", images=list(reversed(images)))


def test_recipe_request_errors_do_not_hide_tokenizer_configuration_errors(tokenizer):
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": True,
    }
    with pytest.raises(sglang_deepseek.RecipeRequestError, match="boolean"):
        render(request, tokenizer, spec="dsv41")
    with pytest.raises(ValueError, match="fast tokenizer") as error:
        render(request, object(), spec="dsv41")
    assert not isinstance(error.value, sglang_deepseek.RecipeRequestError)


@pytest.mark.parametrize("spec", ["dsv4", "dsv41"])
@pytest.mark.parametrize("thinking", [False, True])
def test_recipe_continues_final_assistant_message(tokenizer, spec, thinking):
    request = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "The answer is"},
        ],
        "continue_final_message": True,
        "chat_template_kwargs": {"thinking": thinking},
    }
    original = copy.deepcopy(request)
    if thinking:
        with pytest.raises(
            sglang_deepseek.RecipeRequestError,
            match="continue_final_message with thinking",
        ):
            render(request, tokenizer, spec=spec)
    else:
        ids = render(request, tokenizer, spec=spec)
        prompt = tokenizer.backend_tokenizer.decode(ids, skip_special_tokens=False)
        assert prompt.endswith("The answer is")
    assert request == original


@pytest.mark.parametrize(
    "model_type,architecture,expected",
    [
        ("deepseek_v41", "DeepseekV4ForCausalLM", "dsv41"),
        ("deepseek_v4", "DeepseekV4ForCausalLM", "dsv4"),
        ("other", "OtherForCausalLM", None),
    ],
)
def test_recipe_spec_comes_from_checkpoint(
    tmp_path, model_type, architecture, expected
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]})
    )
    assert sglang_deepseek.resolve_deepseek_spec(str(tmp_path)) == expected


def test_recipe_normalizes_reasoning_history_without_mutating_request(tokenizer):
    request = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": ["Check the weather", "", "Use the tool"],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "chat_template_kwargs": {"thinking": True},
    }
    original = copy.deepcopy(request)
    ids = render(request, tokenizer, spec="dsv41")
    expected = copy.deepcopy(request)
    expected["messages"][1]["reasoning_content"] = "Check the weather\nUse the tool"
    assert ids == render(expected, tokenizer, spec="dsv41")
    assert request == original

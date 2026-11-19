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
from openai.lib.streaming.chat import ChatCompletionStreamState  # noqa: E402
from openai.types.chat import ChatCompletionChunk  # noqa: E402
from sglang.srt.function_call.function_call_parser import (  # noqa: E402
    FunctionCallParser,
)

from dynamo.frontend import sglang_deepseek  # noqa: E402 -- optional SGLang guard
from dynamo.frontend import sglang_processor as module  # noqa: E402
from dynamo.frontend.sglang_prepost import (  # noqa: E402
    build_tool_call_guided_decoding,
    convert_tools,
    create_parsers,
    named_closed_zero_arg_tool,
)
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


def output_processor(
    tokenizer,
    request,
    spec="dsv4",
    default_thinking_mode=None,
    tool_parser_enabled=True,
    structural_tag_mode="on",
):
    suffix = "v41" if spec == "dsv41" else "v4"
    tool_name, reasoning_name = f"deepseek{suffix}", f"deepseek-{suffix}"
    if not tool_parser_enabled:
        tool_name = None
    guided = build_tool_call_guided_decoding(
        request,
        tool_call_parser_name=tool_name,
        tokenizer=tokenizer,
        sglang_tools=convert_tools(request.get("tools")),
        structural_tag_mode=structural_tag_mode,
    )
    tool_parser, reasoning_parser = create_parsers(
        request,
        tool_call_parser_name=tool_name,
        reasoning_parser_name=reasoning_name,
        tokenizer=tokenizer,
        guided_decoding=guided,
    )
    processor = SglangProcessor(
        tokenizer,
        None,
        tool_name,
        reasoning_name,
        [tokenizer.backend_tokenizer.token_to_id("<｜end▁of▁sentence｜>")],
        prompt_renderer=DeepSeekPromptRenderer(
            tokenizer,
            spec,
            tool_call_parser_name=tool_name,
            reasoning_parser_name=reasoning_name,
        ),
        default_thinking_mode=default_thinking_mode,
    )
    return processor._build_postprocessor(
        request,
        {"sampling_options": {"guided_decoding": guided}, "require_reasoning": False},
        tool_call_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        named_zero_arg_tool=(
            named_closed_zero_arg_tool(request)
            if guided == {"regex": r"\{\}"}
            else None
        ),
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
    sdk = ChatCompletionStreamState()
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
        indices = [call["index"] for call in delta.get("tool_calls", [])]
        assert len(indices) == len(set(indices))
        sdk.handle_chunk(
            ChatCompletionChunk(
                id="recipe-test",
                object="chat.completion.chunk",
                created=0,
                model="deepseek",
                choices=[choice],
            )
        )
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
    message = sdk.get_final_completion().choices[0].message
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].function.name == "echo"
    assert message.tool_calls[0].function.arguments == "".join(arguments)


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
    ids = tokenizer.backend_tokenizer.encode("answer!").ids
    stop_id = tokenizer.backend_tokenizer.encode("!").ids[0]
    post = output_processor(tokenizer, {stop_field: [stop_id]})
    output = post.process_output(
        {"token_ids": ids, "finish_reason": "stop", "stop_reason": stop_id}
    )
    assert output["delta"]["content"] == "answer"
    assert output["finish_reason"] == "stop"


@pytest.mark.parametrize("engine_stop", [False, True])
def test_recipe_output_unicode_stop_across_chunks(tokenizer, engine_stop):
    post = output_processor(tokenizer, {"stop": "S🐈OP"})
    ids = tokenizer.backend_tokenizer.encode("answerS🐈OPignored").ids
    # First split a UTF-8 sequence, then hold a decoded stop prefix across calls.
    split = len(tokenizer.backend_tokenizer.encode("answerS").ids) + 1
    prefix_end = len(tokenizer.backend_tokenizer.encode("answerS🐈O").ids)
    chunks = [
        {"token_ids": ids[:split]},
        {"token_ids": ids[split:prefix_end]},
        {"token_ids": ids[prefix_end:]},
    ]
    if engine_stop:
        chunks[-1].update(finish_reason="stop", stop_reason="S🐈OP")
    choices = [choice for chunk in chunks if (choice := post.process_output(chunk))]
    assert "".join(choice["delta"].get("content", "") for choice in choices) == "answer"
    assert choices[-1]["finish_reason"] == "stop"
    assert post.local_stop_reason == "S🐈OP"
    assert post.post_stop_token_count() == len("ignored")


def test_recipe_output_streams_safe_prefix_with_repeated_stop_prefix(tokenizer):
    post = output_processor(tokenizer, {"stop": "ab"})
    assert post._recipe_parser is not None
    batch = tokenizer.backend_tokenizer.encode("xa").ids
    pending_bound = len(tokenizer.backend_tokenizer.encode("ab").ids)
    choices = []
    for _ in range(6):
        choice = post.process_output({"token_ids": batch})
        if choice is not None:
            choices.append(choice)
        assert not post.locally_finished
        assert len(post._recipe_pending_ids) <= pending_bound
    assert any(choice["delta"].get("content") for choice in choices)
    final = post.process_output(
        {"token_ids": tokenizer.backend_tokenizer.encode("b").ids}
    )
    choices.append(final)
    assert final["finish_reason"] == "stop"
    assert post.local_stop_reason == "ab"
    assert (
        "".join(choice["delta"].get("content", "") for choice in choices)
        == ("xa" * 6)[:-1]
    )


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
@pytest.mark.parametrize(
    "tool_choice", [None, "required", "named_zero_args", "legacy_json"]
)
def test_recipe_output_generator_inline_and_pool(
    tokenizer, monkeypatch, use_pool, tool_choice
):
    request = {"model": "deepseek", "messages": [{"role": "user", "content": "Hello"}]}
    tag_mode = "off" if tool_choice == "named_zero_args" else "on"
    answer = "answer"
    if tool_choice == "legacy_json":
        request["guided_json"] = {"type": "object"}
        answer = json.dumps({"literal": "<think>data</think><｜DSML｜tool_calls>"})
    elif tool_choice:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ]
        request["tool_choice"] = (
            {"type": "function", "function": {"name": "echo"}}
            if tool_choice == "named_zero_args"
            else tool_choice
        )
        answer = '<｜DSML｜tool_calls><｜DSML｜invoke name="echo"></｜DSML｜invoke></｜DSML｜tool_calls>'
        if tool_choice == "named_zero_args":
            answer = "{}"
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
        "_w_structural_tag_mode": tag_mode,
    }.items():
        monkeypatch.setattr(module, key, value)
    engine = FakeRoutedEngine(
        items=[
            {
                "token_ids": tokenizer.backend_tokenizer.encode(
                    answer if tool_choice == "legacy_json" else "check</think>" + answer
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
            prompt_renderer=renderer,
            structural_tag_mode=tag_mode,
            default_thinking_mode="enabled",
            preprocess_pool=pool if use_pool else None,
            preprocess_workers=1,
        )

        async def collect():
            return [item async for item in processor.generator(request)]

        choices = [
            choice
            for item in asyncio.run(collect())
            for choice in item["data"]["choices"]
        ]
    assert "".join(c["delta"].get("reasoning_content", "") for c in choices) == (
        "" if tool_choice == "legacy_json" else "check"
    )
    if tool_choice in ("required", "named_zero_args"):
        calls = [call for c in choices for call in c["delta"].get("tool_calls", [])]
        assert (
            next(
                call["function"]["name"]
                for call in calls
                if call["function"].get("name")
            )
            == "echo"
        )
        assert "".join(call["function"]["arguments"] for call in calls) == "{}"
        assert choices[-1]["finish_reason"] == "tool_calls"
        guidance = engine.requests[0]["sampling_options"]["guided_decoding"]
        if tool_choice == "named_zero_args":
            assert guidance == {"regex": r"\{\}"}
        else:
            assert "structural_tag" in guidance
            assert "DSML" in json.dumps(guidance)
    else:
        assert "".join(c["delta"].get("content", "") for c in choices) == answer
        assert choices[-1]["finish_reason"] == "stop"
        if tool_choice == "legacy_json":
            assert engine.requests[0]["sampling_options"]["guided_decoding"] == {
                "json": request["guided_json"]
            }
            assert engine.requests[0]["require_reasoning"] is False


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


@pytest.mark.parametrize(
    "mode,unavailable,named",
    [("on", False, False), ("off", False, False), ("on", True, True)],
)
def test_recipe_output_matches_effective_tool_guidance(
    tokenizer, monkeypatch, mode, unavailable, named
):
    if unavailable:
        monkeypatch.setattr(
            FunctionCallParser, "get_structure_constraint", lambda *args, **kwargs: None
        )
    request = {
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
        "tool_choice": (
            {"type": "function", "function": {"name": "echo"}} if named else "required"
        ),
    }
    native = mode == "on" and not unavailable
    guidance = build_tool_call_guided_decoding(
        request,
        tool_call_parser_name="deepseekv4",
        tokenizer=tokenizer,
        sglang_tools=convert_tools(request["tools"]),
        structural_tag_mode=mode,
    )
    assert ("structural_tag" if native else "json") in guidance
    post = output_processor(tokenizer, request, structural_tag_mode=mode)
    assert (post._recipe_parser is not None) is native
    text = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="echo">'
        '<｜DSML｜parameter name="value" string="true">ok</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜tool_calls>"
        if native
        else '[{"name":"echo","parameters":{"value":"ok"}}]'
    )
    output = post.process_output(
        {
            "token_ids": tokenizer.backend_tokenizer.encode(text).ids,
            "finish_reason": "stop",
        }
    )
    calls = output["delta"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert json.loads(calls[0]["function"]["arguments"]) == {"value": "ok"}
    assert output["finish_reason"] == "tool_calls"


def test_recipe_output_reasoning_only_forced_choice(tokenizer):
    post = output_processor(
        tokenizer,
        {
            "tools": [{"type": "function", "function": {"name": "echo"}}],
            "tool_choice": "required",
        },
        tool_parser_enabled=False,
    )
    # Required JSON selects a tool parser even without a configured native parser.
    assert post._recipe_parser is None
    text = '[{"name":"echo","parameters":{}}]'
    output = post.process_output(
        {
            "token_ids": tokenizer.backend_tokenizer.encode(text).ids,
            "finish_reason": "stop",
        }
    )
    calls = output["delta"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "echo"
    assert calls[0]["function"]["arguments"] == "{}"
    assert output["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    "finish_reason,schedule",
    [
        ("cancelled", "same_chunk"),
        ("abort:cancelled", "split"),
        ("cancelled", "pending_logprobs"),
        ("stop", "split"),
    ],
)
def test_recipe_generator_abort_does_not_flush_tool_call(
    tokenizer, finish_reason, schedule
):
    request = {
        "model": "deepseek",
        "messages": [{"role": "user", "content": "Call echo"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                    },
                },
            }
        ],
        "chat_template_kwargs": {"thinking": False},
    }
    text = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="echo">'
        '<｜DSML｜parameter name="n" string="false">1</｜DSML｜parameter>'
    )
    if finish_reason == "stop":
        text += "</｜DSML｜invoke></｜DSML｜tool_calls>"
    ids = tokenizer.backend_tokenizer.encode(text).ids
    if schedule == "same_chunk":
        chunks = [{"token_ids": ids, "finish_reason": finish_reason}]
    elif schedule == "pending_logprobs":
        request["logprobs"] = True
        chunks = [
            {"token_ids": tokenizer.backend_tokenizer.encode("before ").ids},
            {"token_ids": ids, "log_probs": [-0.1] * len(ids)},
            {
                "token_ids": tokenizer.backend_tokenizer.encode("ignored").ids,
                "finish_reason": finish_reason,
            },
        ]
    else:
        chunks = [
            {"token_ids": ids},
            {"token_ids": [], "finish_reason": finish_reason},
        ]
    engine = FakeRoutedEngine(items=chunks)
    renderer = DeepSeekPromptRenderer(
        tokenizer,
        "dsv4",
        tool_call_parser_name="deepseekv4",
        reasoning_parser_name="deepseek-v4",
    )
    processor = SglangProcessor(
        tokenizer,
        engine,
        "deepseekv4",
        "deepseek-v4",
        [],
        prompt_renderer=renderer,
        stream_interval=4096 if schedule == "pending_logprobs" else 1,
    )

    async def collect():
        return [item async for item in processor.generator(request)]

    async def run():
        return await asyncio.wait_for(collect(), timeout=5)

    choices = [
        choice for item in asyncio.run(run()) for choice in item["data"]["choices"]
    ]
    calls = [
        call for choice in choices for call in choice["delta"].get("tool_calls", [])
    ]
    if finish_reason == "stop":
        assert choices[-1]["finish_reason"] == "tool_calls"
        assert json.loads("".join(call["function"]["arguments"] for call in calls)) == {
            "n": 1
        }
    else:
        assert choices[-1]["finish_reason"] == "stop"
        assert "tool_calls" not in choices[-1]["delta"]
        assert all(choice["finish_reason"] != "tool_calls" for choice in choices)
        if schedule == "split":
            # Already emitted fragments survive, but abort must not complete them.
            assert calls
            assert (
                next(
                    call["function"]["name"]
                    for call in calls
                    if "name" in call["function"]
                )
                == "echo"
            )
            assert not "".join(
                call["function"]["arguments"] for call in calls
            ).endswith("}")
        else:
            assert calls == []
        if schedule == "pending_logprobs":
            assert (
                "".join(choice["delta"].get("content", "") for choice in choices)
                == "before "
            )
    assert engine.stream_released


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

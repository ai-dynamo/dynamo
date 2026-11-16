# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from typing import Any, Protocol

from jinja2.exceptions import TemplateError
from sglang.srt.parser.jinja_template_utils import (
    detect_jinja_template_content_format,
    process_content_for_template_format,
)

from .sglang_deepseek import (
    RecipeRequestError,
    recipe_renderer,
    render_recipe_prompt,
    resolve_deepseek_spec,
)
from .utils import PreprocessError, extract_mm_urls


class PromptRenderer(Protocol):
    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        ...


class JinjaPromptRenderer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        # Build template kwargs -- single call for rendering + tokenization
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
        }
        if resolved_responses_template is not None:
            template_kwargs["chat_template"] = resolved_responses_template
        if template_tools:
            template_kwargs["tools"] = request.get(
                "responses_template_tools", template_tools
            )

        chat_template_kwargs = (
            request.get("chat_template_kwargs")
            or request.get("chat_template_args")
            or {}
        )
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        if (reasoning_effort := request.get("reasoning_effort")) is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort

        template_messages = _normalize_messages_for_template(messages, self._tokenizer)

        try:
            rendered = self._tokenizer.apply_chat_template(
                template_messages, **template_kwargs
            )
        except (TemplateError, TypeError) as exc:
            # Jinja filters such as tojson can raise TypeError for invalid inputs.
            raise PreprocessError(str(exc)) from exc
        return _normalize_prompt_token_ids(rendered)


_KIMI_K3_THINKING_EFFORTS = frozenset({"low", "high", "max"})


def _normalize_kimi_k3_hint(value: Any) -> str:
    return str(value or "").lower().replace("-", "").replace("_", "")


def _should_use_kimi_k3_prompt_bridge(
    request: dict[str, Any],
    *,
    tokenizer,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> bool:
    """Return whether to translate OpenAI controls for Kimi K3's encoder.

    K3 ships no Jinja ``chat_template``; its tokenizer renders XTML directly and
    consumes ``tool_choice``/``response_format``/``thinking_effort`` as template
    kwargs. Limit the translation to K3 so unrelated templates keep receiving
    exactly the caller's template arguments.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        return False

    return any(
        "kimik3" in _normalize_kimi_k3_hint(hint)
        for hint in (
            request.get("model"),
            tool_call_parser_name,
            reasoning_parser_name,
        )
    )


def _kimi_k3_tool_choice_kwargs(
    request: dict[str, Any],
    *,
    has_tools: bool,
) -> dict[str, Any]:
    """Translate an OpenAI ``tool_choice`` into K3 encoder arguments.

    Forward ``required`` and ``none`` when the request declares tools; the K3
    encoder renders the corresponding instruction itself. Named choices use
    a synthetic system message because the encoder implements only the two
    string forms. ``auto`` and an omitted choice need no instruction; an
    explicit ``null`` is equivalent to omission here.
    """
    if not has_tools:
        return {}

    tool_choice = request.get("tool_choice")
    if tool_choice in ("required", "none"):
        return {"tool_choice": tool_choice}
    return {}


def _kimi_k3_thinking_effort_kwargs(
    request: dict[str, Any],
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Map OpenAI ``reasoning_effort`` to K3's ``thinking_effort``.

    An explicit K3 template argument wins. ``"none"`` leaves the effort unset
    and does not toggle thinking; upstream request normalization owns that
    control. Like Dynamo's Rust renderer, reject an unsupported tier while
    thinking is enabled instead of silently substituting the encoder default.
    """
    thinking_disabled = (
        chat_template_kwargs.get("thinking") is False
        or chat_template_kwargs.get("thinking_mode") == "disabled"
    )

    # The native encoder ignores effort entirely when thinking is disabled.
    if thinking_disabled:
        return {}

    if "thinking_effort" in chat_template_kwargs:
        effort = chat_template_kwargs["thinking_effort"]
        # Explicit null suppresses the encoder's default max-effort instruction.
        if effort is None or (
            isinstance(effort, str) and effort in _KIMI_K3_THINKING_EFFORTS
        ):
            return {}
        raise PreprocessError(
            f"Kimi K3 does not support thinking_effort={effort!r}; "
            "supported values are 'low', 'high', and 'max'"
        )

    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = chat_template_kwargs.get("reasoning_effort")
    if reasoning_effort is None:
        return {}
    if reasoning_effort == "none":
        return {}
    if (
        isinstance(reasoning_effort, str)
        and reasoning_effort in _KIMI_K3_THINKING_EFFORTS
    ):
        return {"thinking_effort": reasoning_effort}

    raise PreprocessError(
        f"Kimi K3 does not support reasoning_effort={reasoning_effort!r}; "
        "supported values are 'low', 'high', and 'max'"
    )


def _kimi_k3_response_format_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    response_format = request.get("response_format")
    if not isinstance(response_format, dict):
        return {}
    # The tokenizer may normalize the schema in place. Keep the caller's request
    # object independent of prompt rendering.
    return {"response_format": copy.deepcopy(response_format)}


def _normalize_kimi_k3_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match SGLang's K3 encoder message normalization without mutating input."""
    return [
        {**message, "role": "system"} if message.get("role") == "developer" else message
        for message in messages
    ]


class KimiK3PromptRenderer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        chat_template_kwargs = (
            request.get("chat_template_kwargs")
            or request.get("chat_template_args")
            or {}
        )
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            **_kimi_k3_tool_choice_kwargs(
                request, has_tools=bool(request.get("tools"))
            ),
            **_kimi_k3_thinking_effort_kwargs(request, chat_template_kwargs),
            **_kimi_k3_response_format_kwargs(request),
        }
        if resolved_responses_template is not None:
            template_kwargs["chat_template"] = resolved_responses_template
        if template_tools:
            template_kwargs["tools"] = request.get(
                "responses_template_tools", template_tools
            )
        template_kwargs.update(chat_template_kwargs)
        template_kwargs.pop("reasoning_effort", None)
        template_kwargs["tokenize"] = True
        template_kwargs["return_dict"] = False

        template_messages = _normalize_kimi_k3_messages(
            _normalize_messages_for_template(messages, self._tokenizer)
        )
        try:
            rendered = self._tokenizer.apply_chat_template(
                template_messages, **template_kwargs
            )
        except (TemplateError, TypeError) as exc:
            raise PreprocessError(str(exc)) from exc
        return _normalize_prompt_token_ids(rendered)


class DeepSeekPromptRenderer:
    def __init__(
        self,
        tokenizer: Any,
        spec: str | None,
        *,
        tool_call_parser_name: str | None,
        reasoning_parser_name: str | None,
    ):
        self._tokenizer = tokenizer
        self._spec = spec
        self._tool_call_parser_name = tool_call_parser_name
        self._reasoning_parser_name = reasoning_parser_name
        self._jinja = JinjaPromptRenderer(tokenizer)

    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        if self._spec is None and not _should_use_deepseek_v4_encoding(
            request,
            tokenizer=self._tokenizer,
            tool_call_parser_name=self._tool_call_parser_name,
            reasoning_parser_name=self._reasoning_parser_name,
        ):
            return self._jinja.render(
                request,
                messages=messages,
                template_tools=template_tools,
                resolved_responses_template=resolved_responses_template,
            )
        return _render_deepseek_v4_prompt_token_ids(
            request,
            messages=messages,
            tokenizer=self._tokenizer,
            template_tools=template_tools,
            spec=self._spec or "dsv4",
        )


class _NativePromptRenderer:
    """Select a native encoder using request hints for template-free tokenizers."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        tool_call_parser_name: str | None,
        reasoning_parser_name: str | None,
    ):
        self._tokenizer = tokenizer
        self._tool_call_parser_name = tool_call_parser_name
        self._reasoning_parser_name = reasoning_parser_name
        self._kimi = KimiK3PromptRenderer(tokenizer)
        self._fallback = DeepSeekPromptRenderer(
            tokenizer,
            None,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
        )

    def select(self, request: dict[str, Any]) -> PromptRenderer:
        if _should_use_kimi_k3_prompt_bridge(
            request,
            tokenizer=self._tokenizer,
            tool_call_parser_name=self._tool_call_parser_name,
            reasoning_parser_name=self._reasoning_parser_name,
        ):
            return self._kimi
        return self._fallback

    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        return self.select(request).render(
            request,
            messages=messages,
            template_tools=template_tools,
            resolved_responses_template=resolved_responses_template,
        )


def resolve_prompt_renderer(
    renderer: PromptRenderer, request: dict[str, Any]
) -> PromptRenderer:
    if isinstance(renderer, _NativePromptRenderer):
        return renderer.select(request)
    return renderer


def build_prompt_renderer(
    tokenizer: Any,
    *,
    model_path: str | None = None,
    chat_template: str | None = None,
    tool_call_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
) -> PromptRenderer:
    if chat_template is not None:
        return JinjaPromptRenderer(tokenizer)

    spec = resolve_deepseek_spec(model_path) if model_path is not None else None
    if spec is not None and (
        getattr(tokenizer, "chat_template", None) is None or recipe_renderer is not None
    ):
        return DeepSeekPromptRenderer(
            tokenizer,
            spec,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
        )
    if getattr(tokenizer, "chat_template", None) is None:
        return _NativePromptRenderer(
            tokenizer,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
        )
    return JinjaPromptRenderer(tokenizer)


def _normalize_deepseek_v4_hint(value: Any) -> str:
    return str(value or "").lower().replace("-", "").replace("_", "")


def _should_use_deepseek_v4_encoding(
    request: dict[str, Any],
    *,
    tokenizer,
    tool_call_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> bool:
    if getattr(tokenizer, "chat_template", None) is not None:
        return False

    return any(
        "deepseekv4" in _normalize_deepseek_v4_hint(value)
        for value in (
            request.get("model"),
            tool_call_parser_name,
            reasoning_parser_name,
        )
    )


def _flatten_message_content(content: Any) -> Any:
    """Flatten an OpenAI content-parts array to a plain string for the DSv4 encoder.

    SGLang's DeepSeek-V4 encoder (``encoding_dsv4.encode_messages``) consumes
    string content; SGLang's own OpenAI server flattens parts-list content before
    calling it (``serving_chat._process_messages`` runs the ``"string"`` content
    format over each message). Dynamo invokes ``encode_messages`` directly, so it
    must do the same flattening or the standard structured form
    ``[{"type": "text", "text": "..."}]`` crashes the encoder with
    ``TypeError: sequence item 0: expected str instance, list found``.

    Replicate SGLang's ``"string"``-format behaviour for byte-parity with native
    SGLang serving: join the text parts with a single space and drop non-text
    parts (DSv4 is text-only). Non-list content (``str``/``None``) is returned
    unchanged.
    """
    if not isinstance(content, list):
        return content
    text_parts = [
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ]
    return " ".join(text_parts)


def _render_deepseek_v4_prompt_token_ids(
    request: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tokenizer,
    template_tools: list[dict[str, Any]] | None,
    spec: str = "dsv4",
) -> list[int]:
    """Use SGLang's recipe adapter or its legacy V4 checkpoint encoder."""
    if recipe_renderer is not None:
        try:
            media, _ = extract_mm_urls(messages)
        except ValueError as exc:
            raise PreprocessError(str(exc)) from exc
        images = (media or {}).get("image_url", [])
        if any("Url" not in image for image in images):
            raise PreprocessError("DeepSeek recipe encoding requires image URLs")
        try:
            return render_recipe_prompt(
                request,
                messages=messages,
                tokenizer=tokenizer,
                tools=template_tools,
                spec=spec,
                image_urls=[image["Url"] for image in images],
            )
        except RecipeRequestError as exc:
            raise PreprocessError(str(exc)) from exc

    if spec == "dsv41":
        raise ValueError(
            "DeepSeek-V4.1 preprocessing requires SGLang's deepseek-recipe renderer "
            "(sglang.srt.parser.deepseek_recipe_renderer, SGLang 0.5.23 or newer)."
        )
    try:
        from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
    except ImportError as exc:
        raise ValueError(
            "DeepSeek-V4 preprocessing requires SGLang's "
            "sglang.srt.entrypoints.openai.encoding_dsv4 encoder. "
            "Install an SGLang build that includes the DeepSeek-V4 integration."
        ) from exc

    encoding_messages = copy.deepcopy(messages)
    for msg in encoding_messages:
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        else:
            msg["content"] = _flatten_message_content(content)

        # encoding_dsv4.encode_arguments_to_dsml expects tool_call.arguments as
        # the OpenAI-wire JSON *string* (it json.loads() internally).
        # _normalize_assistant_tool_call_arguments parses arguments to a dict for
        # Jinja chat templates, but a dict reaching this V4 encoder trips its
        # json.loads() fallback into a single name="arguments" parameter wrapping
        # the whole object — which the model then imitates, emitting a spurious
        # nested {"arguments": {...}} on its next call. SGLang native keeps this
        # path string-typed (serving_chat.py only dict-parses for the Jinja
        # branch); mirror that by re-serializing here.
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), (dict, list)):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)

    if template_tools:
        if not encoding_messages or encoding_messages[0].get("role") != "system":
            encoding_messages.insert(0, {"role": "system", "content": ""})
        encoding_messages[0]["tools"] = template_tools

    chat_template_kwargs = dict(
        request.get("chat_template_kwargs") or request.get("chat_template_args") or {}
    )
    thinking_mode = "thinking" if chat_template_kwargs.get("thinking") else "chat"
    reasoning_effort = (
        request.get("reasoning_effort")
        or chat_template_kwargs.get("reasoning_effort")
        or None
    )
    if reasoning_effort not in ("max", "high", None):
        reasoning_effort = None

    prompt = encode_messages(
        encoding_messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    return _normalize_prompt_token_ids(tokenizer.encode(prompt))


def _normalize_prompt_token_ids(prompt_token_ids: Any) -> list[int]:
    """Flatten ``apply_chat_template`` output to ``list[int]``.

    On transformers v5 the default ``TokenizersBackend`` returns a
    ``BatchEncoding`` from ``apply_chat_template(..., tokenize=True)``;
    unwrap to ``.input_ids`` (a flat list for a single conversation).
    """
    ids = getattr(prompt_token_ids, "input_ids", prompt_token_ids)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", prompt_token_ids)
    return list(ids)


def _normalize_messages_for_template(
    messages: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    """Normalize OpenAI media chunks (``image_url``/``video_url``/``audio_url``)
    to the simple ``image``/``video``/``audio`` types most VLM chat templates
    branch on. Without this, templates that gate placeholder emission on
    ``item.type == 'image'`` never fire for raw OpenAI input, and the
    rendered prompt has no slot for the media bytes that
    ``extract_mm_urls()`` forwards in parallel. Mirrors the equivalent
    step in sglang's own OpenAI server and dynamo's Rust default path.
    """
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    content_format = detect_jinja_template_content_format(chat_template)
    # The media-data side outputs are discarded: dynamo's separate
    # ``extract_mm_urls()`` channel is the source of truth for the worker.
    image_sink: list = []
    video_sink: list = []
    audio_sink: list = []
    modality_sink: list = []
    return [
        process_content_for_template_format(
            msg,
            content_format,
            image_data=image_sink,
            video_data=video_sink,
            audio_data=audio_sink,
            modalities=modality_sink,
        )
        for msg in messages
    ]

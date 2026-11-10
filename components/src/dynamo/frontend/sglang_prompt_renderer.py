# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import re
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
from .sglang_tool_call_markers import (
    _marker_free_encoder,
    tool_call_marker_strings,
    tool_call_marker_token_ids,
)
from .utils import PreprocessError, extract_mm_urls

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        tokenizer: Any,
        *,
        tool_call_parser_name: str | None = None,
        isolate_tool_call_markers: bool = False,
    ):
        self._tokenizer = tokenizer
        self._tool_call_parser_name = tool_call_parser_name
        self._isolate_tool_call_markers = isolate_tool_call_markers

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
            if self._isolate_tool_call_markers:
                return encode_chat_prompt_isolating_markers(
                    self._tokenizer,
                    template_messages,
                    template_kwargs,
                    self._tool_call_parser_name,
                )
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
    def __init__(self, tokenizer: Any, *, isolate_tool_call_markers: bool = False):
        self._tokenizer = tokenizer
        self._isolate_tool_call_markers = isolate_tool_call_markers

    def render(
        self,
        request: dict[str, Any],
        *,
        messages: list[dict[str, Any]],
        template_tools: list[dict[str, Any]] | None,
        resolved_responses_template: str | None,
    ) -> list[int]:
        if self._isolate_tool_call_markers:
            # The K3 encoder renders content itself; until it can isolate
            # delimiters, reject rather than encode them unprotected.
            raise PreprocessError(
                "--isolate-tool-call-markers is not supported with the "
                "Kimi K3 prompt encoder"
            )
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
        isolate_tool_call_markers: bool = False,
    ):
        self._tokenizer = tokenizer
        self._spec = spec
        self._tool_call_parser_name = tool_call_parser_name
        self._reasoning_parser_name = reasoning_parser_name
        self._isolate_tool_call_markers = isolate_tool_call_markers
        self._jinja = JinjaPromptRenderer(
            tokenizer,
            tool_call_parser_name=tool_call_parser_name,
            isolate_tool_call_markers=isolate_tool_call_markers,
        )

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
        if self._isolate_tool_call_markers:
            # The DeepSeek encoder renders content itself; until it can isolate
            # delimiters, reject rather than encode them unprotected.
            raise PreprocessError(
                "--isolate-tool-call-markers is not supported with the "
                "DeepSeek-V4 prompt encoder"
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
        isolate_tool_call_markers: bool = False,
    ):
        self._tokenizer = tokenizer
        self._tool_call_parser_name = tool_call_parser_name
        self._reasoning_parser_name = reasoning_parser_name
        self._kimi = KimiK3PromptRenderer(
            tokenizer, isolate_tool_call_markers=isolate_tool_call_markers
        )
        self._fallback = DeepSeekPromptRenderer(
            tokenizer,
            None,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            isolate_tool_call_markers=isolate_tool_call_markers,
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
    isolate_tool_call_markers: bool = False,
) -> PromptRenderer:
    """Build the prompt renderer for a served model.

    With *isolate_tool_call_markers*, message content that contains a
    delimiter of *tool_call_parser_name* is tokenized without the delimiter
    tokens (see :func:`encode_chat_prompt_isolating_markers`); the native
    DeepSeek and Kimi K3 encoders cannot do that yet and reject requests
    while it is on.
    """
    if chat_template is not None:
        return JinjaPromptRenderer(
            tokenizer,
            tool_call_parser_name=tool_call_parser_name,
            isolate_tool_call_markers=isolate_tool_call_markers,
        )

    spec = resolve_deepseek_spec(model_path) if model_path is not None else None
    if spec is not None and (
        getattr(tokenizer, "chat_template", None) is None or recipe_renderer is not None
    ):
        return DeepSeekPromptRenderer(
            tokenizer,
            spec,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            isolate_tool_call_markers=isolate_tool_call_markers,
        )
    if getattr(tokenizer, "chat_template", None) is None:
        return _NativePromptRenderer(
            tokenizer,
            tool_call_parser_name=tool_call_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            isolate_tool_call_markers=isolate_tool_call_markers,
        )
    return JinjaPromptRenderer(
        tokenizer,
        tool_call_parser_name=tool_call_parser_name,
        isolate_tool_call_markers=isolate_tool_call_markers,
    )


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


# A content span is swapped for ``\ue100<index>\ue101`` while the template is
# rendered a second time; the two renderings are then aligned piece by piece.
# Content that itself contains either character is treated as a span, so the
# sentinel cannot be forged from the request.
_CONTENT_SPAN_SENTINEL_RE = re.compile("\ue100(\\d+)\ue101")
_CONTENT_SPAN_SENTINEL_CHARS = ("\ue100", "\ue101")


def _isolate_marker_spans(
    messages: list[dict[str, Any]], markers: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replace every content string that contains a delimiter with a sentinel.

    Returns the substituted messages and the original spans, indexed by the
    number embedded in each sentinel. Content without a delimiter is left as
    is, so its tokenization is byte-for-byte unchanged.

    Templates that keep list content render adjacent text parts back to back,
    so a delimiter may only exist in their concatenation (``"<tool_"`` +
    ``"call>"``). When the joined text of a message's parts contains one,
    every non-empty text part of that message becomes its own span; spans are
    tokenized one at a time, so no token can straddle a part boundary.
    """
    spans: list[str] = []
    needles = markers + _CONTENT_SPAN_SENTINEL_CHARS

    def has_needle(text: str) -> bool:
        return any(n in text for n in needles)

    def substitute(text: str, force: bool = False) -> str:
        if not text or not (force or has_needle(text)):
            return text
        spans.append(text)
        return f"\ue100{len(spans) - 1}\ue101"

    def is_text_part(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )

    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            new_content: Any = substitute(content)
        elif isinstance(content, list):
            joined = "".join(item["text"] for item in content if is_text_part(item))
            force = has_needle(joined)
            new_content = []
            for item in content:
                if is_text_part(item):
                    item = {**item, "text": substitute(item["text"], force)}
                new_content.append(item)
        else:
            new_content = content
        out.append(
            {**msg, "content": new_content} if new_content is not content else msg
        )
    return out, spans


def _align_spans(
    rendered: str, skeleton: str, spans: list[str]
) -> list[tuple[str, bool]] | None:
    """Split *rendered* into ``(text, is_span)`` pieces using the *skeleton*.

    A span may appear with surrounding whitespace stripped (chat templates
    commonly ``strip()`` assistant content), so the stripped variants are
    accepted when the verbatim span does not match. Returns ``None`` when the
    two renderings cannot be reconciled, including a sentinel whose index is
    out of range, so the caller can fail closed.
    """
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for index, part in enumerate(_CONTENT_SPAN_SENTINEL_RE.split(skeleton)):
        if index % 2 == 0:
            if part:
                if not rendered.startswith(part, pos):
                    return None
                pieces.append((part, False))
                pos += len(part)
            continue
        span_index = int(part)
        if span_index >= len(spans):
            return None
        span = spans[span_index]
        for candidate in (span, span.strip(), span.rstrip(), span.lstrip()):
            if candidate and rendered.startswith(candidate, pos):
                pieces.append((candidate, True))
                pos += len(candidate)
                break
        else:
            return None
    return pieces if pos == len(rendered) else None


def encode_chat_prompt_isolating_markers(
    tokenizer: Any,
    template_messages: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
    tool_call_parser_name: str | None,
) -> list[int]:
    """Render and tokenize the chat prompt, keeping delimiters in content as text.

    Content spans that contain a tool-call delimiter are tokenized with the
    delimiter-free encoder; everything the template itself emits (the tool
    instructions, prior assistant ``tool_calls``) keeps the delimiter token
    ids. Prompts whose content has no delimiter tokenize exactly as
    ``apply_chat_template(tokenize=True)`` does.

    Fails closed: when the parser's delimiters are not single tokens for this
    tokenizer, when the spans cannot be aligned with the rendered prompt, or
    when the tokenizer cannot be copied, the request is rejected rather than
    encoded with the delimiter ids.
    """
    strings = tool_call_marker_strings(tool_call_parser_name, tokenizer)
    marker_ids = tool_call_marker_token_ids(tokenizer, tool_call_parser_name)
    if strings and not marker_ids:
        # The parser has delimiters but this tokenizer does not give them
        # single ids, so neither the output gate nor isolation can tell a
        # genuine delimiter from a spelled one. Reject rather than pretend.
        raise PreprocessError(
            "--isolate-tool-call-markers is enabled but the tool-call delimiters "
            f"{strings!r} of parser {tool_call_parser_name!r} are not single "
            "tokens for this tokenizer"
        )
    markers: tuple[str, ...] = strings if strings and marker_ids else ()
    substituted, spans = (
        _isolate_marker_spans(template_messages, markers) if markers else ([], [])
    )
    if not spans:
        return _normalize_prompt_token_ids(
            tokenizer.apply_chat_template(template_messages, **template_kwargs)
        )

    _reject_unsupported_tokenizer_controls(template_kwargs)
    text_kwargs = {**template_kwargs, "tokenize": False}
    rendered = tokenizer.apply_chat_template(template_messages, **text_kwargs)
    skeleton = tokenizer.apply_chat_template(substituted, **text_kwargs)
    pieces = _align_spans(rendered, skeleton, spans)
    if pieces is None:
        logger.warning(
            "Rejecting request: the chat template rewrote a content span that "
            "contains a tool-call delimiter, so it cannot be isolated"
        )
        raise PreprocessError(
            "Message content containing tool-call delimiters could not be encoded "
            "safely for this chat template."
        )
    encoder = _marker_free_encoder(tokenizer, markers, set(marker_ids))
    if encoder is None:
        logger.error(
            "Rejecting request: no delimiter-free tokenizer copy is available, so "
            "tool-call delimiters in message content cannot be isolated"
        )
        raise PreprocessError(
            "Message content containing tool-call delimiters is not supported by "
            "this tokenizer."
        )

    prompt_token_ids: list[int] = []
    for text, is_span in pieces:
        if is_span:
            prompt_token_ids.extend(encoder.encode(text, add_special_tokens=False).ids)
        else:
            prompt_token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return _truncate_prompt_token_ids(tokenizer, prompt_token_ids, template_kwargs)


# apply_chat_template(tokenize=True) forwards these to tokenizer.__call__; the
# isolated path renders with tokenize=False, so it must apply them itself.
_UNSUPPORTED_TOKENIZER_CONTROLS = ("padding", "tokenizer_kwargs", "return_tensors")


def _reject_unsupported_tokenizer_controls(template_kwargs: dict[str, Any]) -> None:
    """Fail closed on tokenizer controls that would change the assembled ids."""
    active = [
        key
        for key in _UNSUPPORTED_TOKENIZER_CONTROLS
        if template_kwargs.get(key) not in (None, False, "do_not_pad", {})
    ]
    if active:
        raise PreprocessError(
            f"chat_template_kwargs {active!r} are not supported for message content "
            "that contains tool-call delimiters while --isolate-tool-call-markers "
            "is enabled"
        )


def _truncate_prompt_token_ids(
    tokenizer: Any, prompt_token_ids: list[int], template_kwargs: dict[str, Any]
) -> list[int]:
    """Apply ``truncation``/``max_length`` as ``tokenizer.__call__`` would."""
    truncation = template_kwargs.get("truncation")
    if truncation not in (True, "longest_first", "only_first"):
        return prompt_token_ids
    limit = template_kwargs.get("max_length") or getattr(
        tokenizer, "model_max_length", None
    )
    if not limit or len(prompt_token_ids) <= limit:
        return prompt_token_ids
    if getattr(tokenizer, "truncation_side", "right") == "left":
        return prompt_token_ids[-limit:]
    return prompt_token_ids[:limit]

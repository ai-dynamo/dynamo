#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared utilities for frontend chat processors (vLLM, SGLang)."""

import json
import logging
import os
import uuid
from typing import Any, Literal

from dynamo.llm.exceptions import HttpError, InvalidArgument

_MASK_64_BITS = (1 << 64) - 1


ChatProcessorBackend = Literal["vllm", "sglang"]


def validate_legacy_guided_decoding_constraints(request: dict[str, Any]) -> None:
    """Reject malformed or multiple legacy guided-decoding constraints."""
    choice = request.get("guided_choice")
    # An empty list is a caller saying "no choices", which is simply no
    # constraint. Any other non-list is malformed: silently ignoring it would
    # generate unconstrained text for a caller who believes it constrained the
    # output. The Rust frontend types this field as Option<Vec<String>> and
    # rejects a scalar, so rejecting here keeps the two paths in step.
    if choice is not None and not isinstance(choice, list):
        raise InvalidArgument(
            "guided_choice must be a list of strings; received "
            f"{type(choice).__name__}"
        )
    constraints = (
        ("json", request.get("guided_json") is not None),
        ("regex", request.get("guided_regex") is not None),
        ("grammar", request.get("guided_grammar") is not None),
        ("choice", bool(choice)),
    )
    active_constraints = [name for name, is_set in constraints if is_set]
    if len(active_constraints) > 1:
        raise InvalidArgument(
            "Only one guided-decoding constraint can be set; received: "
            + ", ".join(active_constraints)
        )


def legacy_guided_decoding(request: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one legacy guided-decoding constraint and its modifier to a dict."""
    validate_legacy_guided_decoding_constraints(request)
    choice = request.get("guided_choice")
    for key, value in (
        ("json", request.get("guided_json")),
        ("regex", request.get("guided_regex")),
        ("grammar", request.get("guided_grammar")),
        ("choice", choice or None),
    ):
        if value is not None:
            guidance = {key: value}
            whitespace_pattern = request.get("guided_whitespace_pattern")
            if whitespace_pattern is not None:
                guidance["whitespace_pattern"] = whitespace_pattern
            return guidance
    return None


def read_jinja_chat_template(
    template_path: str,
    *,
    backend: ChatProcessorBackend,
) -> str:
    """Read a Jinja chat template using backend-specific file semantics."""
    with open(template_path, encoding="utf-8") as f:
        chat_template = f.read()
    if backend == "sglang":
        # Match SGLang TemplateManager._load_jinja_template() for SGLang
        # frontend preprocessing while leaving vLLM's plain file read intact.
        return chat_template.strip("\n").replace("\\n", "\n")
    return chat_template


def resolve_chat_template(
    source_path: str,
    *,
    backend: ChatProcessorBackend = "vllm",
) -> str | None:
    """Return a chat template stored beside the model, or None.

    Covers models (e.g. Qwen3-Omni) whose template lives in chat_template.json
    or chat_template.jinja rather than tokenizer_config.json, which the HF
    tokenizer does not merge. The backend selects native .jinja file semantics.
    """
    jinja_path = os.path.join(source_path, "chat_template.jinja")
    if os.path.exists(jinja_path):
        return read_jinja_chat_template(jinja_path, backend=backend)

    json_path = os.path.join(source_path, "chat_template.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            try:
                return json.load(f).get("chat_template")
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {json_path}: {e}") from e

    return None


def random_uuid() -> str:
    """Generate a random 16-character hex UUID."""
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"


def random_call_id() -> str:
    """Generate a random tool call ID in OpenAI format."""
    return f"call_{uuid.uuid4().int & _MASK_64_BITS:016x}"


def nvext_extra_field_requested(request: dict[str, Any], field: str) -> bool:
    """Return whether a request opted into a response nvext field."""
    nvext = request.get("nvext")
    if not isinstance(nvext, dict):
        return False
    extra_fields = nvext.get("extra_fields")
    return isinstance(extra_fields, list) and field in extra_fields


def worker_warmup() -> bool:
    """Dummy task to ensure a ProcessPoolExecutor worker is fully initialized."""
    return True


class PreprocessError(Exception):
    """Raised by preprocess workers for user-facing errors (e.g., n!=1).

    Carries a plain message because the worker→main-process boundary
    pickles the exception; the main process re-raises a Dynamo-typed
    exception so PyO3 can route it through the proper backend-error path.
    """

    def __init__(self, message: str):
        super().__init__(message)


# Content part types that carry media URLs, mapped to the key used in the
# multimodal data dict sent to the backend handler.
_MEDIA_CONTENT_TYPES = ("image_url", "audio_url", "video_url")


def extract_mm_urls(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]] | None, dict[str, list[str | None]] | None,]:
    """Extract media and vLLM image processor-cache UUIDs from chat messages.

    URL-backed parts become ``Url`` variants. Image parts with no URL and an
    opaque ``uuid`` become ``UuidOnly`` variants for vLLM's multimodal
    processor cache. Cache UUIDs on audio and video are rejected. Image UUID
    lists preserve slot order::

        ({"image_url": [{"Url": "https://..."}, {"UuidOnly": "image-1"}]},
         {"image_url": ["image-1", "image-1"]})

    The UUID map is ``None`` when no user UUID is present. A media content part
    with neither a URL nor UUID is rejected instead of being silently dropped.
    """
    mm_data: dict[str, list[dict[str, str]]] = {}
    mm_uuids: dict[str, list[str | None]] = {}
    has_user_uuid = False

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in _MEDIA_CONTENT_TYPES:
                continue

            media_value = part.get(part_type)
            uuid_value = part.get("uuid")
            if uuid_value is not None and (
                not isinstance(uuid_value, str) or not uuid_value
            ):
                raise ValueError(f"{part_type} uuid must be a non-empty string")
            if uuid_value is not None and part_type != "image_url":
                raise ValueError(
                    "multimodal cache UUIDs are supported only for "
                    "image_url parts with vLLM"
                )

            url = media_value.get("url") if isinstance(media_value, dict) else None
            if isinstance(url, str) and url:
                mm_data.setdefault(part_type, []).append({"Url": url})
            elif isinstance(uuid_value, str):
                mm_data.setdefault(part_type, []).append({"UuidOnly": uuid_value})
            else:
                raise ValueError(
                    f"{part_type} part must contain a non-empty URL or uuid"
                )

            if part_type == "image_url":
                mm_uuids.setdefault(part_type, []).append(uuid_value)
                has_user_uuid |= uuid_value is not None

    return mm_data or None, mm_uuids if has_user_uuid else None


def make_backend_error(engine_response: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-style error dict, guarding against None/missing message."""
    backend_msg = engine_response.get("message") or "unknown backend error"
    return {
        "error": {
            "message": backend_msg,
            "type": "backend_error",
        }
    }


def make_internal_error(request_id: str, detail: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style internal error dict with request-specific fallback."""
    message = detail or f"Invalid engine response for request {request_id}"
    return {
        "error": {
            "message": message,
            "type": "internal_error",
        }
    }


def as_error_envelope(error_payload: dict[str, Any]) -> dict[str, Any]:
    """Tag an error dict so the binding reads it as an error frame.

    Without ``_dynamo_annotated`` the binding reads the dict as a completion
    chunk and fails with ``missing field `id```. The client then gets a 500 and
    never sees the message. The HTTP layer reads the text back off ``comment``.
    See ``depythonize_annotated`` in ``lib/bindings/python/rust/engine.rs``.
    """
    message = (error_payload.get("error") or {}).get("message") or "unknown error"
    return {"_dynamo_annotated": True, "event": "error", "comment": [message]}


_SERIALIZED_BACKEND_INVALID_ARGUMENT_PREFIX = "BackendInvalidArgument: "


def backend_invalid_argument_to_http_error(exc: BaseException) -> HttpError | None:
    """Recover a worker's own HTTP status from a serialized backend error.

    A worker that rejects a request -- an unsupported sampling parameter, an
    unparseable grammar -- fails it with ``{"message": ..., "code": 4xx}``.
    Crossing the Rust boundary turns that into a plain Python exception whose
    text is ``BackendInvalidArgument: {json}``, so a generic ``except
    Exception`` reports a 500 and discards both the status and the reason.

    Rust's HTTP service already performs this recovery
    (``extract_backend_error_if_present`` in
    ``lib/llm/src/http/service/openai.rs``), but the Python chat processors do
    not go through it, so they need their own. Mirror its status rules: honour
    an explicit code, and otherwise fall back to 400 -- the prefix itself is
    the discriminator proving the argument was invalid.

    Returns None when ``exc`` is not that shape, so callers fall through to
    their existing handling instead of inventing a status.
    """
    # Not ``removeprefix``: adapter paths wrap the error in an outer
    # ``<ErrorType>: `` before it reaches Python, so the discriminator is not
    # always at position 0.
    _, prefix, payload = str(exc).partition(_SERIALIZED_BACKEND_INVALID_ARGUMENT_PREFIX)
    if not prefix:
        return None

    try:
        parsed = json.loads(payload)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        # Prefix but no payload: still an invalid argument, just unstructured.
        return HttpError(400, payload)

    code = parsed.get("code")
    if isinstance(code, bool) or not isinstance(code, int):
        code = 400
    elif not 100 <= code < 600:
        # Rust degrades an out-of-range code to a 500, which is exactly what
        # the caller's existing handler already produces -- defer to it.
        return None

    message = parsed.get("message")
    return HttpError(code, message if isinstance(message, str) else payload)


def handle_engine_error(
    engine_response: Any,
    request_id: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Classify an invalid engine response and return an OpenAI-style error dict.

    Called when engine_response is None or missing 'token_ids'.
    """
    if isinstance(engine_response, dict) and engine_response.get("status") == "error":
        err = make_backend_error(engine_response)
        logger.error(
            "Backend error for request %s: %s", request_id, err["error"]["message"]
        )
        return err
    logger.error(
        "No outputs from engine for request %s: %s", request_id, engine_response
    )
    return make_internal_error(request_id)

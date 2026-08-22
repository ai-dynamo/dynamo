# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch SGLang streaming responses into Dynamo's Rust response senders."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


class _ContextWithRequestId:
    """Use Dynamo's request ID as the SGLang RID when no trace ID exists."""

    __slots__ = ("_context", "trace_id")

    def __init__(self, context: Any, rid: str) -> None:
        self._context = context
        self.trace_id = rid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)


@dataclass(slots=True)
class _ResponseEntry:
    rid: str
    handler: Any
    context: Any
    response_sender: Any
    mode: Literal["token", "text", "native"]
    return_tokens_as_token_ids: bool = False
    metadata_upload_enabled: bool = False
    text_counts_per_choice: dict[int, int] = field(default_factory=dict)


def _convert_token(entry: _ResponseEntry, res: Mapping[str, Any]) -> dict | None:
    meta_info = res.get("meta_info") or {}
    if meta_info.get("finish_reason"):
        raise ValueError("response_handler received a final token response")

    output_ids = list(res.get("output_ids") or [])
    if not output_ids or entry.context.is_stopped():
        return None

    out: dict[str, Any] = {
        "index": res.get("index") or 0,
        "token_ids": output_ids,
    }
    if not entry.metadata_upload_enabled:
        log_probs, top_logprobs = entry.handler._extract_logprobs(
            meta_info,
            num_output_tokens_in_chunk=len(output_ids),
            return_tokens_as_token_ids=entry.return_tokens_as_token_ids,
        )
        if log_probs is not None:
            out["log_probs"] = log_probs
        if top_logprobs is not None:
            out["top_logprobs"] = top_logprobs

    engine_data = dict(res.get("engine_data") or {})
    routed_experts = meta_info.get("routed_experts")
    if routed_experts is not None and not entry.metadata_upload_enabled:
        engine_data["routed_experts"] = routed_experts
    if engine_data:
        out["engine_data"] = engine_data
    return out


def _convert_text(entry: _ResponseEntry, res: Mapping[str, Any]) -> dict | None:
    meta_info = res.get("meta_info") or {}
    if meta_info.get("finish_reason"):
        raise ValueError("response_handler received a final text response")
    if entry.context.is_stopped():
        return None

    index = res.get("index") or 0
    text = res.get("text") or ""
    incremental = bool(
        getattr(
            entry.handler.config.server_args,
            "incremental_streaming_output",
            False,
        )
    )
    if incremental:
        delta = text
    else:
        count = entry.text_counts_per_choice.get(index, 0)
        delta = text[count:]
        entry.text_counts_per_choice[index] = len(text)

    choice_data = {
        "index": index,
        "delta": {"role": "assistant", "content": delta},
        "finish_reason": None,
    }
    response: dict[str, Any] = {
        "id": meta_info["id"],
        "created": int(time.time()),
        "choices": [choice_data],
        "model": entry.handler.config.server_args.served_model_name,
        "object": "chat.completion.chunk",
    }
    response_nvext: dict[str, Any] = {}
    routed_experts = meta_info.get("routed_experts")
    if routed_experts is not None and not entry.metadata_upload_enabled:
        response_nvext["routed_experts"] = routed_experts
    if response_nvext:
        response["nvext"] = response_nvext
    return response


def _convert_native(entry: _ResponseEntry, res: Mapping[str, Any]) -> dict | None:
    if entry.context.is_stopped():
        return None
    return {"token_ids": [], "engine_data": {"sglang_response": dict(res)}}


def _convert(entry: _ResponseEntry, res: Mapping[str, Any]) -> dict | None:
    if entry.mode == "token":
        return _convert_token(entry, res)
    if entry.mode == "text":
        return _convert_text(entry, res)
    return _convert_native(entry, res)


class DynamoResponseHandler:
    """Own all intermediate streaming outputs from one SGLang Engine."""

    def __init__(self) -> None:
        self._entries: dict[str, _ResponseEntry] = {}

    def add(self, entry: _ResponseEntry) -> None:
        if entry.rid in self._entries:
            raise ValueError(f"Duplicate SGLang request ID: {entry.rid}")
        self._entries[entry.rid] = entry

    def get(self, rid: str) -> _ResponseEntry | None:
        return self._entries.get(rid)

    def remove(self, rid: str, entry: _ResponseEntry) -> None:
        if self._entries.get(rid) is entry:
            del self._entries[rid]

    def handle_batch(self, outputs: Sequence[Any]) -> None:
        candidates: list[tuple[_ResponseEntry, dict[str, Any]]] = []
        for item in outputs:
            entry = self._entries.get(item.rid)
            if entry is None:
                # Internal SGLang warmup and probe requests do not have a
                # Dynamo response stream. Their terminal iterators still run.
                continue
            try:
                response = _convert(entry, item.output)
            except Exception as error:
                logger.exception(
                    "SGLang response conversion failed for rid=%s", item.rid
                )
                self.remove(item.rid, entry)
                entry.response_sender.close_with_error(str(error))
                entry.context.stop_generating()
                continue
            if response is not None:
                candidates.append((entry, response))

        if not candidates:
            return

        sender_type = type(candidates[0][0].response_sender)
        failed = set(
            sender_type.send_batch(
                [(entry.response_sender, response) for entry, response in candidates]
            )
        )
        for index in failed:
            entry, _ = candidates[index]
            self.remove(entry.rid, entry)
            entry.context.stop_generating()


def create_response_handler() -> DynamoResponseHandler | None:
    """Return a handler only when the installed SGLang supports the API."""

    try:
        from sglang.srt.entrypoints.engine import ResponseHandlerOutput  # noqa: F401
    except ImportError:
        return None
    return DynamoResponseHandler()


def get_response_handler(engine: Any) -> DynamoResponseHandler | None:
    response_handler = getattr(engine, "response_handler", None)
    return (
        response_handler
        if isinstance(response_handler, DynamoResponseHandler)
        else None
    )


def get_response_entry(engine: Any, context: Any) -> _ResponseEntry | None:
    response_handler = get_response_handler(engine)
    if response_handler is None:
        return None
    rid = str(context.trace_id or context.id())
    return response_handler.get(rid)


async def _drive_response_handler_stream(
    stream: AsyncIterator[Any],
    response_handler: DynamoResponseHandler,
    entry: _ResponseEntry,
) -> AsyncGenerator[Any, None]:
    """Send terminal outputs and keep request cleanup on the iterator path."""

    try:
        async for response in stream:
            entry.response_sender.send(response)
        entry.response_sender.close()
    finally:
        response_handler.remove(entry.rid, entry)
    if False:  # pragma: no cover - required async-generator shape for Rust
        yield


def response_handler_capable(func: Any) -> Any:
    """Use the engine response handler when SGLang supports it."""

    from dynamo.sglang.engine_generate import native_generate_payload
    from dynamo.trtllm.request_handlers.push_egress import drive_push_egress_stream

    @functools.wraps(func)
    def dispatch(
        self: Any,
        request: dict[str, Any],
        context: Any = None,
        response_sender: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        if response_sender is None:
            return func(self, request, context, **kwargs)

        response_handler = get_response_handler(self.engine)
        if response_handler is None:
            return drive_push_egress_stream(
                func(self, request, context, **kwargs), response_sender
            )

        native_payload = native_generate_payload(request)
        native_rid = native_payload.get("rid") if native_payload is not None else None
        rid = str(native_rid or context.trace_id or context.id())
        effective_context = context
        if context.trace_id != rid:
            effective_context = _ContextWithRequestId(context, rid)

        mode: Literal["token", "text", "native"]
        if native_payload is not None:
            mode = "native"
        elif self.use_sglang_tokenizer:
            mode = "text"
        else:
            mode = "token"

        output_options = request.get("output_options") or {}
        metadata_upload_enabled = (
            self._metadata_uploader_from_request(request) is not None
        )
        entry = _ResponseEntry(
            rid=rid,
            handler=self,
            context=effective_context,
            response_sender=response_sender,
            mode=mode,
            return_tokens_as_token_ids=bool(
                output_options.get("return_tokens_as_token_ids")
            ),
            metadata_upload_enabled=metadata_upload_enabled,
        )
        response_handler.add(entry)
        stream = func(self, request, effective_context, **kwargs)
        return _drive_response_handler_stream(stream, response_handler, entry)

    del dispatch.__wrapped__
    return dispatch

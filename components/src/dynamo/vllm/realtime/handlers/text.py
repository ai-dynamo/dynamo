# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage text-only Realtime sessions, prefix warming, and response streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from dynamo._core import Context

from ..connection import RealtimeConnection, RealtimeTurn
from ..events import (
    conversation_item_added_event,
    conversation_item_done_event,
    invalid_request_error_event,
    response_content_part_event,
    response_created_event,
    response_done_event,
    response_output_item_added_event,
    response_output_item_done_event,
    response_output_text_event,
    session_updated_event,
)
from ..factories import (
    ChatCompletionFactory,
    TextPrefillFactory,
    build_realtime_text_factories,
)
from .text_utils import (
    _max_output_tokens,
    _normalize_text_item,
    _realtime_usage,
    _text_prompt,
    _validate_text_options,
)

logger = logging.getLogger(__name__)

# Bound uncommitted text independently of the engine's token/context limit.
MAX_TEXT_BUFFER_BYTES = 4 * 1024 * 1024


async def _emit_events(turn: RealtimeTurn, *events: dict[str, Any]) -> None:
    for event in events:
        await turn.events.put(event)


class _TextTurn(RealtimeTurn):
    def __init__(
        self,
        *,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
        wire_max_output_tokens: int | str,
        add_to_conversation: bool,
        items: list[dict[str, Any]],
        metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.messages = messages
        self.max_output_tokens = max_output_tokens
        self.wire_max_output_tokens = wire_max_output_tokens
        self.add_to_conversation = add_to_conversation
        self.items = items
        self.metadata = metadata
        self.previous_item_id = items[-1]["id"] if items else None
        self.text = ""
        self.finished = False

        # Announce the turn even if generation is cancelled before its task starts.
        pending_item = self.item("in_progress")
        started = [
            response_created_event(
                self.response_id,
                output_modalities=["text"],
                max_output_tokens=self.wire_max_output_tokens,
                metadata=self.metadata,
            ),
            response_output_item_added_event(self.response_id, pending_item),
        ]
        if self.add_to_conversation:
            started.append(
                conversation_item_added_event(pending_item, self.previous_item_id)
            )
        started.append(
            response_content_part_event(
                "response.content_part.added", self.response_id, self.item_id, ""
            )
        )
        for event in started:
            self.events.put_nowait(event)

    def item(self, status: str) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "object": "realtime.item",
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": (
                [{"type": "output_text", "text": self.text}] if self.text else []
            ),
        }

    def final_events(
        self,
        *,
        status: str,
        status_details: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Close the response item and update server-side conversation state."""
        if self.finished:
            return []
        self.finished = True
        item = self.item("completed" if status == "completed" else "incomplete")
        events = [
            response_output_text_event(
                "response.output_text.done", self.response_id, self.item_id, self.text
            ),
            response_content_part_event(
                "response.content_part.done", self.response_id, self.item_id, self.text
            ),
            response_output_item_done_event(self.response_id, item),
        ]
        if self.add_to_conversation:
            self.items.append(item)
            events.append(conversation_item_done_event(item, self.previous_item_id))
        events.append(
            response_done_event(
                self.response_id,
                output_modalities=["text"],
                max_output_tokens=self.wire_max_output_tokens,
                output=[item],
                status=status,
                status_details=status_details,
                usage=_realtime_usage(usage),
                metadata=self.metadata,
            )
        )
        return events


class _TextPrefill:
    """Coalesce incremental text for one best-effort prefill request."""

    def __init__(
        self,
        *,
        messages: list[dict[str, str]],
        factory: TextPrefillFactory,
    ) -> None:
        self.text = ""
        self._text_bytes = 0
        self._cancel_requested = False
        self._updated = asyncio.Event()
        self.task: asyncio.Task[None] = asyncio.create_task(
            factory(messages, self.updates())
        )

    async def updates(self) -> AsyncGenerator[str, None]:
        while True:
            await self._updated.wait()
            self._updated.clear()
            yield self.text

    def append(self, text: str) -> None:
        text_bytes = self._text_bytes + len(text.encode("utf-8"))
        if text_bytes > MAX_TEXT_BUFFER_BYTES:
            raise ValueError(f"input text exceeds {MAX_TEXT_BUFFER_BYTES} bytes")
        self.text += text
        self._text_bytes = text_bytes
        if not self.task.done():
            self._updated.set()

    def commit(self) -> None:
        # Final generation reuses completed prefix blocks. Finishing another
        # warming generation here would put speculative work on its critical path.
        if not self._cancel_requested:
            self._cancel_requested = True
            self.task.cancel()

    async def cancel(self) -> None:
        self.commit()
        # Session cancellation must not interrupt the engine's abort cleanup.
        await asyncio.shield(asyncio.gather(self.task, return_exceptions=True))


class RealtimeTextHandler:
    """Serve text conversations and incremental text prefill with vLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        chat_completion_factory: ChatCompletionFactory,
        text_prefill_factory: TextPrefillFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self._chat_completion_factory = chat_completion_factory
        self._text_prefill_factory = text_prefill_factory

    @classmethod
    def from_engine(
        cls,
        *,
        engine_client: Any,
        model_name: str,
        model_path: str,
        chat_template_path: str | None,
    ) -> "RealtimeTextHandler":
        chat_completion, text_prefill = build_realtime_text_factories(
            engine_client=engine_client,
            model_name=model_name,
            model_path=model_path,
            chat_template_path=chat_template_path,
        )
        return cls(
            model_name=model_name,
            chat_completion_factory=chat_completion,
            text_prefill_factory=text_prefill,
        )

    def _validate_session(self, session: Any) -> str | None:
        if not isinstance(session, dict) or session.get("type") != "realtime":
            return "session.type must be 'realtime'"
        if session.get("model") not in (None, self.model_name):
            return "session model mismatch"
        if session.get("audio") is not None:
            return "audio input and output are not supported by this worker"
        if session.get("truncation") not in (None, "disabled"):
            return "automatic truncation is not supported; use truncation='disabled'"
        try:
            _validate_text_options(session, "session")
            _max_output_tokens(session.get("max_output_tokens"))
        except ValueError as exc:
            return str(exc)
        return None

    def _response_options(
        self, value: Any, session: dict[str, Any]
    ) -> tuple[int | None, int | str, str, bool, bool, dict[str, str] | None]:
        response = {} if value is None else value
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        _validate_text_options(response, "response")
        if response.get("input") not in (None, []):
            raise ValueError("response.input items are not supported")
        if response.get("conversation") not in (None, "auto", "none"):
            raise ValueError("response.conversation must be 'auto' or 'none'")
        metadata = response.get("metadata")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            )
        ):
            raise ValueError("response.metadata must be an object with string values")

        instructions = response.get("instructions", session["instructions"])
        max_tokens = response.get("max_output_tokens", session["max_output_tokens"])
        max_output_tokens, wire_max_output_tokens = _max_output_tokens(max_tokens)
        return (
            max_output_tokens,
            wire_max_output_tokens,
            instructions,
            response.get("conversation") != "none",
            response.get("input") != [],
            metadata,
        )

    async def _run_turn(self, turn: _TextTurn, context: Context) -> None:
        usage = None
        finish_reason = None
        stream = None
        generation_active = True

        def cancel_generation(_: asyncio.Future[bool]) -> None:
            if generation_active:
                turn.cancel()

        # Interrupt pending factory/frame awaits on disconnect, not on input EOF.
        stopped = context.async_killed_or_stopped()
        stopped.add_done_callback(cancel_generation)
        try:
            stream = await self._chat_completion_factory(
                turn.messages, turn.max_output_tokens
            )
            async for frame in stream:
                if context.is_stopped():
                    return
                for line in frame.splitlines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        continue
                    payload = json.loads(data)
                    if "error" in payload:
                        raise RuntimeError(
                            payload["error"].get("message", "Chat generation failed")
                        )
                    usage = payload.get("usage") or usage
                    for choice in payload.get("choices", []):
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            if not isinstance(delta, str):
                                raise ValueError(
                                    "chat completion returned non-text content"
                                )
                            turn.text += delta
                            await turn.events.put(
                                response_output_text_event(
                                    "response.output_text.delta",
                                    turn.response_id,
                                    turn.item_id,
                                    delta,
                                )
                            )

            incomplete = finish_reason == "length"
            status = "incomplete" if incomplete else "completed"
            await _emit_events(
                turn,
                *turn.final_events(
                    status=status,
                    status_details=(
                        {"type": "incomplete", "reason": "max_output_tokens"}
                        if incomplete
                        else None
                    ),
                    usage=usage,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate engine failures per response
            logger.exception("realtime text generation failed: %s", exc)
            await _emit_events(
                turn,
                *turn.final_events(
                    status="failed",
                    status_details={
                        "type": "failed",
                        "error": {
                            "type": "server_error",
                            "code": "generation_error",
                        },
                    },
                ),
            )
        finally:
            # A queued stop callback must not cancel the awaited cleanup again.
            generation_active = False
            stopped.remove_done_callback(cancel_generation)
            stopped.cancel()
            # Cancellation may interrupt an output queue put, not stream.__anext__.
            if stream is not None:
                await stream.aclose()

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.model_name,
            "instructions": "",
            "max_output_tokens": "inf",
            "output_modalities": ["text"],
            "truncation": "disabled",
        }
        items: list[dict[str, Any]] = []
        connection = RealtimeConnection[_TextTurn](
            context=context, run_turn=self._run_turn, max_concurrent_turns=1
        )
        active_response: _TextTurn | None = None
        active_prefill: _TextPrefill | None = None
        committed_prefill: _TextPrefill | None = None

        def append_item(item: dict[str, Any]) -> None:
            previous_item_id = items[-1]["id"] if items else None
            items.append(item)
            connection.emit(conversation_item_added_event(item, previous_item_id))
            connection.emit(conversation_item_done_event(item, previous_item_id))

        def emit_error(event: dict[str, Any], code: str, message: str) -> None:
            connection.emit(
                invalid_request_error_event(
                    code, message, client_event_id=event.get("event_id")
                )
            )

        def current_response() -> _TextTurn | None:
            nonlocal active_response
            if active_response is not None and (
                active_response.finished
                or (active_response.task is not None and active_response.task.done())
            ):
                active_response = None
            return active_response

        async def handle_event(
            event: Any, connection: RealtimeConnection[_TextTurn]
        ) -> None:
            nonlocal active_prefill, active_response, committed_prefill, session
            if not isinstance(event, dict):
                connection.emit(
                    invalid_request_error_event(
                        "invalid_event", "event must be an object"
                    )
                )
                return
            event_type = event.get("type")
            running = current_response()

            if event_type == "session.update":
                if active_prefill is not None:
                    emit_error(
                        event,
                        "input_buffer_active",
                        "session cannot change while the text buffer is active",
                    )
                    return
                update = event.get("session")
                if not isinstance(update, dict):
                    emit_error(event, "invalid_session", "session must be an object")
                    return
                candidate = {**session, **update}
                if error := self._validate_session(candidate):
                    emit_error(event, "invalid_session", error)
                    return
                session = candidate
                connection.emit(session_updated_event(session))
            elif event_type == "conversation.item.create":
                if running is not None or active_prefill is not None:
                    emit_error(
                        event,
                        "response_in_progress",
                        "conversation cannot change while input or response is active",
                    )
                    return
                if committed_prefill is not None:
                    emit_error(
                        event,
                        "response_required",
                        "create a response before starting another input turn",
                    )
                    return
                if event.get("previous_item_id") is not None:
                    emit_error(
                        event,
                        "unsupported_item_position",
                        "only appending conversation items is supported",
                    )
                    return
                try:
                    item = _normalize_text_item(event.get("item"))
                    if any(existing["id"] == item["id"] for existing in items):
                        raise ValueError(
                            f"conversation item {item['id']!r} already exists"
                        )
                except ValueError as exc:
                    emit_error(event, "invalid_item", str(exc))
                    return
                append_item(item)
            elif event_type == "input_text.append":
                if running is not None:
                    emit_error(
                        event,
                        "response_in_progress",
                        "text cannot be appended while a response is running",
                    )
                    return
                if committed_prefill is not None:
                    emit_error(
                        event,
                        "response_required",
                        "create a response before starting another input turn",
                    )
                    return
                text = event.get("text")
                if not isinstance(text, str) or not text:
                    emit_error(event, "invalid_text", "text must be a non-empty string")
                    return
                if self._text_prefill_factory is None:
                    emit_error(
                        event,
                        "unsupported_event",
                        "incremental text input is unavailable for this worker",
                    )
                    return
                if active_prefill is None:
                    active_prefill = _TextPrefill(
                        messages=_text_prompt(items, session["instructions"]),
                        factory=self._text_prefill_factory,
                    )
                try:
                    active_prefill.append(text)
                except ValueError as exc:
                    # A rejected first append must not activate the input buffer.
                    if not active_prefill.text:
                        await active_prefill.cancel()
                        active_prefill = None
                    emit_error(event, "invalid_text", str(exc))
            elif event_type == "input_text.commit":
                if active_prefill is None or not active_prefill.text:
                    emit_error(event, "invalid_text", "input text buffer is empty")
                    return
                active_prefill.commit()
                committed_prefill = active_prefill
                text = active_prefill.text
                active_prefill = None
                item = _normalize_text_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
                append_item(item)
            elif event_type == "input_text.clear":
                if active_prefill is not None:
                    await active_prefill.cancel()
                    active_prefill = None
            elif event_type == "response.create":
                if running is not None:
                    emit_error(
                        event, "response_in_progress", "a response is already running"
                    )
                    return
                if active_prefill is not None:
                    emit_error(
                        event,
                        "input_not_committed",
                        "commit the text buffer before creating a response",
                    )
                    return
                try:
                    (
                        max_output_tokens,
                        wire_max_output_tokens,
                        instructions,
                        add_to_conversation,
                        use_conversation,
                        metadata,
                    ) = self._response_options(event.get("response"), session)
                    prompt = _text_prompt(
                        items if use_conversation else [], instructions
                    )
                    if not prompt:
                        raise ValueError(
                            "response requires conversation input or instructions"
                        )
                except ValueError as exc:
                    emit_error(event, "invalid_response", str(exc))
                    return
                if committed_prefill is not None:
                    await committed_prefill.cancel()
                active_response = await connection.ensure_turn(
                    lambda: _TextTurn(
                        messages=prompt,
                        max_output_tokens=max_output_tokens,
                        wire_max_output_tokens=wire_max_output_tokens,
                        add_to_conversation=add_to_conversation,
                        items=items,
                        metadata=metadata,
                    )
                )
                committed_prefill = None
                connection.finish_active_turn()
            elif event_type == "response.cancel":
                if running is None:
                    emit_error(
                        event,
                        "no_active_response",
                        "there is no active response to cancel",
                    )
                    return
                if event.get("response_id") not in (None, running.response_id):
                    emit_error(
                        event,
                        "response_not_found",
                        f"active response is {running.response_id!r}",
                    )
                    return
                active_response = None
                # Preserve response.created and any text deltas already queued;
                # Realtime clients must receive terminal item/content events even
                # when generation is cancelled.
                await connection.cancel_turn_preserving_output(running)
                for terminal_event in running.final_events(
                    status="cancelled",
                    status_details={
                        "type": "cancelled",
                        "reason": "client_cancelled",
                    },
                ):
                    connection.emit(terminal_event)
            else:
                emit_error(
                    event,
                    "unsupported_event",
                    f"unsupported event type: {event_type}",
                )

        try:
            async for event in connection.generate(
                request_stream,
                handle_event=handle_event,
                close_active_turn=lambda turn: None,
            ):
                yield event
        finally:
            if active_prefill is not None:
                await active_prefill.cancel()
            if committed_prefill is not None:
                await committed_prefill.cancel()

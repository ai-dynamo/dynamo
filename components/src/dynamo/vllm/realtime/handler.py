# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime text and transcription sessions for standard vLLM models."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any, Protocol

import numpy as np

from dynamo._core import Context

from .connection import RealtimeConnection, RealtimeTurn
from .events import (
    conversation_item_added_event,
    conversation_item_done_event,
    input_audio_buffer_cleared_event,
    input_audio_buffer_committed_event,
    input_audio_transcription_completed_event,
    input_audio_transcription_delta_event,
    input_audio_transcription_failed_event,
    invalid_request_error_event,
    response_content_part_event,
    response_created_event,
    response_done_event,
    response_output_item_added_event,
    response_output_item_done_event,
    response_output_text_event,
    session_updated_event,
)
from .serving import (
    ChatCompletionFactory,
    StreamingInputFactory,
    TextPrefillFactory,
    build_realtime_serving,
    build_realtime_text_factories,
)

logger = logging.getLogger(__name__)

OPENAI_PCM_SAMPLE_RATE = 24_000
MAX_AUDIO_CHUNK_BYTES = 4 * 1024 * 1024
RESAMPLE_BLOCK_MILLISECONDS = 100
MAX_UTTERANCE_SECONDS = 60

SamplingParamsFactory = Callable[[], Any]


class RealtimeSessionHandler(Protocol):
    def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        ...


class RealtimeHandler:
    """Select one session handler from the initial ``session.update`` event."""

    def __init__(self, handlers: Mapping[str, RealtimeSessionHandler]) -> None:
        self._handlers = dict(handlers)

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        try:
            first_event = await anext(request_stream)
        except StopAsyncIteration:
            return

        if (
            not isinstance(first_event, dict)
            or first_event.get("type") != "session.update"
        ):
            yield invalid_request_error_event(
                "invalid_event",
                "first event must be session.update",
                client_event_id=(
                    first_event.get("event_id")
                    if isinstance(first_event, dict)
                    else None
                ),
            )
            return

        session = first_event.get("session")
        session_type = session.get("type") if isinstance(session, dict) else None
        handler = (
            self._handlers.get(session_type) if isinstance(session_type, str) else None
        )
        if handler is None:
            yield invalid_request_error_event(
                "unsupported_session",
                f"unsupported session type: {session_type!r}",
                client_event_id=first_event.get("event_id"),
            )
            return

        async def replay() -> AsyncGenerator[Any, None]:
            yield first_event
            async for event in request_stream:
                yield event

        async for event in handler.generate(replay(), context):
            yield event


def _max_output_tokens(value: Any) -> tuple[int | None, int | str]:
    if value in (None, "inf"):
        return None, "inf"
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ValueError("max_output_tokens must be an integer from 1 to 4096 or 'inf'")
    return value, value


def _normalize_text_item(value: Any) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(value, dict) or value.get("type") != "message":
        raise ValueError("only message conversation items are supported")
    role = value.get("role")
    if role not in {"system", "user", "assistant"}:
        raise ValueError("message role must be system, user, or assistant")

    content_type = "output_text" if role == "assistant" else "input_text"
    content = value.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be a non-empty array")
    if any(
        not isinstance(part, dict)
        or part.get("type") != content_type
        or not isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError(f"{role} message content must contain only {content_type}")

    item = {
        "id": value.get("id") or f"item_{uuid.uuid4().hex}",
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": role,
        "content": [{"type": content_type, "text": part["text"]} for part in content],
    }
    if not isinstance(item["id"], str):
        raise ValueError("conversation item id must be a string")
    return item, {
        "role": role,
        "content": "".join(part["text"] for part in content),
    }


def _realtime_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if usage is None:
        return None
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


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
        conversation_messages: list[dict[str, str]],
        prefill_task: asyncio.Task[None] | None = None,
    ) -> None:
        super().__init__()
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.messages = messages
        self.max_output_tokens = max_output_tokens
        self.wire_max_output_tokens = wire_max_output_tokens
        self.add_to_conversation = add_to_conversation
        self.items = items
        self.conversation_messages = conversation_messages
        self.prefill_task = prefill_task
        self.previous_item_id = items[-1]["id"] if items else None
        self.text = ""
        self.finished = False

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


class _TextPrefill:
    """Queue incremental text for one best-effort prefill request."""

    def __init__(
        self,
        *,
        messages: list[dict[str, str]],
        factory: TextPrefillFactory,
    ) -> None:
        self.messages = messages
        self._parts: list[str] = []
        self._updates: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        self.task: asyncio.Task[None] = asyncio.create_task(
            factory(messages, self.updates())
        )

    @property
    def text(self) -> str:
        return "".join(self._parts)

    async def updates(self) -> AsyncGenerator[tuple[str, bool], None]:
        while (update := await self._updates.get()) is not None:
            yield update

    def append(self, text: str) -> None:
        self._parts.append(text)
        if not self.task.done():
            self._updates.put_nowait((text, False))

    def commit(self) -> None:
        if not self.task.done():
            self._updates.put_nowait(("", True))
            self._updates.put_nowait(None)

    async def cancel(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


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
        checks = (
            (session.get("model") in (None, self.model_name), "session model mismatch"),
            (
                session.get("output_modalities") in (None, ["text"]),
                "only text output is supported",
            ),
            (
                session.get("audio") is None,
                "audio input and output are not supported by this worker",
            ),
            (session.get("tools") in (None, []), "tools are not supported"),
            (
                session.get("tool_choice") in (None, "none"),
                "tool_choice is not supported",
            ),
        )
        for supported, message in checks:
            if not supported:
                return message
        if any(session.get(field) is not None for field in ("prompt", "reasoning")):
            return "prompt and reasoning configuration are not supported"
        if not isinstance(session.get("instructions", ""), str):
            return "session.instructions must be a string"
        try:
            _max_output_tokens(session.get("max_output_tokens"))
        except ValueError as exc:
            return str(exc)
        return None

    def _response_options(
        self, value: Any, session: dict[str, Any]
    ) -> tuple[int | None, int | str, str, bool, bool]:
        response = {} if value is None else value
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        if response.get("output_modalities") not in (None, ["text"]):
            raise ValueError("only text output is supported")
        if response.get("tools") not in (None, []):
            raise ValueError("tools are not supported")
        if response.get("input") not in (None, []):
            raise ValueError("response.input items are not supported")
        if response.get("conversation") not in (None, "auto", "none"):
            raise ValueError("response.conversation must be 'auto' or 'none'")

        instructions = response.get("instructions", session["instructions"])
        if not isinstance(instructions, str):
            raise ValueError("response.instructions must be a string")
        max_tokens = response.get("max_output_tokens", session["max_output_tokens"])
        max_output_tokens, wire_max_output_tokens = _max_output_tokens(max_tokens)
        return (
            max_output_tokens,
            wire_max_output_tokens,
            instructions,
            response.get("conversation") != "none",
            response.get("input") != [],
        )

    async def _run_turn(self, turn: _TextTurn, context: Context) -> None:
        pending_item = turn.item("in_progress")
        started = [
            response_created_event(
                turn.response_id,
                output_modalities=["text"],
                max_output_tokens=turn.wire_max_output_tokens,
            ),
            response_output_item_added_event(turn.response_id, pending_item),
        ]
        if turn.add_to_conversation:
            started.append(
                conversation_item_added_event(pending_item, turn.previous_item_id)
            )
        started.append(
            response_content_part_event(
                "response.content_part.added", turn.response_id, turn.item_id, ""
            )
        )
        await _emit_events(turn, *started)

        usage = None
        finish_reason = None
        try:
            if turn.prefill_task is not None:
                # Prefill is an optimization. Its failure must not change the
                # exact final chat-completion behavior.
                await asyncio.gather(turn.prefill_task, return_exceptions=True)
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
            item = turn.item("incomplete" if incomplete else "completed")
            completed = [
                response_output_text_event(
                    "response.output_text.done",
                    turn.response_id,
                    turn.item_id,
                    turn.text,
                ),
                response_content_part_event(
                    "response.content_part.done",
                    turn.response_id,
                    turn.item_id,
                    turn.text,
                ),
                response_output_item_done_event(turn.response_id, item),
            ]
            if turn.add_to_conversation:
                turn.items.append(item)
                turn.conversation_messages.append(
                    {"role": "assistant", "content": turn.text}
                )
                completed.append(
                    conversation_item_done_event(item, turn.previous_item_id)
                )
            status = "incomplete" if incomplete else "completed"
            completed.append(
                response_done_event(
                    turn.response_id,
                    output_modalities=["text"],
                    max_output_tokens=turn.wire_max_output_tokens,
                    output=[item],
                    status=status,
                    status_details=(
                        {"type": "incomplete", "reason": "max_output_tokens"}
                        if incomplete
                        else None
                    ),
                    usage=_realtime_usage(usage),
                )
            )
            turn.finished = True
            await _emit_events(turn, *completed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate engine failures per response
            logger.exception("realtime text generation failed: %s", exc)
            turn.finished = True
            await turn.events.put(
                response_done_event(
                    turn.response_id,
                    output_modalities=["text"],
                    max_output_tokens=turn.wire_max_output_tokens,
                    output=[turn.item("incomplete")],
                    status="failed",
                    status_details={
                        "type": "failed",
                        "error": {
                            "type": "server_error",
                            "code": "generation_error",
                        },
                    },
                )
            )

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
        }
        items: list[dict[str, Any]] = []
        messages: list[dict[str, str]] = []
        connection = RealtimeConnection[_TextTurn](
            context=context, run_turn=self._run_turn, max_concurrent_turns=1
        )
        active_response: _TextTurn | None = None
        active_prefill: _TextPrefill | None = None
        committed_prefill: _TextPrefill | None = None

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
                    item, message = _normalize_text_item(event.get("item"))
                    if any(existing["id"] == item["id"] for existing in items):
                        raise ValueError(
                            f"conversation item {item['id']!r} already exists"
                        )
                except ValueError as exc:
                    emit_error(event, "invalid_item", str(exc))
                    return
                previous_item_id = items[-1]["id"] if items else None
                items.append(item)
                messages.append(message)
                connection.emit(conversation_item_added_event(item, previous_item_id))
                connection.emit(conversation_item_done_event(item, previous_item_id))
            elif event_type == "input_text_buffer.append":
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
                    prompt = list(messages)
                    if session["instructions"]:
                        prompt.insert(
                            0,
                            {"role": "system", "content": session["instructions"]},
                        )
                    active_prefill = _TextPrefill(
                        messages=prompt,
                        factory=self._text_prefill_factory,
                    )
                active_prefill.append(text)
            elif event_type == "input_text_buffer.commit":
                if active_prefill is None or not active_prefill.text:
                    emit_error(event, "invalid_text", "input text buffer is empty")
                    return
                active_prefill.commit()
                committed_prefill = active_prefill
                text = active_prefill.text
                active_prefill = None
                item, message = _normalize_text_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
                previous_item_id = items[-1]["id"] if items else None
                items.append(item)
                messages.append(message)
                connection.emit(conversation_item_added_event(item, previous_item_id))
                connection.emit(conversation_item_done_event(item, previous_item_id))
            elif event_type == "input_text_buffer.clear":
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
                    ) = self._response_options(event.get("response"), session)
                    prompt = list(messages) if use_conversation else []
                    if instructions:
                        prompt.insert(0, {"role": "system", "content": instructions})
                    if not prompt:
                        raise ValueError(
                            "response requires conversation input or instructions"
                        )
                except ValueError as exc:
                    emit_error(event, "invalid_response", str(exc))
                    return
                prefill_task = None
                if committed_prefill is not None:
                    if use_conversation and committed_prefill.messages == prompt[:-1]:
                        prefill_task = committed_prefill.task
                    else:
                        await committed_prefill.cancel()
                active_response = await connection.ensure_turn(
                    lambda: _TextTurn(
                        messages=prompt,
                        max_output_tokens=max_output_tokens,
                        wire_max_output_tokens=wire_max_output_tokens,
                        add_to_conversation=add_to_conversation,
                        items=items,
                        conversation_messages=messages,
                        prefill_task=prefill_task,
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
                connection.cancel_turn(running)
                connection.emit(
                    response_done_event(
                        running.response_id,
                        output_modalities=["text"],
                        max_output_tokens=running.wire_max_output_tokens,
                        output=[running.item("incomplete")],
                        status="cancelled",
                        status_details={
                            "type": "cancelled",
                            "reason": "client_cancelled",
                        },
                    )
                )
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


def _default_sampling_params() -> Any:
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    return SamplingParams.from_optional(
        temperature=0.0,
        max_tokens=64,
        output_kind=RequestOutputKind.DELTA,
        skip_clone=True,
    )


def decode_pcm16(audio_b64: str) -> np.ndarray:
    if not isinstance(audio_b64, str):
        raise ValueError("audio must be a base64 string")
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio must be valid base64") from exc
    if not raw:
        raise ValueError("audio chunk is empty")
    if len(raw) > MAX_AUDIO_CHUNK_BYTES:
        raise ValueError("audio chunk exceeds 4 MiB")
    if len(raw) % 2:
        raise ValueError("PCM16 audio must be 2-byte aligned")

    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class _Turn(RealtimeTurn):
    def __init__(self, *, input_rate: int, model_sample_rate: int) -> None:
        super().__init__()
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.request_id = f"rt_{uuid.uuid4().hex}"
        self.input_rate = input_rate
        self.model_sample_rate = model_sample_rate
        self.pending_audio: np.ndarray = np.empty(0, dtype=np.float32)
        self.received_samples = 0
        self.audio: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def _resample(self, waveform: np.ndarray) -> np.ndarray:
        if self.input_rate == self.model_sample_rate:
            return waveform

        # scipy is already a vLLM audio dependency. Polyphase resampling preserves
        # speech bandwidth when adapting OpenAI's fixed 24 kHz PCM stream to the
        # model's native rate (commonly 16 kHz).
        from scipy.signal import resample_poly

        divisor = math.gcd(self.input_rate, self.model_sample_rate)
        return resample_poly(
            waveform,
            self.model_sample_rate // divisor,
            self.input_rate // divisor,
        ).astype(np.float32, copy=False)

    def append_audio(self, waveform: np.ndarray) -> np.ndarray | None:
        received_samples = self.received_samples + len(waveform)
        if received_samples > self.input_rate * MAX_UTTERANCE_SECONDS:
            raise ValueError(f"input audio exceeds {MAX_UTTERANCE_SECONDS} seconds")
        self.received_samples = received_samples
        self.pending_audio = np.concatenate((self.pending_audio, waveform))
        block_size = self.input_rate * RESAMPLE_BLOCK_MILLISECONDS // 1000
        ready_size = len(self.pending_audio) // block_size * block_size
        if not ready_size:
            return None
        ready, self.pending_audio = np.split(self.pending_audio, [ready_size])
        return self._resample(ready)

    def flush_audio(self) -> np.ndarray | None:
        if not len(self.pending_audio):
            return None
        ready = self.pending_audio
        self.pending_audio = np.empty(0, dtype=np.float32)
        return self._resample(ready)

    async def audio_stream(self) -> AsyncGenerator[np.ndarray, None]:
        while True:
            chunk = await self.audio.get()
            if chunk is None:
                return
            yield chunk


class RealtimeTranscriptionHandler:
    """Translate OpenAI realtime transcription events to vLLM streaming input."""

    def __init__(
        self,
        *,
        engine_client: Any,
        model_name: str,
        model_sample_rate: int | float,
        streaming_input_factory: StreamingInputFactory,
        sampling_params_factory: SamplingParamsFactory = _default_sampling_params,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        self.model_sample_rate = int(model_sample_rate)
        self._streaming_input_factory = streaming_input_factory
        self._sampling_params_factory = sampling_params_factory

    @classmethod
    def from_engine(
        cls,
        *,
        engine_client: Any,
        model_name: str,
        model_path: str,
    ) -> "RealtimeTranscriptionHandler":
        serving = build_realtime_serving(
            engine_client=engine_client,
            model_name=model_name,
            model_path=model_path,
        )
        speech_config = serving.model_cls.get_speech_to_text_config(
            serving.model_config, "transcribe"
        )

        def sampling_params() -> Any:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            return SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=serving.model_cls.realtime_max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

        return cls(
            engine_client=engine_client,
            model_name=model_name,
            model_sample_rate=speech_config.sample_rate,
            streaming_input_factory=serving.transcribe_realtime,
            sampling_params_factory=sampling_params,
        )

    async def _run_turn(
        self,
        turn: _Turn,
        context: Context,
    ) -> None:
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input = self._streaming_input_factory(
            turn.audio_stream(), input_stream
        )
        transcript = ""
        input_tokens = 0
        output_tokens = 0

        try:
            result_stream = self.engine_client.generate(
                prompt=streaming_input,
                sampling_params=self._sampling_params_factory(),
                request_id=turn.request_id,
            )
            async for result in result_stream:
                if context.is_stopped():
                    return
                outputs = getattr(result, "outputs", None)
                if not outputs:
                    continue
                candidate = outputs[0]
                delta = getattr(candidate, "text", "") or ""
                token_ids = list(getattr(candidate, "token_ids", None) or [])
                if not input_tokens:
                    input_tokens = len(getattr(result, "prompt_token_ids", None) or [])
                output_tokens += len(token_ids)
                if token_ids:
                    input_stream.put_nowait(token_ids)
                if delta:
                    transcript += delta
                    await turn.events.put(
                        input_audio_transcription_delta_event(turn.item_id, delta)
                    )

            if not context.is_stopped():
                await turn.events.put(
                    input_audio_transcription_completed_event(
                        turn.item_id,
                        transcript,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate engine failures per turn
            logger.exception("realtime transcription failed: %s", exc)
            await turn.events.put(
                input_audio_transcription_failed_event(
                    turn.item_id, "Transcription failed"
                )
            )

    def _validate_session(self, session: Any) -> str | None:
        if not isinstance(session, dict) or session.get("type") != "transcription":
            return "session.type must be 'transcription'"
        audio = session.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("input"), dict):
            return "session.audio.input must be an object"
        audio_input = audio["input"]
        transcription = audio_input.get("transcription")
        if (
            not isinstance(transcription, dict)
            or transcription.get("model") != self.model_name
        ):
            return f"session transcription model must be '{self.model_name}'"
        audio_format = audio_input.get("format")
        if not isinstance(audio_format, dict):
            return "session.audio.input.format must be an object"
        if audio_format.get("type") != "audio/pcm":
            return "only audio/pcm input is supported"
        rate = audio_format.get("rate")
        if rate != OPENAI_PCM_SAMPLE_RATE:
            return f"audio/pcm input rate must be {OPENAI_PCM_SAMPLE_RATE} Hz"
        language = transcription.get("language")
        if language not in (None, "en"):
            return "only English realtime transcription is supported"
        if transcription.get("prompt") not in (None, ""):
            return "transcription prompts are not supported"
        if audio_input.get("noise_reduction") is not None:
            return "input audio noise reduction is not supported"
        if audio_input.get("turn_detection") is not None:
            return "server turn detection is not supported; use local VAD and explicit commits"
        return None

    async def generate(
        self,
        request_stream: AsyncGenerator[Any, None],
        context: Context,
    ) -> AsyncGenerator[dict, None]:
        input_rate = OPENAI_PCM_SAMPLE_RATE

        connection = RealtimeConnection[_Turn](
            context=context,
            run_turn=self._run_turn,
            max_concurrent_turns=1,
            max_queued_turns=1,
        )

        def new_turn() -> _Turn:
            return _Turn(
                input_rate=input_rate,
                model_sample_rate=self.model_sample_rate,
            )

        def close_turn(turn: _Turn) -> None:
            remainder = turn.flush_audio()
            if remainder is not None:
                turn.audio.put_nowait(remainder)
            turn.audio.put_nowait(None)

        async def handle_event(
            event: Any,
            connection: RealtimeConnection[_Turn],
        ) -> None:
            nonlocal input_rate
            if not isinstance(event, dict):
                connection.emit(
                    invalid_request_error_event(
                        "invalid_event", "event must be an object"
                    )
                )
                return
            event_type = event.get("type")
            if event_type == "session.update":
                session = event.get("session")
                error = self._validate_session(session)
                if error:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_session",
                            error,
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                assert isinstance(session, dict)
                input_rate = session["audio"]["input"]["format"]["rate"]
                connection.emit(session_updated_event(session))
            elif event_type == "input_audio_buffer.append":
                try:
                    waveform = decode_pcm16(event.get("audio", ""))
                except ValueError as exc:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio",
                            str(exc),
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                turn = await connection.ensure_turn(new_turn)
                try:
                    ready = turn.append_audio(waveform)
                except ValueError as exc:
                    connection.cancel_active_turn()
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio",
                            str(exc),
                            client_event_id=event.get("event_id"),
                        )
                    )
                    return
                if ready is not None:
                    turn.audio.put_nowait(ready)
            elif event_type == "input_audio_buffer.commit":
                active_turn = connection.active_turn
                if active_turn is None:
                    connection.emit(
                        invalid_request_error_event(
                            "invalid_audio", "input audio buffer is empty"
                        )
                    )
                    return
                remainder = active_turn.flush_audio()
                if remainder is not None:
                    active_turn.audio.put_nowait(remainder)
                await connection.emit_for_turn(
                    active_turn,
                    input_audio_buffer_committed_event(active_turn.item_id),
                )
                active_turn.audio.put_nowait(None)
                connection.finish_active_turn()
            elif event_type == "input_audio_buffer.clear":
                connection.cancel_active_turn()
                connection.emit(input_audio_buffer_cleared_event())
            else:
                connection.emit(
                    invalid_request_error_event(
                        "unsupported_event",
                        f"unsupported event type: {event_type}",
                        client_event_id=event.get("event_id"),
                    )
                )

        async for event in connection.generate(
            request_stream,
            handle_event=handle_event,
            close_active_turn=close_turn,
        ):
            yield event

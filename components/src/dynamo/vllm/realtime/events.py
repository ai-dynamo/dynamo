# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct OpenAI Realtime server events for Python workers."""

from __future__ import annotations

import uuid
from typing import Any


def event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def invalid_request_error_event(
    code: str,
    message: str,
    *,
    client_event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": event_id(),
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "event_id": client_event_id,
        },
    }


def server_error_event(code: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "event_id": event_id(),
        "error": {
            "type": "server_error",
            "code": code,
            "message": message,
        },
    }


def session_updated_event(session: Any) -> dict[str, Any]:
    return {
        "type": "session.updated",
        "event_id": event_id(),
        "session": session,
    }


def input_audio_buffer_committed_event(item_id: str) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.committed",
        "event_id": event_id(),
        "previous_item_id": None,
        "item_id": item_id,
    }


def input_audio_buffer_cleared_event() -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.cleared",
        "event_id": event_id(),
    }


def input_audio_transcription_delta_event(
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": event_id(),
        "item_id": item_id,
        "content_index": 0,
        "delta": delta,
        "logprobs": None,
    }


def input_audio_transcription_completed_event(
    item_id: str,
    transcript: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": event_id(),
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
        "logprobs": None,
        "usage": {
            "type": "tokens",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "audio_tokens": input_tokens,
                "text_tokens": 0,
            },
        },
    }


def input_audio_transcription_failed_event(
    item_id: str,
    message: str,
) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.failed",
        "event_id": event_id(),
        "item_id": item_id,
        "content_index": 0,
        "error": {
            "type": "server_error",
            "code": "transcription_error",
            "message": message,
        },
    }


def response_created_event(
    response_id: str,
    *,
    output_modalities: list[str],
) -> dict[str, Any]:
    return {
        "type": "response.created",
        "event_id": event_id(),
        "response": {
            "id": response_id,
            "max_output_tokens": "inf",
            "object": "realtime.response",
            "output": [],
            "output_modalities": output_modalities,
            "status": "in_progress",
        },
    }


def response_done_event(
    response_id: str,
    *,
    output_modalities: list[str],
    status: str = "completed",
    status_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": output_modalities,
        "status": status,
    }
    if status_details is not None:
        response["status_details"] = status_details
    return {
        "type": "response.done",
        "event_id": event_id(),
        "response": response,
    }


def response_failed_event(
    response_id: str,
    *,
    output_modalities: list[str],
    code: str,
) -> dict[str, Any]:
    return response_done_event(
        response_id,
        output_modalities=output_modalities,
        status="failed",
        status_details={
            "type": "failed",
            "error": {
                "code": code,
                "type": "server_error",
            },
        },
    )


def response_output_audio_delta_event(
    response_id: str,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_audio.delta",
        "event_id": event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def response_output_audio_done_event(
    response_id: str,
    item_id: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_audio.done",
        "event_id": event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }


def response_output_audio_transcript_delta_event(
    response_id: str,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "response.output_audio_transcript.delta",
        "event_id": event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }

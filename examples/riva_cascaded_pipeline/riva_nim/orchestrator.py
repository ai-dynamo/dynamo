# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime orchestrator for the RIVA cascaded voice pipeline.

Serves a ``ModelType.Realtime`` bidirectional endpoint. The user's utterance
streams in as ``input_audio_buffer.append`` events, which are accumulated; an
``input_audio_buffer.commit`` marks end-of-turn and triggers the response:
ASR (audio -> transcript) -> LLM (chat, text -> text) -> TTS (text -> audio),
emitted back as the OpenAI-realtime ``response.*`` event sequence.

The downstream ASR / LLM / TTS workers are reached as Dynamo endpoint clients,
so this engine holds no RIVA connection itself.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import logging
import uuid
from typing import AsyncIterator

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="riva-orchestrator")

DEFAULT_MODEL_NAME = "riva-voice-agent"
DEFAULT_ASR_ENDPOINT = "dynamo.riva-asr.generate"
DEFAULT_TTS_ENDPOINT = "dynamo.riva-tts.generate"
DEFAULT_LLM_ENDPOINT = "dynamo.backend.generate"
DEFAULT_LLM_MODEL = ""


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _error_event(error_type: str, code: str, message: str) -> dict:
    return {
        "type": "error",
        "event_id": _event_id(),
        "error": {"type": error_type, "code": code, "message": message},
    }


def _response_payload(response_id: str, status: str) -> dict:
    """Minimal ``RealtimeResponse`` payload accepted by the frontend's typed reader."""
    return {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
    }


class CascadedPipeline:
    """Drives ASR -> LLM -> TTS for one realtime session.

    Args:
        asr_client: Dynamo client for the RIVA ASR worker.
        llm_client: Dynamo client for the LLM chat worker (text-in-text-out).
        tts_client: Dynamo client for the RIVA TTS worker.
        llm_model: Model name sent in the chat request to the LLM worker.
    """

    def __init__(self, asr_client, llm_client, tts_client, llm_model: str) -> None:
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.tts_client = tts_client
        self.llm_model = llm_model

    async def _recognize(self, audio_base64: str) -> str:
        """Send accumulated audio to the ASR worker and return the transcript."""
        stream = await self.asr_client.round_robin({"audio_base64": audio_base64})
        async for response in stream:
            return response.data()["transcript"]
        return ""

    async def _chat(self, text: str) -> str:
        """Send the transcript to the LLM chat worker and return the reply text.

        The LLM worker (vLLM with ``--use-vllm-tokenizer``) streams
        ``chat.completion.chunk`` dicts; the reply is the concatenation of the
        ``choices[0].delta.content`` deltas.
        """
        request = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": text}],
            "stream": True,
        }
        stream = await self.llm_client.round_robin(request)
        parts = []
        async for response in stream:
            choices = response.data().get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    parts.append(content)
        return "".join(parts)

    async def _synthesize(self, text: str) -> str:
        """Send the reply text to the TTS worker and return base64 audio."""
        stream = await self.tts_client.round_robin({"text": text})
        async for response in stream:
            return response.data()["audio_base64"]
        return ""

    async def _respond(self, audio_base64: str, context) -> AsyncIterator[dict]:
        """Run ASR -> LLM -> TTS for one turn and emit the response.* events.

        Checks ``context.is_stopped()`` between stages so a cancelled turn stops
        before doing further work; cancellation is observed between stages, not
        mid-stage (the in-flight downstream call still completes).
        """
        transcript = await self._recognize(audio_base64)
        if context.is_stopped():
            return
        reply_text = await self._chat(transcript)
        if context.is_stopped():
            return
        reply_audio = await self._synthesize(reply_text)
        if context.is_stopped():
            return

        response_id = f"resp_{uuid.uuid4().hex}"
        item_id = f"item_{uuid.uuid4().hex}"
        yield {
            "type": "response.created",
            "event_id": _event_id(),
            "response": _response_payload(response_id, "in_progress"),
        }
        yield {
            "type": "response.output_audio.delta",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": reply_audio,
        }
        yield {
            "type": "response.output_audio.done",
            "event_id": _event_id(),
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }
        yield {
            "type": "response.done",
            "event_id": _event_id(),
            "response": _response_payload(response_id, "completed"),
        }

    async def handle(self, request_stream, context) -> AsyncIterator[dict]:
        """Bidirectional realtime engine: accumulate audio, respond on commit."""
        audio_chunks: list[str] = []
        async for event in request_stream:
            if context.is_stopped():
                return
            event_type = event.get("type") if isinstance(event, dict) else None

            if event_type == "session.update":
                yield {
                    "type": "session.updated",
                    "event_id": _event_id(),
                    "session": event.get("session"),
                }
            elif event_type == "input_audio_buffer.append":
                audio_chunks.append(event.get("audio", ""))
            elif event_type == "input_audio_buffer.commit":
                chunks = audio_chunks
                audio_chunks = []
                if not chunks:
                    # Commit with no buffered audio: nothing to respond to.
                    continue
                # Concatenate the raw PCM bytes (not the base64 strings). Malformed
                # client audio is a client error, so report it rather than letting
                # the decode exception tear down the realtime stream.
                try:
                    audio = b"".join(
                        base64.b64decode(chunk, validate=True) for chunk in chunks
                    )
                except (binascii.Error, ValueError) as exc:
                    yield _error_event(
                        "invalid_request_error", "invalid_audio", str(exc)
                    )
                    continue
                # A downstream worker failure should surface to the realtime
                # client as an error event, not tear down the whole session.
                try:
                    async for output in self._respond(
                        base64.b64encode(audio).decode(), context
                    ):
                        yield output
                except Exception as exc:
                    logger.exception("Cascaded pipeline turn failed")
                    yield _error_event("server_error", "pipeline_error", str(exc))
            else:
                yield _error_event(
                    "invalid_request_error",
                    "unsupported_client_event",
                    f"orchestrator does not support {event_type}",
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIVA cascaded-pipeline orchestrator")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Realtime model name to register.",
    )
    parser.add_argument(
        "--asr-endpoint", default=DEFAULT_ASR_ENDPOINT, help="ASR worker endpoint."
    )
    parser.add_argument(
        "--llm-endpoint", default=DEFAULT_LLM_ENDPOINT, help="LLM chat worker endpoint."
    )
    parser.add_argument(
        "--tts-endpoint", default=DEFAULT_TTS_ENDPOINT, help="TTS worker endpoint."
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Model name sent to the LLM worker.",
    )
    parser.add_argument(
        "--endpoint",
        default="dynamo.riva-orchestrator.generate",
        help="Realtime endpoint to serve.",
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    asr_client = await runtime.endpoint(args.asr_endpoint).client()
    llm_client = await runtime.endpoint(args.llm_endpoint).client()
    tts_client = await runtime.endpoint(args.tts_endpoint).client()
    pipeline = CascadedPipeline(asr_client, llm_client, tts_client, args.llm_model)

    endpoint = runtime.endpoint(args.endpoint)
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        args.model_name,
        model_name=args.model_name,
        worker_type=WorkerType.Aggregated,
    )
    logger.info("Serving realtime orchestrator endpoint %s", args.endpoint)
    await endpoint.serve_bidirectional_endpoint(pipeline.handle)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker(_parse_args()))

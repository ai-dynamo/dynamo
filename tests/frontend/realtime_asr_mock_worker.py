# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ASR worker using the production ordered-turn connection."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import uuid
from pathlib import Path

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.realtime.connection import RealtimeConnection, RealtimeTurn
from dynamo.vllm.realtime.events import (
    input_audio_buffer_cleared_event,
    input_audio_buffer_committed_event,
    input_audio_transcription_completed_event,
    input_audio_transcription_delta_event,
    invalid_request_error_event,
    session_updated_event,
)


class Turn(RealtimeTurn):
    def __init__(self) -> None:
        super().__init__()
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.audio: asyncio.Queue[str | None] = asyncio.Queue()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--split-deltas", action="store_true")
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    async def run_turn(turn, context):
        parts = []
        while (audio := await turn.audio.get()) is not None:
            text = base64.b64decode(audio).decode()
            parts.append(text)
            for delta in list(text) if args.split_deltas else [text]:
                await turn.events.put(
                    input_audio_transcription_delta_event(turn.item_id, delta)
                )
        while (args.audit_dir / f"{args.worker_id}.hold_final").exists():
            await asyncio.sleep(0.01)
        await turn.events.put(
            input_audio_transcription_completed_event(
                turn.item_id, "".join(parts), input_tokens=0, output_tokens=0
            )
        )

    async def generate(request_stream, context):
        connection_id = uuid.uuid4().hex
        audit_path = args.audit_dir / f"{args.worker_id}-{connection_id}.jsonl"
        connection = RealtimeConnection(
            context=context, run_turn=run_turn, max_concurrent_turns=2
        )

        async def handle_event(event, connection):
            with audit_path.open("a") as audit:
                audit.write(json.dumps(event) + "\n")
            match event["type"]:
                case "session.update":
                    if (args.audit_dir / f"{args.worker_id}.hang_setup").exists():
                        await context.async_killed_or_stopped()
                        return
                    updated = session_updated_event(event["session"])
                    updated["event_id"] = f"{args.worker_id}:{connection_id}"
                    connection.emit(updated)
                case "input_audio_buffer.append":
                    if (args.audit_dir / f"{args.worker_id}.hang_replay").exists():
                        await context.async_killed_or_stopped()
                        return
                    if (args.audit_dir / f"{args.worker_id}.abort_turn").exists():
                        connection.cancel_active_turn()
                        connection.emit(
                            invalid_request_error_event(
                                "invalid_audio", "input audio exceeds 60 seconds"
                            )
                        )
                        return
                    turn = await connection.ensure_turn(Turn)
                    turn.audio.put_nowait(event["audio"])
                case "input_audio_buffer.commit":
                    turn = connection.finish_active_turn()
                    assert turn is not None
                    await connection.emit_for_turn(
                        turn, input_audio_buffer_committed_event(turn.item_id)
                    )
                    turn.audio.put_nowait(None)
                case "input_audio_buffer.clear":
                    connection.cancel_active_turn()
                    connection.emit(input_audio_buffer_cleared_event())
                case _:
                    connection.emit(
                        invalid_request_error_event(
                            "unsupported_event", "unsupported event"
                        )
                    )

        try:
            async for event in connection.generate(
                request_stream,
                handle_event=handle_event,
                close_active_turn=lambda turn: turn.audio.put_nowait(None),
            ):
                yield event
        finally:
            (args.audit_dir / f"{args.worker_id}-{connection_id}.closed").touch()

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(args.endpoint)
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        args.model,
        model_name=args.model,
        worker_type=WorkerType.Aggregated,
    )
    (args.audit_dir / f"{args.worker_id}.ready").touch()
    await endpoint.serve_bidirectional_endpoint(generate)


if __name__ == "__main__":
    uvloop.run(main())

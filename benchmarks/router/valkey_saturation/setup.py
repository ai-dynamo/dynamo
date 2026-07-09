# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from .protocol import Counters, LatencyRecorder, RespCommandError, WorkCommand, read_resp

async def send_batch(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    commands: Sequence[WorkCommand],
    counters: Counters,
    latency: LatencyRecorder | None,
) -> list[Any]:
    if not commands:
        return []
    encoded = b"".join(command.encoded for command in commands)
    sent_ns = time.perf_counter_ns()
    writer.write(encoded)
    await writer.drain()
    counters.commands += len(commands)
    counters.request_wire_bytes += len(encoded)
    for command in commands:
        counters.commands_by_kind[command.kind] = (
            counters.commands_by_kind.get(command.kind, 0) + 1
        )
        counters.logical_payload_bytes += command.logical_payload_bytes
        if command.event_bytes:
            counters.events += 1
            counters.event_bytes += command.event_bytes
            counters.blocks += command.blocks
            event_kind = command.event_kind or command.kind
            counters.events_by_kind[event_kind] = (
                counters.events_by_kind.get(event_kind, 0) + 1
            )
        counters.queries += int(command.query)
        counters.selections += int(command.selection)

    values = []
    for command in commands:
        try:
            response = await read_resp(reader)
        except RespCommandError as error:
            counters.response_wire_bytes += error.wire_bytes
            raise RuntimeError(f"{command.kind} failed: {error}") from error
        counters.response_wire_bytes += response.wire_bytes
        value = command.validator(response)
        values.append(value)
        if latency is not None:
            latency.record(
                command.latency_kind or command.kind,
                time.perf_counter_ns() - sent_ns,
            )
    return values

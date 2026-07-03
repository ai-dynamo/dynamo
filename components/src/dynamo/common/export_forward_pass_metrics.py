# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Dynamo ForwardPassMetrics to a local newline-delimited JSON socket.

This process remains on the Dynamo side of the integration boundary: it uses
``FpmEventSubscriber`` for discovery and event-plane transport, decodes the
versioned Dynamo payload, and forwards a stable record to a local collector
such as Tachometer.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections.abc import Sequence

import msgspec

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode

logger = logging.getLogger(__name__)


class FpmExportRecord(msgspec.Struct, frozen=True, gc=False):  # type: ignore[call-arg]
    """One event-plane FPM record sent to the local collection socket."""

    namespace: str
    component: str
    publisher_id: int
    sequence: int
    published_at_ms: int
    metrics: ForwardPassMetrics


_json_encoder = msgspec.json.Encoder()


def encode_export_record(record: FpmExportRecord) -> bytes:
    """Encode one record as an NDJSON line."""
    return _json_encoder.encode(record) + b"\n"


def _socket_path(value: str) -> str:
    if value.startswith("unix://"):
        return value.removeprefix("unix://")
    return value


async def _connect_sink(path: str, timeout: float) -> asyncio.StreamWriter:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            _reader, writer = await asyncio.open_unix_connection(path)
            return writer
        except OSError as error:
            last_error = error
            await asyncio.sleep(0.2)
    raise TimeoutError(f"Timed out connecting to FPM sink {path}: {last_error}")


async def _receive_component(
    *,
    subscriber,
    namespace: str,
    component: str,
    records: asyncio.Queue[FpmExportRecord],
) -> None:
    while True:
        envelope = await asyncio.to_thread(subscriber.recv_envelope)
        if envelope is None:
            logger.info("FPM stream closed for component=%s", component)
            return

        publisher_id, sequence, published_at_ms, payload = envelope
        metrics = decode(payload)
        if metrics is None:
            continue

        await records.put(
            FpmExportRecord(
                namespace=namespace,
                component=component,
                publisher_id=publisher_id,
                sequence=sequence,
                published_at_ms=published_at_ms,
                metrics=metrics,
            )
        )


async def _write_records(
    writer: asyncio.StreamWriter, records: asyncio.Queue[FpmExportRecord]
) -> None:
    while True:
        record = await records.get()
        writer.write(encode_export_record(record))
        await writer.drain()


async def export(
    *,
    namespace: str,
    components: Sequence[str],
    endpoint_name: str,
    discovery_backend: str,
    request_plane: str,
    socket_path: str,
    connect_timeout: float,
) -> None:
    from dynamo.llm import FpmEventSubscriber
    from dynamo.runtime import DistributedRuntime

    writer = await _connect_sink(socket_path, connect_timeout)
    logger.info("Connected to FPM collection sink at %s", socket_path)

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, discovery_backend, request_plane)
    subscribers = []
    records: asyncio.Queue[FpmExportRecord] = asyncio.Queue(maxsize=100_000)

    for component in components:
        endpoint = runtime.endpoint(f"{namespace}.{component}.{endpoint_name}")
        subscriber = FpmEventSubscriber(endpoint)
        subscribers.append((component, subscriber))

    writer_task = asyncio.create_task(_write_records(writer, records))
    receiver_tasks = [
        asyncio.create_task(
            _receive_component(
                subscriber=subscriber,
                namespace=namespace,
                component=component,
                records=records,
            )
        )
        for component, subscriber in subscribers
    ]

    logger.info(
        "Exporting forward-pass-metrics components=%s via %s event plane",
        ",".join(components),
        os.environ.get("DYN_EVENT_PLANE", "zmq"),
    )

    try:
        tasks = [writer_task, *receiver_tasks]
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise RuntimeError("An FPM export task stopped unexpectedly")
    finally:
        for _component, subscriber in subscribers:
            subscriber.shutdown()
        for task in receiver_tasks:
            task.cancel()
        writer_task.cancel()
        await asyncio.gather(*receiver_tasks, writer_task, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Dynamo ForwardPassMetrics to a local NDJSON socket"
    )
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help="Component to subscribe to; repeat for multiple components",
    )
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument(
        "--discovery-backend",
        default=os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
    )
    parser.add_argument(
        "--request-plane", default=os.environ.get("DYN_REQUEST_PLANE", "tcp")
    )
    parser.add_argument(
        "--socket",
        required=True,
        help="Unix socket path, optionally prefixed with unix://",
    )
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()
    args = _parse_args()
    components = args.components or ["backend"]
    try:
        asyncio.run(
            export(
                namespace=args.namespace,
                components=components,
                endpoint_name=args.endpoint,
                discovery_backend=args.discovery_backend,
                request_plane=args.request_plane,
                socket_path=_socket_path(args.socket),
                connect_timeout=args.connect_timeout,
            )
        )
    except KeyboardInterrupt:
        logger.info("Stopped FPM exporter")


if __name__ == "__main__":
    main()

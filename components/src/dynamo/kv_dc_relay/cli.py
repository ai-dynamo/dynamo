# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line configuration for the KV DC Relay component."""

import argparse
import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class KvDcRelayCliConfig:
    dc_id: str
    namespaces: tuple[str, ...]
    endpoint_prefixes: tuple[str, ...]
    watch_all: bool
    expected_unique_blocks: int
    publication_threshold: int
    publication_delay_ms: int
    recovery_attempt_timeout_ms: int
    bind: str | None
    tls_server_cert: str | None
    tls_server_key: str | None
    tls_client_ca: str | None
    max_message_bytes: int
    keepalive_interval_ms: int
    keepalive_timeout_ms: int
    pool_heartbeat_interval_ms: int
    readiness_heartbeat_interval_ms: int
    load_window_ms: int
    load_fanout_capacity: int
    publication_queue_capacity: int
    publication_queue_bytes: int
    publication_encoding_concurrency: int
    max_catalog_subscribers: int
    max_pool_subscribers: int
    max_readiness_subscribers: int
    max_load_subscribers: int


class RelayShutdownWaiter(Protocol):
    async def wait_for_shutdown(self) -> None:
        ...


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _csv_values(
    value: str, option: str, parser: argparse.ArgumentParser
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(","))
    if not values or any(not item for item in values):
        parser.error(f"{option} requires a comma-separated list of non-empty values")
    if len(set(values)) != len(values):
        parser.error(f"{option} must not contain duplicate values")
    return values


def _environment_bool(
    environment: Mapping[str, str], name: str, parser: argparse.ArgumentParser
) -> bool:
    value = environment.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    parser.error(f"{name} must be a boolean value")


def _numeric_value(
    cli_value: int | None,
    environment: Mapping[str, str],
    environment_name: str,
    default: int,
    parser: argparse.ArgumentParser,
) -> int:
    if cli_value is not None:
        return cli_value
    environment_value = environment.get(environment_name)
    if environment_value is None:
        return default
    try:
        return _positive_int(environment_value)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(f"{environment_name}: {error}")


def _string_value(
    cli_value: str | None, environment: Mapping[str, str], environment_name: str
) -> str | None:
    if cli_value is not None:
        return cli_value
    return environment.get(environment_name)


def parse_args(
    argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> KvDcRelayCliConfig:
    """Parse Relay CLI arguments with command-line values taking precedence over env."""

    environment = os.environ if environment is None else environment
    parser = argparse.ArgumentParser(description="Dynamo DC-scoped KV Relay")
    parser.add_argument("--dc-id")
    parser.add_argument(
        "--namespaces",
        help="Comma-separated Dynamo namespaces containing inference DGDs",
    )
    parser.add_argument(
        "--watch-all",
        action="store_true",
        default=None,
        help="Watch inference model cards in every Dynamo namespace",
    )
    parser.add_argument(
        "--endpoint-prefix",
        action="append",
        dest="endpoint_prefixes",
        help="Endpoint prefix to include; repeat the option for multiple prefixes",
    )

    parser.add_argument("--expected-unique-blocks", type=_positive_int)
    parser.add_argument("--publication-threshold", type=_positive_int)
    parser.add_argument("--publication-delay-ms", type=_positive_int)
    parser.add_argument("--recovery-attempt-timeout-ms", type=_positive_int)

    parser.add_argument("--bind", help="WAN gRPC listen address")
    parser.add_argument("--tls-server-cert")
    parser.add_argument("--tls-server-key")
    parser.add_argument("--tls-client-ca")
    parser.add_argument("--max-message-bytes", type=_positive_int)
    parser.add_argument("--keepalive-interval-ms", type=_positive_int)
    parser.add_argument("--keepalive-timeout-ms", type=_positive_int)
    parser.add_argument("--pool-heartbeat-interval-ms", type=_positive_int)
    parser.add_argument("--readiness-heartbeat-interval-ms", type=_positive_int)
    parser.add_argument("--load-window-ms", type=_positive_int)
    parser.add_argument("--load-fanout-capacity", type=_positive_int)
    parser.add_argument("--publication-queue-capacity", type=_positive_int)
    parser.add_argument("--publication-queue-bytes", type=_positive_int)
    parser.add_argument("--publication-encoding-concurrency", type=_positive_int)
    parser.add_argument("--max-catalog-subscribers", type=_positive_int)
    parser.add_argument("--max-pool-subscribers", type=_positive_int)
    parser.add_argument("--max-readiness-subscribers", type=_positive_int)
    parser.add_argument("--max-load-subscribers", type=_positive_int)

    parsed = parser.parse_args(argv)
    dc_id = _string_value(parsed.dc_id, environment, "DYN_DC_ID")
    if dc_id is None or not dc_id.strip():
        parser.error("--dc-id or DYN_DC_ID is required")
    if dc_id != dc_id.strip():
        parser.error("DC ID must not contain surrounding whitespace")

    if parsed.namespaces is not None and parsed.watch_all:
        parser.error("--namespaces and --watch-all are mutually exclusive")
    if parsed.namespaces is not None:
        namespaces = _csv_values(parsed.namespaces, "--namespaces", parser)
        watch_all = False
    elif parsed.watch_all:
        namespaces = ()
        watch_all = True
    else:
        environment_namespaces = environment.get("DYN_RELAY_NAMESPACES")
        environment_watch_all = _environment_bool(
            environment, "DYN_RELAY_WATCH_ALL", parser
        )
        if environment_namespaces is not None and environment_watch_all:
            parser.error(
                "DYN_RELAY_NAMESPACES and DYN_RELAY_WATCH_ALL are mutually exclusive"
            )
        if environment_namespaces is not None:
            namespaces = _csv_values(
                environment_namespaces, "DYN_RELAY_NAMESPACES", parser
            )
            watch_all = False
        elif environment_watch_all:
            namespaces = ()
            watch_all = True
        else:
            parser.error("one of --namespaces or --watch-all is required")

    if parsed.endpoint_prefixes is not None:
        endpoint_prefixes = tuple(parsed.endpoint_prefixes)
    else:
        environment_prefixes = environment.get("DYN_RELAY_ENDPOINT_PREFIXES")
        endpoint_prefixes = (
            _csv_values(environment_prefixes, "DYN_RELAY_ENDPOINT_PREFIXES", parser)
            if environment_prefixes is not None
            else ()
        )
    if any(
        not prefix.strip() or prefix != prefix.strip() for prefix in endpoint_prefixes
    ):
        parser.error(
            "endpoint prefixes must be non-empty and have no surrounding whitespace"
        )
    if len(set(endpoint_prefixes)) != len(endpoint_prefixes):
        parser.error("endpoint prefixes must not contain duplicates")
    if not watch_all and any(
        not any(
            prefix == namespace or prefix.startswith(f"{namespace}.")
            for namespace in namespaces
        )
        for prefix in endpoint_prefixes
    ):
        parser.error("endpoint prefixes must be inside the selected namespaces")

    bind = _string_value(parsed.bind, environment, "DYN_RELAY_BIND")
    tls_server_cert = _string_value(
        parsed.tls_server_cert, environment, "DYN_RELAY_TLS_SERVER_CERT"
    )
    tls_server_key = _string_value(
        parsed.tls_server_key, environment, "DYN_RELAY_TLS_SERVER_KEY"
    )
    tls_client_ca = _string_value(
        parsed.tls_client_ca, environment, "DYN_RELAY_TLS_CLIENT_CA"
    )
    tls_values = (tls_server_cert, tls_server_key, tls_client_ca)
    if bind is not None and not bind.strip():
        parser.error("--bind must not be empty")
    if any(value is not None and not value for value in tls_values):
        parser.error("TLS paths must not be empty")
    if bind is not None and any(value is None for value in tls_values):
        parser.error(
            "--bind requires --tls-server-cert, --tls-server-key, and --tls-client-ca"
        )
    if bind is None and any(value is not None for value in tls_values):
        parser.error("TLS options require --bind")

    return KvDcRelayCliConfig(
        dc_id=dc_id,
        namespaces=namespaces,
        endpoint_prefixes=endpoint_prefixes,
        watch_all=watch_all,
        expected_unique_blocks=_numeric_value(
            parsed.expected_unique_blocks,
            environment,
            "DYN_RELAY_EXPECTED_UNIQUE_BLOCKS",
            1_048_576,
            parser,
        ),
        publication_threshold=_numeric_value(
            parsed.publication_threshold,
            environment,
            "DYN_RELAY_PUBLICATION_THRESHOLD",
            16,
            parser,
        ),
        publication_delay_ms=_numeric_value(
            parsed.publication_delay_ms,
            environment,
            "DYN_RELAY_PUBLICATION_DELAY_MS",
            1,
            parser,
        ),
        recovery_attempt_timeout_ms=_numeric_value(
            parsed.recovery_attempt_timeout_ms,
            environment,
            "DYN_RELAY_RECOVERY_ATTEMPT_TIMEOUT_MS",
            30_000,
            parser,
        ),
        bind=bind,
        tls_server_cert=tls_server_cert,
        tls_server_key=tls_server_key,
        tls_client_ca=tls_client_ca,
        max_message_bytes=_numeric_value(
            parsed.max_message_bytes,
            environment,
            "DYN_RELAY_MAX_MESSAGE_BYTES",
            8 * 1024 * 1024,
            parser,
        ),
        keepalive_interval_ms=_numeric_value(
            parsed.keepalive_interval_ms,
            environment,
            "DYN_RELAY_KEEPALIVE_INTERVAL_MS",
            20_000,
            parser,
        ),
        keepalive_timeout_ms=_numeric_value(
            parsed.keepalive_timeout_ms,
            environment,
            "DYN_RELAY_KEEPALIVE_TIMEOUT_MS",
            10_000,
            parser,
        ),
        pool_heartbeat_interval_ms=_numeric_value(
            parsed.pool_heartbeat_interval_ms,
            environment,
            "DYN_RELAY_POOL_HEARTBEAT_INTERVAL_MS",
            10_000,
            parser,
        ),
        readiness_heartbeat_interval_ms=_numeric_value(
            parsed.readiness_heartbeat_interval_ms,
            environment,
            "DYN_RELAY_READINESS_HEARTBEAT_INTERVAL_MS",
            10_000,
            parser,
        ),
        load_window_ms=_numeric_value(
            parsed.load_window_ms,
            environment,
            "DYN_RELAY_LOAD_WINDOW_MS",
            1_000,
            parser,
        ),
        load_fanout_capacity=_numeric_value(
            parsed.load_fanout_capacity,
            environment,
            "DYN_RELAY_LOAD_FANOUT_CAPACITY",
            16,
            parser,
        ),
        publication_queue_capacity=_numeric_value(
            parsed.publication_queue_capacity,
            environment,
            "DYN_RELAY_PUBLICATION_QUEUE_CAPACITY",
            16,
            parser,
        ),
        publication_queue_bytes=_numeric_value(
            parsed.publication_queue_bytes,
            environment,
            "DYN_RELAY_PUBLICATION_QUEUE_BYTES",
            16 * 1024 * 1024,
            parser,
        ),
        publication_encoding_concurrency=_numeric_value(
            parsed.publication_encoding_concurrency,
            environment,
            "DYN_RELAY_PUBLICATION_ENCODING_CONCURRENCY",
            2,
            parser,
        ),
        max_catalog_subscribers=_numeric_value(
            parsed.max_catalog_subscribers,
            environment,
            "DYN_RELAY_MAX_CATALOG_SUBSCRIBERS",
            64,
            parser,
        ),
        max_pool_subscribers=_numeric_value(
            parsed.max_pool_subscribers,
            environment,
            "DYN_RELAY_MAX_POOL_SUBSCRIBERS",
            64,
            parser,
        ),
        max_readiness_subscribers=_numeric_value(
            parsed.max_readiness_subscribers,
            environment,
            "DYN_RELAY_MAX_READINESS_SUBSCRIBERS",
            64,
            parser,
        ),
        max_load_subscribers=_numeric_value(
            parsed.max_load_subscribers,
            environment,
            "DYN_RELAY_MAX_LOAD_SUBSCRIBERS",
            64,
            parser,
        ),
    )


async def monitor_relay(
    relay: RelayShutdownWaiter, endpoint_tasks: Sequence[asyncio.Task[object]]
) -> None:
    """Return when Relay cancellation or an endpoint task ends."""

    relay_shutdown = asyncio.create_task(relay.wait_for_shutdown())
    try:
        done, _pending = await asyncio.wait(
            {relay_shutdown, *endpoint_tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task is not relay_shutdown:
                task.result()
    finally:
        if not relay_shutdown.done():
            relay_shutdown.cancel()
            await asyncio.gather(relay_shutdown, return_exceptions=True)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line configuration for the KV DC Relay component."""

import argparse
import asyncio
import os
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")

TUNING_KEYS: tuple[str, ...] = (
    "publication_threshold",
    "publication_delay_ms",
    "recovery_attempt_timeout_ms",
    "max_message_bytes",
    "keepalive_interval_ms",
    "keepalive_timeout_ms",
    "pool_heartbeat_interval_ms",
    "readiness_heartbeat_interval_ms",
    "snapshot_progress_timeout_ms",
    "load_window_ms",
    "load_fanout_capacity",
    "publication_queue_capacity",
    "publication_queue_bytes",
    "publication_encoding_concurrency",
    "max_catalog_subscribers",
    "max_pool_streams_total",
    "max_subscribers_per_pool",
    "max_initialized_pool_hubs",
    "max_readiness_subscribers",
    "max_load_subscribers",
)


@dataclass(frozen=True)
class KvDcRelayCliConfig:
    dc_id: str
    namespaces: tuple[str, ...]
    endpoint_prefixes: tuple[str, ...]
    watch_all: bool
    expected_unique_blocks: int
    bind: str | None
    tuning: tuple[tuple[str, int], ...]


class RelayShutdownWaiter(Protocol):
    def wait_for_shutdown(self) -> Awaitable[None]:
        ...


def schedule_awaitable(awaitable: Awaitable[T]) -> asyncio.Future[T]:
    """Schedule coroutine- and extension-backed asyncio awaitables."""

    return asyncio.ensure_future(awaitable)


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
        "--namespace-filter",
        help=(
            "Legacy single-namespace discovery filter; equivalent to "
            "--namespaces with one value"
        ),
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

    parser.add_argument("--bind", help="Plaintext WAN gRPC listen address")

    parsed = parser.parse_args(argv)
    dc_id = _string_value(parsed.dc_id, environment, "DYN_DC_ID")
    if dc_id is None or not dc_id.strip():
        parser.error("--dc-id or DYN_DC_ID is required")
    if dc_id != dc_id.strip():
        parser.error("DC ID must not contain surrounding whitespace")

    configured_cli_scopes = sum(
        (
            parsed.namespaces is not None,
            parsed.namespace_filter is not None,
            bool(parsed.watch_all),
        )
    )
    if configured_cli_scopes > 1:
        parser.error(
            "--namespace-filter, --namespaces, and --watch-all are mutually exclusive"
        )
    if parsed.namespaces is not None:
        namespaces = _csv_values(parsed.namespaces, "--namespaces", parser)
        watch_all = False
    elif parsed.namespace_filter is not None:
        namespace_filter = parsed.namespace_filter
        if not namespace_filter.strip() or namespace_filter != namespace_filter.strip():
            parser.error(
                "--namespace-filter must be non-empty and have no surrounding whitespace"
            )
        namespaces = (namespace_filter,)
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
        elif "DYN_RELAY_WATCH_ALL" in environment:
            parser.error("DYN_RELAY_WATCH_ALL=false requires DYN_RELAY_NAMESPACES")
        else:
            # Preserve the fresh-main contract: an omitted namespace filter watches all
            # namespaces. --watch-all exists to make that choice explicit in deployment files.
            namespaces = ()
            watch_all = True

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
    if bind is not None and not bind.strip():
        parser.error("--bind must not be empty")

    tuning: list[tuple[str, int]] = []
    for key in TUNING_KEYS:
        environment_name = f"DYN_RELAY_{key.upper()}"
        environment_value = environment.get(environment_name)
        if environment_value is None:
            continue
        try:
            tuning.append((key, _positive_int(environment_value)))
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(f"{environment_name}: {error}")

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
        bind=bind,
        tuning=tuple(tuning),
    )


async def monitor_relay(
    relay: RelayShutdownWaiter, endpoint_tasks: Sequence[asyncio.Task[object]]
) -> None:
    """Return when Relay cancellation or an endpoint task ends."""

    relay_shutdown = schedule_awaitable(relay.wait_for_shutdown())
    try:
        done, _pending = await asyncio.wait(
            {relay_shutdown, *endpoint_tasks}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task is not relay_shutdown:
                task.result()
        if relay_shutdown in done:
            relay_shutdown.result()
    finally:
        if not relay_shutdown.done():
            relay_shutdown.cancel()
            await asyncio.gather(relay_shutdown, return_exceptions=True)

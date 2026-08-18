# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dynamo.kv_dc_relay.cli import (
    TUNING_KEYS,
    monitor_relay,
    parse_args,
    schedule_awaitable,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def test_local_only_defaults_require_explicit_watch_scope() -> None:
    config = parse_args(["--dc-id", "dc-a", "--watch-all"], {})

    assert config.dc_id == "dc-a"
    assert config.watch_all is True
    assert config.namespaces == ()
    assert config.endpoint_prefixes == ()
    assert config.bind is None
    assert config.expected_unique_blocks == 1_048_576
    assert config.tuning == ()


def test_environment_values_and_cli_precedence() -> None:
    environment = {
        "DYN_DC_ID": "environment-dc",
        "DYN_RELAY_NAMESPACES": "environment-a, environment-b",
        "DYN_RELAY_ENDPOINT_PREFIXES": "environment-a.backend",
        "DYN_RELAY_PUBLICATION_THRESHOLD": "7",
    }
    config = parse_args(
        [
            "--dc-id",
            "cli-dc",
            "--namespaces",
            "cli-a, cli-b",
            "--endpoint-prefix",
            "cli-a.backend",
            "--endpoint-prefix",
            "cli-b.backend",
        ],
        environment,
    )

    assert config.dc_id == "cli-dc"
    assert config.namespaces == ("cli-a", "cli-b")
    assert config.endpoint_prefixes == ("cli-a.backend", "cli-b.backend")
    assert config.tuning == (("publication_threshold", 7),)


def test_tuning_is_collected_from_environment_only() -> None:
    config = parse_args(
        ["--dc-id", "dc-a", "--watch-all"],
        {
            "DYN_RELAY_MAX_POOL_STREAMS_TOTAL": "71",
            "DYN_RELAY_MAX_SUBSCRIBERS_PER_POOL": "8",
            "DYN_RELAY_MAX_INITIALIZED_POOL_HUBS": "3",
            "DYN_RELAY_UNRELATED": "9",
        },
    )

    assert dict(config.tuning) == {
        "max_pool_streams_total": 71,
        "max_subscribers_per_pool": 8,
        "max_initialized_pool_hubs": 3,
    }


def test_environment_namespace_csv_and_prefixes_are_parsed() -> None:
    config = parse_args(
        [],
        {
            "DYN_DC_ID": "dc-a",
            "DYN_RELAY_NAMESPACES": "prod-a,prod-b",
            "DYN_RELAY_ENDPOINT_PREFIXES": "prod-a.backend,prod-b.backend",
        },
    )

    assert config.namespaces == ("prod-a", "prod-b")
    assert config.endpoint_prefixes == ("prod-a.backend", "prod-b.backend")
    assert config.watch_all is False


@pytest.mark.parametrize(
    ("argv", "environment"),
    [
        (["--watch-all"], {}),
        (["--dc-id", "dc-a"], {}),
        (
            ["--dc-id", "dc-a", "--watch-all", "--namespaces", "prod"],
            {},
        ),
        (
            ["--dc-id", "dc-a"],
            {
                "DYN_RELAY_NAMESPACES": "prod",
                "DYN_RELAY_WATCH_ALL": "true",
            },
        ),
        (
            [
                "--dc-id",
                "dc-a",
                "--namespaces",
                "prod",
                "--endpoint-prefix",
                "other.backend",
            ],
            {},
        ),
    ],
)
def test_invalid_discovery_scopes_are_rejected(
    argv: list[str], environment: dict[str, str]
) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv, environment)


@pytest.mark.parametrize(
    "argv",
    [
        ["--dc-id", "dc-a", "--watch-all", "--bind", "127.0.0.1:50051"],
        [
            "--dc-id",
            "dc-a",
            "--watch-all",
            "--tls-server-cert",
            "server.crt",
        ],
    ],
)
def test_bind_and_tls_must_be_configured_together(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv, {})


def test_complete_mtls_configuration_is_accepted() -> None:
    config = parse_args(
        [
            "--dc-id",
            "dc-a",
            "--watch-all",
            "--bind",
            "127.0.0.1:50051",
            "--tls-server-cert",
            "server.crt",
            "--tls-server-key",
            "server.key",
            "--tls-client-ca",
            "ca.crt",
        ],
        {},
    )

    assert config.bind == "127.0.0.1:50051"
    assert config.tls_server_cert == "server.crt"
    assert config.tls_server_key == "server.key"
    assert config.tls_client_ca == "ca.crt"


def test_expected_unique_blocks_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            ["--dc-id", "dc-a", "--watch-all", "--expected-unique-blocks", "0"], {}
        )


@pytest.mark.parametrize("key", TUNING_KEYS)
def test_environment_tuning_must_be_positive(key: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            ["--dc-id", "dc-a", "--watch-all"], {f"DYN_RELAY_{key.upper()}": "0"}
        )


@pytest.mark.parametrize(
    "removed_option",
    [
        "--metrics-bind",
        "--telemetry-interval-ms",
        "--publication-threshold",
        "--max-pool-streams-total",
        "--load-window-ms",
    ],
)
def test_removed_options_are_rejected(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--dc-id", "dc-a", "--watch-all", removed_option, "ignored"], {})


def test_fatal_relay_shutdown_releases_launcher_monitor() -> None:
    class FakeRelay:
        def __init__(self) -> None:
            self.cancelled = asyncio.Event()

        async def wait_for_shutdown(self) -> None:
            await self.cancelled.wait()

    async def scenario() -> None:
        relay = FakeRelay()
        endpoint_never_finishes = asyncio.create_task(asyncio.Event().wait())
        relay.cancelled.set()

        await asyncio.wait_for(
            monitor_relay(relay, [endpoint_never_finishes]), timeout=1
        )

        assert not endpoint_never_finishes.done()
        endpoint_never_finishes.cancel()
        await asyncio.gather(endpoint_never_finishes, return_exceptions=True)

    asyncio.run(scenario())


def test_schedule_awaitable_accepts_extension_backed_future() -> None:
    async def scenario() -> None:
        future = asyncio.get_running_loop().create_future()

        scheduled = schedule_awaitable(future)

        assert scheduled is future
        future.set_result("done")
        assert await scheduled == "done"

    asyncio.run(scenario())


def test_monitor_relay_accepts_future_shutdown_waiter() -> None:
    class FakeRelay:
        def __init__(self, shutdown: asyncio.Future[None]) -> None:
            self.shutdown = shutdown

        def wait_for_shutdown(self) -> asyncio.Future[None]:
            return self.shutdown

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        shutdown = loop.create_future()
        endpoint_never_finishes = loop.create_future()
        monitor = asyncio.create_task(
            monitor_relay(FakeRelay(shutdown), [endpoint_never_finishes])
        )

        await asyncio.sleep(0)
        shutdown.set_result(None)
        await asyncio.wait_for(monitor, timeout=1)

        assert not endpoint_never_finishes.done()
        endpoint_never_finishes.cancel()
        await asyncio.gather(endpoint_never_finishes, return_exceptions=True)

    asyncio.run(scenario())

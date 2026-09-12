# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.kv_dc_relay.cli import TUNING_KEYS, WAN_TUNING_KEYS, parse_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def assert_cli_error(
    argv: list[str],
    environment: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    *expected: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(argv, environment)
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err.splitlines()[-1]
    for fragment in expected:
        assert fragment in diagnostic


def test_default_config() -> None:
    config = parse_args(["--dc-id", "dc-a"], {})

    assert config.watch_all is True
    assert config.namespaces == ()
    assert config.endpoint_prefixes == ()
    assert config.bind is None
    assert config.tuning == ()


def test_namespace_filter_selects_one_namespace() -> None:
    config = parse_args(["--dc-id", "dc-a", "--namespace-filter", "production"], {})

    assert config.namespaces == ("production",)
    assert config.watch_all is False


def test_cli_overrides_environment() -> None:
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
            "--bind",
            "[::1]:0",
            "--expected-unique-blocks",
            "128",
        ],
        {
            "DYN_DC_ID": "environment-dc",
            "DYN_RELAY_NAMESPACES": "environment-a, environment-b",
            "DYN_RELAY_ENDPOINT_PREFIXES": "environment-a.backend",
            "DYN_RELAY_BIND": "127.0.0.1:0",
            "DYN_RELAY_EXPECTED_UNIQUE_BLOCKS": "64",
        },
    )

    assert config.dc_id == "cli-dc"
    assert config.namespaces == ("cli-a", "cli-b")
    assert config.watch_all is False
    assert config.endpoint_prefixes == ("cli-a.backend", "cli-b.backend")
    assert config.bind == "[::1]:0"
    assert config.expected_unique_blocks == 128


def test_config_from_environment() -> None:
    config = parse_args(
        [],
        {
            "DYN_DC_ID": "dc-a",
            "DYN_RELAY_NAMESPACES": "prod-a, prod-b",
            "DYN_RELAY_ENDPOINT_PREFIXES": "prod-a.backend, prod-b.backend",
            "DYN_RELAY_BIND": "127.0.0.1:0",
            "DYN_RELAY_EXPECTED_UNIQUE_BLOCKS": "64",
        },
    )

    assert config.dc_id == "dc-a"
    assert config.namespaces == ("prod-a", "prod-b")
    assert config.endpoint_prefixes == ("prod-a.backend", "prod-b.backend")
    assert config.watch_all is False
    assert config.bind == "127.0.0.1:0"
    assert config.expected_unique_blocks == 64


@pytest.mark.parametrize(
    ("argv", "environment", "namespaces", "watch_all"),
    [
        pytest.param(
            ["--watch-all"],
            {"DYN_RELAY_NAMESPACES": "prod"},
            (),
            True,
            id="watch-all-over-env-namespaces",
        ),
        pytest.param(
            ["--namespaces", "prod"],
            {"DYN_RELAY_WATCH_ALL": "true"},
            ("prod",),
            False,
            id="namespaces-over-env-watch-all",
        ),
        pytest.param(
            ["--namespace-filter", "prod"],
            {"DYN_RELAY_WATCH_ALL": "true"},
            ("prod",),
            False,
            id="namespace-filter-over-env-watch-all",
        ),
    ],
)
def test_cli_scope_overrides_environment(
    argv: list[str],
    environment: dict[str, str],
    namespaces: tuple[str, ...],
    watch_all: bool,
) -> None:
    config = parse_args(["--dc-id", "dc-a", *argv], environment)
    assert config.namespaces == namespaces
    assert config.watch_all is watch_all


def test_environment_watch_all_is_normalized() -> None:
    config = parse_args(["--dc-id", "dc-a"], {"DYN_RELAY_WATCH_ALL": " YES "})
    assert config.watch_all is True
    assert config.namespaces == ()


def test_dc_id_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    assert_cli_error([], {}, capsys, "--dc-id or DYN_DC_ID is required")


@pytest.mark.parametrize(
    ("argv", "environment", "diagnostic"),
    [
        pytest.param(
            ["--watch-all", "--namespaces", "prod"],
            {},
            "are mutually exclusive",
            id="conflicting-cli-scopes",
        ),
        pytest.param(
            ["--namespace-filter", "prod", "--namespaces", "prod"],
            {},
            "are mutually exclusive",
            id="conflicting-cli-alias",
        ),
        pytest.param(
            [],
            {"DYN_RELAY_NAMESPACES": "prod", "DYN_RELAY_WATCH_ALL": "true"},
            "DYN_RELAY_NAMESPACES and DYN_RELAY_WATCH_ALL are mutually exclusive",
            id="conflicting-env-scopes",
        ),
        pytest.param(
            ["--namespaces", "prod", "--endpoint-prefix", "other.backend"],
            {},
            "endpoint prefixes must be inside the selected namespaces",
            id="prefix-outside-scope",
        ),
        pytest.param(
            [],
            {"DYN_RELAY_WATCH_ALL": "false"},
            "DYN_RELAY_WATCH_ALL=false requires DYN_RELAY_NAMESPACES",
            id="disabled-watch-all-without-scope",
        ),
        pytest.param(
            [],
            {"DYN_RELAY_WATCH_ALL": "perhaps"},
            "DYN_RELAY_WATCH_ALL must be a boolean value",
            id="invalid-env-bool",
        ),
    ],
)
def test_invalid_discovery_scopes(
    argv: list[str],
    environment: dict[str, str],
    diagnostic: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert_cli_error(["--dc-id", "dc-a", *argv], environment, capsys, diagnostic)


@pytest.mark.parametrize("source", ["cli", "environment"])
@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        pytest.param("", "non-empty values", id="empty"),
        pytest.param("prod,", "non-empty values", id="empty-element"),
        pytest.param("prod, prod", "duplicate values", id="duplicate"),
    ],
)
def test_invalid_namespace_csv(
    source: str,
    value: str,
    diagnostic: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["--namespaces", value] if source == "cli" else []
    environment = {"DYN_RELAY_NAMESPACES": value} if source == "environment" else {}
    assert_cli_error(["--dc-id", "dc-a", *argv], environment, capsys, diagnostic)


@pytest.mark.parametrize(
    ("argv", "environment", "diagnostic"),
    [
        pytest.param(
            ["--endpoint-prefix", ""], {}, "must be non-empty", id="empty-cli-prefix"
        ),
        pytest.param(
            ["--endpoint-prefix", " prod.backend "],
            {},
            "no surrounding whitespace",
            id="untrimmed-cli-prefix",
        ),
        pytest.param(
            ["--endpoint-prefix", "prod.backend", "--endpoint-prefix", "prod.backend"],
            {},
            "must not contain duplicates",
            id="duplicate-cli-prefix",
        ),
        pytest.param(
            [],
            {"DYN_RELAY_ENDPOINT_PREFIXES": "prod.backend,"},
            "non-empty values",
            id="empty-env-element",
        ),
        pytest.param(
            [],
            {"DYN_RELAY_ENDPOINT_PREFIXES": "prod.backend,prod.backend"},
            "duplicate values",
            id="duplicate-env-prefix",
        ),
    ],
)
def test_invalid_endpoint_prefixes(
    argv: list[str],
    environment: dict[str, str],
    diagnostic: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert_cli_error(["--dc-id", "dc-a", *argv], environment, capsys, diagnostic)


def test_known_tuning_is_loaded_and_unknown_environment_keys_are_ignored() -> None:
    config = parse_args(
        ["--dc-id", "dc-a"],
        {
            "DYN_RELAY_BIND": "127.0.0.1:0",
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


@pytest.mark.parametrize("key", WAN_TUNING_KEYS)
def test_wan_tuning_requires_bind(key: str, capsys: pytest.CaptureFixture[str]) -> None:
    name = f"DYN_RELAY_{key.upper()}"
    assert_cli_error(
        ["--dc-id", "dc-a"],
        {name: "1"},
        capsys,
        f"{name} requires --bind or DYN_RELAY_BIND",
    )


def test_producer_tuning_does_not_require_bind() -> None:
    config = parse_args(
        ["--dc-id", "dc-a"],
        {
            "DYN_RELAY_PUBLICATION_THRESHOLD": "7",
            "DYN_RELAY_PUBLICATION_DELAY_MS": "20",
            "DYN_RELAY_RECOVERY_ATTEMPT_TIMEOUT_MS": "1000",
        },
    )
    assert config.bind is None
    assert dict(config.tuning) == {
        "publication_threshold": 7,
        "publication_delay_ms": 20,
        "recovery_attempt_timeout_ms": 1000,
    }


@pytest.mark.parametrize("key", TUNING_KEYS)
def test_environment_tuning_rejects_zero(
    key: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = f"DYN_RELAY_{key.upper()}"
    assert_cli_error(
        ["--dc-id", "dc-a", "--bind", "127.0.0.1:0"],
        {name: "0"},
        capsys,
        f"{name}: must be a positive integer",
    )


@pytest.mark.parametrize("key", ["publication_threshold", "max_pool_streams_total"])
@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        pytest.param("-1", "must be a positive integer", id="negative"),
        pytest.param("invalid", "invalid literal for int", id="not-an-integer"),
    ],
)
def test_environment_tuning_rejects_invalid_numbers(
    key: str,
    value: str,
    diagnostic: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = f"DYN_RELAY_{key.upper()}"
    assert_cli_error(
        ["--dc-id", "dc-a", "--bind", "127.0.0.1:0"],
        {name: value},
        capsys,
        name,
        diagnostic,
    )


@pytest.mark.parametrize("source", ["cli", "environment"])
@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        pytest.param("0", "must be a positive integer", id="zero"),
        pytest.param("-1", "must be a positive integer", id="negative"),
        pytest.param("invalid", "invalid", id="not-an-integer"),
    ],
)
def test_expected_unique_blocks_rejects_invalid_numbers(
    source: str,
    value: str,
    diagnostic: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if source == "cli":
        argv = ["--expected-unique-blocks", value]
        environment = {}
        name = "--expected-unique-blocks"
    else:
        argv = []
        name = "DYN_RELAY_EXPECTED_UNIQUE_BLOCKS"
        environment = {name: value}
    assert_cli_error(["--dc-id", "dc-a", *argv], environment, capsys, name, diagnostic)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.triton.args: CLI parsing, Config defaults/validation,
and the argv -> Config -> tritonserver.Options pipeline."""

import pytest
import tritonserver

from conftest import make_cli_args_fixture
from dynamo.triton import backend_args

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Arbitrary directory other than the default backend directory
BACKEND_DIR = "/opt/tritonserver/include"

# Create Triton-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_triton_cli = make_cli_args_fixture("dynamo.triton")


def test_config_options_passed_to_server(mock_triton_cli):
    mock_triton_cli(
        "--model-repository",
        "/models",
        "--backend-directory",
        BACKEND_DIR,
        "--exit-on-error",
        "false",
        "--buffer-manager-thread-count",
        "4",
        "--backend-config",
        "tensorrt,plugins=/a.so;/b.so",
    )

    options = backend_args.parse_args().to_server_options()

    assert options == {
        "model_repository": "/models",
        "backend_directory": BACKEND_DIR,
        "exit_on_error": False,
        "buffer_manager_thread_count": 4,
        "backend_configuration": {"tensorrt": {"plugins": "/a.so;/b.so"}},
        "metrics": True,
        "log_info": True,
        "log_warn": True,
        "log_error": True,
        "server_id": "triton",
    }


@pytest.mark.parametrize(
    "raw_args, server_args, is_server_option",
    [
        # --- Dynamo worker (Config only; not Triton Options) ---
        pytest.param(
            [("--discovery-backend", "file")],
            {"discovery_backend": "file"},
            False,
            id="discovery-backend",
        ),
        pytest.param(
            [("--request-plane", "nats")],
            {"request_plane": "nats"},
            False,
            id="request-plane",
        ),
        # --- models & backends ---
        pytest.param(
            [("--model-repository", "/models")],
            {"model_repository": "/models"},
            True,
            id="model-repository",
        ),
        pytest.param(
            [("--backend-directory", BACKEND_DIR)],
            {"backend_directory": BACKEND_DIR},
            True,
            id="backend-directory",
        ),
        pytest.param(
            [("--repoagent-directory", "/custom/repoagents")],
            {"repo_agent_directory": "/custom/repoagents"},
            True,
            id="repoagent-directory",
        ),
        pytest.param(
            [("--strict-model-config", "true")],
            {"strict_model_config": True},
            True,
            id="strict-model-config",
        ),
        pytest.param(
            [("--backend-config", "tensorrt,plugins=/a.so;/b.so")],
            {"backend_configuration": {"tensorrt": {"plugins": "/a.so;/b.so"}}},
            True,
            id="backend-config",
        ),
        # --- resources & lifecycle ---
        pytest.param(
            [("--id", "my-server")], {"server_id": "my-server"}, True, id="id"
        ),
        pytest.param(
            [("--exit-on-error", "false")],
            {"exit_on_error": False},
            True,
            id="exit-on-error",
        ),
        pytest.param(
            [("--strict-readiness", "true")],
            {"strict_readiness": True},
            True,
            id="strict-readiness",
        ),
        pytest.param(
            [("--exit-timeout-secs", "45")],
            {"exit_timeout": 45},
            True,
            id="exit-timeout-secs",
        ),
        pytest.param(
            [("--buffer-manager-thread-count", "4")],
            {"buffer_manager_thread_count": 4},
            True,
            id="buffer-manager-thread-count",
        ),
        pytest.param(
            [("--model-load-thread-count", "8")],
            {"model_load_thread_count": 8},
            True,
            id="model-load-thread-count",
        ),
        pytest.param(
            [("--model-load-retry-count", "2")],
            {"model_load_retry_count": 2},
            True,
            id="model-load-retry-count",
        ),
        pytest.param(
            [("--model-namespacing", "true")],
            {"model_namespacing": True},
            True,
            id="model-namespacing",
        ),
        pytest.param(
            [("--enable-peer-access", "true")],
            {"enable_peer_access": True},
            True,
            id="enable-peer-access",
        ),
        pytest.param(
            [("--pinned-memory-pool-byte-size", "1048576")],
            {"pinned_memory_pool_size": 1048576},
            True,
            id="pinned-memory-pool-byte-size",
        ),
        pytest.param(
            [
                ("--cuda-memory-pool-byte-size", "0:2097152"),
                ("--cuda-memory-pool-byte-size", "1:1048576"),
            ],
            {"cuda_memory_pool_sizes": {0: 2097152, 1: 1048576}},
            True,
            id="cuda-memory-pool-byte-size",
        ),
        pytest.param(
            [("--min-supported-compute-capability", "7.5")],
            {"min_supported_compute_capability": 7.5},
            True,
            id="min-supported-compute-capability",
        ),
        pytest.param(
            [("--rate-limit", "EXEC_COUNT")],
            {"rate_limiter_mode": tritonserver.RateLimitMode.EXEC_COUNT},
            True,
            id="rate-limit",
        ),
        # --- logging & metrics ---
        pytest.param(
            [("--log-verbose", "2")], {"log_verbose": 2}, True, id="log-verbose"
        ),
        pytest.param(
            [("--log-file", "/tmp/triton.log")],
            {"log_file": "/tmp/triton.log"},
            True,
            id="log-file",
        ),
        pytest.param(
            [("--log-info", "false")], {"log_info": False}, True, id="log-info"
        ),
        pytest.param(
            [("--log-warning", "false")], {"log_warn": False}, True, id="log-warning"
        ),
        pytest.param(
            [("--log-error", "false")], {"log_error": False}, True, id="log-error"
        ),
        pytest.param(
            [("--log-format", "ISO8601")],
            {"log_format": tritonserver.LogFormat.ISO8601},
            True,
            id="log-format",
        ),
        pytest.param(
            [("--allow-metrics", "true")], {"metrics": True}, True, id="allow-metrics"
        ),
        pytest.param(
            [("--allow-gpu-metrics", "true")],
            {"gpu_metrics": True},
            True,
            id="allow-gpu-metrics",
        ),
        pytest.param(
            [("--allow-cpu-metrics", "true")],
            {"cpu_metrics": True},
            True,
            id="allow-cpu-metrics",
        ),
        pytest.param(
            [("--metrics-interval-ms", "1000")],
            {"metrics_interval": 1000},
            True,
            id="metrics-interval-ms",
        ),
        pytest.param(
            [("--metrics-config", "prometheus,counter_latencies=true")],
            {"metrics_configuration": {"prometheus": {"counter_latencies": "true"}}},
            True,
            id="metrics-config",
        ),
        # --- caching & policies ---
        pytest.param(
            [("--cache-config", "local,size=1048576")],
            {"cache_config": {"local": {"size": "1048576"}}},
            True,
            id="cache-config",
        ),
        pytest.param(
            [("--cache-directory", "/opt/tritonserver/caches")],
            {"cache_directory": "/opt/tritonserver/caches"},
            True,
            id="cache-directory",
        ),
        pytest.param(
            [("--host-policy", "policy0,numa-node=0")],
            {"host_policies": {"policy0": {"numa-node": "0"}}},
            True,
            id="host-policy",
        ),
    ],
)
def test_argv_flows_through_parse_args_into_server(
    mock_triton_cli, raw_args, server_args, is_server_option
):
    """End to end per argument: argv -> parse_args -> to_server_options (the
    kwargs init_worker feeds to tritonserver.Server) or, for Dynamo-only options,
    the Config attribute."""

    def format_args(raw_args, arg_format):
        argv: list[str] = []
        for flag, value in raw_args:
            if value is None:
                argv.append(flag)
            elif arg_format == "split":
                argv.extend([flag, value])
            else:
                argv.append(f"{flag}={value}")
        return argv

    for arg_format in ("split", "equals"):
        argv = format_args(raw_args, arg_format)
        # --model-repository is always required
        if not any(flag == "--model-repository" for flag, _ in raw_args):
            argv = ["--model-repository", "/models", *argv]

        mock_triton_cli(*argv)

        config = backend_args.parse_args()
        assert isinstance(config, backend_args.Config)

        if is_server_option:
            options = config.to_server_options()
            for key, value in server_args.items():
                assert options[key] == value, f"{key} via {arg_format}"
        else:
            for key, value in server_args.items():
                assert getattr(config, key) == value, f"{key} via {arg_format}"


def test_parse_args_applies_config_defaults(mock_triton_cli):
    mock_triton_cli("--model-repository", "/models")
    config = backend_args.parse_args()

    assert isinstance(config, backend_args.Config)
    assert config.model_repository == "/models"
    assert config.log_info is True
    assert config.log_warn is True
    assert config.log_error is True
    assert config.discovery_backend == "etcd"
    assert config.request_plane == "tcp"


def test_parse_args_defaults_model_repository(mock_triton_cli):
    """--model-repository defaults to /models when the user doesn't supply one,
    so `python -m dynamo.triton` works out of the box against the canonical
    container mount point."""
    mock_triton_cli()

    config = backend_args.parse_args()

    assert config.model_repository == "/models"


@pytest.mark.parametrize(
    "argv",
    [
        ["--log-format", "BOGUS"],
        ["--rate-limit", "BOGUS"],
        ["--allow-metrics=maybe"],
    ],
    ids=["log-format", "rate-limit", "allow-metrics"],
)
def test_parse_args_rejects_invalid_values(mock_triton_cli, argv):
    """The _enum_arg/_bool_arg converters reject bad values; argparse turns the
    ArgumentTypeError into a usage error and exits with code 2."""
    mock_triton_cli("--model-repository", "/models", *argv)

    with pytest.raises(SystemExit):
        backend_args.parse_args()


@pytest.mark.parametrize(
    "metrics_flags, expected_options",
    [
        (
            [
                "--allow-metrics=false",
                "--allow-gpu-metrics=true",
                "--allow-cpu-metrics=true",
            ],
            {"metrics": False, "gpu_metrics": False, "cpu_metrics": False},
        ),
        (["--allow-metrics=false"], {"metrics": False}),
        (
            [
                "--allow-metrics=true",
                "--allow-gpu-metrics=true",
                "--allow-cpu-metrics=false",
            ],
            {"metrics": True, "gpu_metrics": True, "cpu_metrics": False},
        ),
        (
            ["--allow-gpu-metrics=true", "--allow-cpu-metrics=true"],
            {"metrics": True, "gpu_metrics": True, "cpu_metrics": True},
        ),
    ],
    ids=[
        "metrics-off-masks-explicit",
        "metrics-off-strips-unset",
        "metrics-on-keeps-explicit",
        "metrics-unset-keeps-explicit",
    ],
)
def test_metrics_flags_flow_through_to_server_options(
    mock_triton_cli, metrics_flags, expected_options
):
    """--allow-metrics/-gpu/-cpu flow through validate() and to_server_options:
    metrics off masks explicit gpu/cpu off, metrics on/unset keeps them, and unset
    options are stripped so the binding applies its own default."""
    mock_triton_cli("--model-repository", "/models", *metrics_flags)

    options = backend_args.parse_args().to_server_options()

    for key in ("metrics", "gpu_metrics", "cpu_metrics"):
        if key in expected_options:
            assert (
                options[key] is expected_options[key]
            ), f"Expected {key} to be {expected_options[key]}, got {options.get(key)}"
        else:
            assert (
                key not in options
            ), f"Expected {key} to be stripped (unset), got {options.get(key)}"

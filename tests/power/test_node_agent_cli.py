# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the node_agent command line (no EPD, no NVML needed)."""

import logging
import os

import pytest

from dynamo.power import config as power_config
from dynamo.power.config import MetricsUrl, PowerControlConfigError
from dynamo.power.node_agent_cli import (
    MIN_TGP_REFUSAL,
    build_parser,
    resolve_node_name,
    settings_from_args,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

REQUIRED = [
    "--pool-name",
    "shared",
    "--rank",
    "0",
    "--nodelist",
    "hecate0001,hecate0002",
    "--log-dir",
    "logs/power_control/shared",
    "--gpu-indices",
    "3,0,1,2",
    "--mode",
    "nupd",
    "--budget-w-per-gpu",
    "1500",
    "--fom",
    "tokens_per_gpu",
]
FRONTEND = "--metrics-url=frontend=http://f:8000/metrics"


def parse(*extra):
    return build_parser().parse_args([*REQUIRED, *extra])


def test_parser_defaults_come_from_the_config_constants():
    args = parse(FRONTEND)
    assert args.interval_s == power_config.DEFAULT_INTERVAL_S
    assert args.perf_threshold == power_config.DEFAULT_PERF_THRESHOLD
    assert args.ubpc_buffer == power_config.DEFAULT_NODE_UBPC_BUFFER
    assert args.warmup_policy == power_config.DEFAULT_WARMUP_POLICY
    assert (args.min_tgp_w, args.max_tgp_w, args.gpus_per_node) == (None, None, None)
    with pytest.raises(SystemExit):
        build_parser().parse_args(REQUIRED[:-2])  # --fom is required
    with pytest.raises(SystemExit):
        parse("--mode", "turbo")


def test_settings_from_args_resolves_every_flag():
    settings = settings_from_args(
        parse(
            FRONTEND,
            "--metrics-url",
            "worker=http://w1:9090/metrics,http://w2:9090/metrics",
            "--interval-s",
            "10",
            "--perf-threshold",
            "0.05",
            "--ubpc-buffer",
            "2",
            "--max-tgp-w",
            "2300",
            "--warmup-policy",
            "fixed",
            "--gpus-per-node",
            "4",
        )
    )
    assert settings.pool_name == "shared"
    assert settings.rank == 0 and settings.is_controller
    assert settings.nodelist == ("hecate0001", "hecate0002")
    assert os.path.isabs(settings.log_dir)
    assert settings.log_dir.endswith(os.path.join("logs", "power_control", "shared"))
    assert settings.gpu_indices == (0, 1, 2, 3)
    assert settings.fom.name == "tokens_per_gpu"
    assert (settings.interval_s, settings.perf_threshold) == (10, 0.05)
    assert (settings.ubpc_buffer, settings.max_tgp_w) == (2.0, 2300)
    assert settings.warmup_policy == "fixed"
    assert settings.gpus_per_node == 4
    assert settings.metrics_urls == (
        MetricsUrl("http://f:8000/metrics", "frontend"),
        MetricsUrl("http://w1:9090/metrics", "worker"),
        MetricsUrl("http://w2:9090/metrics"),
    )
    assert settings.frontend_metrics_urls == ("http://f:8000/metrics",)
    assert settings.worker_metrics_urls == ("http://w1:9090/metrics",)
    assert settings.pool_gpu_count == 8


def test_pool_gpu_count_flag_overrides_the_nodes_times_gpus_estimate():
    settings = settings_from_args(parse(FRONTEND, "--gpus-per-node", "4"))
    assert settings.pool_gpu_count == 8
    settings = settings_from_args(
        parse(FRONTEND, "--gpus-per-node", "4", "--pool-gpu-count", "6")
    )
    assert settings.pool_gpu_count == 6
    with pytest.raises(PowerControlConfigError, match="pool_gpu_count"):
        settings_from_args(parse(FRONTEND, "--pool-gpu-count", "0"))


def test_rank_is_an_int_checked_against_the_roster():
    settings = settings_from_args(parse("--rank", "1"))
    assert settings.rank == 1 and not settings.is_controller
    with pytest.raises(PowerControlConfigError, match="outside the pool roster"):
        settings_from_args(parse("--rank", "2"))
    with pytest.raises(SystemExit):
        parse("--rank", "zero")


def test_rank_0_without_a_metrics_url_is_refused():
    with pytest.raises(PowerControlConfigError, match="metrics-url"):
        settings_from_args(parse())


def test_min_tgp_w_is_refused_and_points_at_the_applier_floor():
    with pytest.raises(PowerControlConfigError, match="not enforced by EPD") as info:
        settings_from_args(parse(FRONTEND, "--min-tgp-w", "1700"))
    assert str(info.value) == MIN_TGP_REFUSAL
    assert "--min-w-per-gpu" in MIN_TGP_REFUSAL
    assert "NUPD_SKIP_HISTORY" in MIN_TGP_REFUSAL


def test_scoped_fom_without_a_matching_tag_is_refused():
    with pytest.raises(PowerControlConfigError, match="worker=<url>"):
        settings_from_args(
            parse(
                FRONTEND,
                "--fom",
                "prom:vllm:prompt_tokens_total:scope=pool_workers:counter:sum",
            )
        )


def test_resolve_node_name_prefers_slurmd_nodename_and_warns_on_roster_mismatch(
    caplog,
):
    settings = settings_from_args(parse(FRONTEND))
    assert resolve_node_name(settings, {}) == "hecate0001"
    assert (
        resolve_node_name(settings, {"SLURMD_NODENAME": "hecate0001"}) == "hecate0001"
    )
    with caplog.at_level(logging.WARNING):
        assert resolve_node_name(settings, {"SLURMD_NODENAME": "hecate0002"}) == (
            "hecate0002"
        )
    assert "regardless of roster order" in caplog.text
    with pytest.raises(PowerControlConfigError, match="not in the pool roster"):
        resolve_node_name(settings, {"SLURMD_NODENAME": "elsewhere"})

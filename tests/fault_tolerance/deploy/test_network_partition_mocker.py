# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mocker-backed network-partition fault test. NetworkPolicy +
# conntrack flush sever the Frontend<->decode TCP request plane for
# ~20s; the policy is lifted automatically when the partition's
# duration expires. Validates the framework's NetworkPartition event
# and the report's pre/post bucketing for transient partitions.
# Companion to test_network_partition_vllm.py.

import pytest

from tests.fault_tolerance.deploy.checks import LoadStopped, MaxErrors, MinRequests
from tests.fault_tolerance.deploy.events import (
    NetworkPartition,
    StartLoad,
    StopLoad,
    Wait,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    FaultToleranceReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_network_partition_mocker(
    namespace, image, skip_service_restart, storage_class
):
    """Block Frontend -> decode mid-load; verify it surfaces in the report."""
    spec = DeploymentSpec.from_backend("mocker", "agg")
    spec["decode"].set_arg(
        "--planner-profile-data",
        "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D",
    )
    # Shorten the frontend's per-request inactivity timeout. Default is
    # unset (= no timeout, frontend buffers forever — masks the partition
    # at aiperf level). Setting this to 5s makes push_router give up after
    # 5s of silence from the worker and quarantine the instance, so
    # in-flight requests fail fast and follow-on requests see 5xx.
    spec["Frontend"].set_env_var("DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS", "5")
    served_model = spec["decode"].model

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(
                load_config=LoadConfig(
                    model_name=served_model,
                    tokenizer=served_model,
                    duration_minutes=2,
                    concurrency=4,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                    # Tighten aiperf's request timeout so even if the
                    # frontend buffers, in-flight requests give up
                    # comfortably before the 20s partition lifts.
                    request_timeout_seconds=10,
                    # Force a fresh TCP connection per request so the
                    # NetworkPolicy applied mid-load actually rejects
                    # new flows instead of being bypassed by an existing
                    # conntrack-tracked socket.
                    connection_reuse_strategy="never",
                )
            ),
            Wait(duration=15),
            # Transient: partition fires, holds for 20s, heals — all
            # inside the single event. No separate "remove" step needed.
            NetworkPartition(source="Frontend", target="decode", duration=20),
            Wait(duration=10),  # observe recovery after the partition heals
            StopLoad(),
        ],
        checks=[
            LoadStopped(),
            MinRequests(min_count=20),
            # With conntrack flush the partition severs established
            # sockets and aiperf hammers the closed connection for 20s.
            # The point of the test is to demonstrate the rupture and
            # recovery, not to bound errors precisely — keep the cap
            # permissive.
            MaxErrors(max_errors=1_000_000),
        ],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_network_partition_mocker",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )

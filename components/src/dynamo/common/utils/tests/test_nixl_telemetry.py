# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.nixl_telemetry."""

import pytest

from dynamo.common.utils.nixl_telemetry import (
    MAX_COLOCATED_NIXL_EXPORTERS,
    MAX_PORT,
    derive_nixl_prometheus_port,
    nixl_prometheus_base_port,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# The ports the operator injects into a worker container today.
OPERATOR_ENV = {
    "NIXL_TELEMETRY_ENABLE": "y",
    "NIXL_TELEMETRY_EXPORTER": "prometheus",
    "NIXL_TELEMETRY_PROMETHEUS_PORT": "19090",
    "DYN_SYSTEM_PORT": "9090",
    "DYN_FORWARDPASS_METRIC_PORT": "20380",
}


class TestDeriveNixlPrometheusPort:
    def test_colocated_ranks_never_share_an_exporter_port(self):
        base = int(OPERATOR_ENV["NIXL_TELEMETRY_PROMETHEUS_PORT"])
        ports = {
            derive_nixl_prometheus_port(base, rank, env=OPERATOR_ENV)
            for rank in range(MAX_COLOCATED_NIXL_EXPORTERS)
        }
        assert len(ports) == MAX_COLOCATED_NIXL_EXPORTERS

    def test_derived_ports_stay_inside_the_reserved_range(self):
        """A port past the reserved range binds where nothing scrapes it."""
        base = int(OPERATOR_ENV["NIXL_TELEMETRY_PROMETHEUS_PORT"])
        ports = [
            derive_nixl_prometheus_port(base, rank, env=OPERATOR_ENV)
            for rank in range(MAX_COLOCATED_NIXL_EXPORTERS)
        ]
        assert min(ports) >= base
        assert max(ports) <= base + MAX_COLOCATED_NIXL_EXPORTERS - 1

    def test_rank_beyond_the_reserved_range_is_rejected(self):
        with pytest.raises(ValueError, match="outside the reserved"):
            derive_nixl_prometheus_port(
                19090, MAX_COLOCATED_NIXL_EXPORTERS, env=OPERATOR_ENV
            )

    @pytest.mark.parametrize(
        "env_name", ["DYN_SYSTEM_PORT", "DYN_FORWARDPASS_METRIC_PORT"]
    )
    @pytest.mark.parametrize("local_rank", [0, 1])
    def test_base_that_would_overlap_another_listener_is_rejected(
        self, env_name, local_rank
    ):
        # One below the listener's own base: rank 0 lands just clear of it and
        # only later ranks collide, so rejecting rank 0 is what stops the pod
        # from starting one scheduler and failing the rest.
        overlapping_base = int(OPERATOR_ENV[env_name]) - 1
        with pytest.raises(ValueError, match=env_name):
            derive_nixl_prometheus_port(overlapping_base, local_rank, env=OPERATOR_ENV)

    @pytest.mark.parametrize("local_rank", [0, 1])
    def test_base_too_high_for_the_reserved_range_is_rejected(self, local_rank):
        # Rank 0 fits at MAX_PORT on its own; rejecting it is what stops the pod
        # from starting one scheduler and failing every rank after it.
        with pytest.raises(ValueError, match="exceeds the maximum port"):
            derive_nixl_prometheus_port(MAX_PORT, local_rank, env=OPERATOR_ENV)


class TestNixlPrometheusBasePort:
    def test_operator_defaults_are_recognized(self):
        assert nixl_prometheus_base_port(OPERATOR_ENV) == 19090

    @pytest.mark.parametrize(
        "override",
        [
            {"NIXL_TELEMETRY_ENABLE": "n"},
            {"NIXL_TELEMETRY_ENABLE": ""},
            {"NIXL_TELEMETRY_EXPORTER": "file"},
        ],
    )
    def test_disabled_telemetry_has_no_base_port(self, override):
        assert nixl_prometheus_base_port({**OPERATOR_ENV, **override}) is None

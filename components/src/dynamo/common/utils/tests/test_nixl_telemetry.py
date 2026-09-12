# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.nixl_telemetry."""

import pytest

from dynamo.common.utils.nixl_telemetry import (
    DEFAULT_NIXL_PROMETHEUS_PORT,
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
    def test_base_that_would_overlap_another_listener_is_rejected(self, env_name):
        # One below the listener's own base: rank 0 lands just clear of it and
        # only later ranks collide, so rejecting rank 0 is what stops the pod
        # from starting one scheduler and failing the rest.
        overlapping_base = int(OPERATOR_ENV[env_name]) - 1
        with pytest.raises(ValueError, match=env_name):
            derive_nixl_prometheus_port(overlapping_base, 0, env=OPERATOR_ENV)

    def test_base_too_high_for_the_reserved_range_is_rejected(self):
        # Rank 0 fits at MAX_PORT on its own; rejecting it is what stops the pod
        # from starting one scheduler and failing every rank after it.
        with pytest.raises(ValueError, match="exceeds the maximum port"):
            derive_nixl_prometheus_port(MAX_PORT, 0, env=OPERATOR_ENV)

    def test_a_narrower_launch_is_measured_against_its_own_width(self):
        """The pod reserves one port per rank it places, not the maximum."""
        base = MAX_PORT - 3
        ports = [
            derive_nixl_prometheus_port(base, rank, max_ranks=4, env=OPERATOR_ENV)
            for rank in range(4)
        ]
        assert ports == [MAX_PORT - 3, MAX_PORT - 2, MAX_PORT - 1, MAX_PORT]

    def test_a_rank_outside_a_narrower_launch_is_rejected(self):
        with pytest.raises(ValueError, match="outside the reserved"):
            derive_nixl_prometheus_port(19090, 4, max_ranks=4, env=OPERATOR_ENV)

    def test_a_launch_wider_than_the_pod_reserves_is_rejected(self):
        """Ranks past the declared container ports would be scraped by nobody."""
        with pytest.raises(ValueError, match="cannot each be given"):
            derive_nixl_prometheus_port(
                19090,
                0,
                max_ranks=MAX_COLOCATED_NIXL_EXPORTERS + 1,
                env=OPERATOR_ENV,
            )


class TestNixlPrometheusBasePort:
    def test_operator_defaults_are_recognized(self):
        assert nixl_prometheus_base_port(OPERATOR_ENV) == 19090

    @pytest.mark.parametrize(
        "enabled_value", ["y", "1", "yes", "on", "true", "enable", "TRUE"]
    )
    def test_nixl_truthy_token_is_recognized(self, enabled_value):
        env = {**OPERATOR_ENV, "NIXL_TELEMETRY_ENABLE": enabled_value}
        assert nixl_prometheus_base_port(env) == 19090

    def test_unset_exporter_is_not_prometheus(self):
        env = dict(OPERATOR_ENV)
        del env["NIXL_TELEMETRY_EXPORTER"]
        assert nixl_prometheus_base_port(env) is None

    def test_unset_port_uses_nixl_default(self):
        env = dict(OPERATOR_ENV)
        del env["NIXL_TELEMETRY_PROMETHEUS_PORT"]
        assert nixl_prometheus_base_port(env) == DEFAULT_NIXL_PROMETHEUS_PORT

    def test_hexadecimal_port_is_recognized(self):
        env = {**OPERATOR_ENV, "NIXL_TELEMETRY_PROMETHEUS_PORT": "0x4A92"}
        assert nixl_prometheus_base_port(env) == 19090

    @pytest.mark.parametrize(
        "port_value", ["abc", "0", "99999", " 9090", "+9090", "9_090"]
    )
    def test_unusable_port_is_not_recognized(self, port_value):
        env = {**OPERATOR_ENV, "NIXL_TELEMETRY_PROMETHEUS_PORT": port_value}
        assert nixl_prometheus_base_port(env) is None

    def test_oversized_decimal_port_is_not_recognized(self):
        # This exceeds Python's default integer-string conversion limit.
        env = {**OPERATOR_ENV, "NIXL_TELEMETRY_PROMETHEUS_PORT": "9" * 5000}
        assert nixl_prometheus_base_port(env) is None

    @pytest.mark.parametrize(
        "override",
        [
            {"NIXL_TELEMETRY_ENABLE": "n"},
            {"NIXL_TELEMETRY_ENABLE": ""},
            {"NIXL_TELEMETRY_ENABLE": " y"},
            {"NIXL_TELEMETRY_ENABLE": "y "},
            {"NIXL_TELEMETRY_EXPORTER": "file"},
            {"NIXL_TELEMETRY_EXPORTER": "PROMETHEUS"},
            {"NIXL_TELEMETRY_EXPORTER": "prometheus "},
        ],
    )
    def test_inactive_configuration_has_no_base_port(self, override):
        assert nixl_prometheus_base_port({**OPERATOR_ENV, **override}) is None

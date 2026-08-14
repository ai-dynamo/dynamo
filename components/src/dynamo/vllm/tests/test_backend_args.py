# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for vLLM backend arguments.

[gluo NOTE] currently the test cover is being added as part of multimodal related test coverage,
need to add more tests to cover different code paths of DynamoVllmConfig.
"""

import pytest

from dynamo.vllm.backend_args import DisaggregationMode, DynamoVllmConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def create_config() -> DynamoVllmConfig:
    """
    Create a config with default values. This is needed as the config
    is instantiated by the argparse parser with dynamically generated fields,
    so we need to create a config with default values manually if not using
    from_cli_args() method.

    All multimodal flags are False, disaggregation mode is None.
    Returns:
        DynamoVllmConfig: A config with default values.
    """
    config = DynamoVllmConfig()
    config.disaggregation_mode = None
    config.multimodal_worker = False
    config.multimodal_encode_worker = False
    config.multimodal_decode_worker = False
    config.enable_multimodal = False
    config.embedding_worker = False
    config.embedding_worker_processes = 1
    config.headless = False
    config.benchmark_mode = None
    config.use_vllm_tokenizer = False
    config.frontend_decoding = False
    return config


class TestResolveDisaggregationModeFromLegacyMultimodalFlags:
    """
    Test suite for resolving disaggregation mode when legacy multimodal flags are set.
    """

    def test_pd_alias_resolves_to_aggregated(self):
        config = create_config()
        config.disaggregation_mode = "pd"
        config.is_prefill_worker = False
        config.is_decode_worker = False

        config._resolve_disaggregation_mode()

        assert config.disaggregation_mode == DisaggregationMode.AGGREGATED

    @pytest.mark.parametrize(
        "mode",
        [
            None,  # Not specified
            DisaggregationMode.AGGREGATED,
            # DisaggregationMode.PREFILL, # test in 'test_prefill_worker' below
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_agg_worker(self, mode):
        config = create_config()
        config.disaggregation_mode = mode
        config.multimodal_worker = True
        with pytest.warns(DeprecationWarning):
            if mode is None or mode == DisaggregationMode.AGGREGATED:
                config._resolve_disaggregation_model_from_legacy_multimodal_flags()
                assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
            else:
                with pytest.raises(ValueError):
                    config._resolve_disaggregation_model_from_legacy_multimodal_flags()

    # special case of 'test_agg_worker' above, test the prefill worker case
    def test_prefill_worker(self):
        config = create_config()
        config.disaggregation_mode = DisaggregationMode.PREFILL
        config.multimodal_worker = True
        with pytest.warns(DeprecationWarning):
            config._resolve_disaggregation_model_from_legacy_multimodal_flags()
            assert config.disaggregation_mode == DisaggregationMode.PREFILL

    @pytest.mark.parametrize(
        "mode",
        [
            None,  # Not specified
            DisaggregationMode.AGGREGATED,
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_encode_worker(self, mode):
        config = create_config()
        config.disaggregation_mode = mode
        config.multimodal_encode_worker = True
        with pytest.warns(DeprecationWarning):
            if mode is None or mode == DisaggregationMode.ENCODE:
                config._resolve_disaggregation_model_from_legacy_multimodal_flags()
                assert config.disaggregation_mode == DisaggregationMode.ENCODE
            else:
                with pytest.raises(ValueError):
                    config._resolve_disaggregation_model_from_legacy_multimodal_flags()

    @pytest.mark.parametrize(
        "mode",
        [
            None,  # Not specified
            DisaggregationMode.AGGREGATED,
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_decode_worker(self, mode):
        config = create_config()
        config.disaggregation_mode = mode
        config.multimodal_decode_worker = True
        with pytest.warns(DeprecationWarning):
            if mode is None or mode == DisaggregationMode.DECODE:
                config._resolve_disaggregation_model_from_legacy_multimodal_flags()
                assert config.disaggregation_mode == DisaggregationMode.DECODE
            else:
                with pytest.raises(ValueError):
                    config._resolve_disaggregation_model_from_legacy_multimodal_flags()


class TestEmbeddingWorkerExclusivity:
    """--embedding-worker rejects combinations that don't make sense for a
    pooling engine (non-aggregated disagg, multimodal, benchmark-mode).
    """

    def test_baseline_aggregated_is_accepted(self):
        config = create_config()
        config.embedding_worker = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        # Must not raise.
        config._validate_embedding_worker_exclusivity()

    @pytest.mark.parametrize(
        "mode",
        [
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_non_aggregated_disagg_rejected(self, mode):
        config = create_config()
        config.embedding_worker = True
        config.disaggregation_mode = mode
        with pytest.raises(ValueError, match="disaggregation-mode=agg"):
            config._validate_embedding_worker_exclusivity()

    def test_multimodal_combination_rejected(self):
        config = create_config()
        config.embedding_worker = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.enable_multimodal = True
        with pytest.raises(ValueError, match="multimodal"):
            config._validate_embedding_worker_exclusivity()

    def test_benchmark_mode_rejected(self):
        # The bug surfaced by review: --embedding-worker + --benchmark-mode
        # silently injected InstrumentedScheduler (a generation scheduler) on
        # the pooling engine. Validation must reject the combination upfront.
        config = create_config()
        config.embedding_worker = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.benchmark_mode = "agg"
        with pytest.raises(ValueError, match="benchmark-mode"):
            config._validate_embedding_worker_exclusivity()

    def test_no_op_when_embedding_worker_disabled(self):
        # Validator must not punish callers that have benchmark_mode set
        # but are not running an embedding worker.
        config = create_config()
        config.embedding_worker = False
        config.benchmark_mode = "agg"
        config._validate_embedding_worker_exclusivity()


class TestValidateCustomEncoder:
    """--custom-encoder-class is an in-process, aggregated-only multimodal
    component, so validation must require --enable-multimodal and reject any
    non-aggregated disaggregation mode (where the custom-encoder branch is
    never reached) up front.
    """

    def test_requires_enable_multimodal(self):
        # Without the gate the custom encoder processes images while multimodal
        # is disabled, bypassing the normal multimodal enable check.
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.enable_multimodal = False
        with pytest.raises(ValueError, match="enable-multimodal"):
            config._validate_custom_encoder()

    @pytest.mark.parametrize(
        "mode",
        [
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_non_aggregated_mode_rejected(self, mode):
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.enable_multimodal = True
        config.disaggregation_mode = mode
        with pytest.raises(ValueError, match="agg"):
            config._validate_custom_encoder()

    def test_use_vllm_tokenizer_rejected(self):
        # --use-vllm-tokenizer routes to text mode, which never invokes the
        # custom encoder, so the encoder would load but sit unused. Reject it.
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.enable_multimodal = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.use_vllm_tokenizer = True
        with pytest.raises(ValueError, match="use-vllm-tokenizer"):
            config._validate_custom_encoder()

    @pytest.mark.parametrize(
        "role_flag",
        [
            "multimodal_worker",
            "multimodal_encode_worker",
            "multimodal_decode_worker",
        ],
    )
    def test_legacy_multimodal_role_rejected(self, role_flag):
        # The custom encoder is its own aggregated multimodal path; combining it
        # with a legacy multimodal role flag sets up two conflicting multimodal
        # paths (and --multimodal-worker resolves to agg, slipping past the
        # disaggregation-mode check), so reject the combination up front.
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.enable_multimodal = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        setattr(config, role_flag, True)
        with pytest.raises(ValueError, match="legacy multimodal role flags"):
            config._validate_custom_encoder()

    def test_frontend_decoding_rejected(self):
        # --frontend-decoding pre-decodes images to tensors; the custom encoder
        # consumes URLs, so the decoded inputs would fail extraction. Reject it.
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.enable_multimodal = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.frontend_decoding = True
        with pytest.raises(ValueError, match="frontend-decoding"):
            config._validate_custom_encoder()

    def test_accepted_when_agg_and_multimodal(self):
        config = create_config()
        config.custom_encoder_class = "my_pkg.MyEncoder"
        config.enable_multimodal = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        # Must not raise.
        config._validate_custom_encoder()

    def test_no_op_when_unset(self):
        # No custom encoder → validator must not touch unrelated configs.
        config = create_config()
        config.custom_encoder_class = None
        config.enable_multimodal = False
        config._validate_custom_encoder()


class TestValidateBenchmarkConfig:
    @pytest.mark.parametrize(
        "axis",
        [
            "benchmark_prefill_kv_read_granularity",
            "benchmark_prefill_batch_granularity",
        ],
    )
    @pytest.mark.parametrize("value", [0, -1, 1025])
    def test_rejects_out_of_range_grid_axis(self, axis, value):
        config = create_config()
        config.benchmark_mode = "prefill"
        setattr(config, axis, value)

        with pytest.raises(ValueError, match="must be between 1 and 1024"):
            config._validate_benchmark_config()

    def test_caps_prefill_cartesian_grid(self):
        config = create_config()
        config.benchmark_mode = "prefill"
        config.benchmark_prefill_granularity = 16
        config.benchmark_prefill_kv_read_granularity = 16
        config.benchmark_prefill_batch_granularity = 16
        config._validate_benchmark_config()

        config.benchmark_prefill_batch_granularity = 17
        with pytest.raises(ValueError, match="requests 4352 grid points"):
            config._validate_benchmark_config()

    def test_decode_mode_ignores_inactive_prefill_grid(self):
        config = create_config()
        config.benchmark_mode = "decode"
        config.benchmark_prefill_granularity = 1024
        config.benchmark_prefill_kv_read_granularity = 1024
        config.benchmark_prefill_batch_granularity = 1024

        config._validate_benchmark_config()


class TestEmbeddingWorkerProcesses:
    @pytest.fixture(autouse=True)
    def clear_port_and_failover_env(self, monkeypatch):
        """Keep validation tests independent of the launcher environment."""
        for env_name in (
            "DYN_SYSTEM_PORT",
            "DYN_TCP_RPC_PORT",
            "DYN_FORWARDPASS_METRIC_PORT",
            "NIXL_TELEMETRY_ENABLE",
            "NIXL_TELEMETRY_EXPORTER",
            "NIXL_TELEMETRY_PROMETHEUS_PORT",
            "DYN_VLLM_EMBEDDING_PROCESS_ROLE",
            "ENGINE_ID",
            "CONTAINER_NAME",
            "FAILOVER_LOCK_PATH",
        ):
            monkeypatch.delenv(env_name, raising=False)

    def test_default_single_process_is_accepted(self):
        config = create_config()
        config._validate_embedding_worker_processes()

    def test_multiple_processes_require_embedding_worker(self):
        config = create_config()
        config.embedding_worker_processes = 4
        with pytest.raises(ValueError, match="requires --embedding-worker"):
            config._validate_embedding_worker_processes()

    def test_multiple_embedding_processes_are_accepted(self):
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 8
        config._validate_embedding_worker_processes()

    @pytest.mark.parametrize("count", [0, -1])
    def test_process_count_must_be_positive(self, count):
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = count
        with pytest.raises(ValueError, match="at least 1"):
            config._validate_embedding_worker_processes()

    def test_headless_is_rejected(self):
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config.headless = True
        with pytest.raises(ValueError, match="--headless"):
            config._validate_embedding_worker_processes()

    def test_system_port_range_that_overflows_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "65534")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        with pytest.raises(ValueError, match="exceeds the maximum port 65535"):
            config._validate_embedding_worker_processes()

    def test_system_port_range_that_fits_is_accepted(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "19401")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config._validate_embedding_worker_processes()

    def test_system_port_range_collision_with_fpm_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "20379")
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "20380")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3

        with pytest.raises(
            ValueError,
            match=(
                "DYN_SYSTEM_PORT reserves 20379-20381, while "
                "DYN_FORWARDPASS_METRIC_PORT reserves 20380"
            ),
        ):
            config._validate_embedding_worker_processes()

    def test_system_port_range_adjacent_to_fpm_is_accepted(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "20377")
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "20380")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3
        config._validate_embedding_worker_processes()

    def test_enabled_nixl_prometheus_collision_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "19089")
        monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "y")
        monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
        monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3

        with pytest.raises(
            ValueError,
            match="NIXL_TELEMETRY_PROMETHEUS_PORT reserves 19090",
        ):
            config._validate_embedding_worker_processes()

    def test_disabled_nixl_prometheus_port_is_not_reserved(self, monkeypatch):
        monkeypatch.setenv("DYN_SYSTEM_PORT", "19089")
        monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "n")
        monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
        monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3
        config._validate_embedding_worker_processes()

    def test_active_non_system_listeners_cannot_overlap(self, monkeypatch):
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "19090")
        monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "y")
        monkeypatch.setenv("NIXL_TELEMETRY_EXPORTER", "prometheus")
        monkeypatch.setenv("NIXL_TELEMETRY_PROMETHEUS_PORT", "19090")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3

        with pytest.raises(
            ValueError,
            match=(
                "DYN_FORWARDPASS_METRIC_PORT reserves 19090, while "
                "NIXL_TELEMETRY_PROMETHEUS_PORT reserves 19090"
            ),
        ):
            config._validate_embedding_worker_processes()

    def test_fixed_tcp_rpc_port_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DYN_TCP_RPC_PORT", "25000")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config.request_plane = "tcp"

        with pytest.raises(ValueError, match="DYN_TCP_RPC_PORT cannot be fixed"):
            config._validate_embedding_worker_processes()

    def test_fixed_tcp_rpc_port_is_ignored_for_nats(self, monkeypatch):
        monkeypatch.setenv("DYN_TCP_RPC_PORT", "25000")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config.request_plane = "nats"
        config._validate_embedding_worker_processes()

    def test_intra_pod_failover_is_rejected(self, monkeypatch):
        monkeypatch.setenv("ENGINE_ID", "1")
        monkeypatch.setenv("CONTAINER_NAME", "engine-1")
        monkeypatch.setenv("FAILOVER_LOCK_PATH", "/shared/failover.lock")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4

        with pytest.raises(ValueError, match="intra-pod failover"):
            config._validate_embedding_worker_processes()

    def test_inter_pod_failover_marker_is_not_rejected(self, monkeypatch):
        monkeypatch.setenv("ENGINE_ID", "1")
        monkeypatch.setenv("CONTAINER_NAME", "main")
        monkeypatch.setenv("FAILOVER_LOCK_PATH", "/shared/failover.lock")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config._validate_embedding_worker_processes()

    @pytest.mark.parametrize("raw", ["-1", "0", "", "not-a-port"])
    def test_disabled_or_unparseable_system_port_skips_range_check(
        self, monkeypatch, raw
    ):
        """No range is reserved unless the parent asked for a real port."""
        monkeypatch.setenv("DYN_SYSTEM_PORT", raw)
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4096
        config._validate_embedding_worker_processes()

    def test_child_skips_phantom_system_port_overflow(self, monkeypatch):
        """A child near the top of the port space must not validate base+i..base+i+N-1.

        Parent DYN_SYSTEM_PORT=65533 with N=3 claims 65533-65535 and is legal.
        After _child_environment, child index 1 sees 65534 and would otherwise
        check 65534-65536, which exceeds MAX_PORT.
        """
        monkeypatch.setenv("DYN_VLLM_EMBEDDING_PROCESS_ROLE", "child")
        monkeypatch.setenv("DYN_SYSTEM_PORT", "65534")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3
        config._validate_embedding_worker_processes()

    def test_child_skips_phantom_adjacent_fpm_collision(self, monkeypatch):
        """A child must not treat ports past the parent's range as reserved.

        Parent base=20377 with N=3 claims 20377-20379; FPM at 20380 is adjacent
        and legal. Child index 1 sees 20378 and would otherwise check 20378-20380.
        """
        monkeypatch.setenv("DYN_VLLM_EMBEDDING_PROCESS_ROLE", "child")
        monkeypatch.setenv("DYN_SYSTEM_PORT", "20378")
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "20380")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 3
        config._validate_embedding_worker_processes()

    def test_child_still_rejects_headless(self, monkeypatch):
        """Skipping the phantom range check must not skip the other N>1 guards."""
        monkeypatch.setenv("DYN_VLLM_EMBEDDING_PROCESS_ROLE", "child")
        config = create_config()
        config.embedding_worker = True
        config.embedding_worker_processes = 4
        config.headless = True
        with pytest.raises(ValueError, match="--headless"):
            config._validate_embedding_worker_processes()

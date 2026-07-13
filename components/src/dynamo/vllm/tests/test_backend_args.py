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
    config.benchmark_mode = None
    config.use_vllm_tokenizer = False
    config.frontend_decoding = False
    config.frontend_video_preprocessing = False
    config.route_to_encoder = False
    config.custom_encoder_class = None
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


class TestValidateFrontendVideoPreprocessing:
    def _enabled_config(self) -> DynamoVllmConfig:
        config = create_config()
        config.frontend_video_preprocessing = True
        config.frontend_decoding = True
        config.enable_multimodal = True
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        return config

    def test_requires_frontend_decoding(self):
        config = self._enabled_config()
        config.frontend_decoding = False
        with pytest.raises(ValueError, match="frontend-decoding"):
            config._validate_frontend_video_preprocessing()

    def test_requires_multimodal(self):
        config = self._enabled_config()
        config.enable_multimodal = False
        with pytest.raises(ValueError, match="enable-multimodal"):
            config._validate_frontend_video_preprocessing()

    @pytest.mark.parametrize(
        "mode",
        [
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
            DisaggregationMode.ENCODE,
        ],
    )
    def test_rejects_disaggregated_modes(self, mode):
        config = self._enabled_config()
        config.disaggregation_mode = mode
        with pytest.raises(ValueError, match="aggregated"):
            config._validate_frontend_video_preprocessing()

    def test_accepts_aggregated_multimodal(self):
        self._enabled_config()._validate_frontend_video_preprocessing()

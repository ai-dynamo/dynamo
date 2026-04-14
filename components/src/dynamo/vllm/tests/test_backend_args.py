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
    return config


class TestResolveDisaggregationModeFromLegacyMultimodalFlags:
    """
    Test suite for resolving disaggregation mode when legacy multimodal flags are set.
    """

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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Priority-chain tests for ``fetch_pre_deployment_metrics``.

Order (highest → lowest):
    1. ``get_perf_metrics`` endpoint
    2. AIC interpolation (if spec is set)
    3. File fallback (NPZ / JSON under ``profile_results_dir``)
"""

from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.monitoring import perf_metrics as pm

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _make_spec() -> AICInterpolationSpec:
    pick = PickedParallelConfig(tp=1, dp=8, moe_tp=1, moe_ep=8)
    return AICInterpolationSpec(
        hf_id="x",
        system="h200_sxm",
        backend="trtllm",
        isl=1000,
        osl=200,
        sweep_max_context_length=4096,
        prefill_interpolation_granularity=4,
        decode_interpolation_granularity=4,
        prefill_pick=pick,
        decode_pick=pick,
    )


def _fpm():
    """Opaque sentinel FPM — identity comparison is enough for these tests."""
    return object()


class TestPriorityChain:
    @pytest.mark.asyncio
    async def test_endpoint_wins(self):
        """When the endpoint returns FPMs, AIC and files are never consulted."""
        endpoint_fpms = [_fpm(), _fpm()]
        with patch.object(
            pm, "_try_endpoint", return_value=endpoint_fpms
        ) as mock_ep, patch.object(
            pm, "_try_aic_interpolation"
        ) as mock_aic, patch.object(
            pm, "_convert_profiling_data_to_fpms"
        ) as mock_files:
            got = await pm.fetch_pre_deployment_metrics(
                runtime=MagicMock(),
                namespace="dynamo",
                worker_info=MagicMock(component_name="worker"),
                profile_results_dir="/tmp/profile",
                component_type=SubComponentType.PREFILL,
                aic_spec=_make_spec(),
            )
        assert got is endpoint_fpms
        mock_ep.assert_awaited_once()
        mock_aic.assert_not_called()
        mock_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_aic_fallback_when_endpoint_empty(self):
        """Endpoint returns [] → AIC runs. Files never consulted."""
        aic_fpms = [_fpm()]
        with patch.object(pm, "_try_endpoint", return_value=[]), patch.object(
            pm, "_try_aic_interpolation", return_value=aic_fpms
        ) as mock_aic, patch.object(
            pm, "_convert_profiling_data_to_fpms"
        ) as mock_files:
            got = await pm.fetch_pre_deployment_metrics(
                runtime=MagicMock(),
                namespace="dynamo",
                worker_info=MagicMock(component_name="worker"),
                profile_results_dir="/tmp/profile",
                component_type=SubComponentType.PREFILL,
                aic_spec=_make_spec(),
            )
        assert got is aic_fpms
        mock_aic.assert_called_once()
        mock_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_fallback_when_no_spec(self):
        """Endpoint returns [] and aic_spec is None → files load."""
        file_fpms = [_fpm()]
        with patch.object(pm, "_try_endpoint", return_value=[]), patch.object(
            pm, "_try_aic_interpolation"
        ) as mock_aic, patch.object(
            pm, "_convert_profiling_data_to_fpms", return_value=file_fpms
        ) as mock_files:
            got = await pm.fetch_pre_deployment_metrics(
                runtime=MagicMock(),
                namespace="dynamo",
                worker_info=MagicMock(component_name="worker"),
                profile_results_dir="/tmp/profile",
                component_type=SubComponentType.DECODE,
                aic_spec=None,
            )
        assert got is file_fpms
        mock_aic.assert_not_called()
        mock_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_fallback_when_aic_fails(self):
        """Endpoint empty, AIC raises at runtime → files are consulted."""
        file_fpms = [_fpm()]
        with patch.object(pm, "_try_endpoint", return_value=[]), patch.object(
            pm, "_try_aic_interpolation", side_effect=RuntimeError("aic boom")
        ), patch.object(
            pm, "_convert_profiling_data_to_fpms", return_value=file_fpms
        ) as mock_files:
            got = await pm.fetch_pre_deployment_metrics(
                runtime=MagicMock(),
                namespace="dynamo",
                worker_info=MagicMock(component_name="worker"),
                profile_results_dir="/tmp/profile",
                component_type=SubComponentType.DECODE,
                aic_spec=_make_spec(),
            )
        assert got is file_fpms
        mock_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_aic_missing_package_falls_through_to_files(self):
        """aiconfigurator not installed → ImportError is caught, files run."""
        file_fpms = [_fpm()]
        with patch.object(pm, "_try_endpoint", return_value=[]), patch.object(
            pm,
            "_try_aic_interpolation",
            side_effect=ImportError("no module named aiconfigurator"),
        ), patch.object(
            pm, "_convert_profiling_data_to_fpms", return_value=file_fpms
        ) as mock_files:
            got = await pm.fetch_pre_deployment_metrics(
                runtime=MagicMock(),
                namespace="dynamo",
                worker_info=MagicMock(component_name="worker"),
                profile_results_dir="/tmp/profile",
                component_type=SubComponentType.DECODE,
                aic_spec=_make_spec(),
            )
        assert got is file_fpms
        mock_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        """No endpoint, no spec, no files → RuntimeError."""
        with patch.object(pm, "_try_endpoint", return_value=[]), patch.object(
            pm, "_try_aic_interpolation"
        ) as mock_aic, patch.object(
            pm,
            "_convert_profiling_data_to_fpms",
            side_effect=FileNotFoundError("no npz"),
        ):
            with pytest.raises(RuntimeError, match="Failed to obtain"):
                await pm.fetch_pre_deployment_metrics(
                    runtime=MagicMock(),
                    namespace="dynamo",
                    worker_info=MagicMock(component_name="worker"),
                    profile_results_dir="/tmp/profile",
                    component_type=SubComponentType.PREFILL,
                    aic_spec=None,
                )
        mock_aic.assert_not_called()

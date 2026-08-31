# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os

import pytest

from tests.serve.common import WORKSPACE_DIR, run_serve_deployment
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import images_payload_default

logger = logging.getLogger(__name__)

sample_dir = os.path.join(WORKSPACE_DIR, "examples/backends/sample")

sample_diffusion_configs = {
    "aggregated": EngineConfig(
        name="aggregated",
        directory=sample_dir,
        script_name="agg_diffusion.sh",
        script_args=["--model-name", "sample-diffusion-model"],
        marks=[
            pytest.mark.gpu_0,
            pytest.mark.timeout(300),
            pytest.mark.pre_merge,
            pytest.mark.unified,
        ],
        model="sample-diffusion-model",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[images_payload_default()],
    ),
}


def _params_without_model_mark(configs):
    """Like ``params_with_model_mark`` but WITHOUT the per-model marker.

    The sample diffusion engine registers name-only (no Hugging Face model),
    so a ``model`` marker would make ``predownload_models`` try to fetch
    ``sample-diffusion-model`` from the Hub and fail."""
    return [pytest.param(name, marks=list(cfg.marks)) for name, cfg in configs.items()]


@pytest.fixture(params=_params_without_model_mark(sample_diffusion_configs))
def sample_diffusion_config_test(request):
    return sample_diffusion_configs[request.param]


@pytest.mark.e2e
def test_sample_diffusion_deployment(
    sample_diffusion_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
):
    """Smoke test for the raw-media (DiffusionEngine) path through the unified
    Worker: frontend -> RawEngineAdapter -> SampleDiffusionEngine, served at
    /v1/images/generations. CPU-only; no model download (name-only worker)."""
    config = dataclasses.replace(
        sample_diffusion_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)

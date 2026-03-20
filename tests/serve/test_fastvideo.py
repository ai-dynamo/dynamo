# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import os
import sys
from dataclasses import dataclass

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import video_generation_payload_default

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastvideo") is None,
    reason="fastvideo is not installed",
)


@dataclass
class FastVideoConfig(EngineConfig):
    """Configuration for FastVideo serve smoke tests."""


fastvideo_local_dir = os.path.join(WORKSPACE_DIR, "examples/diffusers/local")

fastvideo_configs = {
    "aggregated": FastVideoConfig(
        name="aggregated",
        directory=fastvideo_local_dir,
        command=["bash", os.path.join(fastvideo_local_dir, "run_local.sh")],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.fastvideo,
            pytest.mark.nightly,
            pytest.mark.slow,
            pytest.mark.timeout(1800),
        ],
        model="FastVideo/LTX2-Distilled-Diffusers",
        timeout=1800,
        env={},
        request_payloads=[video_generation_payload_default()],
    ),
}


@pytest.fixture(params=params_with_model_mark(fastvideo_configs))
def fastvideo_config_test(request):
    """Fixture that provides FastVideo serve test configurations."""
    return fastvideo_configs[request.param]


@pytest.mark.fastvideo
@pytest.mark.e2e
def test_fastvideo_deployment(
    fastvideo_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """Smoke test the built-in FastVideo backend behind the shared frontend."""
    config = dataclasses.replace(
        fastvideo_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    runtime_dir = f"/tmp/dynamo-fastvideo-serve-{dynamo_dynamic_ports.frontend_port}"
    config.env.update(
        {
            "MODEL": config.model,
            "PYTHON_BIN": sys.executable,
            "HTTP_PORT": str(dynamo_dynamic_ports.frontend_port),
            "DISCOVERY_DIR": f"{runtime_dir}/discovery",
            "LOG_DIR": f"{runtime_dir}/logs",
        }
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)

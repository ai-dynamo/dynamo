# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for aggregated vLLM serving on a single-host Google TPU slice.

No NIXL/KVBM support on this path yet, so only the aggregated (non-disaggregated)
scenario is covered here -- see examples/backends/vllm/deploy/tpu/README.md for the
scope of the current TPU integration.

Requires a real single-host TPU slice; there is no simulated/CPU fallback for these
tests (unlike gpu_0-marked tests), so they only run where a `tpu_1`-labeled runner
is available.
"""

import os

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload_default,
    completion_payload_default,
    metric_payload_default,
)

vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)

vllm_tpu_configs = {
    "aggregated": EngineConfig(
        name="aggregated_tpu",
        directory=vllm_dir,
        script_name="tpu/agg_tpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.tpu_1,
            # No profiled_vram_gib / requested_vllm_kv_cache_bytes markers yet --
            # those are measured on real hardware; add once a TPU CI runner exists.
            pytest.mark.timeout(300),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(vllm_tpu_configs))
def vllm_tpu_config_test(request):
    """Fixture that provides different vLLM-on-TPU test configurations"""
    return vllm_tpu_configs[request.param]


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.parametrize("num_system_ports", [1], indirect=True)
def test_serve_deployment(
    vllm_tpu_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Aggregated vLLM+TPU deployment: chat + completion + metrics sanity."""
    assert num_system_ports >= 1, "serve tests require at least SYSTEM_PORT1"
    run_serve_deployment(vllm_tpu_config_test, request, ports=dynamo_dynamic_ports)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Shadow Engine Failover Test for vLLM."""

import os

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

from .utils.common import run_shadow_failover_test
from .utils.vllm import VLLMWithGMSProcess

TENSOR_PARALLEL_SIZE = int(os.environ.get("FAULT_TOLERANCE_TP", "1"))


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover(
    request, runtime_services, gms_ports, predownload_models
):
    ports = gms_ports
    tp = TENSOR_PARALLEL_SIZE

    run_shadow_failover_test(
        request,
        ports,
        make_shadow=lambda: VLLMWithGMSProcess(
            request,
            "shadow",
            ports["shadow_system"],
            ports["shadow_kv_event"],
            ports["shadow_nixl"],
            ports["frontend"],
            tensor_parallel_size=tp,
        ),
        make_primary=lambda: VLLMWithGMSProcess(
            request,
            "primary",
            ports["primary_system"],
            ports["primary_kv_event"],
            ports["primary_nixl"],
            ports["frontend"],
            tensor_parallel_size=tp,
        ),
        tp_size=tp,
    )

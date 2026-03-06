# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service shadow-engine failover test for TensorRT-LLM."""

import pytest

from .utils.common import run_shadow_failover_test
from .utils.trtllm import TRTLLM_GMS_MODEL_NAME, TRTLLMWithGMSProcess


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(TRTLLM_GMS_MODEL_NAME)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover(
    request, runtime_services, gms_ports, predownload_models
):
    ports = gms_ports

    run_shadow_failover_test(
        request,
        ports,
        make_shadow=lambda: TRTLLMWithGMSProcess(
            request,
            "shadow",
            ports["shadow_system"],
            ports["frontend"],
        ),
        make_primary=lambda: TRTLLMWithGMSProcess(
            request,
            "primary",
            ports["primary_system"],
            ports["frontend"],
        ),
        model=TRTLLM_GMS_MODEL_NAME,
    )

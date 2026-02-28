# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Shadow Engine Failover Test for SGLang."""

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

from .utils.common import run_shadow_failover_test
from .utils.sglang import SGLangWithGMSProcess


@pytest.mark.sglang
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

    run_shadow_failover_test(
        request,
        ports,
        make_shadow=lambda: SGLangWithGMSProcess(
            request,
            "shadow",
            ports["shadow_system"],
            ports["shadow_sglang"],
            ports["frontend"],
        ),
        make_primary=lambda: SGLangWithGMSProcess(
            request,
            "primary",
            ports["primary_system"],
            ports["primary_sglang"],
            ports["frontend"],
        ),
    )

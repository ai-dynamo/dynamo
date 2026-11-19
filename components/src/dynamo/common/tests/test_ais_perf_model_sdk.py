# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configs of ``dynamo.common.ais_perf_model`` next to the AISimulate config class."""

import pytest

from dynamo.common import ais_perf_model
from tests.utils.ais_perf_model import IDENTITY, normalized

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_router_perf_config_equals_forward_pass_config_dict() -> None:
    untyped = {**IDENTITY, "tp": 2, "backend_version": "1.0"}
    entry = {**untyped, "worker_type": "prefill"}
    # The entry with the defaults of the AISimulate config class.
    expected = normalized(entry)

    assert ais_perf_model.router_perf_config(entry) == expected
    assert ais_perf_model.router_perf_config(untyped, "prefill") == expected
    # Only a direct call can give a config without worker_type.
    with pytest.raises(ValueError, match="^ais_perf_model requires worker_type$"):
        ais_perf_model.normalized_config(untyped)

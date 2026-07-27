# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

# Pytest applies marker selection after importing test modules. Importing
# disagg_utils loads TensorRT-LLM bindings, which require libcuda even though
# these tests do not execute CUDA kernels, so skip collection on CPU runners.
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU not available, but tensorrt_llm import requires CUDA.",
        allow_module_level=True,
    )

from dynamo.trtllm.utils import disagg_utils

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
]


def test_rc22_machine_id_is_split_losslessly():
    with (
        patch.object(disagg_utils, "_TRTLLM_DISAGG_ID_HAS_PROCESS_ID", True),
        patch.object(
            disagg_utils,
            "_trtllm_get_global_disagg_request_id",
            return_value=123,
        ) as generate,
    ):
        result = disagg_utils.get_compatible_global_disagg_request_id(1020)

    assert result == 123
    generate.assert_called_once_with(15, 60)


def test_rc22_all_dynamo_machine_ids_are_in_range_and_unique():
    pairs = {
        divmod(machine_id, disagg_utils._TRTLLM_PROCESS_ID_SPACE)
        for machine_id in range(disagg_utils._DYNAMO_DISAGG_MACHINE_ID_SPACE)
    }

    assert len(pairs) == disagg_utils._DYNAMO_DISAGG_MACHINE_ID_SPACE
    assert all(
        0 <= node_id < disagg_utils._TRTLLM_NODE_ID_SPACE for node_id, _ in pairs
    )
    assert all(
        0 <= process_id < disagg_utils._TRTLLM_PROCESS_ID_SPACE
        for _, process_id in pairs
    )


def test_rc21_keeps_legacy_machine_id():
    with (
        patch.object(disagg_utils, "_TRTLLM_DISAGG_ID_HAS_PROCESS_ID", False),
        patch.object(
            disagg_utils,
            "_trtllm_get_global_disagg_request_id",
            return_value=456,
        ) as generate,
    ):
        result = disagg_utils.get_compatible_global_disagg_request_id(1020)

    assert result == 456
    generate.assert_called_once_with(1020)


@pytest.mark.parametrize("machine_id", [-1, 1021])
def test_invalid_dynamo_machine_id_fails_fast(machine_id):
    with pytest.raises(ValueError, match="machine_id must be in range"):
        disagg_utils.get_compatible_global_disagg_request_id(machine_id)

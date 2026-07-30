# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt_llm")

from dynamo.trtllm.args import parse_args  # noqa: E402
from dynamo.trtllm.backend_args import DEFAULT_DISAGG_NODE_ID_MODULUS  # noqa: E402
from dynamo.trtllm.utils.disagg_utils import get_disagg_node_id  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.unified,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_get_disagg_node_id_uses_legacy_modulus():
    assert get_disagg_node_id(1020, DEFAULT_DISAGG_NODE_ID_MODULUS) == 1020
    assert get_disagg_node_id(1021, DEFAULT_DISAGG_NODE_ID_MODULUS) == 0


def test_parse_args_rejects_invalid_disagg_node_id_modulus():
    with pytest.raises(ValueError, match="--disagg-node-id-modulus"):
        parse_args(["--disagg-node-id-modulus", "0"])

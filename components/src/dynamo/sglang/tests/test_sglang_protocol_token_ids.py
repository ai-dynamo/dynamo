# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.sglang.protocol import PreprocessedRequest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

IDS = [0, 1, 128000, 2**31 - 1]


def _request(token_ids):
    return {"token_ids": token_ids, "stop_conditions": {}, "sampling_options": {}}


def test_packed_token_ids_validate_as_a_list():
    packed = b"".join(i.to_bytes(4, "little") for i in IDS)
    assert PreprocessedRequest.model_validate(_request(packed)).token_ids == IDS


def test_list_token_ids_are_unchanged():
    assert PreprocessedRequest.model_validate(_request(list(IDS))).token_ids == IDS

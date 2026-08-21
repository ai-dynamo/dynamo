# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo import _core

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_native_response_egress_replaces_ownership_based_api_name():
    assert hasattr(_core, "NativeResponseEgress")
    assert hasattr(_core, "RequestKey")
    assert not hasattr(_core, "OwnedTokenEgress")


def test_response_event_requires_registration_identity_and_sequence():
    egress = _core.NativeResponseEgress(shards=1, queue_depth=1)

    assert (
        egress.process_batch(
            [
                {
                    "client_id": 1,
                    "generation": 1,
                    "sequence": 0,
                    "outputs": [],
                }
            ]
        )
        == []
    )
    with pytest.raises(ValueError, match="generation"):
        egress.process_batch([{"client_id": 1, "sequence": 0, "outputs": []}])
    with pytest.raises(TypeError):
        _core.RequestKey()

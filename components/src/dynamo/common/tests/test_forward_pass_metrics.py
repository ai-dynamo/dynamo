# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode, encode

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_timing_scope_survives_msgpack_round_trip():
    metrics = ForwardPassMetrics(timing_scope="model_step_cuda")

    decoded = decode(encode(metrics))

    assert decoded is not None
    assert decoded.timing_scope == "model_step_cuda"
    reencoded = msgspec.msgpack.decode(encode(decoded))
    assert reencoded["timing_scope"] == "model_step_cuda"


def test_legacy_record_still_omits_timing_scope_after_reencoding():
    legacy = msgspec.msgpack.encode({"version": 1})

    decoded = decode(legacy)

    assert decoded is not None
    assert decoded.timing_scope is msgspec.UNSET
    reencoded = msgspec.msgpack.decode(encode(decoded))
    assert "timing_scope" not in reencoded

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest

from dynamo.common.export_forward_pass_metrics import (
    FpmExportRecord,
    encode_export_record,
)
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_encode_export_record_is_ndjson_and_preserves_envelope_metadata():
    record = FpmExportRecord(
        namespace="dynamo",
        component="backend",
        publisher_id=17,
        sequence=23,
        published_at_ms=1_700_000_000_123,
        metrics=ForwardPassMetrics(
            worker_id="worker-1",
            dp_rank=2,
            counter_id=99,
            wall_time=0.025,
            scheduled_requests=ScheduledRequestMetrics(
                num_decode_requests=4,
                sum_decode_kv_tokens=4096,
            ),
        ),
    )

    encoded = encode_export_record(record)

    assert encoded.endswith(b"\n")
    decoded = msgspec.json.decode(encoded)
    assert decoded["namespace"] == "dynamo"
    assert decoded["component"] == "backend"
    assert decoded["publisher_id"] == 17
    assert decoded["sequence"] == 23
    assert decoded["published_at_ms"] == 1_700_000_000_123
    assert decoded["metrics"]["worker_id"] == "worker-1"
    assert decoded["metrics"]["dp_rank"] == 2
    assert decoded["metrics"]["counter_id"] == 99
    assert decoded["metrics"]["scheduled_requests"]["num_decode_requests"] == 4

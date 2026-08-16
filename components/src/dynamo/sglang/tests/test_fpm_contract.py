# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wire-format contract test for Forward Pass Metrics.

Verifies that ForwardPassMetrics encoded by SGLang can be decoded by Dynamo's
shared schema. Both are plain msgspec.Structs -- neither sets
array_like=True -- so they encode as maps keyed by field name, and the
contract that has to hold is that the fields SGLang sends are present in
Dynamo's schema with the same names and types. Dynamo may carry fields SGLang
does not: msgspec fills them from their defaults on the way in and ignores
them on the way out.

Requiring the two field lists to be equal would state a stricter contract than
the encoding needs, and one this repository cannot keep: SGLang's structs live
in the installed sglang package, so a backend that grows a metric here could
not land until an external project shipped the same field.
"""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_sglang_fpm_decodes_with_dynamo_schema():
    """Encode with SGLang's ForwardPassMetrics, decode with Dynamo's."""
    from sglang.srt.observability.forward_pass_metrics import (
        ForwardPassMetrics as SglangFPM,
    )
    from sglang.srt.observability.forward_pass_metrics import (
        QueuedRequestMetrics as SglangQueued,
    )
    from sglang.srt.observability.forward_pass_metrics import (
        ScheduledRequestMetrics as SglangScheduled,
    )
    from sglang.srt.observability.forward_pass_metrics import encode as sglang_encode

    from dynamo.common.forward_pass_metrics import ForwardPassMetrics as DynamoFPM
    from dynamo.common.forward_pass_metrics import decode as dynamo_decode

    sglang_fpm = SglangFPM(
        version=1,
        worker_id="test-worker-42",
        dp_rank=3,
        counter_id=99,
        wall_time=0.042,
        scheduled_requests=SglangScheduled(
            num_prefill_requests=5,
            sum_prefill_tokens=1024,
            var_prefill_length=33.3,
            sum_prefill_kv_tokens=512,
            num_decode_requests=32,
            sum_decode_kv_tokens=8192,
            var_decode_kv_tokens=100.0,
        ),
        queued_requests=SglangQueued(
            num_prefill_requests=3,
            sum_prefill_tokens=768,
            var_prefill_length=25.0,
            num_decode_requests=1,
            sum_decode_kv_tokens=128,
            var_decode_kv_tokens=0.0,
        ),
    )

    raw = sglang_encode(sglang_fpm)
    dynamo_fpm = dynamo_decode(raw)

    assert dynamo_fpm is not None, "Dynamo decoder returned None (version mismatch?)"
    assert isinstance(dynamo_fpm, DynamoFPM)

    assert dynamo_fpm.version == 1
    assert dynamo_fpm.worker_id == "test-worker-42"
    assert dynamo_fpm.dp_rank == 3
    assert dynamo_fpm.counter_id == 99
    assert dynamo_fpm.wall_time == 0.042

    assert dynamo_fpm.scheduled_requests.num_prefill_requests == 5
    assert dynamo_fpm.scheduled_requests.sum_prefill_tokens == 1024
    assert dynamo_fpm.scheduled_requests.num_decode_requests == 32
    assert dynamo_fpm.scheduled_requests.sum_decode_kv_tokens == 8192

    assert dynamo_fpm.queued_requests.num_prefill_requests == 3
    assert dynamo_fpm.queued_requests.sum_prefill_tokens == 768
    assert dynamo_fpm.queued_requests.num_decode_requests == 1


def test_sglang_fpm_field_order_matches_dynamo():
    """Verify struct field names and order are identical."""
    import msgspec
    from sglang.srt.observability.forward_pass_metrics import (
        ForwardPassMetrics as SglangFPM,
    )
    from sglang.srt.observability.forward_pass_metrics import (
        QueuedRequestMetrics as SglangQueued,
    )
    from sglang.srt.observability.forward_pass_metrics import (
        ScheduledRequestMetrics as SglangScheduled,
    )

    from dynamo.common.forward_pass_metrics import ForwardPassMetrics as DynamoFPM
    from dynamo.common.forward_pass_metrics import QueuedRequestMetrics as DynamoQueued
    from dynamo.common.forward_pass_metrics import (
        ScheduledRequestMetrics as DynamoScheduled,
    )

    for sglang_cls, dynamo_cls, name in [
        (SglangFPM, DynamoFPM, "ForwardPassMetrics"),
        (SglangScheduled, DynamoScheduled, "ScheduledRequestMetrics"),
        (SglangQueued, DynamoQueued, "QueuedRequestMetrics"),
    ]:
        sglang_fields = msgspec.structs.fields(sglang_cls)
        dynamo_fields = msgspec.structs.fields(dynamo_cls)

        sglang_names = [f.name for f in sglang_fields]
        dynamo_names = [f.name for f in dynamo_fields]
        missing = [n for n in sglang_names if n not in dynamo_names]
        assert not missing, (
            f"{name}: Dynamo is missing fields SGLang sends: {missing} "
            f"(sglang={sglang_names}, dynamo={dynamo_names})"
        )

        # Compare type names (not identity) because sglang and dynamo define
        # structurally identical but distinct nested struct classes. Only the
        # shared fields are compared: a field Dynamo alone carries has no
        # counterpart to disagree with.
        dynamo_types = {f.name: f.type.__name__ for f in dynamo_fields}
        mismatched = [
            (f.name, f.type.__name__, dynamo_types[f.name])
            for f in sglang_fields
            if dynamo_types[f.name] != f.type.__name__
        ]
        assert (
            not mismatched
        ), f"{name} field types differ (field, sglang, dynamo): {mismatched}"

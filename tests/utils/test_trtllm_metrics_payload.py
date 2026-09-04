# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.payload_builder import metric_payload_default
from tests.utils.payloads import TRTLLMMetricsPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _spec_decode_check(payload: TRTLLMMetricsPayload):
    return next(
        check
        for check in payload._get_backend_specific_checks()
        if check.name == "trtllm_spec_decode_num_draft_tokens_total"
    )


def test_trtllm_metrics_payload_accepts_spec_decode_draft_tokens():
    payload = metric_payload_default(
        min_num_requests=1,
        backend="trtllm",
        min_spec_decode_draft_tokens=1,
    )

    assert isinstance(payload, TRTLLMMetricsPayload)
    check = _spec_decode_check(payload)
    payload._validate_metric_checks(
        [check],
        'trtllm_spec_decode_num_draft_tokens_total{model="test"} 4\n',
    )


def test_trtllm_metrics_payload_rejects_zero_spec_decode_draft_tokens():
    payload = metric_payload_default(
        min_num_requests=1,
        backend="trtllm",
        min_spec_decode_draft_tokens=1,
    )

    assert isinstance(payload, TRTLLMMetricsPayload)
    check = _spec_decode_check(payload)
    with pytest.raises(AssertionError, match="expected at least 1"):
        payload._validate_metric_checks(
            [check],
            'trtllm_spec_decode_num_draft_tokens_total{model="test"} 0\n',
        )

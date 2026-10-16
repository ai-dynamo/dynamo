# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm.exceptions import WorkerOverloaded
from dynamo.trtllm.errors import engine_error

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "failure,validation,expected",
    [
        (ValueError("bad input"), True, "InvalidRequest"),
        (TypeError("bad input"), True, "InvalidRequest"),
        (NotImplementedError("unsupported"), True, "NotImplemented"),
        (ValueError("executor invariant"), False, "Internal"),
        (RuntimeError("request failure"), False, "Internal"),
    ],
)
def test_submission_validation_is_distinct_from_executor_failure(
    failure, validation, expected
):
    error = engine_error(failure, validation=validation)
    assert error.error_class == expected
    assert str(failure) not in str(error.public_details)


def test_native_request_error_is_not_necessarily_client_fault():
    executor = pytest.importorskip("tensorrt_llm.executor.utils")
    error = engine_error(executor.RequestError("executor request failed"))
    assert error.error_class == "Internal"


def test_existing_capacity_classification_is_preserved():
    failure = WorkerOverloaded("full")
    assert engine_error(failure) is failure

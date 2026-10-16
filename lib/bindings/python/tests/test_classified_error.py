# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm.exceptions import ClassifiedError, DynamoException, classified_error

pytestmark = [pytest.mark.gpu_0, pytest.mark.unit, pytest.mark.pre_merge]


def test_semantic_factory_keeps_private_diagnostic_out_of_exception_message():
    error = classified_error(
        "CapacityExhausted",
        reason="capacity.worker_overloaded",
        diagnostic="private credential https://internal.invalid/?secret=value",
        public_details={"type": "size_limit", "limit": 4, "actual": 5},
    )
    assert isinstance(error, ClassifiedError)
    assert isinstance(error, DynamoException)
    assert error.error_class == "CapacityExhausted"
    assert error.reason == "capacity.worker_overloaded"
    assert error.public_details == {"type": "size_limit", "limit": 4, "actual": 5}
    assert "private" not in str(error)
    assert "secret" not in str(error)


@pytest.mark.parametrize(
    "error_class,reason",
    [
        ("MissingClass", None),
        ("Unavailable", "capacity.worker_overloaded"),
        ("Unavailable", "arbitrary.reason"),
    ],
)
def test_semantic_factory_rejects_invalid_identity(error_class, reason):
    with pytest.raises(ValueError):
        classified_error(error_class, reason=reason)


def test_private_diagnostic_is_bounded_without_becoming_public():
    error = classified_error("Internal", diagnostic="é" * 5000)
    assert len(error.diagnostic.encode("utf-8")) <= 4096
    assert error.public_details is None
    assert "é" not in str(error)


def test_cancelled_identity_is_retained():
    error = classified_error("Cancelled", diagnostic="private cancellation")
    assert error.error_class == "Cancelled"
    assert error.public_details is None


@pytest.mark.parametrize(
    "details",
    [
        {"type": "unknown"},
        {"type": "size_limit", "limit": -1},
        {"type": "message", "message": 3},
    ],
)
def test_semantic_factory_rejects_invalid_public_details(details):
    with pytest.raises(ValueError, match="Invalid public error details"):
        classified_error("InvalidRequest", public_details=details)

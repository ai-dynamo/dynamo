# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native parameter identities are public; exception messages remain private."""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.fixture
def native_errors():
    # Marker-only collection runs without a native framework installation.
    # Import at execution time so real contract tests still require vLLM.
    from vllm import exceptions

    from dynamo.vllm.errors import vllm_client_error_to_http_error

    return exceptions, vllm_client_error_to_http_error


@pytest.mark.parametrize("parameter", ["reasoning_effort", "prompt_logprobs", "suffix"])
def test_declared_native_field_is_preserved(parameter, native_errors):
    exceptions, adapt = native_errors
    native = exceptions.VLLMValidationError("private diagnostic", parameter=parameter)
    error = adapt(native)
    assert error.code == 400
    assert error.param == parameter
    assert error.message == str(native)


@pytest.mark.parametrize(
    "parameter", [None, "private_secret", "messages[0].role", "", "a/b"]
)
def test_unknown_or_nested_parameter_is_not_promoted(parameter, native_errors):
    exceptions, adapt = native_errors
    native = exceptions.VLLMValidationError(
        "reasoning_effort: private diagnostic", parameter=parameter
    )
    error = adapt(native)
    assert error.code == 400
    assert error.param is None


@pytest.mark.parametrize(
    "exception_name, kwargs, status",
    [
        ("VLLMNotFoundError", {}, 404),
        ("VLLMUnprocessableEntityError", {"parameter": "messages"}, 422),
    ],
)
def test_other_error_categories_keep_existing_adapter_behavior(
    exception_name, kwargs, status, native_errors
):
    exceptions, adapt = native_errors
    native = getattr(exceptions, exception_name)("private diagnostic", **kwargs)
    error = adapt(native)
    assert error.code == status
    assert error.param is None

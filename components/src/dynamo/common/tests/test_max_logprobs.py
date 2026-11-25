# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.max_logprobs import (
    MAX_LOGPROBS_RUNTIME_KEY,
    publish_max_logprobs,
    read_max_logprobs,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize("max_logprobs", [0, 20, 32, 151936])
def test_publish_writes_a_json_integer(max_logprobs):
    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    publish_max_logprobs(runtime_config, max_logprobs)

    runtime_config.set_engine_specific.assert_called_once_with(
        MAX_LOGPROBS_RUNTIME_KEY, str(max_logprobs)
    )


@pytest.mark.parametrize("max_logprobs", [-1, 32.0, "32", True, None])
def test_publish_rejects_anything_but_a_non_negative_integer(max_logprobs):
    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    with pytest.raises(ValueError, match="non-negative integer"):
        publish_max_logprobs(runtime_config, max_logprobs)

    runtime_config.set_engine_specific.assert_not_called()


def test_read_returns_what_publish_wrote():
    runtime_data = {}
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda key, value: runtime_data.update({key: int(value)})
    )
    publish_max_logprobs(runtime_config, 32)

    assert read_max_logprobs({"runtime_data": runtime_data}) == 32


@pytest.mark.parametrize(
    "runtime_config",
    [
        None,
        {},
        {"runtime_data": None},
        {"runtime_data": {}},
        {"runtime_data": {MAX_LOGPROBS_RUNTIME_KEY: -1}},
        {"runtime_data": {MAX_LOGPROBS_RUNTIME_KEY: "32"}},
        {"runtime_data": {MAX_LOGPROBS_RUNTIME_KEY: 32.0}},
        {"runtime_data": {MAX_LOGPROBS_RUNTIME_KEY: True}},
    ],
)
def test_read_treats_missing_or_malformed_values_as_unpublished(runtime_config):
    assert read_max_logprobs(runtime_config) is None

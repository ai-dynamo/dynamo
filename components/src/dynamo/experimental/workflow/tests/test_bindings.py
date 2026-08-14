# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.experimental.workflow import (
    InlineBinding,
    RemoteBinding,
    WorkflowValidationError,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def test_inline_binding_owns_initialized_runner() -> None:
    runner = SimpleNamespace(contract=object(), run=lambda: None)

    assert InlineBinding(runner).runner is runner


def test_inline_binding_rejects_non_runner() -> None:
    with pytest.raises(WorkflowValidationError, match="implement StageRunner"):
        InlineBinding(object())


def test_remote_binding_uses_stable_discovery_identity() -> None:
    binding = RemoteBinding("namespace.component.endpoint")

    assert binding.endpoint_id == "namespace.component.endpoint"
    assert binding.routing_policy == "round_robin"


def test_remote_binding_rejects_invalid_endpoint_and_routing_policy() -> None:
    with pytest.raises(WorkflowValidationError, match="namespace.component.endpoint"):
        RemoteBinding("component.endpoint")
    with pytest.raises(WorkflowValidationError, match="unsupported"):
        RemoteBinding("namespace.component.endpoint", routing_policy="random")

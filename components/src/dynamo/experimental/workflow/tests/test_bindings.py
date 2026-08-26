# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.experimental.workflow import InlineBinding, WorkflowValidationError

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

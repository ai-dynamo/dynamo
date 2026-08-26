# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical invocation targets for authored workflow stages."""

from __future__ import annotations

from dataclasses import dataclass

from dynamo.experimental.workflow.runtime import StageRunner
from dynamo.experimental.workflow.types import WorkflowValidationError


@dataclass(frozen=True)
class InlineBinding:
    """Bind one logical stage to an initialized in-process runner."""

    runner: StageRunner

    def __post_init__(self) -> None:
        if not isinstance(self.runner, StageRunner):
            raise WorkflowValidationError("inline runner must implement StageRunner")


Binding = InlineBinding

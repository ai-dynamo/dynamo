# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expose the bound workflow to the existing Dynamo frontend."""

from typing import Any

from dynamo.workflow import WorkflowOrchestrator
from examples.custom_backend.workflow_hello_world.dynamo.workflow import (
    compile_remote_workflow,
)


async def provide_workflow(runtime: Any) -> WorkflowOrchestrator:
    return await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
    )

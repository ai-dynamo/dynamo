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

"""Dynamo contracts and runner adapters for the shared stages."""

from collections.abc import Mapping
from typing import Any

from dynamo.workflow import StageContext, StageContract, ValueSpec
from examples.custom_backend.workflow_hello_world.common import HelloStage as HelloLogic
from examples.custom_backend.workflow_hello_world.common import MergeStage as MergeLogic
from examples.custom_backend.workflow_hello_world.common import WorldStage as WorldLogic

REQUEST = ValueSpec(type="json")
TEXT = ValueSpec(type="text")


class HelloStage:
    contract = StageContract(
        id="hello",
        inputs={"request": REQUEST},
        outputs={"text": TEXT},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"text": await HelloLogic().run(inputs["request"])}


class WorldStage:
    contract = StageContract(
        id="world",
        inputs={"request": REQUEST},
        outputs={"text": TEXT},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"text": await WorldLogic().run(inputs["request"])}


class MergeStage:
    contract = StageContract(
        id="merge",
        inputs={"hello": TEXT, "world": TEXT},
        outputs={"chunk": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        text = await MergeLogic().run(inputs["hello"], inputs["world"])
        return {
            "chunk": {
                "token_ids": [],
                "text": text,
                "index": 0,
                "finish_reason": "stop",
            }
        }


STAGES: dict[str, type[Any]] = {
    "hello": HelloStage,
    "world": WorldStage,
    "merge": MergeStage,
}

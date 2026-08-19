# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage contracts and implementations for the Hello World workflow."""

from collections.abc import Mapping
from typing import Any

from dynamo.workflow import StageContext, StageContract, ValueSpec

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
        return {"text": "Hello, "}


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
        return {"text": "World!"}


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
        return {
            "chunk": {
                "token_ids": [],
                "text": f'{inputs["hello"]}{inputs["world"]}',
                "index": 0,
                "finish_reason": "stop",
            }
        }


STAGES: dict[str, type[Any]] = {
    "hello": HelloStage,
    "world": WorldStage,
    "merge": MergeStage,
}

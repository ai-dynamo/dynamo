# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage contracts and implementations for the Hello World workflow."""

from collections.abc import Mapping
from typing import Any

from dynamo.workflow import StageContext, StageContract


class HelloStage:
    contract = StageContract(
        id="hello",
        inputs={"request"},
        outputs={"text"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        return {"text": "Hello, "}


class WorldStage:
    contract = StageContract(
        id="world",
        inputs={"request"},
        outputs={"text"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        return {"text": "World!"}


class MergeStage:
    contract = StageContract(
        id="merge",
        inputs={"hello", "world"},
        outputs={"chunk"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
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

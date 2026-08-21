# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from dynamo.experimental.workflow import StageContext, StageContract


class TextEncoder:
    contract = StageContract(
        id="text-encoder",
        inputs={"text"},
        outputs={"tokens"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"tokens": inputs["text"].lower().split()}


class KeywordClassifier:
    contract = StageContract(
        id="keyword-classifier",
        inputs={"tokens"},
        outputs={"scores"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        tokens = inputs["tokens"]
        workflow_hits = sum(token == "workflow" for token in tokens)
        score = workflow_hits / max(1, len(tokens))
        return {"scores": {"workflow": score, "other": 1.0 - score}}


class TextGenerator:
    contract = StageContract(
        id="text-generator",
        inputs={"tokens"},
        outputs={"text"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"text": " ".join(reversed(inputs["tokens"]))}


STAGES = {
    "encoder": TextEncoder,
    "classifier": KeywordClassifier,
    "generator": TextGenerator,
}

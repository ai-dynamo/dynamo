# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from dynamo.workflow import StageContext, StageContract, ValueSpec

TOKENS = ValueSpec(type="json")


class TextEncoder:
    contract = StageContract(
        id="text-encoder",
        inputs={"text": ValueSpec(type="text")},
        outputs={"tokens": TOKENS},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        return {"tokens": inputs["text"].lower().split()}


class KeywordClassifier:
    contract = StageContract(
        id="keyword-classifier",
        inputs={"tokens": TOKENS},
        outputs={"scores": ValueSpec(type="json")},
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
        inputs={"tokens": TOKENS},
        outputs={"text": ValueSpec(type="text")},
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for stock Dynamo vLLM workflow stages."""

from dynamo.workflow import StageContract, StreamSpec, ValueSpec

REQUEST = ValueSpec(type="json")


class DynamoVllmStage:
    """Contracts implemented by a stock aggregated Dynamo vLLM worker."""

    request_complete_contract = StageContract(
        id="dynamo-vllm-request-complete",
        inputs={"request": REQUEST},
        outputs={"completion": ValueSpec(type="json")},
    )

    # This declares the intended streaming ABI. The workflow compiler rejects
    # StreamSpec execution until scheduling and frontend propagation are added.
    request_stream_contract = StageContract(
        id="dynamo-vllm-request-stream",
        inputs={"request": REQUEST},
        outputs={"chunks": StreamSpec(item=ValueSpec(type="json"))},
    )

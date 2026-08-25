# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for a stock Dynamo vLLM workflow stage."""

from dynamo.experimental.workflow import StageContract


class DynamoVllmStage:
    """Contract implemented by a stock aggregated Dynamo vLLM worker."""

    request_complete_contract = StageContract(
        id="dynamo-vllm-request-complete",
        inputs={"request"},
        outputs={"completion"},
    )

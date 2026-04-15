# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import AsyncGenerator

from dynamo._core import Context
from dynamo.trtllm.request_handlers.handler_base import (
    HandlerBase,
    RequestHandlerConfig,
)


class PrefillDecodeHandler(HandlerBase):
    """
    Handler for decode workers in conditional prefill mode.

    Handles both:
    - Disaggregated requests (with prefill_result containing disaggregated_params)
      → normal decode path using KV state from prefill worker
    - Aggregated requests (no prefill_result, prefill was skipped by the router)
      → full prefill+decode on this worker, same as aggregated mode
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[dict, None]:
        logging.debug(f"PrefillDecode Request ID: {context.id()}")

        prefill_result = request.get("prefill_result")
        if prefill_result and "disaggregated_params" in prefill_result:
            logging.debug(
                f"PrefillDecode: using disaggregated params from prefill for request {context.id()}"
            )
        else:
            logging.debug(
                f"PrefillDecode: no prefill result, running as aggregated for request {context.id()}"
            )

        async for res in self.generate_locally(request, context):
            yield res

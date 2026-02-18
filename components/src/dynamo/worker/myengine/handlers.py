# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample handler that echoes input tokens back 5 times, streamed token by token."""

from typing import Any, AsyncGenerator, Dict, List

from dynamo._core import Context
from dynamo.common.backend import BaseHandler

_REPEAT_COUNT = 5


class MyEngineHandler(BaseHandler):
    """Echoes the request's input tokens repeated 5 times, one token at a time."""

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        input_ids: List[int] = request.get("input_ids", [])
        output_ids = input_ids * _REPEAT_COUNT
        total = len(output_ids)

        for i, token_id in enumerate(output_ids):
            out: Dict[str, Any] = {"token_ids": [token_id]}
            if i == total - 1:
                out["finish_reason"] = "stop"
            yield out

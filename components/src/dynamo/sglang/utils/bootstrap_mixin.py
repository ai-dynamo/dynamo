# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
import socket
from typing import Any, AsyncIterator, Dict, Optional

from sglang.srt.utils import get_ip

from dynamo.sglang.protocol import DisaggPreprocessedRequest

logger = logging.getLogger(__name__)


class BootstrapMixin:
    """Mixin for bootstrap room operations in disaggregated SGLang deployments"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()

    def _get_bootstrap_info(self) -> tuple[str, int]:
        """Extract bootstrap host/port from tokenizer manager"""
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return bootstrap_host, bootstrap_port

    def _generate_bootstrap_room(self) -> int:
        """Generate a unique bootstrap room ID"""
        return random.randint(0, 2**63 - 1)

    def _build_bootstrap_info(self, bootstrap_room: int) -> Dict[str, Any]:
        """Build bootstrap info dictionary"""
        return {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

    async def _get_bootstrap_from_prefill(
        self,
        request: Any,
        sampling_params: dict,
        request_wrapper: Optional[type] = None,
    ) -> Dict[str, Any]:
        """Get bootstrap info from prefill worker (for decode workers)"""
        if request_wrapper is None:
            # Default to regular sglang request wrapper, but allow override
            request_wrapper = DisaggPreprocessedRequest

        prefill_stream = await self.prefill_client.generate(
            request_wrapper(
                request=request,
                sampling_params=sampling_params,
            ).model_dump_json()
        )

        bootstrap_info = None
        async for info in prefill_stream:
            bootstrap_info = info.data()
            break

        if not bootstrap_info:
            raise RuntimeError("No bootstrap info received from prefill worker")

        return bootstrap_info

    async def _yield_bootstrap_and_process(
        self, process_func, *process_args, **process_kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield bootstrap info first, then process request (for prefill workers)"""
        bootstrap_room = self._generate_bootstrap_room()
        bootstrap_info = self._build_bootstrap_info(bootstrap_room)

        yield bootstrap_info

        try:
            await process_func(
                *process_args, bootstrap_room=bootstrap_room, **process_kwargs
            )
        except Exception as e:
            logger.error(f"Error in bootstrap processing: {e}", exc_info=True)
            # You might want to yield error info here depending on your error handling
            raise

    def _add_bootstrap_to_generation(
        self, generation_kwargs: dict, bootstrap_info: dict
    ) -> dict:
        """Add bootstrap parameters to SGLang generation kwargs"""
        generation_kwargs.update(
            {
                "bootstrap_host": bootstrap_info["bootstrap_host"],
                "bootstrap_port": bootstrap_info["bootstrap_port"],
                "bootstrap_room": bootstrap_info["bootstrap_room"],
            }
        )
        return generation_kwargs

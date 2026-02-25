# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FFN Worker Handler for AFD (Attention-FFN Disaggregation).

AFD separates stateful Attention layers (KV-cache dominated) from 
stateless FFN layers (compute-intensive) during the decode phase.

Architecture: r Attention instances -> 1 shared FFN instance

The FFN worker:
- Receives activations from multiple Attention workers
- Performs FFN computation (compute-bound with sufficient batching)
- Returns results to Attention workers

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class AFDFFNHandler(BaseWorkerHandler):
    """Handler for FFN workers in AFD disaggregated mode.
    
    In AFD mode, the FFN worker is stateless and compute-intensive,
    receiving activations from multiple Attention workers.
    
    Key characteristics:
    - Stateless computation (no KV cache)
    - Becomes compute-bound with sufficient batching
    - Shared by multiple Attention workers (r:1 topology)
    - Aggregates batch from all Attention instances
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
        attention_ratio: int = 1,
    ) -> None:
        """Initialize FFN worker handler for AFD mode.
        
        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
            attention_ratio: The r in r:1 AFD topology (number of attention workers per FFN).
        """
        super().__init__(
            component, engine, config, publisher, generate_endpoint, shutdown_event
        )
        self.attention_ratio = attention_ratio
        self._pending_requests = asyncio.Queue()
        
        logging.info(
            f"AFD FFN handler initialized - "
            f"attention_ratio={attention_ratio} (shared by {attention_ratio} Attention workers)"
        )

    def cleanup(self) -> None:
        """Shutdown the FFN engine and cleanup resources."""
        super().cleanup()
        self.engine.shutdown()
        logging.info("AFD FFN engine shutdown")

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process FFN computation for activations from Attention workers.
        
        In AFD mode, the FFN worker:
        1. Receives activations from Attention workers
        2. Aggregates batch for efficient computation
        3. Performs FFN matrix multiplications (compute-bound)
        4. Returns results to respective Attention workers
        
        Args:
            request: Request dict containing activations from Attention worker.
            context: Context object for cancellation handling.
            
        Yields:
            Response dicts with FFN computation results.
        """
        logging.debug(f"AFD FFN Request ID: {context.id()}")
        trace_id = context.trace_id
        
        # Get trace header if tracing enabled
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        
        # TODO: Implement actual AFD communication protocol
        # The actual implementation needs:
        # 1. NIXL-based activation reception from Attention workers
        # 2. Batch aggregation across Attention workers
        # 3. FFN computation (matrix multiplications)
        # 4. Result distribution back to Attention workers
        
        logging.warning(
            "AFD FFN mode is currently a placeholder. "
            "Full implementation requires NIXL activation transfer protocol."
        )
        
        # Placeholder: yield empty response
        yield {
            "token_ids": [],
            "text": None,
            "finish_reason": None,
            "meta_info": {
                "afd_mode": "ffn",
                "attention_ratio": self.attention_ratio,
                "batch_size": request.get("batch_size", 1),
            },
        }

    async def aggregate_batch(
        self, requests: list[Dict[str, Any]], timeout_ms: int = 100
    ) -> Dict[str, Any]:
        """Aggregate activations from multiple Attention workers.
        
        This method collects activations from Attention workers within
        a timeout window to form an efficient batch for FFN computation.
        
        Args:
            requests: List of activation requests from Attention workers.
            timeout_ms: Maximum time to wait for batch aggregation.
            
        Returns:
            Aggregated batch ready for FFN computation.
        """
        aggregated = {
            "activations": [],
            "request_ids": [],
            "batch_size": 0,
        }
        
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        
        for req in requests:
            if asyncio.get_event_loop().time() > deadline:
                break
            aggregated["activations"].append(req.get("activations"))
            aggregated["request_ids"].append(req.get("request_id"))
            aggregated["batch_size"] += 1
        
        logging.debug(f"Aggregated batch size: {aggregated['batch_size']}")
        return aggregated

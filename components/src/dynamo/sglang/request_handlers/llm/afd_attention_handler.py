# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Attention Worker Handler for AFD (Attention-FFN Disaggregation).

AFD separates stateful Attention layers (KV-cache dominated) from 
stateless FFN layers (compute-intensive) during the decode phase.

Architecture: r Attention instances -> 1 shared FFN instance

The Attention worker:
- Maintains KV cache state
- Performs attention computation (memory-bound)
- Transfers activations to FFN worker
- Receives outputs from FFN worker

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


class AFDAttentionHandler(BaseWorkerHandler):
    """Handler for Attention workers in AFD disaggregated mode.
    
    In AFD mode, Attention workers are stateful and memory-bound,
    dominated by KV cache reads. Multiple Attention instances feed
    into a single shared FFN worker.
    
    Key characteristics:
    - Each Attention instance maintains its own microbatch of requests
    - Attention computation time grows with sequence length (KV cache size)
    - Uses microbatch pipelining to overlap communication with computation
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
        ffn_endpoint: Optional[str] = None,
        attention_ratio: int = 1,
    ) -> None:
        """Initialize Attention worker handler for AFD mode.
        
        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
            ffn_endpoint: Endpoint for communicating with FFN worker.
            attention_ratio: The r in r:1 AFD topology (number of attention workers per FFN).
        """
        super().__init__(
            component, engine, config, publisher, generate_endpoint, shutdown_event
        )
        self.ffn_endpoint = ffn_endpoint
        self.attention_ratio = attention_ratio
        self._consume_tasks = set()
        
        logging.info(
            f"AFD Attention handler initialized - "
            f"attention_ratio={attention_ratio}, ffn_endpoint={ffn_endpoint}"
        )

    def cleanup(self) -> None:
        """Shutdown the Attention engine and cleanup resources."""
        for task in self._consume_tasks:
            if not task.done():
                task.cancel()
        self._consume_tasks.clear()
        
        super().cleanup()
        self.engine.shutdown()
        logging.info("AFD Attention engine shutdown")

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate attention output and transfer activations to FFN worker.
        
        In AFD mode, the Attention worker:
        1. Performs attention computation (reads KV cache)
        2. Transfers intermediate activations to FFN worker
        3. Waits for FFN computation results
        4. Yields final output tokens
        
        Args:
            request: Request dict with input tokens and sampling parameters.
            context: Context object for cancellation handling.
            
        Yields:
            Response dicts with token_ids and metadata.
        """
        logging.debug(f"AFD Attention Request ID: {context.id()}")
        trace_id = context.trace_id
        
        # Extract sampling parameters
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)
        
        # Get trace header if tracing enabled
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        
        # TODO: Implement actual AFD communication protocol with FFN worker
        # For now, this is a placeholder that demonstrates the architecture
        # The actual implementation needs:
        # 1. NIXL-based activation transfer to FFN worker
        # 2. Synchronization with FFN computation
        # 3. Result aggregation from FFN worker
        
        logging.warning(
            "AFD Attention mode is currently a placeholder. "
            "Full implementation requires NIXL activation transfer protocol."
        )
        
        # Placeholder: yield empty response
        yield {
            "token_ids": [],
            "text": None,
            "finish_reason": None,
            "meta_info": {"afd_mode": "attention", "attention_ratio": self.attention_ratio},
        }

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format."""
        if self.skip_tokenizer_init:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}

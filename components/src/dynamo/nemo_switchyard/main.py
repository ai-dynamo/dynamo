#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
RouteLLM Model Router for Dynamo Hierarchical Planner

Usage: python -m dynamo.nemo_switchyard [args]

A first-class Dynamo runtime component that sits between the Frontend and
per-pool local KV routers. It uses RouteLLM to classify query complexity
and routes requests to either a "strong" or "weak" model pool.

Architecture:
    Frontend (HTTP + preprocess + tokenize)
        │  dyn://
        ▼
    Model Router (this component)
        │  - Detokenizes token_ids → prompt text
        │  - RouteLLM classifies complexity → "strong" or "weak"
        │  - Forwards to appropriate pool via dyn://
        ├──────────────────────┐
        ▼                      ▼
    Strong Pool            Weak Pool
    (local KV router       (local KV router
     + workers)             + workers)
"""

import argparse
import asyncio
import logging
import os
from typing import Optional

import uvloop
from routellm.controller import Controller
from transformers import AutoTokenizer

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class ModelRouterHandler:
    """
    Routes requests to strong/weak model pools based on RouteLLM classification.

    Receives PreprocessedRequest dicts (with token_ids) from the Frontend,
    detokenizes to recover the prompt text, classifies with RouteLLM, and
    forwards to the appropriate pool's local router via dyn:// protocol.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        strong_pool_endpoint: str,
        weak_pool_endpoint: str,
        model_path: str,
        router_type: str = "mf",
        threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
    ):
        self.runtime = runtime
        self.strong_pool_endpoint = strong_pool_endpoint
        self.weak_pool_endpoint = weak_pool_endpoint
        self.model_path = model_path
        self.router_type = router_type
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path

        self.strong_client = None
        self.weak_client = None
        self.controller: Optional[Controller] = None
        self.tokenizer = None

        # Routing statistics
        self.stats = {
            "total": 0,
            "strong_routes": 0,
            "weak_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
        }

    async def initialize(self):
        """Initialize RouteLLM controller, tokenizer, and dyn:// clients."""

        # ── 1. Load tokenizer for detokenization ──
        logger.info("Loading tokenizer from %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # ── 2. Initialize RouteLLM controller ──
        logger.info(
            "Initializing RouteLLM: router=%s, threshold=%.2f",
            self.router_type,
            self.threshold,
        )
        if self.checkpoint_path:
            router_config = {
                self.router_type: {"checkpoint_path": self.checkpoint_path}
            }
        else:
            router_config = None

        self.controller = Controller(
            routers=[self.router_type],
            strong_model="strong",
            weak_model="weak",
            config=router_config,
        )

        # ── 3. Create dyn:// clients to each pool's local router ──
        for label, endpoint_path in [
            ("strong", self.strong_pool_endpoint),
            ("weak", self.weak_pool_endpoint),
        ]:
            parts = endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path for {label} pool: {endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )
            ns, comp, ep = parts
            client = await self.runtime.namespace(ns).component(comp).endpoint(ep).client()
            if label == "strong":
                self.strong_client = client
            else:
                self.weak_client = client
            logger.info("Connected to %s pool at %s", label, endpoint_path)

        logger.info("Model Router initialized successfully")

    def _classify(self, token_ids: list) -> str:
        """
        Detokenize token_ids and classify with RouteLLM.

        Returns:
            "strong" or "weak"
        """
        self.stats["total"] += 1

        # Detokenize to recover prompt text
        try:
            prompt = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            logger.exception("Failed to detokenize, falling back to strong pool")
            self.stats["fallback_routes"] += 1
            return "strong"

        if not prompt or not prompt.strip():
            logger.warning("Empty prompt after detokenization, falling back to strong pool")
            self.stats["fallback_routes"] += 1
            return "strong"

        # Classify with RouteLLM
        try:
            routed = self.controller.route(prompt, self.router_type, self.threshold)
            if routed == "strong":
                self.stats["strong_routes"] += 1
            else:
                self.stats["weak_routes"] += 1
            logger.debug(
                "Routed to %s (prompt: %.80s...)", routed, prompt.strip()
            )
            return routed
        except Exception:
            logger.exception("RouteLLM classification failed, falling back to strong pool")
            self.stats["fallback_routes"] += 1
            self.stats["errors"] += 1
            return "strong"

    async def generate(self, request):
        """
        Classify the request and forward to the appropriate pool.

        Args:
            request: PreprocessedRequest dict with token_ids, model, etc.

        Yields:
            Response dicts from the downstream pool worker.
        """
        token_ids = request.get("token_ids", [])
        choice = self._classify(token_ids)

        if choice == "strong":
            client = self.strong_client
        else:
            client = self.weak_client

        # Forward the entire PreprocessedRequest to the pool's local router.
        # Pass annotated=False so client.round_robin() returns raw data dicts
        # instead of Annotated wrapper objects. The serving ingress will wrap
        # our yields into Annotated<LLMEngineOutput> automatically.
        async for response in await client.round_robin(request, annotated=False):
            yield response


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dynamo Model Router: RouteLLM-powered routing between strong/weak model pools.\n"
            "Sits between the Frontend and per-pool local KV routers in a hierarchical deployment."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Model & endpoint configuration ──
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "HuggingFace model ID or local path for the tokenizer.\n"
            "Used for: (1) Frontend preprocessing (tokenization),\n"
            "          (2) Model Router detokenization for RouteLLM classification.\n"
            "Should match the tokenizer used by the model pools (e.g., same model family)."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Virtual model name that clients use in the 'model' field.\n"
            "Defaults to --model-path value. The Frontend will register\n"
            "this name and clients send requests to it."
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.environ.get("DYN_NAMESPACE", "hierarchical"),
        help="Dynamo namespace for the model router (default: hierarchical).",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="nemo_switchyard",
        help="Component name for service registration (default: nemo_switchyard).",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="generate",
        help="Endpoint name (default: generate).",
    )

    # ── Pool endpoints ──
    parser.add_argument(
        "--strong-pool-endpoint",
        type=str,
        required=True,
        help=(
            "Strong pool router endpoint in namespace.component.endpoint format.\n"
            "Example: strong_pool.router.generate"
        ),
    )
    parser.add_argument(
        "--weak-pool-endpoint",
        type=str,
        required=True,
        help=(
            "Weak pool router endpoint in namespace.component.endpoint format.\n"
            "Example: weak_pool.router.generate"
        ),
    )

    # ── RouteLLM configuration ──
    parser.add_argument(
        "--router-type",
        type=str,
        choices=["mf", "causal_llm", "bert", "sw_ranking"],
        default="mf",
        help="RouteLLM router algorithm (default: mf).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "RouteLLM routing threshold 0.0-1.0 (default: 0.5).\n"
            "Higher values route more requests to the weak model."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local path for the RouteLLM router checkpoint.",
    )

    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function for the Model Router."""
    args = parse_args()

    logger.info("Starting Model Router")
    logger.info("  Namespace:      %s", args.namespace)
    logger.info("  Model path:     %s", args.model_path)
    logger.info("  Strong pool:    %s", args.strong_pool_endpoint)
    logger.info("  Weak pool:      %s", args.weak_pool_endpoint)
    logger.info("  Router type:    %s", args.router_type)
    logger.info("  Threshold:      %.2f", args.threshold)

    # ── Register as a Dynamo component ──
    component = runtime.namespace(args.namespace).component(args.component)
    await component.create_service()

    endpoint = component.endpoint(args.endpoint_name)

    # Register with the Frontend so it discovers us and routes requests here.
    # ModelInput.Tokens tells the Frontend that we accept pre-tokenized requests.
    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        args.model_path,
        args.model_name,
    )
    logger.info(
        "Registered model '%s' at %s.%s.%s",
        args.model_name or args.model_path,
        args.namespace,
        args.component,
        args.endpoint_name,
    )

    # ── Initialize the handler ──
    handler = ModelRouterHandler(
        runtime=runtime,
        strong_pool_endpoint=args.strong_pool_endpoint,
        weak_pool_endpoint=args.weak_pool_endpoint,
        model_path=args.model_path,
        router_type=args.router_type,
        threshold=args.threshold,
        checkpoint_path=args.checkpoint_path,
    )
    await handler.initialize()

    # ── Serve the generate endpoint ──
    logger.info("Serving generate endpoint...")
    await endpoint.serve_endpoint(
        handler.generate,
        graceful_shutdown=True,
        metrics_labels=[("service", "nemo_switchyard")],
    )


def main():
    """Entry point for the model router."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()


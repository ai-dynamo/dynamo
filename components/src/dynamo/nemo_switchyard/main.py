#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
NeMo Switchyard — Pluggable Model Router for Dynamo Hierarchical Planner

Usage: python -m dynamo.nemo_switchyard [args]

A first-class Dynamo runtime component that sits between the Frontend and
per-pool local KV routers.  Uses a pluggable router abstraction to classify
requests and route them to the appropriate model pool.

Architecture:
    Frontend (HTTP + preprocess + tokenize)
        |  dyn://
        v
    Model Router (this component)
        |  - Router classifies request -> pool name
        |  - Forwards to appropriate pool via dyn://
        +--------- ... ---------+
        v                       v
    Pool A                  Pool B  ...
    (local KV router        (local KV router
     + workers)              + workers)
"""

import argparse
import logging
import os
import sys
import warnings

import uvloop

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# Known RouteLLM algorithm names — used to detect legacy --router-type usage
_ROUTELLM_ALGORITHMS = {"mf", "causal_llm", "bert", "sw_ranking"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dynamo Model Router: pluggable routing between N model pools.\n"
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
            "          (2) RouteLLM router detokenization for classification.\n"
            "Should match the tokenizer used by the model pools."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Virtual model name that clients use in the 'model' field.\n"
            "Defaults to --model-path value."
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

    # ── Pool endpoints (new syntax) ──
    parser.add_argument(
        "--pool",
        action="append",
        default=[],
        metavar="NAME=ENDPOINT",
        help=(
            "Add a pool: --pool strong=strong_pool.router.generate\n"
            "Repeat for each pool. Endpoint format: namespace.component.endpoint"
        ),
    )
    parser.add_argument(
        "--fallback-pool",
        type=str,
        default=None,
        help="Which pool to route to on error (default: first pool).",
    )

    # ── Legacy pool flags (backward compat) ──
    parser.add_argument(
        "--strong-pool-endpoint",
        type=str,
        default=None,
        help="[Deprecated] Use --pool strong=ENDPOINT instead.",
    )
    parser.add_argument(
        "--weak-pool-endpoint",
        type=str,
        default=None,
        help="[Deprecated] Use --pool weak=ENDPOINT instead.",
    )

    # ── Router selection ──
    parser.add_argument(
        "--router-type",
        type=str,
        default="routellm",
        help=(
            "Router implementation to use (default: routellm).\n"
            "Use --list-routers to see all available routers."
        ),
    )
    parser.add_argument(
        "--list-routers",
        action="store_true",
        help="Print available router types and exit.",
    )

    # ── RouteLLM-specific options ──
    parser.add_argument(
        "--routellm-algorithm",
        type=str,
        default=None,
        help="RouteLLM algorithm (e.g. mf, bert, causal_llm, sw_ranking). Default: mf.",
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


def _resolve_args(args):
    """Handle backward compatibility for legacy flags and auto-detect old --router-type usage."""

    # ── Legacy --strong-pool-endpoint / --weak-pool-endpoint ──
    if args.strong_pool_endpoint:
        warnings.warn(
            "--strong-pool-endpoint is deprecated; use --pool strong=ENDPOINT",
            DeprecationWarning,
            stacklevel=2,
        )
        args.pool.append(f"strong={args.strong_pool_endpoint}")
    if args.weak_pool_endpoint:
        warnings.warn(
            "--weak-pool-endpoint is deprecated; use --pool weak=ENDPOINT",
            DeprecationWarning,
            stacklevel=2,
        )
        args.pool.append(f"weak={args.weak_pool_endpoint}")

    # ── Auto-detect old RouteLLM algorithm names passed as --router-type ──
    if args.router_type in _ROUTELLM_ALGORITHMS:
        warnings.warn(
            f"--router-type {args.router_type} is deprecated; "
            f"use --router-type routellm --routellm-algorithm {args.router_type}",
            DeprecationWarning,
            stacklevel=2,
        )
        args.routellm_algorithm = args.router_type
        args.router_type = "routellm"

    # ── Parse --pool key=value pairs ──
    pool_endpoints = {}
    for spec in args.pool:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --pool format: {spec!r}. Expected NAME=ENDPOINT "
                "(e.g. strong=strong_pool.router.generate)"
            )
        name, endpoint = spec.split("=", 1)
        if name in pool_endpoints:
            raise ValueError(f"Duplicate pool name: {name!r}")
        pool_endpoints[name] = endpoint

    if not pool_endpoints:
        raise ValueError(
            "No pools configured. Use --pool NAME=ENDPOINT (at least one required)."
        )

    return pool_endpoints


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function for the Model Router."""
    args = parse_args()

    # Import routers package to trigger auto-registration
    from . import routers as _routers  # noqa: F401
    from .base import RouterConfig
    from .handler import ModelRouterHandler
    from .pool import PoolManager
    from .registry import create_router, list_routers

    # ── --list-routers ──
    if args.list_routers:
        print("Available routers:", ", ".join(list_routers()) or "(none)")
        return

    pool_endpoints = _resolve_args(args)
    pool_names = list(pool_endpoints.keys())

    logger.info("Starting Model Router")
    logger.info("  Namespace:      %s", args.namespace)
    logger.info("  Model path:     %s", args.model_path)
    logger.info("  Router type:    %s", args.router_type)
    for name, ep in pool_endpoints.items():
        logger.info("  Pool %-10s: %s", name, ep)

    # ── Build router config ──
    extra = {}
    if args.router_type == "routellm":
        extra["model_path"] = args.model_path
        extra["routellm_algorithm"] = args.routellm_algorithm or "mf"
        extra["threshold"] = args.threshold
        if args.checkpoint_path:
            extra["checkpoint_path"] = args.checkpoint_path
        logger.info("  Algorithm:      %s", extra["routellm_algorithm"])
        logger.info("  Threshold:      %.2f", args.threshold)

    config = RouterConfig(
        pool_names=pool_names,
        fallback_pool=args.fallback_pool,
        extra=extra,
    )

    router = create_router(args.router_type, config)
    pool_manager = PoolManager(runtime, pool_endpoints)
    handler = ModelRouterHandler(router, pool_manager)

    # ── Register as a Dynamo component ──
    component = runtime.namespace(args.namespace).component(args.component)
    await component.create_service()

    endpoint = component.endpoint(args.endpoint_name)

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

    # ── Initialize and serve ──
    await handler.initialize()

    try:
        logger.info("Serving generate endpoint...")
        await endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("service", "nemo_switchyard")],
        )
    except Exception as e:
        logger.error("Failed to serve endpoint: %s", e)
        raise
    finally:
        logger.debug("Cleaning up model router")
        handler.cleanup()


def main():
    """Entry point for the model router."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()

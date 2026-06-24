# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""

Arguments required for aggregated, streaming generation against an
externally-launched Megatron coordinator are defined here. Disagg, KV events,
and metrics support will be added later
"""

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    coordinator_addr: str
    model: str
    served_model_name: str | None
    context_length: int
    namespace: str
    component: str
    endpoint: str
    discovery_backend: str
    request_plane: str
    event_plane: str
    role: str  # "aggregated" | "prefill" | "decode"


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.megatron",
        description=(
            "Dynamo worker that proxies requests to a Megatron-LM "
            "DataParallelInferenceCoordinator launched separately via "
            "tools/run_dynamic_text_generation_server.py --frontend dynamo."
        ),
    )
    parser.add_argument(
        "--coordinator-addr",
        required=True,
        help=(
            "ZMQ ROUTER address of the Megatron coordinator, e.g. "
            "tcp://10.0.0.1:5555. Read from the MEGATRON_COORDINATOR_ADDR=<addr> "
            "line printed by the Megatron launcher on stdout."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "HuggingFace model id or local path used by the Dynamo frontend to "
            "load the tokenizer + config.json. Must match the tokenizer the "
            "Megatron engine was built with."
        ),
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Display name advertised to clients. Defaults to --model.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Max context length to advertise on the model card.",
    )
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument(
        "--component",
        default=None,
        help=(
            "Dynamo component name. Defaults to 'backend' for aggregated/decode "
            "and 'prefill' for --role prefill."
        ),
    ) # For use later when adding disagg support
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument("--discovery-backend", default="etcd")
    parser.add_argument("--request-plane", default="nats")
    parser.add_argument("--event-plane", default="nats")
    parser.add_argument(
        "--role",
        choices=["aggregated"],
        default="aggregated",
        help=(
            "Disagg role. 'aggregated' (default)"
        ),
    ) # For use later when adding disagg support
    args = parser.parse_args(argv)

    component = args.component
    if component is None:
        component = "backend"

    return Config(
        coordinator_addr=args.coordinator_addr,
        model=args.model,
        served_model_name=args.served_model_name or args.model,
        context_length=args.context_length,
        namespace=args.namespace,
        component=component,
        endpoint=args.endpoint,
        discovery_backend=args.discovery_backend,
        request_plane=args.request_plane,
        event_plane=args.event_plane,
        role=args.role,
    )

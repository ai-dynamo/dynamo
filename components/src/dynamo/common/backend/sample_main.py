# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the sample backend.

Usage:
    python -m dynamo.common.backend.sample_main --model-name test-model
"""

import argparse

import uvloop

from dynamo.common.backend import BackendConfig, DynamoBackend
from dynamo.common.backend.sample_engine import SampleDynamoEngine


async def worker():
    parser = argparse.ArgumentParser(description="Sample Dynamo backend")
    parser.add_argument("--model-name", default="sample-model")
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--component", default="sample")
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--endpoint-types", default="chat,completions")
    parser.add_argument("--discovery-backend", default="etcd")
    parser.add_argument("--request-plane", default="nats")
    parser.add_argument("--event-plane", default="nats")
    args = parser.parse_args()

    engine = SampleDynamoEngine(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        delay=args.delay,
    )
    backend_config = BackendConfig(
        namespace=args.namespace,
        component=args.component,
        endpoint=args.endpoint,
        model_name=args.model_name,
        served_model_name=args.model_name,
        endpoint_types=args.endpoint_types,
        discovery_backend=args.discovery_backend,
        request_plane=args.request_plane,
        event_plane=args.event_plane,
    )
    model = DynamoBackend(backend_config, engine)
    await model.run()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Mock NIM Frontend - Polls the mock NIM backend for metrics

This script demonstrates how to poll a custom backend for metrics using
the Dynamo runtime in static mode (no etcd required, uses NATS only).
"""
import asyncio
import json
import signal

import uvloop

from dynamo.runtime import DistributedRuntime


async def poll_custom_backend_metrics(
    runtime, namespace_component_endpoint, interval_secs
):
    """Poll custom backend metrics and print the data"""

    print(
        f"Starting custom backend metrics polling: endpoint={namespace_component_endpoint}, interval={interval_secs}s"
    )

    # Parse endpoint string (namespace.component.endpoint)
    parts = namespace_component_endpoint.split(".")
    if len(parts) != 3:
        print(f"ERROR: Invalid endpoint format: {namespace_component_endpoint}")
        return

    namespace, component_name, endpoint_name = parts
    print(f"Polling {namespace}/{component_name}/{endpoint_name}")

    try:
        # Get the component and endpoint
        ns = runtime.namespace(namespace)
        component = ns.component(component_name)
        endpoint = component.endpoint(endpoint_name)

        # Get client (in static mode, no need to wait for instances)
        client = await endpoint.client()
        print("Client created for static endpoint")

    except Exception as e:
        print(f"ERROR during polling setup: {e}")
        import traceback

        traceback.print_exc()
        return

    # Poll loop
    print(f"Starting polling loop (every {interval_secs}s)...")
    while True:
        try:
            await asyncio.sleep(interval_secs)
            print(f"\n{'='*60}")
            print(f"Polling tick at {asyncio.get_event_loop().time():.2f}")

            # Send request and collect responses
            # In static mode, use client.static() or client.generate()
            response_stream = await client.generate("")
            responses = []
            async for response in response_stream:
                if response.data():
                    responses.append(response.data())

            print(f"Received {len(responses)} responses")
            for idx, data in enumerate(responses):
                print(f"\nResponse #{idx+1}:")
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        print(json.dumps(parsed, indent=2))
                    except json.JSONDecodeError:
                        print(data)
                else:
                    print(data)
            print(f"{'='*60}\n")

        except asyncio.CancelledError:
            print("Polling cancelled")
            break
        except Exception as e:
            print(f"ERROR polling backend: {e}")
            import traceback

            traceback.print_exc()
            await asyncio.sleep(interval_secs)


async def graceful_shutdown(runtime):
    """Gracefully shutdown the runtime"""
    print("\nShutting down...")
    runtime.shutdown()


async def async_main():
    """Main async function - similar to frontend/main.py"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mock NIM Frontend - Poll backend for metrics"
    )
    parser.add_argument(
        "--custom-backend-metrics-endpoint",
        type=str,
        default="nim.backend.runtime_stats",
        help="Custom backend metrics endpoint in format 'namespace.component.endpoint' (default: 'nim.backend.runtime_stats')",
    )
    parser.add_argument(
        "--polling-interval",
        type=float,
        default=3.0,
        help="Polling interval in seconds (default: 3.0)",
    )
    args = parser.parse_args()

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Create DistributedRuntime - similar to frontend/main.py line 246
    runtime = DistributedRuntime(loop, "file", "nats")  # type: ignore[call-arg]

    # Setup signal handlers for graceful shutdown
    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    print("Mock NIM Frontend starting...")
    print(f"Target endpoint: {args.custom_backend_metrics_endpoint}")
    print(f"Polling interval: {args.polling_interval}s")
    print("Static mode: No etcd required, using NATS only\n")

    try:
        # Start polling
        await poll_custom_backend_metrics(
            runtime, args.custom_backend_metrics_endpoint, args.polling_interval
        )
    except asyncio.exceptions.CancelledError:
        pass


def main():
    """Entry point - similar to frontend/main.py"""
    uvloop.run(async_main())


if __name__ == "__main__":
    main()

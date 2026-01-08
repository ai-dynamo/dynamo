# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI entry point for scale testing tool.

Provides commands to start, run, and cleanup scale test DGD deployments
on Kubernetes.
"""

import argparse
import asyncio
import logging
import sys

from tests.scale_test.load_generator import LoadGenerator
from tests.scale_test.scale_manager import ScaleManager
from tests.scale_test.utils import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.scale_test",
        description="Scale testing tool for Dynamo DGD deployments on Kubernetes",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'start' command - start deployments and wait for manual testing
    start_parser = subparsers.add_parser(
        "start",
        help="Deploy N DGDs and wait (for manual testing)",
    )
    _add_common_args(start_parser)

    # 'run' command - start + load + cleanup
    run_parser = subparsers.add_parser(
        "run",
        help="Run full test: deploy DGDs + load test + cleanup",
    )
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Load test duration in seconds (default: 60)",
    )
    run_parser.add_argument(
        "--qps",
        type=float,
        default=1.0,
        help="Queries per second to send (default: 1.0)",
    )

    # 'cleanup' command - cleanup any leftover DGDs
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Cleanup any leftover scale test DGDs",
    )
    cleanup_parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)",
    )
    cleanup_parser.add_argument(
        "--name-prefix",
        type=str,
        default="scale-test",
        help="DGD name prefix to match (default: scale-test)",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between 'start' and 'run' commands."""
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of DGD deployments to create (default: 5)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model path for tokenizer (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=10.0,
        help="Mocker speedup multiplier (default: 10.0)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace for DGD deployments (default: default)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="nvcr.io/nvidia/ai-dynamo/dynamo-base:latest",
        help="Container image for all services (default: nvcr.io/nvidia/ai-dynamo/dynamo-base:latest)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for DGDs to become ready (default: 600)",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="scale-test",
        help="Prefix for DGD names (default: scale-test)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not cleanup DGDs on exit (useful for debugging)",
    )


async def cmd_start_async(args: argparse.Namespace) -> int:
    """Handle the 'start' command - deploy DGDs and wait."""
    logger.info(f"Starting {args.count} DGD deployments...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Speedup ratio: {args.speedup_ratio}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Image: {args.image}")

    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        kubernetes_namespace=args.namespace,
        image=args.image,
        timeout=args.timeout,
        name_prefix=args.name_prefix,
        cleanup_on_exit=not args.no_cleanup,
    )

    try:
        print(f"\nDeploying {args.count} DGD resources to Kubernetes...")
        print(f"Namespace: {args.namespace}")
        print(f"Image: {args.image}")

        await manager._init_kubernetes()
        await manager.deploy_dgds()

        print("\nWaiting for all DGDs to be ready...")
        if not await manager.wait_for_dgds_ready():
            print("ERROR: Not all DGDs became ready in time")
            await manager.cleanup()
            return 1

        print("\n" + "=" * 60)
        print("All DGDs ready!")
        print("=" * 60)

        print("\nDeployment Names:")
        for name in manager._deployment_names:
            print(f"  - {name}")

        frontend_urls = await manager.get_frontend_urls()
        print("\nFrontend Service URLs (cluster-internal):")
        for url in frontend_urls:
            print(f"  - {url}")

        print("\nTo run load tests from within the cluster:")
        print(f"  kubectl exec -it <pod> -- curl {frontend_urls[0]}/health")

        print("\nPress Ctrl+C to cleanup and exit...")

        # Wait indefinitely
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
    except Exception as e:
        logger.exception(f"Error during start: {e}")
        await manager.cleanup()
        return 1
    finally:
        if manager.cleanup_on_exit:
            await manager.cleanup()

    return 0


async def cmd_run_async(args: argparse.Namespace) -> int:
    """Handle the 'run' command - deploy, load test, and cleanup."""
    logger.info(f"Running scale test with {args.count} DGD deployments...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Speedup ratio: {args.speedup_ratio}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Duration: {args.duration}s")
    logger.info(f"QPS: {args.qps}")

    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        kubernetes_namespace=args.namespace,
        image=args.image,
        timeout=args.timeout,
        name_prefix=args.name_prefix,
        cleanup_on_exit=True,  # Always cleanup after run
    )

    try:
        print(f"\nDeploying {args.count} DGD resources to Kubernetes...")
        await manager._init_kubernetes()
        await manager.deploy_dgds()

        print("\nWaiting for all DGDs to be ready...")
        if not await manager.wait_for_dgds_ready():
            print("ERROR: Not all DGDs became ready in time")
            await manager.cleanup()
            return 1

        print("\n" + "=" * 60)
        print("All DGDs ready!")
        print("=" * 60)

        # Get frontend URLs for load generation
        frontend_urls = await manager.get_frontend_urls()
        print(f"\nGenerating load for {args.duration} seconds at {args.qps} QPS...")
        print(f"Targeting {len(frontend_urls)} frontends...")

        if not frontend_urls:
            print("ERROR: No frontend URLs found")
            await manager.cleanup()
            return 1

        load_generator = LoadGenerator(
            frontend_urls=frontend_urls,
            model=args.model_path,
        )

        # Run load generation
        await load_generator.generate_load(
            duration_sec=args.duration,
            qps=args.qps,
        )

        print("\nLoad generation complete.")
        load_generator.print_summary()

        # Cleanup
        print("\nCleaning up DGDs...")
        await manager.cleanup()
        print("All DGDs deleted.")

        return 0

    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
        await manager.cleanup()
        return 0
    except Exception as e:
        logger.exception(f"Error during run: {e}")
        await manager.cleanup()
        return 1


async def cmd_cleanup_async(args: argparse.Namespace) -> int:
    """Handle the 'cleanup' command - delete any leftover DGDs."""
    from kubernetes_asyncio import client, config
    from kubernetes_asyncio.client import exceptions

    logger.info(f"Cleaning up scale test DGDs in namespace {args.namespace}...")

    try:
        # Initialize Kubernetes
        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        custom_api = client.CustomObjectsApi(k8s_client)

        # List all DGDs in the namespace
        dgds = await custom_api.list_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=args.namespace,
            plural="dynamographdeployments",
        )

        # Filter by name prefix
        matching_dgds = [
            dgd["metadata"]["name"]
            for dgd in dgds.get("items", [])
            if dgd["metadata"]["name"].startswith(args.name_prefix)
        ]

        if not matching_dgds:
            print(
                f"No DGDs found with prefix '{args.name_prefix}' in namespace '{args.namespace}'"
            )
            return 0

        print(f"Found {len(matching_dgds)} DGDs to delete:")
        for name in matching_dgds:
            print(f"  - {name}")

        # Delete each DGD
        deleted_count = 0
        for name in matching_dgds:
            try:
                await custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=args.namespace,
                    plural="dynamographdeployments",
                    name=name,
                )
                print(f"Deleted: {name}")
                deleted_count += 1
            except exceptions.ApiException as e:
                if e.status == 404:
                    print(f"Already deleted: {name}")
                else:
                    logger.warning(f"Error deleting {name}: {e}")

        print(f"\nDeleted {deleted_count} DGDs.")
        return 0

    except Exception as e:
        logger.exception(f"Error during cleanup: {e}")
        return 1


def cmd_start(args: argparse.Namespace) -> int:
    """Wrapper to run async start command."""
    return asyncio.run(cmd_start_async(args))


def cmd_run(args: argparse.Namespace) -> int:
    """Wrapper to run async run command."""
    return asyncio.run(cmd_run_async(args))


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Wrapper to run async cleanup command."""
    return asyncio.run(cmd_cleanup_async(args))


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to appropriate command handler
    if args.command == "start":
        return cmd_start(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "cleanup":
        return cmd_cleanup(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

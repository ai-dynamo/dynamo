# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI entry point for scale testing tool.

Provides commands to start, run, and cleanup scale test deployments.
"""

import argparse
import asyncio
import logging
import sys
import time

from tests.scale_test.load_generator import LoadGenerator
from tests.scale_test.scale_manager import ScaleManager, setup_signal_handlers
from tests.scale_test.utils import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.scale_test",
        description="Scale testing tool for Dynamo mocker instances",
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
        help="Start N deployments and wait (for manual testing)",
    )
    _add_common_args(start_parser)

    # 'run' command - start + load + cleanup
    run_parser = subparsers.add_parser(
        "run",
        help="Run full test: start + load + cleanup",
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

    # 'cleanup' command - cleanup any leftover processes
    subparsers.add_parser(
        "cleanup",
        help="Cleanup any leftover scale test processes",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between 'start' and 'run' commands."""
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of deployments to create (default: 5)",
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
        "--base-port",
        type=int,
        default=8001,
        help="Starting frontend port (default: 8001)",
    )
    parser.add_argument(
        "--display-output",
        action="store_true",
        help="Display process output to console",
    )


def cmd_start(args: argparse.Namespace) -> int:
    """Handle the 'start' command - start and wait."""
    logger.info(f"Starting {args.count} deployments...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Speedup ratio: {args.speedup_ratio}")
    logger.info(f"Base frontend port: {args.base_port}")

    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        base_frontend_port=args.base_port,
        display_output=args.display_output,
    )

    # Set up signal handlers for graceful cleanup
    setup_signal_handlers(manager)

    try:
        print("\nStarting shared NATS and etcd...")
        manager.start_infrastructure()
        print(f"NATS started on port {manager.nats_port}")
        print(f"etcd started on port {manager.etcd_port}")

        print(f"\nStarting {args.count} mocker processes...")
        manager.start_mockers()

        print(f"\nStarting {args.count} frontend processes...")
        manager.start_frontends()

        # Wait for all services to be ready
        print("\nWaiting for all services to be ready...")
        if not manager.wait_for_all_ready(timeout=120):
            print("ERROR: Not all services became ready in time")
            manager.cleanup()
            return 1

        print("\n" + "=" * 60)
        print("All services ready!")
        print("=" * 60)
        print("\nFrontend URLs:")
        for url in manager.get_frontend_urls():
            print(f"  {url}")
        print("\nPress Ctrl+C to stop all services...")

        # Wait indefinitely
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
        manager.cleanup()
        return 0
    except Exception as e:
        logger.exception(f"Error during start: {e}")
        manager.cleanup()
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the 'run' command - start, load test, and cleanup."""
    logger.info(f"Running scale test with {args.count} deployments...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Speedup ratio: {args.speedup_ratio}")
    logger.info(f"Base frontend port: {args.base_port}")
    logger.info(f"Duration: {args.duration}s")
    logger.info(f"QPS: {args.qps}")

    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        base_frontend_port=args.base_port,
        display_output=args.display_output,
    )

    # Set up signal handlers for graceful cleanup
    setup_signal_handlers(manager)

    try:
        print("\nStarting shared NATS and etcd...")
        manager.start_infrastructure()
        print(f"NATS started on port {manager.nats_port}")
        print(f"etcd started on port {manager.etcd_port}")

        print(f"\nStarting {args.count} mocker processes...")
        manager.start_mockers()

        print(f"\nStarting {args.count} frontend processes...")
        manager.start_frontends()

        # Wait for all services to be ready
        print("\nWaiting for all services to be ready...")
        if not manager.wait_for_all_ready(timeout=120):
            print("ERROR: Not all services became ready in time")
            manager.cleanup()
            return 1

        print("\n" + "=" * 60)
        print("All services ready!")
        print("=" * 60)

        # Run load generation
        frontend_urls = manager.get_frontend_urls()
        print(f"\nGenerating load for {args.duration} seconds at {args.qps} QPS...")
        print(f"Targeting {len(frontend_urls)} frontends...")

        load_generator = LoadGenerator(
            frontend_urls=frontend_urls,
            model=args.model_path,
        )

        # Run the async load generator
        asyncio.run(
            load_generator.generate_load(
                duration_sec=args.duration,
                qps=args.qps,
            )
        )

        print("\nLoad generation complete.")
        load_generator.print_summary()

        # Cleanup
        print("\nCleaning up...")
        manager.cleanup()
        print("All processes terminated.")

        return 0

    except KeyboardInterrupt:
        print("\nReceived interrupt, cleaning up...")
        manager.cleanup()
        return 0
    except Exception as e:
        logger.exception(f"Error during run: {e}")
        manager.cleanup()
        return 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Handle the 'cleanup' command - kill any leftover processes."""
    import psutil

    logger.info("Cleaning up any leftover scale test processes...")

    # Patterns to look for in command lines
    patterns = [
        "dynamo.mocker",
        "dynamo.frontend",
        "scale-test-",
    ]

    killed_count = 0

    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = proc.cmdline()
            cmdline_str = " ".join(cmdline) if cmdline else ""

            for pattern in patterns:
                if pattern in cmdline_str:
                    logger.info(f"Killing process {proc.pid}: {cmdline_str[:80]}...")
                    proc.terminate()
                    killed_count += 1
                    break

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    print(f"Killed {killed_count} processes.")
    return 0


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

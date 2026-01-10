# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import sys

from tests.scale_test.scale_manager import ScaleManager
from tests.scale_test.utils import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.scale_test",
        description="Scale testing tool for Dynamo DGD deployments on Kubernetes",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'start' command
    start_parser = subparsers.add_parser("start", help="Deploy N DGDs and wait")
    _add_common_args(start_parser)

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Deploy DGDs + load test + cleanup")
    _add_common_args(run_parser)
    _add_load_args(run_parser)

    # 'load' command
    load_parser = subparsers.add_parser("load", help="Load test existing DGDs")
    load_parser.add_argument("--namespace", type=str, default="default")
    load_parser.add_argument("--name-prefix", type=str, default="scale-test")
    load_parser.add_argument(
        "--model-path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    _add_load_args(load_parser)

    # 'cleanup' command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup leftover DGDs")
    cleanup_parser.add_argument("--namespace", type=str, default="default")
    cleanup_parser.add_argument("--name-prefix", type=str, default="scale-test")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--count", type=int, default=5, help="Number of DGD deployments"
    )
    parser.add_argument(
        "--model-path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--speedup-ratio", type=float, default=10.0)
    parser.add_argument("--namespace", type=str, default="default")
    parser.add_argument(
        "--timeout", type=int, default=1200, help="DGD ready timeout in seconds"
    )
    parser.add_argument("--name-prefix", type=str, default="scale-test")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep DGDs on exit")
    parser.add_argument(
        "--worker-replicas",
        type=int,
        default=1,
        help="Number of mocker worker replicas per deployment",
    )


def _add_load_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--duration", type=int, default=60, help="Load test duration in seconds"
    )
    parser.add_argument("--qps", type=float, default=1.0, help="Queries per second")
    parser.add_argument(
        "--load-gen-pods", type=int, default=1, help="Parallel load generator pods"
    )
    parser.add_argument(
        "--load-gen-processes", type=int, default=1, help="Processes per pod"
    )


async def cmd_start_async(args: argparse.Namespace) -> int:
    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        kubernetes_namespace=args.namespace,
        timeout=args.timeout,
        name_prefix=args.name_prefix,
        cleanup_on_exit=not args.no_cleanup,
        worker_replicas=args.worker_replicas,
    )

    try:
        print(f"\nDeploying {args.count} DGDs to namespace {args.namespace}...")
        await manager._init_kubernetes()
        await manager.deploy_dgds()

        print("Waiting for DGDs to be ready...")
        if not await manager.wait_for_dgds_ready():
            print("ERROR: Not all DGDs became ready")
            await manager.cleanup()
            return 1

        print("\n" + "=" * 60)
        print("All DGDs ready!")
        print("=" * 60)

        for name in manager._deployment_names:
            print(f"  - {name}")

        frontend_urls = await manager.get_frontend_urls()
        print("\nFrontend URLs:")
        for url in frontend_urls:
            print(f"  - {url}")

        print("\nPress Ctrl+C to cleanup and exit...")
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print("\nCleaning up...")
    finally:
        if manager.cleanup_on_exit:
            await manager.cleanup()

    return 0


async def cmd_run_async(args: argparse.Namespace) -> int:
    manager = ScaleManager(
        num_deployments=args.count,
        model_path=args.model_path,
        speedup_ratio=args.speedup_ratio,
        kubernetes_namespace=args.namespace,
        timeout=args.timeout,
        name_prefix=args.name_prefix,
        cleanup_on_exit=True,
        worker_replicas=args.worker_replicas,
    )

    try:
        print(f"\nDeploying {args.count} DGDs to namespace {args.namespace}...")
        await manager._init_kubernetes()
        await manager.deploy_dgds()

        print("Waiting for DGDs to be ready...")
        if not await manager.wait_for_dgds_ready():
            print("ERROR: Not all DGDs became ready")
            await manager.cleanup()
            return 1

        print("\nAll DGDs ready!")
        print(f"\nGenerating load for {args.duration}s at {args.qps} QPS...")

        num_pods = getattr(args, "load_gen_pods", 1)
        num_processes = getattr(args, "load_gen_processes", 1)
        if num_pods > 1 or num_processes > 1:
            print(f"Parallelism: {num_pods} pod(s) x {num_processes} process(es)")

        success = await manager.run_load_generator_job(
            model=args.model_path,
            duration_sec=args.duration,
            qps=args.qps,
            timeout=args.duration + 300,
            num_pods=num_pods,
            num_processes_per_pod=num_processes,
        )
        if not success:
            print("ERROR: Load generation failed")
            await manager.cleanup()
            return 1

        print("\nCleaning up...")
        await manager.cleanup()
        return 0

    except KeyboardInterrupt:
        print("\nCleaning up...")
        await manager.cleanup()
        return 0


async def cmd_load_async(args: argparse.Namespace) -> int:
    from kubernetes_asyncio import client, config
    from kubernetes_asyncio.client import exceptions

    try:
        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        custom_api = client.CustomObjectsApi(k8s_client)
        core_api = client.CoreV1Api(k8s_client)

        dgds = await custom_api.list_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=args.namespace,
            plural="dynamographdeployments",
        )

        matching_dgds = [
            dgd["metadata"]["name"]
            for dgd in dgds.get("items", [])
            if dgd["metadata"]["name"].startswith(args.name_prefix)
        ]

        if not matching_dgds:
            print(
                f"ERROR: No DGDs found with prefix '{args.name_prefix}' in namespace '{args.namespace}'"
            )
            return 1

        print(f"Found {len(matching_dgds)} DGDs")

        frontend_urls = []
        for dgd_name in sorted(matching_dgds):
            service_name = f"{dgd_name}-frontend"
            try:
                await core_api.read_namespaced_service(
                    name=service_name, namespace=args.namespace
                )
                url = f"http://{service_name}.{args.namespace}.svc.cluster.local:8000"
                frontend_urls.append(url)
            except exceptions.ApiException as e:
                if e.status != 404:
                    logger.warning(f"Error getting service {service_name}: {e}")

        if not frontend_urls:
            print("ERROR: No frontend services found")
            return 1

        print(f"Targeting {len(frontend_urls)} frontends")
        print(f"Generating load for {args.duration}s at {args.qps} QPS...")

        num_pods = getattr(args, "load_gen_pods", 1)
        num_processes = getattr(args, "load_gen_processes", 1)

        from tests.scale_test.load_generator_job import LoadGeneratorJob

        batch_api = client.BatchV1Api(k8s_client)
        job = LoadGeneratorJob(
            namespace=args.namespace,
            frontend_urls=frontend_urls,
            model=args.model_path,
            duration_sec=args.duration,
            qps=args.qps,
            num_pods=num_pods,
            num_processes_per_pod=num_processes,
        )

        success = await job.create_and_wait(
            batch_api, core_api, timeout=args.duration + 300
        )
        await job.delete()

        if not success:
            print("ERROR: Load generation failed")
            return 1

        print("\nLoad generation complete.")
        return 0

    except KeyboardInterrupt:
        print("\nExiting...")
        return 0


async def cmd_cleanup_async(args: argparse.Namespace) -> int:
    from kubernetes_asyncio import client, config
    from kubernetes_asyncio.client import exceptions

    try:
        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        custom_api = client.CustomObjectsApi(k8s_client)

        dgds = await custom_api.list_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=args.namespace,
            plural="dynamographdeployments",
        )

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

        print(f"Deleting {len(matching_dgds)} DGDs...")
        for name in matching_dgds:
            try:
                await custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=args.namespace,
                    plural="dynamographdeployments",
                    name=name,
                )
                print(f"  Deleted: {name}")
            except exceptions.ApiException as e:
                if e.status == 404:
                    print(f"  Already deleted: {name}")
                else:
                    logger.warning(f"Error deleting {name}: {e}")

        return 0
    except Exception as e:
        logger.exception(f"Error during cleanup: {e}")
        return 1


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "start": cmd_start_async,
        "run": cmd_run_async,
        "load": cmd_load_async,
        "cleanup": cmd_cleanup_async,
    }

    handler = handlers.get(args.command)
    if handler:
        return asyncio.run(handler(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

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

    # 'aiperf' command - run AIPerf directly against URLs
    aiperf_parser = subparsers.add_parser(
        "aiperf", help="Run AIPerf load test directly against URL(s)"
    )
    aiperf_parser.add_argument(
        "--url",
        type=str,
        nargs="+",
        required=True,
        help="Frontend URL(s) to target (e.g., http://localhost:8000)",
    )
    aiperf_parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name",
    )
    aiperf_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model)",
    )
    aiperf_parser.add_argument("--namespace", type=str, default="default")
    aiperf_parser.add_argument(
        "--duration", type=int, default=60, help="Load test duration in seconds"
    )
    aiperf_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Container image with AIPerf installed (e.g., Dynamo image)",
    )
    # AIPerf parameters
    aiperf_parser.add_argument(
        "--isl-mean", type=int, default=512, help="Mean input sequence length"
    )
    aiperf_parser.add_argument(
        "--isl-stddev", type=int, default=0, help="Input sequence length stddev"
    )
    aiperf_parser.add_argument(
        "--osl-mean", type=int, default=128, help="Mean output sequence length"
    )
    aiperf_parser.add_argument(
        "--osl-stddev", type=int, default=0, help="Output sequence length stddev"
    )
    aiperf_parser.add_argument(
        "--concurrency", type=int, default=10, help="Number of concurrent requests"
    )
    aiperf_parser.add_argument(
        "--request-count",
        type=int,
        default=0,
        help="Total requests to send, 0 for duration-based",
    )
    aiperf_parser.add_argument(
        "--prefix-prompt-length",
        type=int,
        default=0,
        help="Shared prefix prompt length for prefix caching simulation",
    )
    aiperf_parser.add_argument(
        "--num-prefix-prompts",
        type=int,
        default=0,
        help="Number of distinct prefix prompts",
    )
    aiperf_parser.add_argument(
        "--collect-frontend-logs",
        action="store_true",
        help="Collect frontend pod logs after the load test",
    )
    aiperf_parser.add_argument(
        "--frontend-pod",
        type=str,
        nargs="*",
        default=None,
        help="Frontend pod name(s) to collect logs from (auto-detected if not specified)",
    )
    aiperf_parser.add_argument(
        "--log-output-dir",
        type=str,
        default=None,
        help="Directory to save frontend logs (defaults to ./frontend_logs)",
    )
    aiperf_parser.add_argument(
        "--log-flush-delay",
        type=int,
        default=5,
        help="Seconds to wait after aiperf completes before collecting logs (default: 5). "
        "Allows frontend pod to flush buffered logs.",
    )

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
    parser.add_argument(
        "--load-generator",
        type=str,
        choices=["simple", "aiperf"],
        default="simple",
        help="Load generator type: 'simple' (lightweight Python) or 'aiperf' (full benchmarking)",
    )

    # Simple load generator args
    parser.add_argument("--qps", type=float, default=1.0, help="Queries per second (simple mode)")
    parser.add_argument(
        "--load-gen-pods", type=int, default=1, help="Parallel load generator pods (simple mode)"
    )
    parser.add_argument(
        "--load-gen-processes", type=int, default=1, help="Processes per pod (simple mode)"
    )

    # AIPerf args
    parser.add_argument(
        "--isl-mean", type=int, default=512, help="Mean input sequence length (aiperf mode)"
    )
    parser.add_argument(
        "--isl-stddev", type=int, default=0, help="Input sequence length stddev (aiperf mode)"
    )
    parser.add_argument(
        "--osl-mean", type=int, default=128, help="Mean output sequence length (aiperf mode)"
    )
    parser.add_argument(
        "--osl-stddev", type=int, default=0, help="Output sequence length stddev (aiperf mode)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Number of concurrent requests (aiperf mode)"
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=0,
        help="Total requests to send, 0 for duration-based (aiperf mode)",
    )
    parser.add_argument(
        "--prefix-prompt-length",
        type=int,
        default=0,
        help="Shared prefix prompt length for prefix caching (aiperf mode)",
    )
    parser.add_argument(
        "--num-prefix-prompts",
        type=int,
        default=0,
        help="Number of distinct prefix prompts (aiperf mode)",
    )
    parser.add_argument(
        "--aiperf-image",
        type=str,
        default=None,
        help="Container image with AIPerf installed (required for aiperf mode). "
        "Use the same Dynamo image used for mocker workers.",
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

        load_generator = getattr(args, "load_generator", "simple")

        if load_generator == "aiperf":
            print(f"\nRunning AIPerf load test for {args.duration}s...")
            print(f"  Concurrency: {args.concurrency}")
            print(f"  ISL: {args.isl_mean} (stddev: {args.isl_stddev})")
            print(f"  OSL: {args.osl_mean} (stddev: {args.osl_stddev})")

            from tests.scale_test.aiperf_load_generator_job import AIPerfConfig

            aiperf_config = AIPerfConfig(
                isl_mean=args.isl_mean,
                isl_stddev=args.isl_stddev,
                osl_mean=args.osl_mean,
                osl_stddev=args.osl_stddev,
                concurrency=args.concurrency,
                request_count=getattr(args, "request_count", 0),
                prefix_prompt_length=getattr(args, "prefix_prompt_length", 0),
                num_prefix_prompts=getattr(args, "num_prefix_prompts", 0),
            )

            success = await manager.run_aiperf_load_generator(
                model=args.model_path,
                duration_sec=args.duration,
                config=aiperf_config,
                timeout=args.duration + 300,
                image=getattr(args, "aiperf_image", None),
            )
        else:
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

        load_generator = getattr(args, "load_generator", "simple")
        batch_api = client.BatchV1Api(k8s_client)

        if load_generator == "aiperf":
            print(f"Running AIPerf load test for {args.duration}s...")
            print(f"  Concurrency: {args.concurrency}")
            print(f"  ISL: {args.isl_mean} (stddev: {args.isl_stddev})")
            print(f"  OSL: {args.osl_mean} (stddev: {args.osl_stddev})")

            from tests.scale_test.aiperf_load_generator_job import (
                AIPerfConfig,
                MultiTargetAIPerfJob,
            )

            aiperf_config = AIPerfConfig(
                isl_mean=args.isl_mean,
                isl_stddev=args.isl_stddev,
                osl_mean=args.osl_mean,
                osl_stddev=args.osl_stddev,
                concurrency=args.concurrency,
                request_count=getattr(args, "request_count", 0),
                prefix_prompt_length=getattr(args, "prefix_prompt_length", 0),
                num_prefix_prompts=getattr(args, "num_prefix_prompts", 0),
            )

            job = MultiTargetAIPerfJob(
                namespace=args.namespace,
                frontend_urls=frontend_urls,
                model=args.model_path,
                duration_sec=args.duration,
                config=aiperf_config,
                image=getattr(args, "aiperf_image", None),
            )
        else:
            print(f"Generating load for {args.duration}s at {args.qps} QPS...")

            num_pods = getattr(args, "load_gen_pods", 1)
            num_processes = getattr(args, "load_gen_processes", 1)

            from tests.scale_test.load_generator_job import LoadGeneratorJob

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


async def cmd_aiperf_async(args: argparse.Namespace) -> int:
    """Run AIPerf directly against specified URL(s)."""
    from kubernetes_asyncio import client, config

    k8s_client = None
    log_streaming_task = None
    log_file_path = None

    try:
        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        batch_api = client.BatchV1Api(k8s_client)
        core_api = client.CoreV1Api(k8s_client)

        urls = args.url
        print(f"Running AIPerf load test against {len(urls)} URL(s):")
        for url in urls:
            print(f"  - {url}")
        print(f"\nConfiguration:")
        print(f"  Duration: {args.duration}s")
        print(f"  Concurrency: {args.concurrency}")
        print(f"  ISL: {args.isl_mean} (stddev: {args.isl_stddev})")
        print(f"  OSL: {args.osl_mean} (stddev: {args.osl_stddev})")
        if args.prefix_prompt_length > 0:
            print(f"  Prefix: {args.prefix_prompt_length} tokens, {args.num_prefix_prompts} prompts")
        print()

        # Start log streaming BEFORE the aiperf job begins
        if getattr(args, "collect_frontend_logs", False):
            log_streaming_task, log_file_path = await _start_frontend_log_streaming(
                args, core_api
            )

        from tests.scale_test.aiperf_load_generator_job import (
            AIPerfConfig,
            AIPerfLoadGeneratorJob,
            MultiTargetAIPerfJob,
        )

        aiperf_config = AIPerfConfig(
            isl_mean=args.isl_mean,
            isl_stddev=args.isl_stddev,
            osl_mean=args.osl_mean,
            osl_stddev=args.osl_stddev,
            concurrency=args.concurrency,
            request_count=args.request_count,
            prefix_prompt_length=args.prefix_prompt_length,
            num_prefix_prompts=args.num_prefix_prompts,
        )

        if len(urls) == 1:
            job = AIPerfLoadGeneratorJob(
                namespace=args.namespace,
                frontend_url=urls[0],
                model=args.model,
                duration_sec=args.duration,
                config=aiperf_config,
                image=args.image,
                tokenizer=args.tokenizer,
            )
        else:
            job = MultiTargetAIPerfJob(
                namespace=args.namespace,
                frontend_urls=urls,
                model=args.model,
                duration_sec=args.duration,
                config=aiperf_config,
                image=args.image,
                tokenizer=args.tokenizer,
            )

        success = await job.create_and_wait(
            batch_api, core_api, timeout=args.duration + 300
        )
        await job.delete()

        if not success:
            print("ERROR: AIPerf load generation failed")
            return 1

        print("\nAIPerf load generation complete.")

        # Stop log streaming and finalize
        if log_streaming_task is not None:
            await _stop_frontend_log_streaming(args, log_streaming_task, log_file_path)

        return 0

    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    finally:
        # Ensure log streaming is stopped on any exit
        if log_streaming_task is not None and not log_streaming_task.done():
            log_streaming_task.cancel()
            try:
                await log_streaming_task
            except asyncio.CancelledError:
                pass
        if k8s_client:
            await k8s_client.close()


async def _get_frontend_pod_names(
    args: argparse.Namespace,
    core_api,  # kubernetes_asyncio.client.CoreV1Api
) -> list:
    """Get frontend pod names from args or auto-detect."""
    from kubernetes_asyncio.client import exceptions

    pod_names = list(args.frontend_pod or [])

    if not pod_names:
        # Try to auto-detect frontend pods from the URLs
        for url in args.url:
            try:
                host_part = url.replace("http://", "").replace("https://", "")
                host = host_part.split(":")[0]
                service_name = host.split(".")[0]

                for label_selector in [
                    f"app={service_name}",
                    f"nvidia.com/dgd-service=Frontend",
                ]:
                    try:
                        pod_list = await core_api.list_namespaced_pod(
                            namespace=args.namespace,
                            label_selector=label_selector,
                        )
                        for pod in pod_list.items:
                            if pod.status.phase == "Running":
                                pod_names.append(pod.metadata.name)
                        if pod_names:
                            break
                    except exceptions.ApiException:
                        pass
            except Exception as e:
                logger.debug(f"Could not parse URL {url}: {e}")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(pod_names))


async def _start_frontend_log_streaming(
    args: argparse.Namespace,
    core_api,  # kubernetes_asyncio.client.CoreV1Api
) -> tuple:
    """Start streaming frontend logs to a file BEFORE the load test begins.

    Returns:
        Tuple of (streaming_task, log_file_path) or (None, None) if no pods found.
    """
    from kubernetes_asyncio.client import exceptions

    pod_names = await _get_frontend_pod_names(args, core_api)

    if not pod_names:
        print("\nNo frontend pods found for log collection.")
        print("Specify pod names with --frontend-pod or ensure pods have expected labels.")
        return None, None

    # For now, stream from the first pod (could be extended to multiple)
    pod_name = pod_names[0]
    if len(pod_names) > 1:
        print(f"\nNote: Multiple frontend pods found, streaming logs from: {pod_name}")

    output_dir = args.log_output_dir or "./frontend_logs"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"{pod_name}_{timestamp}.log")

    print(f"\nStarting log streaming from {pod_name} to {log_file_path}")

    # Create the streaming task
    task = asyncio.create_task(
        _stream_pod_logs_to_file(
            core_api,
            args.namespace,
            pod_name,
            log_file_path,
        )
    )

    # Give it a moment to start
    await asyncio.sleep(0.5)

    return task, log_file_path


async def _stream_pod_logs_to_file(
    core_api,
    namespace: str,
    pod_name: str,
    log_file_path: str,
) -> None:
    """Stream logs from a pod to a file continuously until cancelled."""
    from kubernetes_asyncio.client import exceptions

    logger.info(f"Starting log stream from {pod_name}")

    try:
        with open(log_file_path, "w") as f:
            # Poll for new logs continuously
            # kubernetes_asyncio doesn't support true streaming easily,
            # so we poll with a sliding window
            last_lines_seen = set()
            poll_interval = 1.0  # Poll every second
            first_poll = True

            while True:
                try:
                    # On first poll, get ALL existing logs (no since_seconds)
                    # On subsequent polls, get only recent logs (last 60 seconds)
                    if first_poll:
                        logs = await core_api.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                        )
                        first_poll = False
                    else:
                        logs = await core_api.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=namespace,
                            since_seconds=60,
                        )

                    if logs:
                        # Write only new lines we haven't seen
                        for line in logs.split("\n"):
                            if line:
                                line_hash = hash(line)
                                if line_hash not in last_lines_seen:
                                    last_lines_seen.add(line_hash)
                                    f.write(line + "\n")
                                    f.flush()

                        # Limit memory usage by keeping only recent hashes
                        if len(last_lines_seen) > 100000:
                            # Keep roughly half
                            last_lines_seen = set(list(last_lines_seen)[-50000:])

                except exceptions.ApiException as e:
                    if e.status == 404:
                        logger.warning(f"Pod {pod_name} no longer exists")
                        break
                    logger.debug(f"Error polling logs: {e}")

                await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        logger.info(f"Log streaming cancelled for {pod_name}")
        raise
    except Exception as e:
        logger.error(f"Error streaming logs from {pod_name}: {e}")


async def _stop_frontend_log_streaming(
    args: argparse.Namespace,
    task: asyncio.Task,
    log_file_path: str,
) -> None:
    """Stop log streaming and wait for final flush."""
    # Wait for any final logs to be written
    flush_delay = getattr(args, "log_flush_delay", 5)
    if flush_delay > 0:
        print(f"\nWaiting {flush_delay}s for final logs to flush...")
        await asyncio.sleep(flush_delay)

    # Cancel the streaming task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Report the saved file
    if log_file_path and os.path.exists(log_file_path):
        size = os.path.getsize(log_file_path)
        print(f"Frontend logs saved: {log_file_path} ({size} bytes)")


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
        "aiperf": cmd_aiperf_async,
    }

    handler = handlers.get(args.command)
    if handler:
        return asyncio.run(handler(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

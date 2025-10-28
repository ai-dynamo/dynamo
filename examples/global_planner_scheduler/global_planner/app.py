# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import uvloop
from aiohttp import web
from pydantic import BaseModel

from dynamo.planner.kube import KubernetesAPI

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GPUResourceTracker:
    """
    Tracks GPU resource requirements and usage across components.
    Reads GPU specifications from DynamoGraphDeployment resources.
    """

    def __init__(self, k8s_namespace: Optional[str] = None):
        """
        Initialize the GPU resource tracker.
        Args:
            k8s_namespace: Kubernetes namespace. If None, auto-detect from environment.
        """
        self.kube_api = KubernetesAPI(k8s_namespace)
        self._deployment_cache: Dict[str, dict] = {}  # Cache by DGD name
        logger.info("GPUResourceTracker initialized")

    async def get_deployment(self, dgd_name: str, refresh: bool = False) -> Optional[dict]:
        """
        Get a specific DynamoGraphDeployment by name.
        Args:
            dgd_name: Name of the DynamoGraphDeployment
            refresh: If True, force refresh from Kubernetes API
        Returns:
            The deployment dict or None if not found
        """
        if dgd_name not in self._deployment_cache or refresh:
            try:
                self._deployment_cache[dgd_name] = self.kube_api._get_graph_deployment_from_name(dgd_name)
            except Exception as e:
                logger.error(f"Failed to get deployment '{dgd_name}': {e}")
                return None
        return self._deployment_cache.get(dgd_name)

    def get_component_gpu_requirement(
        self, deployment: dict, component_name: str
    ) -> int:
        """
        Get the GPU requirement for a specific component.
        Reads from: deployment["spec"]["services"][component_name]["resources"]["limits"]["gpu"]
        Args:
            deployment: The DynamoGraphDeployment dict
            component_name: Name of the component
        Returns:
            Number of GPUs required per replica (default 0 if not specified)
        """
        try:
            gpu_str = (
                deployment.get("spec", {})
                .get("services", {})
                .get(component_name, {})
                .get("resources", {})
                .get("limits", {})
                .get("gpu", "0")
            )
            # GPU value is typically a string like "1" or "2"
            gpu_count = int(gpu_str)
            logger.debug(
                f"Component '{component_name}' requires {gpu_count} GPU(s) per replica"
            )
            return gpu_count
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to parse GPU requirement for component '{component_name}': {e}. "
                f"Defaulting to 0."
            )
            return 0

    def get_current_replicas(self, deployment: dict, component_name: str) -> int:
        """
        Get the current replica count for a component.
        Args:
            deployment: The DynamoGraphDeployment dict
            component_name: Name of the component
        Returns:
            Current number of replicas (default 1 if not specified)
        """
        replicas = (
            deployment.get("spec", {})
            .get("services", {})
            .get(component_name, {})
            .get("replicas", 1)
        )
        return replicas

    def calculate_total_gpu_usage(
        self, deployment: dict, replicas_map: Dict[str, int]
    ) -> int:
        """
        Calculate total GPU usage for a given replica configuration.
        Args:
            deployment: The DynamoGraphDeployment dict
            replicas_map: Dictionary mapping component names to replica counts
        Returns:
            Total number of GPUs required
        """
        total_gpus = 0
        for component_name, replica_count in replicas_map.items():
            gpus_per_replica = self.get_component_gpu_requirement(
                deployment, component_name
            )
            component_gpus = replica_count * gpus_per_replica
            total_gpus += component_gpus
            logger.debug(
                f"Component '{component_name}': {replica_count} replicas × "
                f"{gpus_per_replica} GPU/replica = {component_gpus} GPUs"
            )

        logger.info(f"Total GPU usage: {total_gpus}")
        return total_gpus

    def get_current_total_gpu_usage(self, deployment: dict) -> int:
        """
        Calculate current total GPU usage across all components.
        Args:
            deployment: The DynamoGraphDeployment dict
        Returns:
            Total number of GPUs currently in use
        """
        services = deployment.get("spec", {}).get("services", {})
        replicas_map = {}

        for component_name, config in services.items():
            replicas_map[component_name] = config.get("replicas", 1)

        return self.calculate_total_gpu_usage(deployment, replicas_map)

    def calculate_gpu_delta(
        self,
        deployment: dict,
        component_name: str,
        current_replicas: int,
        target_replicas: int,
    ) -> int:
        """
        Calculate the GPU difference for a scaling operation.
        Args:
            deployment: The DynamoGraphDeployment dict
            component_name: Name of the component
            current_replicas: Current replica count
            target_replicas: Desired replica count
        Returns:
            GPU delta (positive for scale-up, negative for scale-down)
        """
        gpus_per_replica = self.get_component_gpu_requirement(deployment, component_name)
        delta = (target_replicas - current_replicas) * gpus_per_replica
        logger.debug(
            f"GPU delta for '{component_name}': "
            f"({target_replicas} - {current_replicas}) × {gpus_per_replica} = {delta}"
        )
        return delta

class ScaleRequest(BaseModel):
    """Request model for scaling operations"""

    target_replicas: Dict[str, int]
    priority: int
    planner_id: str
    dgd_name: str  # Name of the DynamoGraphDeployment to scale
    blocking: bool = False
    timestamp: float = 0.0  # When the request was received


class ScaleResponse(BaseModel):
    """Response model for scaling operations"""

    success: bool
    message: str
    components_scaled: Dict[str, int]
    components_partial: Dict[str, int] = {}
    components_dropped: List[str] = []


class ScalerService:
    """
    Service that handles scaling operations with priority-based batching.
    Collects scaling requests over a batching window, then processes them
    in priority order while respecting GPU resource limits.
    """

    def __init__(
        self,
        dynamo_namespace: str,
        batching_window: int,
        max_gpu_limit: int,
        k8s_namespace: Optional[str] = None,
    ):
        """
        Initialize the scaler service.
        Args:
            dynamo_namespace: Dynamo namespace
            batching_window: Time window in seconds to batch requests
            max_gpu_limit: Maximum total GPUs allowed in the cluster
            k8s_namespace: Kubernetes namespace (auto-detect if None)
        """
        self.dynamo_namespace = dynamo_namespace
        self.batching_window = batching_window
        self.max_gpu_limit = max_gpu_limit

        # Global planner doesn't need a connector for itself - it updates DGDs directly via K8s API
        self.gpu_tracker = GPUResourceTracker(k8s_namespace=k8s_namespace)

        # Request queue: list of ScaleRequest objects
        self.request_queue: List[ScaleRequest] = []
        self.queue_lock = asyncio.Lock()

        logger.info(
            f"ScalerService initialized: batching_window={batching_window}s, "
            f"max_gpu_limit={max_gpu_limit}"
        )

    async def queue_scale_request(self, request: ScaleRequest) -> ScaleResponse:
        """
        Add a scaling request to the queue.
        Args:
            request: The scale request to queue
        Returns:
            Immediate response indicating the request was queued
        """
        request.timestamp = time.time()

        async with self.queue_lock:
            self.request_queue.append(request)
            # Log incoming request details
            component_summary = ", ".join(
                f"{comp}:{reps}" for comp, reps in request.target_replicas.items()
            )
            logger.info(
                f"📥 RECEIVED scale request from planner '{request.planner_id}' "
                f"(priority={request.priority}, DGD='{request.dgd_name}')"
            )
            logger.info(f"   Requested replicas: {component_summary}")
            logger.info(f"   Queue size: {len(self.request_queue)}")

        return ScaleResponse(
            success=True,
            message="Request queued for batch processing",
            components_scaled={},
        )

    async def process_batch(self):
        """
        Process all queued requests in priority order, respecting GPU limits.
        This is called periodically (every batching_window seconds).
        Handles scaling across multiple DynamoGraphDeployments.
        """
        async with self.queue_lock:
            if not self.request_queue:
                logger.debug("No requests in queue to process")
                return

            # Sort by priority (higher priority first)
            self.request_queue.sort(key=lambda r: r.priority, reverse=True)

            logger.info("=" * 80)
            logger.info(
                f"🔄 BATCH START: Processing {len(self.request_queue)} request(s)"
            )
            logger.info(
                f"   DGDs: {sorted(set(r.dgd_name for r in self.request_queue))}"
            )
            logger.info(
                f"   Priority order: {[(r.planner_id, r.priority) for r in self.request_queue]}"
            )

            # Calculate total current GPU usage across ALL deployments
            current_gpu_usage = 0
            all_deployments = {}

            # Get all unique DGD names from requests
            dgd_names = set(r.dgd_name for r in self.request_queue)

            for dgd_name in dgd_names:
                deployment = await self.gpu_tracker.get_deployment(dgd_name, refresh=True)
                if deployment is None:
                    logger.error(f"Failed to get deployment '{dgd_name}', skipping its requests")
                    continue
                all_deployments[dgd_name] = deployment
                dgd_usage = self.gpu_tracker.get_current_total_gpu_usage(deployment)
                current_gpu_usage += dgd_usage
                logger.info(f"DGD '{dgd_name}' current usage: {dgd_usage} GPUs")

            logger.info(
                f"Total GPU usage across all DGDs: {current_gpu_usage}/{self.max_gpu_limit}"
            )

            # Track planned scaling operations per DGD
            planned_operations: Dict[str, Dict[str, int]] = {}  # dgd_name -> {component -> replicas}

            # Process each request in priority order
            for request in self.request_queue:
                # Skip if we couldn't get this DGD's deployment
                if request.dgd_name not in all_deployments:
                    logger.warning(
                        f"Skipping request from '{request.planner_id}' for DGD '{request.dgd_name}' "
                        f"(deployment not found)"
                    )
                    continue

                deployment = all_deployments[request.dgd_name]

                logger.info(
                    f"⚙️  PROCESSING request from planner '{request.planner_id}' "
                    f"for DGD '{request.dgd_name}' (priority={request.priority})"
                )

                if request.dgd_name not in planned_operations:
                    planned_operations[request.dgd_name] = {}

                fulfilled = {}
                partial = {}
                dropped = []

                for component_name, target_replicas in request.target_replicas.items():
                    current_replicas = self.gpu_tracker.get_current_replicas(
                        deployment, component_name
                    )

                    # Calculate GPU delta for this component
                    gpu_delta = self.gpu_tracker.calculate_gpu_delta(
                        deployment, component_name, current_replicas, target_replicas
                    )

                    # Check if we can fulfill this request
                    if current_gpu_usage + gpu_delta <= self.max_gpu_limit:
                        # Full fulfillment
                        planned_operations[request.dgd_name][component_name] = target_replicas
                        current_gpu_usage += gpu_delta
                        fulfilled[component_name] = target_replicas
                        logger.info(
                            f"  ✅ ACCEPTED '{component_name}': {current_replicas} → {target_replicas} "
                            f"({gpu_delta:+d} GPUs, total={current_gpu_usage}/{self.max_gpu_limit})"
                        )
                    elif gpu_delta > 0:
                        # Scaling up but would exceed limit - try partial fulfillment
                        available_gpus = self.max_gpu_limit - current_gpu_usage
                        gpus_per_replica = self.gpu_tracker.get_component_gpu_requirement(
                            deployment, component_name
                        )

                        if available_gpus > 0 and gpus_per_replica > 0:
                            # Partial fulfillment
                            max_additional_replicas = available_gpus // gpus_per_replica
                            actual_replicas = min(
                                target_replicas,
                                current_replicas + max_additional_replicas,
                            )

                            if actual_replicas > current_replicas:
                                planned_operations[request.dgd_name][component_name] = actual_replicas
                                actual_gpu_delta = self.gpu_tracker.calculate_gpu_delta(
                                    deployment,
                                    component_name,
                                    current_replicas,
                                    actual_replicas,
                                )
                                current_gpu_usage += actual_gpu_delta
                                partial[component_name] = actual_replicas
                                logger.info(
                                    f"  ⚠️  PARTIAL '{component_name}': {current_replicas} → {actual_replicas} "
                                    f"(requested {target_replicas}, {actual_gpu_delta:+d} GPUs, "
                                    f"total={current_gpu_usage}/{self.max_gpu_limit})"
                                )
                            else:
                                dropped.append(component_name)
                                logger.info(
                                    f"  ❌ DENIED '{component_name}': no GPUs available "
                                    f"(requested {current_replicas} → {target_replicas})"
                                )
                        else:
                            dropped.append(component_name)
                            logger.info(
                                f"  ❌ DENIED '{component_name}': GPU limit reached "
                                f"(requested {current_replicas} → {target_replicas}, "
                                f"would need {gpu_delta:+d} GPUs, have {available_gpus})"
                            )
                    else:
                        # Scaling down - always allow
                        planned_operations[request.dgd_name][component_name] = target_replicas
                        current_gpu_usage += gpu_delta
                        fulfilled[component_name] = target_replicas
                        logger.info(
                            f"  ✅ ACCEPTED '{component_name}': {current_replicas} → {target_replicas} "
                            f"({gpu_delta:+d} GPUs, total={current_gpu_usage}/{self.max_gpu_limit})"
                        )

                # Log summary for this request
                if fulfilled or partial or dropped:
                    logger.info(
                        f"📊 SUMMARY for planner '{request.planner_id}' (priority={request.priority}):"
                    )
                    if fulfilled:
                        logger.info(f"   ✅ ACCEPTED (full): {list(fulfilled.keys())}")
                    if partial:
                        logger.info(f"   ⚠️  ACCEPTED (partial): {list(partial.keys())}")
                    if dropped:
                        logger.info(f"   ❌ DENIED: {dropped}")

            # Execute all planned operations for each DGD
            if planned_operations:
                logger.info(f"🚀 EXECUTING scaling operations across {len(planned_operations)} DGD(s)")
                for dgd_name, operations in planned_operations.items():
                    if operations:
                        logger.info(f"   Scaling DGD '{dgd_name}': {operations}")
                        try:
                            # Update replicas directly via Kubernetes API
                            for component_name, replicas in operations.items():
                                await self.gpu_tracker.kube_api.update_graph_replicas(
                                    dgd_name, component_name, replicas
                                )
                            logger.info(f"   ✅ Successfully applied scaling to DGD '{dgd_name}'")
                        except Exception as e:
                            logger.error(f"   ❌ Failed to scale DGD '{dgd_name}': {e}")
                logger.info(f"✅ BATCH COMPLETE: Processed {len(self.request_queue)} requests")
            else:
                logger.info("ℹ️  BATCH COMPLETE: No scaling operations needed")

            logger.info("=" * 80)

            # Clear the queue
            self.request_queue.clear()

    async def run_batch_processing_loop(self):
        """
        Main loop that processes batches at regular intervals.
        """
        logger.info(
            f"Starting batch processing loop (interval: {self.batching_window}s)"
        )

        while True:
            await asyncio.sleep(self.batching_window)
            try:
                await self.process_batch()
            except Exception as e:
                logger.error(f"Error during batch processing: {e}", exc_info=True)


async def http_scale_handler(request: web.Request, scaler_service: ScalerService):
    """Simple HTTP endpoint for scaling requests"""
    try:
        # Parse JSON request body
        data = await request.json()
        scale_request = ScaleRequest(**data)

        logger.info(f"📥 HTTP: Received scale request from planner '{scale_request.planner_id}'")

        # Queue the request
        response = await scaler_service.queue_scale_request(scale_request)

        # Return JSON response
        return web.json_response(response.model_dump())

    except Exception as e:
        logger.error(f"HTTP: Error processing scale request: {e}", exc_info=True)
        return web.json_response(
            {
                "success": False,
                "message": f"Error: {str(e)}",
                "components_scaled": {}
            },
            status=500
        )


async def start_http_server(scaler_service: ScalerService, port: int = 9000):
    """Start simple HTTP server for scaling requests"""
    app = web.Application()

    # Add route
    app.router.add_post('/scale', lambda req: http_scale_handler(req, scaler_service))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    logger.info(f"✅ HTTP server started on port {port}")
    logger.info(f"   Scaling endpoint: http://0.0.0.0:{port}/scale")

    # Keep running
    await asyncio.Event().wait()


async def run_global_planner():
    """Initialize and run the global planner service"""
    args = parse_args()

    # Create the scaler service
    scaler_service = ScalerService(
        dynamo_namespace=args.namespace,
        batching_window=args.batching_window,
        max_gpu_limit=args.max_gpu_limit,
        k8s_namespace=args.k8s_namespace,
    )

    # Start HTTP server
    logger.info("Starting HTTP server on port 9000...")
    http_task = asyncio.create_task(start_http_server(scaler_service, port=9000))
    logger.info("HTTP server task launched")

    # Start batch processing loop
    logger.info("Starting batch processing loop...")
    batch_task = asyncio.create_task(scaler_service.run_batch_processing_loop())
    logger.info("📋 Batch processing task launched")

    logger.info("🚀 Global Planner service fully initialized and ready to receive requests")
    logger.info(f"   Namespace: {args.namespace}")
    logger.info(f"   K8s Namespace: {args.k8s_namespace or 'auto-detected'}")
    logger.info(f"   HTTP Endpoint: http://0.0.0.0:9000/scale")
    logger.info(f"   Max GPU Limit: {args.max_gpu_limit}")
    logger.info(f"   Batching Window: {args.batching_window}s")

    try:
        # Run HTTP server and batch processing concurrently
        await asyncio.gather(
            http_task,
            batch_task
        )
    except asyncio.CancelledError:
        logger.info("Shutting down global planner service...")
        http_task.cancel()
        batch_task.cancel()
        try:
            await asyncio.gather(http_task, batch_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        logger.info("Global planner service shut down complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamo Scaling Controller Service")
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.getenv("DYN_NAMESPACE", "dynamo"),
        help="Dynamo namespace (default: from DYN_NAMESPACE env or 'dynamo')",
    )
    parser.add_argument(
        "--k8s-namespace",
        type=str,
        default=None,
        help="Kubernetes namespace (default: auto-detect or 'default')",
    )
    parser.add_argument(
        "--batching-window",
        type=int,
        default=int(os.getenv("BATCHING_WINDOW", "60")),
        help="Batching window in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-gpu-limit",
        type=int,
        default=int(os.getenv("MAX_GPU_LIMIT", "100")),
        help="Maximum total GPUs in cluster (default: 100)",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(run_global_planner())


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import uuid
from typing import Optional

import aiohttp

from dynamo.planner.planner_connector import PlannerConnector

logger = logging.getLogger(__name__)

# Import TargetReplica for type hints
try:
    from dynamo.planner import TargetReplica
except ImportError:
    TargetReplica = None  # Will be defined elsewhere


class GlobalConnector(PlannerConnector):
    """
    Connector that sends scaling requests to a centralized Global Planner service via HTTP.
    
    The Global Planner service batches requests and applies priority-based scaling logic
    with GPU resource constraints.
    
    Service Discovery:
        Uses Kubernetes DNS-based discovery to find the global planner service.
        DNS format: <service-name>.<k8s-namespace>.svc.cluster.local:<port>
    """

    def __init__(
        self,
        namespace: str,
        global_planner_service: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
    ):
        """
        Initialize the GlobalConnector.
        
        Args:
            namespace: Dynamo namespace
            global_planner_service: Global planner service name (defaults to "dynamo-global-planner")
            k8s_namespace: Kubernetes namespace (auto-detects if not provided)
            
        Environment Variables:
            PLANNER_PRIORITY: Priority level for this planner (higher = higher priority, default: 0)
            GLOBAL_PLANNER_SERVICE: Global planner service name (default: "dynamo-global-planner")
            GLOBAL_PLANNER_PORT: HTTP port (default: 9000)
            K8S_NAMESPACE: Kubernetes namespace (auto-detects if not provided)
            DYN_PARENT_DGD_K8S_NAME: Required - DGD name to scale
        """
        self.namespace = namespace
        self.priority = int(os.getenv("PLANNER_PRIORITY", "0"))
        self.planner_id = f"planner-{namespace}-{uuid.uuid4().hex[:8]}"
        self.dgd_name = os.getenv("DYN_PARENT_DGD_K8S_NAME")

        # HTTP endpoint for global planner service (K8s DNS-based discovery)
        # DNS format: <service-name>.<k8s-namespace>.svc.cluster.local:<port>
        service_name = global_planner_service or os.getenv(
            "GLOBAL_PLANNER_SERVICE", "dynamo-global-planner"
        )
        k8s_ns = k8s_namespace or os.getenv("K8S_NAMESPACE", self._auto_detect_namespace())
        planner_port = int(os.getenv("GLOBAL_PLANNER_PORT", "9000"))

        self.global_planner_url = (
            f"http://{service_name}.{k8s_ns}.svc.cluster.local:{planner_port}/scale"
        )

        logger.info(
            f"GlobalConnector initialized for namespace: {namespace}, "
            f"DGD: {self.dgd_name}, priority: {self.priority}, planner_id: {self.planner_id}"
        )
        logger.info(f"   Global Planner endpoint: {self.global_planner_url}")

    async def _async_init(self):
        """
        Async initialization to test connectivity with global planner.
        This is called by planner_core after the connector is created.
        """
        await self._test_global_planner_connection()

    async def _test_global_planner_connection(self):
        """Test if we can reach the global planner service."""
        logger.info("Testing connection to Global Planner...")
        
        try:
            # Send a minimal test request to verify connectivity
            test_payload = {
                "target_replicas": {},  # Empty - just a connectivity test
                "priority": self.priority,
                "planner_id": f"{self.planner_id}-test",
                "dgd_name": self.dgd_name or "test-dgd",
                "blocking": False,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.global_planner_url,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info(
                            f"✅ Successfully connected to Global Planner at {self.global_planner_url}"
                        )
                        logger.info(f"   Response: {response_data.get('message', 'OK')}")
                    else:
                        logger.warning(
                            f"⚠️  Global Planner responded with status {response.status}"
                        )
        except aiohttp.ClientError as e:
            logger.error(
                f"❌ Failed to connect to Global Planner at {self.global_planner_url}: {e}"
            )
            logger.error(
                "   Make sure the global planner service is running and accessible via DNS"
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error testing Global Planner connection: {e}")

    def _auto_detect_namespace(self) -> str:
        """Auto-detect Kubernetes namespace from pod environment"""
        try:
            with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning("Could not auto-detect K8s namespace, using 'default'")
            return "default"

    async def add_component(self, component_name: str, blocking: bool = True):
        """
        Add a component by increasing its replica count by 1.
        Note: This is not implemented as it requires querying current state.
        Use set_component_replicas instead for direct replica control.
        """
        raise NotImplementedError(
            "add_component is not supported by GlobalConnector. "
            "Use set_component_replicas instead for direct replica control."
        )

    async def remove_component(self, component_name: str, blocking: bool = True):
        """
        Remove a component by decreasing its replica count by 1.
        Note: This is not implemented as it requires querying current state.
        Use set_component_replicas instead for direct replica control.
        """
        raise NotImplementedError(
            "remove_component is not supported by GlobalConnector. "
            "Use set_component_replicas instead for direct replica control."
        )

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ):
        """
        Validate deployment (no-op for GlobalConnector).
        The global planner handles deployment validation when processing scaling requests.
        """
        logger.info("Skipping deployment validation for GlobalConnector (handled by global planner)")
        pass

    def get_model_name(self) -> str:
        """
        Get model name (returns placeholder for GlobalConnector).
        Model name is not used in global connector mode.
        """
        logger.warning("get_model_name() called on GlobalConnector - returning placeholder")
        return "unknown-model"

    async def wait_for_deployment_ready(self):
        """
        Wait for deployment to be ready (no-op for GlobalConnector).
        The global planner manages deployment readiness.
        """
        logger.info("Skipping deployment ready wait for GlobalConnector")
        pass

    async def set_component_replicas(
        self, target_replicas: list, blocking: bool = True
    ):
        """
        Set the replicas for multiple components by sending HTTP request to Global Planner service.
        Args:
            target_replicas: List of TargetReplica objects with component names and desired replica counts
            blocking: Whether to wait for scaling to complete (note: not fully supported
                     in batched mode, request is queued regardless)
        """
        if not target_replicas:
            logger.warning("Empty target_replicas list, skipping scaling request")
            return

        if not self.dgd_name:
            logger.error("DYN_PARENT_DGD_K8S_NAME not set, cannot send scaling request")
            raise ValueError("DYN_PARENT_DGD_K8S_NAME environment variable must be set")

        # Convert list of TargetReplica objects to dict
        target_replicas_dict = {
            tr.component_name: tr.desired_replicas for tr in target_replicas
        }

        try:
            # Prepare the request payload
            request_data = {
                "target_replicas": target_replicas_dict,
                "priority": self.priority,
                "planner_id": self.planner_id,
                "dgd_name": self.dgd_name,
                "blocking": blocking,
            }

            logger.info(
                f"Sending HTTP scale request to Global Planner: "
                f"DGD={self.dgd_name}, components={list(target_replicas_dict.keys())}, priority={self.priority}"
            )
            logger.debug(f"Request payload: {request_data}")
            logger.debug(f"Global Planner URL: {self.global_planner_url}")

            # Send HTTP POST request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.global_planner_url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_data = await response.json()

                    if response.status == 200 and response_data.get("success"):
                        logger.info(
                            f"✅ Scaling request accepted: {response_data.get('message')}"
                        )
                    else:
                        logger.error(
                            f"❌ Scaling request failed: {response_data.get('message')}"
                        )

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error sending scaling request: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"Failed to send scaling request to Global Planner service: {e}", exc_info=True
            )
            raise
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Fault Injection Helpers for ManagedDeployment

This module provides a clean interface for integrating hardware fault injection
(GPU failures, CUDA errors) into the ephemeral deployment test framework.

Toggle-based approach (no pod restarts during test):
    async with ManagedDeployment(
        namespace="test",
        deployment_spec=spec,
        enable_hw_faults=True,
        hw_fault_config={'xid_type': 79}
    ) as deployment:
        # 1. Setup patches deployment with library (passthrough mode - faults disabled)
        #    Pods restart ONCE to load library, then faults can be toggled without restarts

        # 2. Inject XID fault (optional - for NVSentinel detection)
        fault_id = await deployment.inject_hw_fault('xid', xid_type=79, gpu_id=0)

        # 3. Toggle CUDA faults ON (no restart - writes to hostPath file)
        await deployment.toggle_cuda_faults(enable=True)

        # 4. Test crash/recovery behavior...

        # 5. Toggle CUDA faults OFF (no restart)
        await deployment.toggle_cuda_faults(enable=False)
"""

import logging
import os
import socket
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException

# Suppress urllib3 deprecation warnings that break kubernetes client error handling
# This is a known issue with kubernetes python client and urllib3 v2.x
warnings.filterwarnings(
    "ignore", message=".*HTTPResponse.getheaders.*", category=DeprecationWarning
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")


@contextmanager
def suppress_deprecation_warnings():
    """Context manager to suppress deprecation warnings during kubernetes API calls.

    The kubernetes client's exception handling calls deprecated urllib3 methods,
    which can cause issues in certain Python/urllib3 version combinations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


def _get_hw_fault_helpers_dir() -> Path:
    """Get the directory containing hardware fault injection helpers."""
    # Navigate from tests/utils to tests/fault_tolerance/hardware/fault_injection_service/helpers
    tests_dir = Path(__file__).parent.parent
    return (
        tests_dir
        / "fault_tolerance"
        / "hardware"
        / "fault_injection_service"
        / "helpers"
    )


def _get_cuda_lib_dir() -> Path:
    """Get the directory containing CUDA fault injection library."""
    tests_dir = Path(__file__).parent.parent
    return (
        tests_dir
        / "fault_tolerance"
        / "hardware"
        / "fault_injection_service"
        / "cuda_fault_injection"
    )


@dataclass
class HWFaultConfig:
    """Configuration for hardware fault injection."""

    enabled: bool = False
    xid_type: int = 79  # Default XID type (GPU fell off bus)
    target_node: Optional[str] = None  # Auto-select if None
    api_url: Optional[str] = None  # Auto-detect if None

    # Timeouts
    setup_timeout: int = 300  # 5 minutes for CUDA library setup
    fault_timeout: int = 600  # 10 minutes for fault effects

    # Service names to patch (for CUDA injection)
    # Supports vLLM, SGLang, and TensorRT-LLM backends
    service_names: List[str] = field(
        default_factory=lambda: [
            "VllmDecodeWorker",
            "VllmPrefillWorker",  # vLLM
            "decode",
            "prefill",  # SGLang
            "TRTLLMDecodeWorker",
            "TRTLLMPrefillWorker",
            "TRTLLMWorker",  # TensorRT-LLM
        ]
    )

    @classmethod
    def from_dict(cls, config: Optional[Dict]) -> "HWFaultConfig":
        """Create config from dictionary."""
        if config is None:
            return cls(enabled=False)

        return cls(
            enabled=config.get("enabled", True),
            xid_type=config.get("xid_type", 79),
            target_node=config.get("target_node"),
            api_url=config.get("api_url"),
            setup_timeout=config.get("setup_timeout", 300),
            fault_timeout=config.get("fault_timeout", 600),
            service_names=config.get(
                "service_names",
                [
                    "VllmDecodeWorker",
                    "VllmPrefillWorker",  # vLLM
                    "decode",
                    "prefill",  # SGLang
                    "TRTLLMDecodeWorker",
                    "TRTLLMPrefillWorker",
                    "TRTLLMWorker",  # TensorRT-LLM
                ],
            ),
        )


class HWFaultManager:
    """
    Manages hardware fault injection lifecycle for ManagedDeployment.

    Handles:
    - CUDA fault library build and deployment
    - XID fault injection via API
    - Cleanup of all artifacts
    """

    VALID_XID_TYPES = {79, 48, 94, 95, 43, 74}

    def __init__(
        self,
        namespace: str,
        deployment_name: str,
        config: HWFaultConfig,
        in_cluster: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.config = config
        self.in_cluster = in_cluster
        self.logger = logger or logging.getLogger(__name__)

        # Load kubernetes config for the synchronous client
        # (ManagedDeployment loads config for kubernetes_asyncio, but that's a different library)
        self._load_k8s_config()

        # State tracking
        self._cuda_injector = None
        self._fault_ids: List[str] = []
        self._cuda_setup_done = False
        self._cuda_passthrough_enabled = False  # Library loaded but faults disabled
        self._cuda_faults_active = False  # Faults currently toggled ON
        self._target_node: Optional[str] = None
        self._api_port_forward = None
        self._api_local_port: Optional[int] = None

        # API URL - will be set after port-forward if needed
        self.api_url = self._get_api_url()

    def _load_k8s_config(self):
        """Load kubernetes config for the synchronous kubernetes client."""
        try:
            if self.in_cluster:
                k8s_config.load_incluster_config()
                self.logger.debug("[HW Faults] Loaded in-cluster kubernetes config")
            else:
                k8s_config.load_kube_config()
                self.logger.debug("[HW Faults] Loaded kubeconfig file")
        except Exception as e:
            self.logger.warning(f"[HW Faults] Failed to load kubernetes config: {e}")

    def _get_api_url(self) -> str:
        """Get fault injection API URL based on execution context."""
        if self.config.api_url:
            return self.config.api_url

        if self.in_cluster:
            return "http://fault-injection-api.fault-injection-system.svc.cluster.local:8080"

        # Check for explicit env var
        env_url = os.getenv("FAULT_INJECTION_API")
        if env_url:
            return env_url

        # If we have a port-forwarded port, use that
        if self._api_local_port:
            return f"http://localhost:{self._api_local_port}"

        # Default - will be updated after port-forward setup
        return "http://localhost:8080"

    def _find_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _setup_api_port_forward(self) -> bool:
        """
        Set up port-forwarding to the fault injection API service using kubectl.

        Called automatically during setup if not running in-cluster.
        Uses subprocess to run `kubectl port-forward` which is more reliable
        than the kubernetes python client's websocket-based approach.

        Returns:
            True if port-forward was set up or not needed, False on failure
        """
        if self.in_cluster:
            self.logger.debug("[HW Faults] Running in-cluster - no port-forward needed")
            return True

        if self._api_port_forward:
            self.logger.debug("[HW Faults] Port-forward already active")
            return True

        FAULT_API_NAMESPACE = "fault-injection-system"
        FAULT_API_SERVICE = "fault-injection-api"
        FAULT_API_PORT = 8080

        try:
            # Find a free local port
            local_port = self._find_free_port()

            self.logger.info(
                f"[HW Faults] Starting port-forward: localhost:{local_port} → "
                f"svc/{FAULT_API_SERVICE}:{FAULT_API_PORT}"
            )

            # Start kubectl port-forward as a subprocess
            cmd = [
                "kubectl",
                "port-forward",
                f"svc/{FAULT_API_SERVICE}",
                f"{local_port}:{FAULT_API_PORT}",
                "-n",
                FAULT_API_NAMESPACE,
            ]

            # Run in background, suppress output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent process group
            )

            # Give it a moment to start
            time.sleep(2)

            # Check if process is still running
            if process.poll() is not None:
                # Process exited - read stderr
                stderr = process.stderr.read().decode() if process.stderr else ""
                self.logger.warning(
                    f"[HW Faults] Port-forward failed to start: {stderr}"
                )
                return False

            # Verify the port is actually listening
            max_retries = 5
            for i in range(max_retries):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        result = s.connect_ex(("localhost", local_port))
                        if result == 0:
                            break
                except Exception:
                    pass
                time.sleep(1)
            else:
                self.logger.warning(
                    "[HW Faults] Port-forward started but port not listening"
                )
                process.terminate()
                return False

            self._api_port_forward = process
            self._api_local_port = local_port
            self.api_url = f"http://localhost:{local_port}"

            self.logger.info(f"[HW Faults] Port-forward active: {self.api_url}")
            return True

        except FileNotFoundError:
            self.logger.warning(
                "[HW Faults] kubectl not found - port-forward unavailable"
            )
            return False
        except Exception as e:
            self.logger.warning(f"[HW Faults] Failed to set up port-forward: {e}")
            return False

    def _cleanup_api_port_forward(self):
        """Clean up the API port-forward subprocess."""
        if self._api_port_forward:
            try:
                # Terminate the kubectl port-forward process
                self._api_port_forward.terminate()
                try:
                    self._api_port_forward.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._api_port_forward.kill()
                self.logger.debug("[HW Faults] Port-forward terminated")
            except Exception as e:
                self.logger.debug(f"[HW Faults] Error terminating port-forward: {e}")
            self._api_port_forward = None
            self._api_local_port = None

    def _get_cuda_injector(self):
        """Lazy-load CUDA fault injector."""
        if self._cuda_injector is None:
            # Add helpers to path
            helpers_dir = _get_hw_fault_helpers_dir()
            if str(helpers_dir) not in sys.path:
                sys.path.insert(0, str(helpers_dir))

            from cuda_fault_injection import CUDAFaultInjector

            self._cuda_injector = CUDAFaultInjector(lib_dir=_get_cuda_lib_dir())

        return self._cuda_injector

    async def setup(self) -> bool:
        """
        Setup hardware fault injection infrastructure.

        - Sets up port-forward to fault injection API (if not in-cluster)
        - Builds CUDA fault library (if needed)
        - Creates ConfigMap with library source

        Returns:
            True if setup succeeded
        """
        if not self.config.enabled:
            self.logger.info("[HW Faults] Disabled - skipping setup")
            return True

        self.logger.info("[HW Faults] Setting up hardware fault injection...")

        # Set up port-forward to fault injection API (optional - XID injection only)
        if not self.in_cluster:
            if self._setup_api_port_forward():
                self.logger.info(f"[HW Faults] API available at {self.api_url}")
            else:
                self.logger.warning(
                    "[HW Faults] API port-forward not available - XID injection will be skipped"
                )

        try:
            cuda_injector = self._get_cuda_injector()

            # Build library (or verify it exists)
            self.logger.info("[HW Faults] Building CUDA fault injection library...")
            if not cuda_injector.build_library():
                self.logger.error("[HW Faults] Failed to build CUDA library")
                return False

            # Create ConfigMap with library source
            self.logger.info("[HW Faults] Creating ConfigMap with library source...")
            if not cuda_injector.create_configmap_with_library(self.namespace):
                self.logger.error("[HW Faults] Failed to create ConfigMap")
                return False

            self._cuda_setup_done = True
            self.logger.info("[HW Faults] Setup complete - ready for fault injection")
            return True

        except Exception as e:
            self.logger.error(f"[HW Faults] Setup failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def set_target_node(self, node_name: str):
        """Set the target node for fault injection."""
        self._target_node = node_name
        self.logger.info(f"[HW Faults] Target node set to: {node_name}")

    async def setup_cuda_passthrough(
        self,
        xid_type: Optional[int] = None,
        target_node: Optional[str] = None,
    ) -> bool:
        """
        Setup CUDA library in passthrough mode (faults disabled).

        Patches deployment to load the CUDA intercept library with faults DISABLED.
        Pods will restart ONCE to load the library. After that, use toggle_cuda_faults()
        to enable/disable faults without additional restarts.

        Args:
            xid_type: XID error type to configure (uses config default if None)
            target_node: Node to target (uses config/auto-detected if None)

        Returns:
            True if successful
        """
        if not self._cuda_setup_done:
            self.logger.error("[HW Faults] CUDA setup not done - call setup() first")
            return False

        xid = xid_type or self.config.xid_type
        node = target_node or self._target_node or self.config.target_node

        if xid not in self.VALID_XID_TYPES:
            self.logger.error(
                f"[HW Faults] Invalid XID type: {xid}. Valid: {self.VALID_XID_TYPES}"
            )
            return False

        self.logger.info(
            f"[HW Faults] Setting up CUDA passthrough (XID {xid}) on deployment..."
        )
        self.logger.info(
            "[HW Faults] Library will load with faults DISABLED - use toggle_cuda_faults() to enable"
        )
        if node:
            self.logger.info(f"[HW Faults] Target node: {node}")

        try:
            cuda_injector = self._get_cuda_injector()

            success = cuda_injector.patch_deployment_for_cuda_fault(
                self.deployment_name,
                self.namespace,
                target_node=node,
                xid_type=xid,
                passthrough_mode=True,  # Library loaded but faults disabled
            )

            if success:
                self._cuda_passthrough_enabled = True
                self.logger.info(
                    "[HW Faults] CUDA passthrough configured - pods will restart with library (faults disabled)"
                )
            else:
                self.logger.error("[HW Faults] Failed to setup CUDA passthrough")

            return success

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to setup CUDA passthrough: {e}")
            return False

    async def toggle_cuda_faults(self, enable: bool = True) -> bool:
        """
        Toggle CUDA faults ON or OFF without restarting pods.

        Writes to hostPath file that the CUDA intercept library reads at runtime.
        Requires setup_cuda_passthrough() to have been called first.

        Args:
            enable: True to enable faults, False to disable

        Returns:
            True if toggle succeeded
        """
        if not self._cuda_passthrough_enabled:
            self.logger.error(
                "[HW Faults] CUDA passthrough not enabled - call setup_cuda_passthrough() first"
            )
            return False

        action = "Enabling" if enable else "Disabling"
        self.logger.info(
            f"[HW Faults] {action} CUDA faults via hostPath toggle (no restart)..."
        )

        try:
            cuda_injector = self._get_cuda_injector()

            # Get pods for this deployment
            pods = self._get_deployment_pods()
            if not pods:
                self.logger.error("[HW Faults] No pods found for deployment")
                return False

            # With soft affinity, only toggle pods on target node (they have hostPath mounted)
            # Pods on other nodes don't have the volume and will fail
            target = self._target_node or self.config.target_node
            success = cuda_injector.enable_cuda_faults_via_toggle(
                pods=pods,
                namespace=self.namespace,
                enable=enable,
                target_node=target,
            )

            if success:
                self._cuda_faults_active = enable
                state = "ACTIVE" if enable else "DISABLED"
                self.logger.info(f"[HW Faults] CUDA faults {state}")
            else:
                self.logger.error("[HW Faults] Failed to toggle CUDA faults")

            return success

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to toggle CUDA faults: {e}")
            return False

    def _get_deployment_pods(self) -> List[client.V1Pod]:
        """Get pods belonging to this deployment."""
        try:
            v1 = client.CoreV1Api()
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self.deployment_name}"
            )
            with suppress_deprecation_warnings():
                pods = v1.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=label_selector,
                )
            return pods.items
        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to get deployment pods: {e}")
            return []

    async def remove_node_affinity(self, delete_pods: bool = True) -> bool:
        """
        Remove node affinity from worker pods to allow rescheduling.

        This is needed during recovery because:
        1. CUDA passthrough setup pins workers to a specific node
        2. When that node is cordoned, workers can't reschedule
        3. Removing affinity allows workers to schedule on healthy nodes

        Note: Cordoned nodes don't evict running pods - they only prevent NEW scheduling.
        So we must delete pods to force them to reschedule to healthy nodes.

        Args:
            delete_pods: If True (default), delete pods to force immediate reschedule.
                        This is required for simulated faults where the node is still
                        healthy but cordoned. If False, just update spec.

        Returns:
            True if successful
        """
        self.logger.info(
            "[HW Faults] Removing node affinity to allow pod rescheduling..."
        )

        try:
            if delete_pods:
                # Full cleanup - removes all CUDA artifacts and deletes pods
                # Required for fault simulation where node is cordoned but healthy
                cuda_injector = self._get_cuda_injector()
                success = cuda_injector.cleanup_cuda_fault_injection(
                    deployment_name=self.deployment_name,
                    namespace=self.namespace,
                )
            else:
                # Light-weight: just remove node affinity from DGD spec (no pod deletion)
                # Only use this if pods will crash/restart naturally
                success = await self._remove_node_affinity_only()

            if success:
                self.logger.info(
                    "[HW Faults] Node affinity removed - pods can reschedule"
                )
            else:
                self.logger.error("[HW Faults] Failed to remove node affinity")

            return success

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to remove node affinity: {e}")
            return False

    async def _remove_node_affinity_only(self) -> bool:
        """
        Remove ONLY the node affinity from DGD spec without deleting pods or other artifacts.

        This allows pods to reschedule when they naturally restart (e.g., due to CUDA faults)
        without forcing an immediate restart.
        """
        try:
            k8s_custom = client.CustomObjectsApi()

            with suppress_deprecation_warnings():
                dgd = k8s_custom.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name,
                )

            # Remove node affinity from all worker services
            services = dgd.get("spec", {}).get("services", {})
            patched = False

            for service_name in self.config.service_names:
                if service_name in services:
                    service = services[service_name]
                    if (
                        "extraPodSpec" in service
                        and "affinity" in service["extraPodSpec"]
                    ):
                        del service["extraPodSpec"]["affinity"]
                        patched = True
                        self.logger.debug(
                            f"[HW Faults] Removed affinity from {service_name}"
                        )

            if patched:
                with suppress_deprecation_warnings():
                    k8s_custom.patch_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=self.deployment_name,
                        body=dgd,
                    )
                self.logger.info(
                    "[HW Faults] Node affinity removed from DGD spec (no pod restart)"
                )
            else:
                self.logger.info("[HW Faults] No node affinity found to remove")

            return True

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to remove node affinity: {e}")
            return False

    async def cleanup_cuda_spec_without_restart(self) -> bool:
        """
        Remove CUDA fault injection from DGD spec WITHOUT deleting pods.

        This is used when relying on NVSentinel's node-drainer to evict pods.
        The spec is cleaned up so that when pods are evicted by node-drainer,
        the new pods will come up WITHOUT:
        - CUDA LD_PRELOAD library
        - Node affinity pinning

        Returns:
            True if successful
        """
        try:
            k8s_custom = client.CustomObjectsApi()
            k8s_core = client.CoreV1Api()

            self.logger.info(
                "[HW Faults] Cleaning up DGD spec (keeping pods running)..."
            )

            with suppress_deprecation_warnings():
                dgd = k8s_custom.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name,
                )

            services = dgd.get("spec", {}).get("services", {})
            patched_services = []

            for service_name in self.config.service_names:
                if service_name not in services:
                    continue

                service = services[service_name]
                changed = False

                # Remove node affinity
                if "extraPodSpec" in service:
                    if "affinity" in service["extraPodSpec"]:
                        del service["extraPodSpec"]["affinity"]
                        changed = True

                    # Remove init container for CUDA library
                    # Names used by inject_into_pods.py: compile-cuda-fault-lib, decode-cuda-fault-lib
                    if "initContainers" in service["extraPodSpec"]:
                        original_count = len(service["extraPodSpec"]["initContainers"])
                        cuda_init_names = [
                            "compile-cuda-fault-lib",
                            "decode-cuda-fault-lib",
                            "cuda-fault-init",
                        ]
                        service["extraPodSpec"]["initContainers"] = [
                            ic
                            for ic in service["extraPodSpec"]["initContainers"]
                            if ic.get("name") not in cuda_init_names
                        ]
                        if (
                            len(service["extraPodSpec"]["initContainers"])
                            < original_count
                        ):
                            changed = True
                        if not service["extraPodSpec"]["initContainers"]:
                            del service["extraPodSpec"]["initContainers"]

                    # Remove volumes for CUDA library
                    # Names used by inject_into_pods.py: cuda-fault-lib-source, cuda-fault-lib, node-fault-marker
                    if "volumes" in service["extraPodSpec"]:
                        original_count = len(service["extraPodSpec"]["volumes"])
                        cuda_volume_names = [
                            "cuda-fault-lib",
                            "cuda-fault-lib-source",
                            "node-fault-marker",
                            "host-fault-dir",
                        ]
                        service["extraPodSpec"]["volumes"] = [
                            v
                            for v in service["extraPodSpec"]["volumes"]
                            if v.get("name") not in cuda_volume_names
                        ]
                        if len(service["extraPodSpec"]["volumes"]) < original_count:
                            changed = True
                        if not service["extraPodSpec"]["volumes"]:
                            del service["extraPodSpec"]["volumes"]

                    # Remove main container modifications
                    if "mainContainer" in service["extraPodSpec"]:
                        main = service["extraPodSpec"]["mainContainer"]

                        # Remove volume mounts
                        # Names used by inject_into_pods.py: cuda-fault-lib, node-fault-marker
                        if "volumeMounts" in main:
                            original_count = len(main["volumeMounts"])
                            cuda_mount_names = [
                                "cuda-fault-lib",
                                "cuda-fault-lib-source",
                                "node-fault-marker",
                                "host-fault-dir",
                            ]
                            main["volumeMounts"] = [
                                vm
                                for vm in main["volumeMounts"]
                                if vm.get("name") not in cuda_mount_names
                            ]
                            if len(main["volumeMounts"]) < original_count:
                                changed = True
                            if not main["volumeMounts"]:
                                del main["volumeMounts"]

                        if not main:
                            del service["extraPodSpec"]["mainContainer"]

                    if not service["extraPodSpec"]:
                        del service["extraPodSpec"]

                # Remove envs that were added for CUDA fault injection
                if "envs" in service:
                    cuda_envs = [
                        "LD_PRELOAD",
                        "CUDA_FAULT_INJECTION_ENABLED",
                        "CUDA_XID_TYPE",
                    ]
                    original_count = len(service["envs"])
                    service["envs"] = [
                        e for e in service["envs"] if e.get("name") not in cuda_envs
                    ]
                    if len(service["envs"]) < original_count:
                        changed = True
                    if not service["envs"]:
                        del service["envs"]

                if changed:
                    patched_services.append(service_name)

            if patched_services:
                # Use replace instead of patch to ensure array items are properly removed
                # Strategic merge patch doesn't remove array items correctly
                self.logger.info(
                    f"[HW Faults] Replacing DGD spec for services: {patched_services}"
                )

                # Debug: Log what we're about to set (should all be False after cleanup)
                for svc in patched_services:
                    svc_spec = dgd.get("spec", {}).get("services", {}).get(svc, {})
                    has_init = "initContainers" in svc_spec.get("extraPodSpec", {})
                    has_affinity = "affinity" in svc_spec.get("extraPodSpec", {})
                    has_ldpreload = any(
                        e.get("name") == "LD_PRELOAD" for e in svc_spec.get("envs", [])
                    )
                    self.logger.info(
                        f"[HW Faults]   {svc}: initContainers={has_init}, affinity={has_affinity}, LD_PRELOAD={has_ldpreload}"
                    )
                    if has_init or has_affinity or has_ldpreload:
                        self.logger.error(
                            f"[HW Faults]   ⚠️  Cleanup FAILED for {svc} - items still present!"
                        )

                with suppress_deprecation_warnings():
                    k8s_custom.replace_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=self.deployment_name,
                        body=dgd,
                    )
                self.logger.info("[HW Faults] DGD spec replaced successfully")

                # Verify the change actually took effect
                with suppress_deprecation_warnings():
                    verify_dgd = k8s_custom.get_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=self.deployment_name,
                    )
                for svc in patched_services:
                    svc_spec = (
                        verify_dgd.get("spec", {}).get("services", {}).get(svc, {})
                    )
                    has_init = "initContainers" in svc_spec.get("extraPodSpec", {})
                    has_affinity = "affinity" in svc_spec.get("extraPodSpec", {})
                    if has_init or has_affinity:
                        self.logger.error(
                            f"[HW Faults]   ❌ VERIFICATION FAILED: {svc} still has initContainers={has_init}, affinity={has_affinity}"
                        )
                    else:
                        self.logger.info(f"[HW Faults]   ✓ Verified {svc} is clean")
            else:
                self.logger.info("[HW Faults] No CUDA artifacts found in DGD spec")

            # Delete ConfigMap (but NOT pods)
            try:
                k8s_core.delete_namespaced_config_map(
                    name="cuda-fault-injection-lib",
                    namespace=self.namespace,
                )
                self.logger.info("[HW Faults] ConfigMap deleted")
            except ApiException as e:
                if e.status != 404:
                    self.logger.warning(f"[HW Faults] Failed to delete ConfigMap: {e}")

            self.logger.info("[HW Faults] DGD spec cleaned - pods still running")
            self.logger.info(
                "[HW Faults] When node-drainer evicts pods, new ones will be clean"
            )

            return True

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to cleanup DGD spec: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def enable_cuda_faults(
        self,
        xid_type: Optional[int] = None,
        target_node: Optional[str] = None,
    ) -> bool:
        """
        Enable CUDA fault injection on deployment.

        DEPRECATED: Use setup_cuda_passthrough() + toggle_cuda_faults() for
        better test flow (no restarts during test execution).

        This method patches the deployment which causes pods to restart.
        Use the toggle-based approach instead for cleaner test execution.

        Args:
            xid_type: XID error type (uses config default if None)
            target_node: Node to target (uses config/auto-detected if None)

        Returns:
            True if successful
        """
        if not self._cuda_setup_done:
            self.logger.error("[HW Faults] CUDA setup not done - call setup() first")
            return False

        xid = xid_type or self.config.xid_type
        node = target_node or self._target_node or self.config.target_node

        if xid not in self.VALID_XID_TYPES:
            self.logger.error(
                f"[HW Faults] Invalid XID type: {xid}. Valid: {self.VALID_XID_TYPES}"
            )
            return False

        self.logger.warning(
            "[HW Faults] Using deprecated enable_cuda_faults() - consider toggle-based approach"
        )
        self.logger.info(
            f"[HW Faults] Enabling CUDA faults (XID {xid}) on deployment..."
        )
        if node:
            self.logger.info(f"[HW Faults] Target node: {node}")

        try:
            cuda_injector = self._get_cuda_injector()

            success = cuda_injector.patch_deployment_for_cuda_fault(
                self.deployment_name,
                self.namespace,
                target_node=node,
                xid_type=xid,
                passthrough_mode=False,  # Faults enabled immediately
            )

            if success:
                self.logger.info(
                    "[HW Faults] CUDA faults enabled - pods will restart with library"
                )
            else:
                self.logger.error("[HW Faults] Failed to enable CUDA faults")

            return success

        except Exception as e:
            self.logger.error(f"[HW Faults] Failed to enable CUDA faults: {e}")
            return False

    def _is_api_reachable(self, timeout: float = 2.0) -> bool:
        """Quick check if the fault injection API is reachable."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    async def inject_xid_fault(
        self,
        xid_type: Optional[int] = None,
        gpu_id: int = 0,
        node_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Inject XID error via fault injection API.

        This is OPTIONAL - if the fault injection API is not running, XID injection
        will be skipped. CUDA faults via LD_PRELOAD can still be used for testing.

        Args:
            xid_type: XID error type (uses config default if None)
            gpu_id: GPU ID to target
            node_name: Node name (uses auto-detected if None)

        Returns:
            Fault ID if successful, None otherwise (including when API not available)
        """
        xid = xid_type or self.config.xid_type
        node = node_name or self._target_node or self.config.target_node

        if not node:
            self.logger.error("[HW Faults] No target node specified for XID injection")
            return None

        # Check if API is reachable before attempting injection
        if not self._is_api_reachable():
            self.logger.warning(
                f"[HW Faults] Fault injection API not available at {self.api_url}"
            )
            self.logger.warning(
                "[HW Faults] XID injection skipped - use CUDA faults (enable_cuda_faults) instead"
            )
            return None

        self.logger.info(
            f"[HW Faults] Injecting XID {xid} on node {node}, GPU {gpu_id}..."
        )

        try:
            response = requests.post(
                f"{self.api_url}/api/v1/faults/gpu/inject/xid-{xid}",
                json={
                    "node_name": node,
                    "xid_type": xid,
                    "gpu_id": gpu_id,
                },
                timeout=60,
            )

            if response.status_code == 200:
                fault_id = response.json().get("fault_id")
                self._fault_ids.append(fault_id)
                self.logger.info(
                    f"[HW Faults] XID {xid} injected (fault_id: {fault_id})"
                )
                return fault_id
            else:
                self.logger.error(f"[HW Faults] XID injection failed: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"[HW Faults] XID injection error: {e}")
            return None

    async def wait_for_pods_to_crash(
        self,
        timeout: Optional[int] = None,
        label_selector: Optional[str] = None,
    ) -> bool:
        """
        Wait for pods to crash due to CUDA errors.

        Args:
            timeout: Max wait time in seconds
            label_selector: Pod label selector (auto-generated if None)

        Returns:
            True if pods crashed, False if timeout
        """
        timeout = timeout or self.config.fault_timeout
        node = self._target_node or self.config.target_node

        if not node:
            self.logger.error("[HW Faults] No target node for crash monitoring")
            return False

        if label_selector is None:
            label_selector = (
                f"nvidia.com/dynamo-component-type=worker,"
                f"nvidia.com/dynamo-graph-deployment-name={self.deployment_name}"
            )

        try:
            cuda_injector = self._get_cuda_injector()
            return cuda_injector.wait_for_pods_to_crash(
                namespace=self.namespace,
                label_selector=label_selector,
                node_name=node,
                timeout=timeout,
            )
        except Exception as e:
            self.logger.error(f"[HW Faults] Error waiting for crashes: {e}")
            return False

    async def cleanup(self) -> bool:
        """
        Clean up all hardware fault injection artifacts.

        - Uncordons the target node (if it was cordoned)
        - Removes CUDA library from deployment
        - Deletes ConfigMap
        - Cleans up fault API injections

        Returns:
            True if cleanup succeeded
        """
        self.logger.info("[HW Faults] Cleaning up hardware fault injection...")

        success = True

        # FIRST: Uncordon the target node (most important for cluster health)
        # Try multiple sources for target node in case it wasn't persisted
        target_node = self._target_node or self.config.target_node
        self.logger.info(f"[HW Faults] Cleanup target node: {target_node}")

        if target_node:
            try:
                if self.is_node_cordoned(target_node):
                    self.logger.info(f"[HW Faults] Uncordoning node {target_node}...")
                    if self.uncordon_node(target_node):
                        self.logger.info(f"[HW Faults] ✓ Node {target_node} uncordoned")
                    else:
                        self.logger.warning(
                            f"[HW Faults] Failed to uncordon node {target_node}"
                        )
                        success = False
                else:
                    self.logger.info(
                        f"[HW Faults] Node {target_node} already schedulable"
                    )
            except Exception as e:
                self.logger.warning(f"[HW Faults] Error uncordoning node: {e}")
                success = False
        else:
            self.logger.warning(
                "[HW Faults] No target node to uncordon - node may need manual uncordon"
            )

        # Clean up CUDA injection
        if self._cuda_setup_done:
            try:
                cuda_injector = self._get_cuda_injector()
                if not cuda_injector.cleanup_cuda_fault_injection(
                    self.deployment_name,
                    self.namespace,
                    force_delete_pods=False,  # Let ManagedDeployment handle pod cleanup
                    service_names=self.config.service_names,
                ):
                    self.logger.warning("[HW Faults] CUDA cleanup had issues")
                    success = False
                else:
                    self.logger.info("[HW Faults] CUDA artifacts cleaned up")
            except Exception as e:
                self.logger.warning(f"[HW Faults] CUDA cleanup error: {e}")
                success = False

        # Clean up fault API injections
        for fault_id in self._fault_ids:
            try:
                response = requests.delete(
                    f"{self.api_url}/api/v1/faults/{fault_id}",
                    timeout=10,
                )
                if response.status_code in (200, 404):
                    self.logger.info(f"[HW Faults] Cleaned up fault {fault_id}")
                else:
                    self.logger.warning(f"[HW Faults] Failed to clean fault {fault_id}")
            except Exception as e:
                self.logger.warning(f"[HW Faults] Error cleaning fault {fault_id}: {e}")

        self._fault_ids.clear()
        self._cuda_setup_done = False

        # Clean up port-forward
        self._cleanup_api_port_forward()

        self.logger.info("[HW Faults] Cleanup complete")
        return success

    def check_api_health(self) -> bool:
        """Check if fault injection API is healthy."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_node_cordon(
        self,
        node_name: Optional[str] = None,
        timeout: int = 180,
    ) -> bool:
        """
        Wait for NVSentinel to cordon (mark unschedulable) a node.

        Args:
            node_name: Node to monitor (uses target node if None)
            timeout: Max wait time in seconds

        Returns:
            True if node was cordoned, False if timeout
        """
        node = node_name or self._target_node or self.config.target_node

        if not node:
            self.logger.error(
                "[HW Faults] No target node specified for cordon monitoring"
            )
            return False

        self.logger.info(
            f"[HW Faults] Waiting for NVSentinel to cordon {node} (timeout: {timeout}s)..."
        )

        try:
            k8s_core = client.CoreV1Api()
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    with suppress_deprecation_warnings():
                        node_obj = k8s_core.read_node(node)

                    if node_obj.spec.unschedulable:
                        elapsed = time.time() - start_time
                        self.logger.info(
                            f"[HW Faults] ✓ Node {node} cordoned after {elapsed:.1f}s"
                        )
                        return True

                    elapsed = time.time() - start_time
                    if int(elapsed) % 30 == 0 and elapsed > 0:
                        self.logger.info(
                            f"[HW Faults] [{elapsed:.0f}s] Waiting for cordon..."
                        )

                except Exception as e:
                    self.logger.warning(f"[HW Faults] Error checking node status: {e}")

                time.sleep(5)

            self.logger.warning(
                f"[HW Faults] ✗ Timeout waiting for node cordon ({timeout}s)"
            )
            return False

        except Exception as e:
            self.logger.error(f"[HW Faults] Error in cordon wait: {e}")
            return False

    def is_node_cordoned(self, node_name: Optional[str] = None) -> bool:
        """Check if a node is currently cordoned."""
        node = node_name or self._target_node or self.config.target_node

        if not node:
            return False

        try:
            k8s_core = client.CoreV1Api()
            with suppress_deprecation_warnings():
                node_obj = k8s_core.read_node(node)
            return bool(node_obj.spec.unschedulable)
        except Exception:
            return False

    def uncordon_node(self, node_name: Optional[str] = None) -> bool:
        """
        Uncordon a node (restore schedulability).
        Used for test cleanup.

        Args:
            node_name: Node to uncordon (uses target node if None)

        Returns:
            True if successful
        """
        node = node_name or self._target_node or self.config.target_node

        if not node:
            return False

        try:
            k8s_core = client.CoreV1Api()
            patch = {"spec": {"unschedulable": None}}
            with suppress_deprecation_warnings():
                k8s_core.patch_node(node, patch)
            self.logger.info(f"[HW Faults] Node {node} uncordoned")
            return True
        except Exception as e:
            self.logger.warning(f"[HW Faults] Failed to uncordon {node}: {e}")
            return False

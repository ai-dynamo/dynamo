# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
CUDA fault injection utilities for GPU failure simulation.

Provides tools to inject CUDA faults that simulate GPU hardware failures
(like XID 79 - GPU falls off the bus).
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from kubernetes import client
from kubernetes.client.rest import ApiException


class CUDAFaultInjector:
    """Manages CUDA fault injection library and deployment patching."""

    VALID_XID_TYPES = {79, 48, 94, 95, 43, 74}

    def __init__(self, lib_dir: Optional[Path] = None):
        """
        Initialize CUDA fault injector.

        Args:
            lib_dir: Directory containing CUDA fault injection library.
                    If None, uses default relative to this module.
        """
        if lib_dir is None:
            lib_dir = Path(__file__).parent.parent / "cuda_fault_injection"

        self.lib_dir = lib_dir
        self.lib_path = lib_dir / "cuda_intercept.so"
        self.lib_built = False

    def build_library(self) -> bool:
        """
        Build the CUDA fault injection library.

        Returns:
            True if build succeeded or library already exists
        """
        print("\n[→] Building CUDA fault injection library...")

        if not self.lib_dir.exists():
            print(f"    ✗ Directory not found: {self.lib_dir}")
            return False

        if self.lib_path.exists():
            print(f"    ✓ Library already exists: {self.lib_path}")
            self.lib_built = True
            return True

        # Build using make
        result = subprocess.run(
            ["make"], cwd=self.lib_dir, capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"    ✗ Build failed: {result.stderr}")
            return False

        if not self.lib_path.exists():
            print(f"    ✗ Library not created: {self.lib_path}")
            return False

        print(f"    ✓ Library built: {self.lib_path}")
        self.lib_built = True
        return True

    def create_configmap_with_library(self, namespace: str) -> bool:
        """
        Create ConfigMap with CUDA fault injection library source.

        The library will be compiled in an init container to ensure
        Linux compatibility.

        Args:
            namespace: Kubernetes namespace

        Returns:
            True if ConfigMap created successfully
        """
        sys.path.insert(0, str(self.lib_dir))

        try:
            from inject_into_pods import create_cuda_fault_configmap

            return create_cuda_fault_configmap(namespace)
        except Exception as e:
            print(f"    ✗ Failed to create ConfigMap: {e}")
            import traceback

            traceback.print_exc()
            return False

    def check_if_cuda_library_deployed(
        self, deployment_name: str, namespace: str
    ) -> bool:
        """
        Check if CUDA fault injection is already deployed to the deployment.

        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace

        Returns:
            True if CUDA fault library is already deployed, False otherwise
        """
        try:
            k8s_custom = client.CustomObjectsApi()

            # Get the DynamoGraphDeployment
            dgd = k8s_custom.get_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="dynamographdeployments",
                name=deployment_name,
            )

            # Check for LD_PRELOAD in worker container env
            spec = dgd.get("spec", {})
            worker_spec = spec.get("workerSpec", {})
            pod_spec = worker_spec.get("podSpec", {})
            containers = pod_spec.get("containers", [])

            for container in containers:
                if container.get("name") in ["vllm-worker", "worker"]:
                    env = container.get("env", [])
                    for env_var in env:
                        if env_var.get("name") == "LD_PRELOAD":
                            return True

            return False

        except Exception:
            # If we can't read the deployment, assume it's not deployed
            return False

    def patch_deployment_for_cuda_fault(
        self,
        deployment_name: str,
        namespace: str,
        target_node: Optional[str] = None,
        xid_type: int = 79,
        passthrough_mode: bool = False,
    ) -> bool:
        """
        Patch deployment to enable CUDA fault injection.

        Adds:
        - ConfigMap volume with library source
        - Init container to compile library
        - LD_PRELOAD environment variable
        - CUDA_XID_TYPE environment variable
        - CUDA_FAULT_INJECTION_ENABLED (0 in passthrough mode, 1 otherwise)
        - Node affinity (if target_node specified)

        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            target_node: Node to pin pods to (simulates real XID behavior)
            xid_type: XID error type to simulate (79, 48, 94, 95, 43, 74). Default: 79
            passthrough_mode: If True, set CUDA_FAULT_INJECTION_ENABLED=0
                            (library loaded but faults disabled for baseline)

        Returns:
            True if patch succeeded
        """
        if xid_type not in self.VALID_XID_TYPES:
            print(
                f"    ✗ Invalid xid_type: {xid_type}. Valid values: {sorted(self.VALID_XID_TYPES)}"
            )
            return False

        print(
            f"\n[→] Patching deployment to enable CUDA fault injection (XID {xid_type})..."
        )

        sys.path.insert(0, str(self.lib_dir))

        try:
            from inject_into_pods import patch_deployment_env

            return patch_deployment_env(
                deployment_name,
                namespace,
                enable=True,
                use_configmap=True,
                target_node=target_node,
                xid_type=xid_type,
                passthrough_mode=passthrough_mode,
            )

        except Exception as e:
            print(f"    ✗ Failed to patch: {e}")
            import traceback

            traceback.print_exc()
            return False

    def cleanup_cuda_fault_injection(
        self,
        deployment_name: str,
        namespace: str,
        force_delete_pods: bool = True,
        service_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Remove CUDA fault injection from deployment.

        Removes:
        - LD_PRELOAD environment variable
        - ConfigMap volume mounts
        - Init container
        - Node affinity
        - ConfigMap

        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            force_delete_pods: If True, force delete pods to apply clean spec
            service_names: Service names to check (default: ["VllmDecodeWorker", "VllmPrefillWorker"])

        Returns:
            True if cleanup succeeded
        """
        if service_names is None:
            service_names = ["VllmDecodeWorker", "VllmPrefillWorker"]
        print("\n[→] Cleaning up CUDA fault injection...")

        sys.path.insert(0, str(self.lib_dir))

        try:
            from inject_into_pods import (
                delete_cuda_fault_configmap,
                patch_deployment_env,
            )

            k8s_core = client.CoreV1Api()
            k8s_custom = client.CustomObjectsApi()

            # Step 1: Remove from deployment spec
            print("    → Removing CUDA fault artifacts from deployment...")
            if not patch_deployment_env(
                deployment_name, namespace, enable=False, use_configmap=True
            ):
                print("    ✗ Failed to patch deployment")
                return False

            # Step 2: Verify spec is clean
            print("    → Verifying deployment spec is clean...")
            spec_cleaned = False

            for attempt in range(6):
                time.sleep(5)
                try:
                    dgd = k8s_custom.get_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=namespace,
                        plural="dynamographdeployments",
                        name=deployment_name,
                    )

                    # Check for CUDA fault artifacts
                    has_artifacts = False
                    artifact_details = []

                    for service_name in service_names:
                        service = (
                            dgd.get("spec", {})
                            .get("services", {})
                            .get(service_name, {})
                        )

                        # Check for LD_PRELOAD
                        env_vars = (
                            service.get("extraPodSpec", {})
                            .get("mainContainer", {})
                            .get("env", [])
                        )
                        for env in env_vars:
                            if (
                                isinstance(env, dict)
                                and env.get("name") == "LD_PRELOAD"
                            ):
                                has_artifacts = True
                                artifact_details.append(f"{service_name}: LD_PRELOAD")
                                break

                        # Check for node affinity
                        affinity = service.get("extraPodSpec", {}).get("affinity")
                        if (
                            affinity
                            and isinstance(affinity, dict)
                            and "nodeAffinity" in affinity
                        ):
                            has_artifacts = True
                            artifact_details.append(f"{service_name}: nodeAffinity")

                        # Check for CUDA fault volumes
                        volumes = service.get("extraPodSpec", {}).get("volumes", [])
                        for vol in volumes:
                            if vol.get("name") in [
                                "cuda-fault-lib",
                                "cuda-fault-lib-source",
                            ]:
                                has_artifacts = True
                                artifact_details.append(
                                    f"{service_name}: cuda-fault volume"
                                )
                                break

                    if not has_artifacts:
                        print(
                            f"    ✓ Deployment spec verified clean after {(attempt+1)*5}s"
                        )
                        spec_cleaned = True
                        break
                    else:
                        print(
                            f"    ... {(attempt+1)*5}s: Artifacts: {', '.join(artifact_details)}"
                        )

                except Exception as e:
                    print(f"    ... {(attempt+1)*5}s: Error checking spec: {e}")

            if not spec_cleaned:
                print("    ⚠ Could not verify spec is clean, continuing anyway...")

            # Step 3: Delete ConfigMap
            print("    → Deleting ConfigMap...")
            try:
                delete_cuda_fault_configmap(namespace)
                print("    ✓ ConfigMap deleted")
            except Exception as e:
                print(f"    ⚠ ConfigMap deletion: {e}")

            # Step 4: Force delete pods if requested
            if force_delete_pods:
                print("    → Force deleting ALL worker pods to apply clean spec...")
                try:
                    all_pods = k8s_core.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={deployment_name}",
                    )

                    deleted_count = 0
                    for pod in all_pods.items:
                        try:
                            k8s_core.delete_namespaced_pod(
                                name=pod.metadata.name,
                                namespace=namespace,
                                grace_period_seconds=0,
                            )
                            print(f"      ✓ Deleted: {pod.metadata.name}")
                            deleted_count += 1
                        except ApiException as e:
                            if e.status != 404:
                                print(
                                    f"      ⚠ Failed to delete {pod.metadata.name}: {e}"
                                )

                    if deleted_count > 0:
                        print(f"    ✓ Deleted {deleted_count} pod(s)")
                    else:
                        print("    [i] No pods to delete")

                except Exception as e:
                    print(f"    ⚠ Pod deletion: {e}")

            print("[✓] CUDA fault injection cleaned up successfully")
            return True

        except Exception as e:
            print(f"[✗] Cleanup failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def enable_cuda_faults_via_toggle(
        self, pods: List[client.V1Pod], namespace: str, enable: bool = True
    ) -> bool:
        """
        Enable or disable CUDA faults on running pods via environment variable toggle.

        This modifies the CUDA_FAULT_INJECTION_ENABLED env var in running pods
        without restarting them. Requires the CUDA library to already be loaded.

        Args:
            pods: List of pods to toggle faults on
            namespace: Kubernetes namespace
            enable: True to enable faults, False to disable

        Returns:
            True if toggle succeeded
        """
        if not pods:
            return False

        toggle_value = "1" if enable else "0"
        action = "Enabling" if enable else "Disabling"

        print(f"\n[→] {action} CUDA faults via toggle on {len(pods)} pods...")

        success_count = 0
        failed_pods = []

        for pod in pods:
            pod_name = pod.metadata.name

            try:
                # Get the main container name from pod spec
                container_name = (
                    pod.spec.containers[0].name if pod.spec.containers else None
                )
                if not container_name:
                    failed_pods.append((pod_name, "No container found"))
                    continue

                # Write toggle file to hostPath (persists across pod restarts on same node)
                # This simulates persistent hardware failure!
                exec_command = [
                    "sh",
                    "-c",
                    f'mkdir -p /host-fault && echo "{toggle_value}" > /host-fault/cuda_fault_enabled && cat /host-fault/cuda_fault_enabled',
                ]

                result = subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        namespace,
                        pod_name,
                        "-c",
                        container_name,
                        "--",
                    ]
                    + exec_command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    actual_value = result.stdout.strip()
                    if actual_value == toggle_value:
                        print(
                            f"    ✓ Toggle={toggle_value} in {pod_name}/{container_name}"
                        )
                        success_count += 1
                    else:
                        failed_pods.append(
                            (
                                pod_name,
                                f"Verify failed: expected '{toggle_value}', got '{actual_value}'",
                            )
                        )
                else:
                    failed_pods.append(
                        (pod_name, f"Exec failed: {result.stderr.strip()}")
                    )

            except Exception as e:
                failed_pods.append((pod_name, str(e)))
                continue

        if failed_pods:
            print(f"    ⚠ Failed to toggle {len(failed_pods)} pods:")
            for pod_name, error in failed_pods:
                print(f"       - {pod_name}: {error}")

        print(f"    → Result: {success_count}/{len(pods)} pods toggled successfully")
        return success_count > 0

    def disable_cuda_faults_via_toggle(
        self, pods: List[client.V1Pod], namespace: str
    ) -> bool:
        """
        Disable CUDA faults on running pods via toggle.

        Args:
            pods: List of pod objects to disable faults on
            namespace: Kubernetes namespace

        Returns:
            True if disable succeeded
        """
        return self.enable_cuda_faults_via_toggle(pods, namespace, enable=False)

    def cleanup_node_fault_markers(
        self, pods: List[client.V1Pod], namespace: str
    ) -> bool:
        """
        Remove persistent fault marker files from node hostPath.
        This cleans up /host-fault/cuda_fault_enabled to prevent future tests from failing.

        Args:
            pods: List of pods (to access nodes)
            namespace: Kubernetes namespace

        Returns:
            True if cleanup succeeded
        """
        if not pods:
            return True

        print("    [->] Cleaning persistent fault markers from nodes...")

        success_count = 0
        nodes_cleaned = set()

        for pod in pods:
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name

            # Skip if we already cleaned this node
            if node_name in nodes_cleaned:
                continue

            try:
                container_name = (
                    pod.spec.containers[0].name if pod.spec.containers else None
                )
                if not container_name:
                    continue

                # Remove the persistent marker file from hostPath
                exec_command = [
                    "sh",
                    "-c",
                    'rm -f /host-fault/cuda_fault_enabled 2>/dev/null; echo "ok"',
                ]

                result = subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        namespace,
                        pod_name,
                        "-c",
                        container_name,
                        "--",
                    ]
                    + exec_command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    print(f"    ✓ Cleaned fault marker on node {node_name}")
                    nodes_cleaned.add(node_name)
                    success_count += 1

            except Exception:
                continue

        return success_count > 0

    def verify_env_var_set(
        self,
        deployment_name: str,
        namespace: str,
        expected_value: str,
        max_wait: int = 30,
    ) -> bool:
        """
        Verify that CUDA_FAULT_INJECTION_ENABLED env var is set to expected value.
        Polls until the value matches or timeout.

        Args:
            deployment_name: Name of the DynamoGraphDeployment
            namespace: Kubernetes namespace
            expected_value: Expected value ("0" or "1")
            max_wait: Maximum seconds to wait

        Returns:
            True if verified
        """
        k8s_custom = client.CustomObjectsApi()
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                dgd = k8s_custom.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="dynamographdeployments",
                    name=deployment_name,
                )

                # Check both worker services
                for service_name in ["VllmDecodeWorker", "VllmPrefillWorker"]:
                    if service_name in dgd["spec"]["services"]:
                        service = dgd["spec"]["services"][service_name]
                        env_vars = (
                            service.get("extraPodSpec", {})
                            .get("mainContainer", {})
                            .get("env", [])
                        )

                        for env_var in env_vars:
                            if env_var.get("name") == "CUDA_FAULT_INJECTION_ENABLED":
                                if env_var.get("value") != expected_value:
                                    time.sleep(1)
                                    break  # Try again
                        else:
                            continue  # This service is good
                        break  # Inner loop broke, try again
                else:
                    # All services verified
                    return True

            except Exception:
                time.sleep(1)

        return False

    def trigger_pod_restart(self, pods: List[client.V1Pod], namespace: str):
        """
        Delete pods to trigger restart with new env vars.

        Args:
            pods: List of pod objects to restart
            namespace: Kubernetes namespace
        """
        k8s_core = client.CoreV1Api()

        print("\n[→] Deleting pods to trigger restart with CUDA fault injection...")

        for pod in pods:
            try:
                k8s_core.delete_namespaced_pod(
                    name=pod.metadata.name, namespace=namespace, grace_period_seconds=0
                )
                print(f"    ✓ Deleted: {pod.metadata.name}")
            except ApiException as e:
                print(f"    ✗ Failed to delete {pod.metadata.name}: {e}")

    def wait_for_pods_to_crash(
        self, namespace: str, label_selector: str, node_name: str, timeout: int = 420
    ) -> bool:
        """
        Wait for pods to enter crash state due to CUDA errors.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector for pods
            node_name: Node to monitor
            timeout: Max seconds to wait (default: 420 = 7 minutes)

        Returns:
            True if pods crashed, False if timeout
        """
        k8s_core = client.CoreV1Api()

        print("\n[→] Waiting for pods to crash due to CUDA errors...")
        print(f"    Monitoring pods on {node_name}")
        print(f"    Timeout: {timeout}s")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pods = k8s_core.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector,
                    field_selector=f"spec.nodeName={node_name}",
                )

                if len(pods.items) == 0:
                    print(f"    ⚠ No pods found on {node_name}")
                    time.sleep(30)
                    continue

                crashed_count = 0
                for pod in pods.items:
                    if pod.status.container_statuses:
                        cs = pod.status.container_statuses[0]
                        if (
                            cs.state.waiting
                            and cs.state.waiting.reason in ["CrashLoopBackOff", "Error"]
                        ) or cs.state.terminated:
                            crashed_count += 1

                elapsed = int(time.time() - start_time)
                print(f"    ... {elapsed}s: {crashed_count}/{len(pods.items)} crashed")

                if crashed_count >= len(pods.items) and crashed_count > 0:
                    print(f"    ✓ All pods crashed after {elapsed}s!")
                    return True

            except Exception as e:
                print(f"    ... Error checking pods: {e}")

            time.sleep(30)

        print(f"    ✗ Not all pods crashed within {timeout}s")
        return False

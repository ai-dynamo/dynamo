# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            lib_dir = Path(__file__).parent.parent / "cuda-fault-injection"

        self.lib_dir = lib_dir
        self.lib_path = lib_dir / "fake_cuda_xid79.so"
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

    def patch_deployment_for_cuda_fault(
        self,
        deployment_name: str,
        namespace: str,
        target_node: Optional[str] = None,
        xid_type: int = 79,
    ) -> bool:
        """
        Patch deployment to enable CUDA fault injection.

        Adds:
        - ConfigMap volume with library source
        - Init container to compile library
        - LD_PRELOAD environment variable
        - CUDA_XID_TYPE environment variable
        - Node affinity (if target_node specified)

        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            target_node: Node to pin pods to (simulates real XID behavior)
            xid_type: XID error type to simulate (79, 48, 94, 95, 43, 74). Default: 79

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

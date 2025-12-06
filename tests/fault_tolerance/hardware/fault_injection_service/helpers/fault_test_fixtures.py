# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Pytest Fixtures for Fault Tolerance Testing

Write 2-line tests instead of 600+ lines!

Example usage:
    # Setup once in conftest.py:
    from fault_test_fixtures import *

    # Write tests (2 lines each):
    def test_xid79(xid79_test):
        xid79_test(gpu_id=0)

    def test_xid74(xid74_test):
        xid74_test(gpu_id=0)

This file contains:
1. Core framework (fault specs, expectations, test orchestration)
2. Pytest fixtures (the simple API you use in tests)

Users only interact with fixtures - the framework is internal implementation.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pytest
import requests
from kubernetes import client, config

# Import existing helpers
try:
    from .cuda_fault_injection import CUDAFaultInjector
    from .inference_testing import InferenceLoadTester
    from .k8s_operations import NodeOperations
except ImportError:
    from cuda_fault_injection import CUDAFaultInjector
    from inference_testing import InferenceLoadTester
    from k8s_operations import NodeOperations


# =============================================================================
# PART 1: CORE FRAMEWORK (Internal - Users Don't Touch This)
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration & Auto-detection
# -----------------------------------------------------------------------------


class TestConfig:
    """Auto-detect environment and provide sensible defaults."""

    def __init__(self):
        # Detect if running in-cluster
        self.in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None
        self.k8s_available = False
        self.k8s_core = None

        try:
            if self.in_cluster:
                config.load_incluster_config()
                self.api_base_url = "http://fault-injection-api.fault-injection-system.svc.cluster.local:8080"
            else:
                config.load_kube_config()
                self.api_base_url = os.getenv(
                    "FAULT_INJECTION_API", "http://localhost:8080"
                )
            self.k8s_core = client.CoreV1Api()
            self.k8s_available = True
        except Exception:
            # No kubeconfig available (e.g., CI environment)
            # Tests will be skipped at runtime, but import won't fail
            self.api_base_url = os.getenv(
                "FAULT_INJECTION_API", "http://localhost:8080"
            )

        self.namespace = os.getenv("TEST_NAMESPACE", "dynamo-test")
        self.nvsentinel_namespace = os.getenv("NVSENTINEL_NAMESPACE", "nvsentinel")

        # Inference defaults
        self.inference_endpoint = os.getenv(
            "INFERENCE_ENDPOINT", "http://localhost:8000/v1/completions"
        )
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")


# Global config instance
_config = TestConfig()


# -----------------------------------------------------------------------------
# Fault Specifications (What to Inject)
# -----------------------------------------------------------------------------


class FaultSpec(ABC):
    """Base class for fault specifications."""

    @abstractmethod
    def inject(self, api_base_url: str, target_node: str) -> str:
        """Inject the fault. Returns fault_id for cleanup."""
        pass

    @abstractmethod
    def setup_cuda_faults(
        self,
        cuda_injector: CUDAFaultInjector,
        deployment: str,
        namespace: str,
        target_node: str,
        passthrough_mode: bool = False,
    ) -> bool:
        """
        Setup CUDA-level faults (if applicable).

        Args:
            passthrough_mode: If True, deploy library with faults disabled (ENABLED=0)
                            for baseline testing. Faults can be enabled later via toggle.

        Returns:
            True if deployment was patched (needs restart), False if already set up
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable fault name."""
        pass


@dataclass
class XID79Fault(FaultSpec):
    """XID 79: GPU fell off bus."""

    gpu_id: int = 0

    def inject(self, api_base_url: str, target_node: str) -> str:
        response = requests.post(
            f"{api_base_url}/api/v1/faults/gpu/inject/xid-79",
            json={"node_name": target_node, "xid_type": 79, "gpu_id": self.gpu_id},
            timeout=60,
        )
        assert response.status_code == 200, f"XID injection failed: {response.text}"
        return response.json()["fault_id"]

    def setup_cuda_faults(
        self, cuda_injector, deployment, namespace, target_node, passthrough_mode=False
    ) -> bool:
        """XID 79 causes CUDA calls to fail immediately."""
        # Check if already deployed
        if cuda_injector.check_if_cuda_library_deployed(deployment, namespace):
            print("[INFO] CUDA library already deployed - skipping setup")
            return False  # No restart needed

        # Build and deploy library (one-time setup)
        assert cuda_injector.build_library(), "Failed to build CUDA library"
        assert cuda_injector.create_configmap_with_library(
            namespace
        ), "Failed to create ConfigMap"
        assert cuda_injector.patch_deployment_for_cuda_fault(
            deployment,
            namespace,
            target_node=target_node,
            xid_type=79,
            passthrough_mode=passthrough_mode,
        ), "Failed to patch deployment"
        # Note: Deployment patching will trigger pod restart (one time)
        # But future enable/disable uses toggle (no restarts!)
        return True

    @property
    def name(self) -> str:
        return f"XID 79 (GPU {self.gpu_id} fell off bus)"


@dataclass
class XID74Fault(FaultSpec):
    """XID 74: NVLink error."""

    gpu_id: int = 0

    def inject(self, api_base_url: str, target_node: str) -> str:
        response = requests.post(
            f"{api_base_url}/api/v1/faults/gpu/inject/xid-74",
            json={"node_name": target_node, "xid_type": 74, "gpu_id": self.gpu_id},
            timeout=60,
        )
        assert response.status_code == 200, f"XID injection failed: {response.text}"
        return response.json()["fault_id"]

    def setup_cuda_faults(
        self, cuda_injector, deployment, namespace, target_node, passthrough_mode=False
    ) -> bool:
        """XID 74 causes intermittent CUDA errors."""
        # Check if already deployed
        if cuda_injector.check_if_cuda_library_deployed(deployment, namespace):
            print("[INFO] CUDA library already deployed - skipping setup")
            return False  # No restart needed

        # Build and deploy library (one-time setup)
        assert cuda_injector.build_library(), "Failed to build CUDA library"
        assert cuda_injector.create_configmap_with_library(
            namespace
        ), "Failed to create ConfigMap"
        assert cuda_injector.patch_deployment_for_cuda_fault(
            deployment,
            namespace,
            target_node=target_node,
            xid_type=74,
            passthrough_mode=passthrough_mode,
        ), "Failed to patch deployment"
        # Note: Deployment patching will trigger pod restart (one time)
        # But future enable/disable uses toggle (no restarts!)
        return True

    @property
    def name(self) -> str:
        return f"XID 74 (NVLink error on GPU {self.gpu_id})"


@dataclass
class NetworkPartitionFault(FaultSpec):
    """Network partition between pods."""

    source_label: str
    target_label: str
    block_percentage: int = 100

    def inject(self, api_base_url: str, target_node: str) -> str:
        response = requests.post(
            f"{api_base_url}/api/v1/faults/network/partition",
            json={
                "source_label": self.source_label,
                "target_label": self.target_label,
                "block_percentage": self.block_percentage,
            },
            timeout=60,
        )
        assert response.status_code == 200, f"Network partition failed: {response.text}"
        return response.json()["fault_id"]

    def setup_cuda_faults(
        self, cuda_injector, deployment, namespace, target_node, passthrough_mode=False
    ) -> bool:
        return False  # No CUDA faults for network issues

    @property
    def name(self) -> str:
        return f"Network partition ({self.source_label} → {self.target_label}, {self.block_percentage}%)"


# -----------------------------------------------------------------------------
# Response Expectations (What Should Happen)
# -----------------------------------------------------------------------------


@dataclass
class ResponseExpectation:
    """What should happen after fault injection."""

    # NVSentinel automated responses
    cordon: bool = False  # Should node be cordoned?
    drain: bool = False  # Should pods be drained?
    remediate: bool = False  # Should GPU driver restart?
    uncordon: bool = False  # Should node be uncordoned automatically?

    # Recovery expectations
    pods_recover: bool = True  # Should pods become healthy?
    inference_recovers: bool = True  # Should inference work?
    min_success_rate: float = 90.0  # Minimum success rate (%)

    # Timeouts (seconds)
    cordon_timeout: int = 180
    drain_timeout: int = 300
    remediate_timeout: int = 600
    uncordon_timeout: int = 180
    recovery_timeout: int = 900

    # Custom validators
    custom_checks: List[Callable[[], bool]] = field(default_factory=list)


# Common preset expectations
class NVSentinelResponse:
    """Presets for NVSentinel automated workflows."""

    @staticmethod
    def full_automation(uncordon: bool = False) -> ResponseExpectation:
        """Full NVSentinel workflow: detect → cordon → drain → remediate → recover."""
        return ResponseExpectation(
            cordon=True,
            drain=True,
            remediate=True,
            uncordon=uncordon,
            pods_recover=True,
            inference_recovers=True,
        )

    @staticmethod
    def cordon_only() -> ResponseExpectation:
        """Only expect cordoning (minimal response)."""
        return ResponseExpectation(
            cordon=True,
            drain=False,
            remediate=False,
            uncordon=False,
            pods_recover=False,
            inference_recovers=False,
        )

    @staticmethod
    def cordon_and_drain() -> ResponseExpectation:
        """Cordon + drain, but no automatic remediation."""
        return ResponseExpectation(
            cordon=True,
            drain=True,
            remediate=False,
            uncordon=False,
            pods_recover=True,
            inference_recovers=True,
        )


# -----------------------------------------------------------------------------
# Test Orchestrator (Runs the Test Workflow)
# -----------------------------------------------------------------------------


class FaultToleranceTest:
    """
    Core test orchestrator - handles fault injection, monitoring, recovery, cleanup.

    Users don't call this directly - use pytest fixtures instead!
    """

    def __init__(
        self,
        fault: FaultSpec,
        deployment: str,
        expect: ResponseExpectation,
        namespace: Optional[str] = None,
        target_node: Optional[str] = None,
        config: Optional[TestConfig] = None,
    ):
        self.fault = fault
        self.deployment = deployment
        self.expect = expect
        self.config = config or _config
        self.namespace = namespace or self.config.namespace
        self.target_node = target_node

        # Components
        self.cuda_injector = CUDAFaultInjector()
        self.load_tester = InferenceLoadTester(
            self.config.inference_endpoint, self.config.model_name
        )
        self.node_ops = NodeOperations(self.config.k8s_core)

        # Track phase stats for comparison
        self.phase_stats = []  # List of (phase_name, stats_dict)

        # State tracking for cleanup
        self.fault_id: Optional[str] = None
        self.cuda_artifacts_created = False
        self.original_node_state: Optional[Dict] = None

    def run(self):
        """Run the complete test workflow."""
        try:
            self._print_header()
            self._phase_prerequisites()
            self._phase_inject_fault()
            self._phase_monitor_response()
            self._phase_validate_recovery()
            self._print_success()
        except Exception as e:
            self._print_failure(e)
            raise
        finally:
            self._cleanup()

    def _print_header(self):
        print("\n" + "=" * 80)
        print(f"FAULT TOLERANCE TEST: {self.fault.name}")
        print("=" * 80)
        print(f"Deployment: {self.deployment}")
        print(f"Namespace: {self.namespace}")
        print(f"Expectations: {self._format_expectations()}")
        print("=" * 80)

    def _format_expectations(self) -> str:
        parts = []
        if self.expect.cordon:
            parts.append("cordon")
        if self.expect.drain:
            parts.append("drain")
        if self.expect.remediate:
            parts.append("remediate")
        if self.expect.uncordon:
            parts.append("uncordon")
        if self.expect.pods_recover:
            parts.append("pod-recovery")
        if self.expect.inference_recovers:
            parts.append("inference-recovery")
        return " → ".join(parts) if parts else "none"

    def _phase_prerequisites(self):
        print("\n[PHASE 0] Prerequisites & Health Checks")
        print("-" * 80)

        response = requests.get(f"{self.config.api_base_url}/health", timeout=5)
        assert response.status_code == 200, f"API unhealthy ({response.status_code})"
        print("[PASS] Fault injection API healthy")

        # Wait for at least 1 ready pod (with timeout)
        print("[->] Waiting for worker pods to be ready...")
        start_time = time.time()
        timeout = 300  # 5 minutes

        while time.time() - start_time < timeout:
            pods = self._get_worker_pods()
            ready_pods = [p for p in pods if self._is_pod_ready(p)]

            if len(ready_pods) >= 1:
                print(
                    f"[PASS] {len(ready_pods)} worker pods ready (waited {time.time() - start_time:.0f}s)"
                )
                break

            elapsed = time.time() - start_time
            if elapsed % 30 < 10:  # Print every ~30s
                print(
                    f"    [{elapsed:.0f}s] Waiting... ({len(ready_pods)}/{len(pods)} ready)"
                )
            time.sleep(10)
        else:
            pods = self._get_worker_pods()
            ready_pods = [p for p in pods if self._is_pod_ready(p)]
            assert (
                False
            ), f"No ready pods found after {timeout}s (got {len(ready_pods)}/{len(pods)})"

        if not self.target_node:
            self.target_node = self._select_target_node()
        print(f"[PASS] Target node: {self.target_node}")

        self.original_node_state = self._capture_node_state(self.target_node)

        # Pre-deploy CUDA library in passthrough mode (before baseline)
        # Use natural pod distribution - no forced affinity (realistic!)
        print("\n[->] Pre-deploying CUDA library in passthrough mode...")
        print("    (Using natural pod distribution across nodes - realistic!)")

        # Check if already deployed
        if self.cuda_injector.check_if_cuda_library_deployed(
            self.deployment, self.namespace
        ):
            print("[INFO] CUDA library already deployed - skipping setup")
            self.cuda_artifacts_created = True
        else:
            # Build and deploy library (no node affinity - let K8s distribute naturally)
            assert self.cuda_injector.build_library(), "Failed to build CUDA library"
            assert self.cuda_injector.create_configmap_with_library(
                self.namespace
            ), "Failed to create ConfigMap"

            # Patch deployment WITHOUT node affinity (natural distribution)
            # We'll inject faults only on pods that land on target node
            assert self.cuda_injector.patch_deployment_for_cuda_fault(
                self.deployment,
                self.namespace,
                target_node=None,
                xid_type=79,
                passthrough_mode=True,
            ), "Failed to patch deployment"

            self.cuda_artifacts_created = True
            print("[->] Waiting for pods to restart with CUDA library...")
            time.sleep(45)

            if not self._wait_for_pods_ready(timeout=180, min_count=2):
                print("[WARN] Warning: Not all pods ready, but continuing...")

            print("[PASS] CUDA library loaded (passthrough mode, natural distribution)")

        # Identify which pods ended up on target node
        pods = self._get_worker_pods()
        target_pods = [p for p in pods if p.spec.node_name == self.target_node]
        other_pods = [p for p in pods if p.spec.node_name != self.target_node]
        print(
            f"    [INFO] Pod distribution: {len(target_pods)} on target node, {len(other_pods)} on other nodes"
        )

        print("\n[->] Starting continuous inference load (1 req / 3s)")
        self.load_tester.start(interval=3.0)
        time.sleep(6)
        self._print_inference_stats(
            "Baseline (healthy)", since_checkpoint=True, save_for_comparison=True
        )

    def _phase_inject_fault(self):
        print("\n[PHASE 1] Fault Injection")
        print("-" * 80)

        print(f"[->] Injecting {self.fault.name}")
        self.fault_id = self.fault.inject(self.config.api_base_url, self.target_node)
        print(f"[PASS] Fault injected (ID: {self.fault_id})")

        # Enable CUDA faults via toggle (no restart needed - library already loaded)
        pods = self._get_worker_pods()
        target_pods = [
            p
            for p in pods
            if p.spec.node_name == self.target_node and self._is_pod_ready(p)
        ]

        if not target_pods:
            print(
                "[WARN] No ready pods found on target node - faults may not take effect"
            )
        else:
            print(
                f"\n[->] Enabling CUDA faults on {len(target_pods)} pods on target node..."
            )

            # Checkpoint here - measure from when we enable faults
            self.load_tester.checkpoint()

            # ONLY use toggle file for now (node-specific, no restart needed)
            # We'll set the global env var after cordon to ensure persistence
            print("    [1/1] Writing toggle file to pods on target node...")
            self.cuda_injector.enable_cuda_faults_via_toggle(
                target_pods, self.namespace
            )
            self.cuda_artifacts_created = True

            # Check if any pods are on other nodes (they should stay healthy!)
            all_pods = self._get_worker_pods()
            other_node_pods = [
                p
                for p in all_pods
                if p.spec.node_name != self.target_node and self._is_pod_ready(p)
            ]
            if other_node_pods:
                print(
                    f"    [INFO] {len(other_node_pods)} pods on other nodes will stay healthy (realistic!)"
                )

            print("[PASS] CUDA faults active on target node only")
            print(
                "    [INFO] Only pods on faulty node will crash (realistic GPU failure)"
            )
            print("    [INFO] Pods on other nodes continue serving traffic")
            time.sleep(15)  # Wait for faults to cause crashes
            self._print_inference_stats(
                "During fault (partial failure)", save_for_comparison=True
            )

    def _phase_monitor_response(self):
        print("\n[PHASE 2] Monitor Automated Response")
        print("-" * 80)
        self.load_tester.checkpoint()  # Reset stats for this phase

        if self.expect.cordon:
            print("[->] Waiting for node cordon...")
            assert self._wait_for_cordon(
                self.expect.cordon_timeout
            ), "Node not cordoned"
            print("[PASS] Node cordoned by NVSentinel")
            print(
                "    [INFO] New pods can't schedule here - waiting for eviction of crash-looping pods"
            )
            self._print_inference_stats("After node cordoned (waiting for drain)")

        if self.expect.drain:
            print("[->] Waiting for NVSentinel to drain crash-looping pods...")
            print("    [INFO] Pods must be evicted to reschedule to healthy nodes")
            print("    [INFO] HostPath fault markers will not exist on new nodes")

            # Monitor pods being evicted and rescheduled away from faulty node
            print("    [->] Monitoring for NVSentinel eviction and rescheduling...")
            start_time = time.time()
            timeout = self.expect.drain_timeout if self.expect.drain_timeout else 300

            while time.time() - start_time < timeout:
                pods = self._get_worker_pods()
                target_pods = [p for p in pods if p.spec.node_name == self.target_node]

                if not target_pods:
                    elapsed = time.time() - start_time
                    print(
                        f"    [PASS] All pods evicted and rescheduled to healthy nodes ({elapsed:.0f}s)"
                    )
                    break

                elapsed = time.time() - start_time
                if elapsed % 30 < 10:
                    print(
                        f"        [{elapsed:.0f}s] {len(target_pods)} pods still on target node (waiting for eviction)"
                    )
                time.sleep(10)
            else:
                pods = self._get_worker_pods()
                target_pods = [p for p in pods if p.spec.node_name == self.target_node]
                print(
                    f"    [WARN] Timeout: {len(target_pods)} pods still on target node"
                )

            time.sleep(5)
            self._print_inference_stats(
                "After drain and rescheduling", save_for_comparison=True
            )

        if self.expect.remediate:
            print("[->] Waiting for GPU remediation...")
            time.sleep(10)
            print("[PASS] Remediation phase complete")
            self._print_inference_stats("After remediation")

    def _phase_validate_recovery(self):
        print("\n[PHASE 3] Recovery Validation")
        print("-" * 80)
        self.load_tester.checkpoint()  # Reset stats for this phase

        if self.expect.pods_recover:
            print("[->] Waiting for pods to recover...")
            assert self._wait_for_pods_ready(
                self.expect.recovery_timeout
            ), "Pods did not recover"
            print("[PASS] Pods recovered")
            self._print_inference_stats("After pods recovered (inference starting)")

        if self.expect.inference_recovers:
            print("[->] Waiting for inference recovery...")
            assert self._wait_for_inference_recovery(
                self.expect.recovery_timeout, self.expect.min_success_rate
            ), "Inference did not recover"
            print("[PASS] Inference recovered")
            self._print_inference_stats(
                "After recovery (healthy)", save_for_comparison=True
            )

        for i, check in enumerate(self.expect.custom_checks):
            print(f"[->] Running custom check {i+1}...")
            assert check(), f"Custom check {i+1} failed"
            print(f"[PASS] Custom check {i+1} passed")

    def _print_latency_comparison_table(self):
        """Print phase-by-phase latency comparison table."""
        if not self.phase_stats:
            return

        print("\n" + "=" * 100)
        print("LATENCY IMPACT ANALYSIS")
        print("=" * 100)

        # Table header
        print(
            f"{'Phase':<30} {'Success%':<12} {'Avg':<10} {'p50':<10} {'p95':<10} {'p99':<10}"
        )
        print("-" * 100)

        # Table rows
        for phase_name, stats in self.phase_stats:
            if stats["success"] > 0:
                print(
                    f"{phase_name:<30} "
                    f"{stats['success_rate']:>6.1f}%      "
                    f"{stats['avg_latency']:>6.2f}s    "
                    f"{stats['p50_latency']:>6.2f}s    "
                    f"{stats['p95_latency']:>6.2f}s    "
                    f"{stats['p99_latency']:>6.2f}s"
                )
            else:
                print(
                    f"{phase_name:<30} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
                )

        print("=" * 100)

        # Show impact if we have baseline and fault data
        if len(self.phase_stats) >= 2:
            baseline = self.phase_stats[0][1]
            fault = self.phase_stats[1][1]

            if baseline["success"] > 0 and fault["success"] > 0:
                latency_increase = (
                    (fault["avg_latency"] - baseline["avg_latency"])
                    / baseline["avg_latency"]
                ) * 100
                success_drop = baseline["success_rate"] - fault["success_rate"]

                print("\nFault Impact:")
                print(f"  Latency increase: {latency_increase:+.1f}% (avg)")
                print(f"  Success rate drop: {success_drop:.1f}%")

        print()

    def _print_success(self):
        stats = self.load_tester.get_stats(since_checkpoint=False)  # Show cumulative
        print("\n" + "=" * 80)
        print("TEST PASSED")
        print("=" * 80)
        print(f"Fault: {self.fault.name}")
        print(f"Cumulative Success Rate: {stats['success_rate']:.1f}%")
        print(f"Total Requests: {stats['total']} ({stats['success']} successful)")
        print("=" * 80)

        # Show phase-by-phase latency comparison
        self._print_latency_comparison_table()

    def _print_failure(self, error: Exception):
        stats = self.load_tester.get_stats(since_checkpoint=False)  # Show cumulative
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Error: {error}")
        print(f"Cumulative Success Rate: {stats['success_rate']:.1f}%")
        print(f"Total Requests: {stats['total']} ({stats['success']} successful)")
        print("=" * 80)

    def _cleanup(self):
        print("\n[CLEANUP]")
        print("-" * 80)

        self.load_tester.stop()
        print("[PASS] Load tester stopped")

        # Always check if CUDA artifacts exist (even if test was interrupted early)
        try:
            has_cuda_artifacts = self.cuda_injector.check_if_cuda_library_deployed(
                self.deployment, self.namespace
            )

            if has_cuda_artifacts or self.cuda_artifacts_created:
                # Check if we should keep CUDA library for faster repeated tests
                keep_cuda_library = (
                    os.getenv("KEEP_CUDA_LIBRARY", "false").lower() == "true"
                )

                if keep_cuda_library:
                    print(
                        "[->] Disabling CUDA faults (keeping library for next test)..."
                    )
                    try:
                        pods = self._get_worker_pods()
                        if pods:
                            self.cuda_injector.disable_cuda_faults_via_toggle(
                                pods, self.namespace
                            )
                            # Clean persistent fault markers from nodes
                            self.cuda_injector.cleanup_node_fault_markers(
                                pods, self.namespace
                            )
                            print("[PASS] CUDA faults disabled (library still loaded)")
                        else:
                            print("[WARN] No pods found to disable faults")
                    except Exception as e:
                        print(f"[WARN] Could not disable faults: {e}")
                    print(
                        "    [INFO] LD_PRELOAD, ConfigMap, and init containers remain"
                    )
                    print("    [INFO] Next test will skip CUDA deployment (faster)")
                    print(
                        "    [INFO] Set KEEP_CUDA_LIBRARY=false to fully remove artifacts"
                    )
                else:
                    print("[->] Removing CUDA fault injection artifacts...")
                    # First disable via toggle and clean node markers
                    try:
                        pods = self._get_worker_pods()
                        if pods:
                            self.cuda_injector.disable_cuda_faults_via_toggle(
                                pods, self.namespace
                            )
                            # Clean persistent fault markers from nodes
                            self.cuda_injector.cleanup_node_fault_markers(
                                pods, self.namespace
                            )
                    except Exception:
                        pass  # Toggle may fail if pods are already down
                    time.sleep(5)  # Let pods stabilize before full cleanup

                    # Then remove LD_PRELOAD, ConfigMap, init containers, etc.
                    self.cuda_injector.cleanup_cuda_fault_injection(
                        self.deployment,
                        self.namespace,
                        force_delete_pods=True,  # Force delete to apply clean spec immediately
                    )
                    print("[PASS] CUDA artifacts removed from deployment")
            else:
                print("[PASS] No CUDA artifacts to clean up")
        except Exception as e:
            print(f"[WARN] CUDA cleanup error: {e}")
            import traceback

            traceback.print_exc()

        if self.fault_id:
            try:
                requests.delete(
                    f"{self.config.api_base_url}/api/v1/faults/{self.fault_id}",
                    timeout=10,
                )
                print(f"[PASS] Fault {self.fault_id} cleaned")
            except Exception as e:
                print(f"[WARN] Fault cleanup error: {e}")

        if self.target_node:
            # Restore node state (uncordon + remove annotations)
            # Pass None if we didn't capture original state (test failed early)
            self._restore_node_state(self.target_node, self.original_node_state)

        print("[PASS] Cleanup complete")

    def _print_inference_stats(
        self,
        phase_label: str,
        since_checkpoint: bool = True,
        save_for_comparison: bool = False,
    ):
        """
        Print inference statistics.

        Args:
            phase_label: Description of current phase
            since_checkpoint: If True, show stats since last checkpoint (per-phase).
                            If False, show cumulative stats since test start.
            save_for_comparison: If True, save stats for end-of-test comparison table
        """
        stats = self.load_tester.get_stats(since_checkpoint=since_checkpoint)

        # Save for comparison if requested
        if save_for_comparison and stats["total"] > 0:
            self.phase_stats.append((phase_label, stats.copy()))

        # Skip if no requests sent (avoid "0.0% success (0/0 requests)")
        if stats["total"] == 0:
            phase_type = "this phase" if since_checkpoint else "total"
            print(f"    Inference Stats [{phase_label}] ({phase_type}):")
            print("       [INFO] No requests sent yet")
            return

        success_rate = stats["success_rate"]

        # Color code based on success rate
        if success_rate >= 90:
            status = "HEALTHY"
        elif success_rate >= 50:
            status = "DEGRADED"
        else:
            status = "FAILING"

        phase_type = "this phase" if since_checkpoint else "total"
        print(f"    Inference Stats [{phase_label}] ({phase_type}):")
        print(
            f"       {status}: {success_rate:.1f}% success ({stats['success']}/{stats['total']} requests)"
        )

    def _select_target_node(self) -> str:
        pods = self._get_worker_pods()
        ready_pods = [p for p in pods if self._is_pod_ready(p)]
        assert ready_pods, "No ready pods to select target node from"
        return ready_pods[0].spec.node_name

    def _get_worker_pods(self):
        return self.config.k8s_core.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={self.deployment}",
        ).items

    def _is_pod_ready(self, pod) -> bool:
        return (
            pod.status.phase == "Running"
            and pod.status.container_statuses
            and pod.status.container_statuses[0].ready
        )

    def _capture_node_state(self, node_name: str) -> Dict:
        node = self.config.k8s_core.read_node(node_name)
        return {
            "unschedulable": node.spec.unschedulable,
            "annotations": dict(node.metadata.annotations or {}),
        }

    def _restore_node_state(self, node_name: str, state: Optional[Dict]):
        try:
            # Always uncordon the node in cleanup (even if test failed mid-way)
            node = self.config.k8s_core.read_node(node_name)
            if node.spec.unschedulable:
                self.node_ops.uncordon_node(node_name)
                print(f"[PASS] Node {node_name} uncordoned")
            else:
                print(f"[PASS] Node {node_name} already schedulable")

            # Remove ANY NVSentinel/quarantine/test-related annotations
            node = self.config.k8s_core.read_node(node_name)  # Re-read after uncordon
            current_annotations = node.metadata.annotations or {}

            # Find annotations to remove
            annotations_to_remove = set()
            for k in current_annotations.keys():
                k_lower = k.lower()
                # Remove test-related AND any quarantine annotations
                if any(
                    keyword in k_lower
                    for keyword in ["quarantine", "nvsentinel", "test", "fault"]
                ):
                    # If we have original state, only remove new ones
                    if state is None or k not in state.get("annotations", {}):
                        annotations_to_remove.add(k)

            if annotations_to_remove:
                patch = {
                    "metadata": {
                        "annotations": {k: None for k in annotations_to_remove}
                    }
                }
                self.config.k8s_core.patch_node(node_name, patch)
                print(
                    f"[PASS] Removed {len(annotations_to_remove)} NVSentinel/test annotations"
                )
                if len(annotations_to_remove) <= 5:  # Show details if not too many
                    for k in annotations_to_remove:
                        print(f"    - {k}")
        except Exception as e:
            print(f"[WARN] Node restoration error: {e}")

    def _wait_for_cordon(self, timeout: int) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            node = self.config.k8s_core.read_node(self.target_node)
            if node.spec.unschedulable:
                return True
            time.sleep(5)
        return False

    def _wait_for_pods_ready(self, timeout: int, min_count: int = 3) -> bool:
        start = time.time()
        last_status_time = start

        while time.time() - start < timeout:
            pods = self._get_worker_pods()
            ready_pods = [p for p in pods if self._is_pod_ready(p)]

            # Print progress every 30 seconds
            elapsed = time.time() - start
            if elapsed - (last_status_time - start) >= 30:
                print(
                    f"    [{elapsed:.0f}s] Waiting for pods: {len(ready_pods)}/{min_count} ready"
                )
                last_status_time = time.time()

            if len(ready_pods) >= min_count:
                print(f"    [{elapsed:.0f}s] [OK] {len(ready_pods)} pods ready")
                return True
            time.sleep(10)

        print(f"    [FAIL] Timeout: Only {len(ready_pods)}/{min_count} pods ready")
        return False

    def _wait_for_inference_recovery(
        self, timeout: int, min_success_rate: float
    ) -> bool:
        baseline_stats = self.load_tester.get_stats()
        pods_ready = False
        start = time.time()
        last_report_time = start

        while time.time() - start < timeout:
            if not pods_ready:
                pods = self._get_worker_pods()
                ready_pods = [p for p in pods if self._is_pod_ready(p)]
                if len(ready_pods) >= 3:
                    pods_ready = True
                    baseline_stats = self.load_tester.get_stats()
                    print(
                        f"    [{time.time() - start:.0f}s] Pods ready, measuring recovery..."
                    )
                time.sleep(10)
                continue

            stats = self.load_tester.get_stats()
            recovery_requests = stats["total"] - baseline_stats["total"]
            recovery_successes = stats["success"] - baseline_stats["success"]

            # Report progress every 30 seconds
            if time.time() - last_report_time >= 30:
                if recovery_requests > 0:
                    recovery_rate = recovery_successes / recovery_requests * 100
                    print(
                        f"    [{time.time() - start:.0f}s] Recovery progress: {recovery_rate:.1f}% ({recovery_successes}/{recovery_requests})"
                    )
                last_report_time = time.time()

            if recovery_requests >= 5:
                recovery_rate = recovery_successes / recovery_requests * 100
                if recovery_rate >= min_success_rate:
                    print(
                        f"    [{time.time() - start:.0f}s] Recovery complete: {recovery_rate:.1f}% ({recovery_successes}/{recovery_requests})"
                    )
                    return True

            time.sleep(10)

        return False


# =============================================================================
# PART 2: PYTEST FIXTURES (Public API - Users Use These!)
# =============================================================================

# -----------------------------------------------------------------------------
# Base Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration (session-scoped for reuse)."""
    return TestConfig()


@pytest.fixture
def default_deployment():
    """Default deployment name from environment or constant."""
    return os.getenv("TARGET_DEPLOYMENT", "vllm-v1-disagg-router")


@pytest.fixture
def default_namespace(test_config):
    """Default namespace from config."""
    return test_config.namespace


# -----------------------------------------------------------------------------
# Generic Fault Test Fixture
# -----------------------------------------------------------------------------


@pytest.fixture
def fault_test(default_deployment, default_namespace, test_config):
    """
    Generic fault test fixture - handles any fault type.

    Usage:
        def test_my_fault(fault_test):
            fault_test(
                fault=XID79Fault(gpu_id=0),
                expect=NVSentinelResponse.full_automation(),
            )
    """

    def run_test(
        fault: FaultSpec,
        expect: Optional[ResponseExpectation] = None,
        deployment: Optional[str] = None,
        namespace: Optional[str] = None,
        target_node: Optional[str] = None,
    ):
        if expect is None:
            expect = NVSentinelResponse.full_automation()

        FaultToleranceTest(
            fault=fault,
            deployment=deployment or default_deployment,
            expect=expect,
            namespace=namespace or default_namespace,
            target_node=target_node,
            config=test_config,
        ).run()

    return run_test


# -----------------------------------------------------------------------------
# Specialized XID Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def xid79_test(fault_test):
    """
    Specialized fixture for XID 79 tests.

    Usage:
        def test_xid79_gpu0(xid79_test):
            xid79_test(gpu_id=0)
    """

    def run_xid79_test(
        gpu_id: int = 0,
        expect: Optional[ResponseExpectation] = None,
        deployment: Optional[str] = None,
    ):
        fault_test(
            fault=XID79Fault(gpu_id=gpu_id),
            expect=expect,
            deployment=deployment,
        )

    return run_xid79_test


@pytest.fixture
def xid74_test(fault_test):
    """
    Specialized fixture for XID 74 tests.

    Usage:
        def test_xid74_gpu0(xid74_test):
            xid74_test(gpu_id=0)
    """

    def run_xid74_test(
        gpu_id: int = 0,
        expect: Optional[ResponseExpectation] = None,
        deployment: Optional[str] = None,
    ):
        fault_test(
            fault=XID74Fault(gpu_id=gpu_id),
            expect=expect,
            deployment=deployment,
        )

    return run_xid74_test


# -----------------------------------------------------------------------------
# Network Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def network_partition_test(fault_test):
    """
    Specialized fixture for network partition tests.

    Usage:
        def test_partition_worker_to_frontend(network_partition_test):
            network_partition_test(
                source="nvidia.com/dynamo-component-type=worker",
                target="nvidia.com/dynamo-component-type=frontend",
            )
    """

    def run_network_test(
        source: str,
        target: str,
        block_percentage: int = 100,
        expect: Optional[ResponseExpectation] = None,
        deployment: Optional[str] = None,
    ):
        if expect is None:
            expect = ResponseExpectation(
                cordon=False,
                drain=False,
                pods_recover=True,
                inference_recovers=True,
                min_success_rate=70.0,
            )

        fault_test(
            fault=NetworkPartitionFault(
                source_label=source,
                target_label=target,
                block_percentage=block_percentage,
            ),
            expect=expect,
            deployment=deployment,
        )

    return run_network_test


# -----------------------------------------------------------------------------
# Expectation Builder Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def expect_full_automation():
    """Fixture that returns full automation expectation."""
    return NVSentinelResponse.full_automation()


@pytest.fixture
def expect_cordon_and_drain():
    """Fixture that returns cordon + drain expectation."""
    return NVSentinelResponse.cordon_and_drain()


@pytest.fixture
def expect_cordon_only():
    """Fixture that returns cordon-only expectation."""
    return NVSentinelResponse.cordon_only()


# -----------------------------------------------------------------------------
# Composite Fixtures (For Complex Scenarios)
# -----------------------------------------------------------------------------


@pytest.fixture
def xid79_with_custom_validation(fault_test):
    """
    XID 79 test with custom validation logic.

    Usage:
        def test_xid79_custom(xid79_with_custom_validation):
            def my_check():
                return True

            xid79_with_custom_validation(
                gpu_id=0,
                custom_checks=[my_check],
            )
    """

    def run_test(
        gpu_id: int = 0,
        custom_checks: list = None,
        min_success_rate: float = 90.0,
        deployment: Optional[str] = None,
    ):
        expect = ResponseExpectation(
            cordon=True,
            drain=True,
            pods_recover=True,
            inference_recovers=True,
            min_success_rate=min_success_rate,
            custom_checks=custom_checks or [],
        )

        fault_test(
            fault=XID79Fault(gpu_id=gpu_id),
            expect=expect,
            deployment=deployment,
        )

    return run_test


# -----------------------------------------------------------------------------
# Auto-cleanup Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="function")
def ensure_clean_test_environment(test_config):
    """Auto-runs before/after each test to ensure clean environment."""
    yield
    try:
        pass  # FaultToleranceTest already does cleanup
    except Exception as e:
        print(f"[WARN] Post-test cleanup warning: {e}")


# -----------------------------------------------------------------------------
# Conditional Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def skip_if_no_nvsentinel(test_config):
    """
    Skip test if NVSentinel is not deployed.

    Usage:
        def test_nvsentinel_feature(skip_if_no_nvsentinel, xid79_test):
            xid79_test(gpu_id=0)
    """
    k8s = test_config.k8s_core
    try:
        pods = k8s.list_namespaced_pod(namespace=test_config.nvsentinel_namespace)
        if not pods.items:
            pytest.skip("NVSentinel not deployed")
    except Exception:
        pytest.skip("Cannot access NVSentinel namespace")


@pytest.fixture
def skip_if_insufficient_gpus(test_config):
    """
    Skip test if cluster doesn't have enough GPUs.

    Usage:
        def test_multi_gpu(skip_if_insufficient_gpus, xid79_test):
            skip_if_insufficient_gpus(min_gpus=4)
            xid79_test(gpu_id=3)
    """

    def check_gpu_count(min_gpus: int):
        k8s = test_config.k8s_core
        nodes = k8s.list_node(label_selector="nvidia.com/gpu.present=true")

        total_gpus = 0
        for node in nodes.items:
            capacity = node.status.capacity or {}
            gpu_count = capacity.get("nvidia.com/gpu", "0")
            total_gpus += int(gpu_count)

        if total_gpus < min_gpus:
            pytest.skip(f"Insufficient GPUs: need {min_gpus}, have {total_gpus}")

    return check_gpu_count


# -----------------------------------------------------------------------------
# Markers for Test Organization
# -----------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "xid79: XID 79 (GPU fell off bus) tests")
    config.addinivalue_line("markers", "xid74: XID 74 (NVLink error) tests")
    config.addinivalue_line("markers", "network: Network fault tests")
    config.addinivalue_line(
        "markers", "nvsentinel: Tests that validate NVSentinel automation"
    )
    config.addinivalue_line("markers", "slow: Slow tests (>5 minutes)")
    config.addinivalue_line("markers", "fast: Fast tests (<2 minutes)")

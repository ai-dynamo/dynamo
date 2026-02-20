#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test script for v1beta1 Pydantic models.

Validates that the generated Pydantic models can be imported and used correctly.
"""

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    start = Path(__file__).parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=start,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Fallback: walk up until we find go.mod (same logic as generate_pydantic_from_go.py)
    p = start
    while p != p.parent:
        if (p / "go.mod").exists():
            return p
        p = p.parent
    return start


# Add the components src to path so we can import the generated models
sys.path.insert(0, str(_repo_root() / "components" / "src"))

import pydantic  # noqa: E402

from dynamo.profiler.utils.dgdr_v1beta1_types import (  # noqa: E402
    BackendType,
    DeploymentInfoStatus,
    DGDRPhase,
    DynamoGraphDeploymentRequestSpec,
    DynamoGraphDeploymentRequestStatus,
    FeaturesSpec,
    MockerSpec,
    ModelCacheSpec,
    OptimizationType,
    PlannerPreDeploymentSweepMode,
    PlannerSpec,
    ProfilingPhase,
    SearchStrategy,
    SLASpec,
    WorkloadSpec,
)

print("✓ Successfully imported all Pydantic models")


def test_simple_dgdr():
    """Test creating a simple DGDR (minimal spec)"""
    spec = DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-32B",
    )
    print("✓ Created simple DGDR spec")

    assert spec.model == "Qwen/Qwen3-32B"
    assert spec.backend == BackendType.Auto  # kubebuilder:default=auto
    assert spec.autoApply is True  # kubebuilder:default=true
    print("✓ Simple DGDR spec validation passed")


def test_full_dgdr():
    """Test creating a full DGDR with all fields"""
    spec = DynamoGraphDeploymentRequestSpec(
        model="meta-llama/Llama-3.1-405B",
        backend=BackendType.Vllm,
        image="nvcr.io/nvidia/dynamo-runtime:latest",
        workload=WorkloadSpec(
            isl=1024,
            osl=512,
            concurrency=10.0,
        ),
        sla=SLASpec(
            ttft=100.0,
            itl=10.0,
        ),
        modelCache=ModelCacheSpec(
            pvcName="model-cache",
            pvcModelPath="llama-3.1-405b",
        ),
        features=FeaturesSpec(
            planner=PlannerSpec(enabled=True),
            mocker=MockerSpec(enabled=False),
        ),
        searchStrategy=SearchStrategy.Rapid,
        autoApply=True,
    )
    print("✓ Created full DGDR spec")

    assert spec.model == "meta-llama/Llama-3.1-405B"
    assert spec.backend == BackendType.Vllm
    assert spec.workload.isl == 1024
    assert spec.sla.ttft == 100.0
    assert spec.sla.itl == 10.0
    assert spec.modelCache.pvcName == "model-cache"
    assert spec.modelCache.pvcModelPath == "llama-3.1-405b"
    assert spec.features.planner.enabled is True
    assert spec.features.mocker.enabled is False
    print("✓ Full DGDR spec validation passed")


def test_sla_defaults_and_validation():
    """Test SLASpec defaults and mutual-exclusivity validator"""
    # Default mode: ttft + itl with python-defaults
    sla = SLASpec()
    assert sla.ttft == 2000.0
    assert sla.itl == 30.0
    assert sla.e2eLatency is None
    assert sla.optimizationType is None
    print("✓ SLASpec defaults correct")

    # explicit ttft+itl mode: OK
    SLASpec(ttft=100.0, itl=10.0)

    # e2eLatency mode: OK (null out ttft/itl)
    SLASpec(ttft=None, itl=None, e2eLatency=500.0)

    # optimizationType mode: OK (null out ttft/itl)
    SLASpec(ttft=None, itl=None, optimizationType=OptimizationType.Throughput)

    # mixing modes should raise
    try:
        SLASpec(ttft=100.0, itl=10.0, e2eLatency=500.0)
        raise AssertionError("expected ValidationError for mixed SLA modes")
    except pydantic.ValidationError:
        pass

    # ttft without itl should raise
    try:
        SLASpec(itl=None, ttft=100.0)
        raise AssertionError("expected ValidationError for ttft without itl")
    except pydantic.ValidationError:
        pass

    print("✓ SLASpec validation correct")


def test_workload_defaults():
    """Test WorkloadSpec kubebuilder defaults"""
    w = WorkloadSpec()
    assert w.isl == 4000
    assert w.osl == 1000
    print("✓ WorkloadSpec defaults correct")


def test_enums():
    """Test enum values"""
    # DGDRPhase — TitleCase suffix from Go const names
    assert DGDRPhase.Pending == "Pending"
    assert DGDRPhase.Profiling == "Profiling"
    assert DGDRPhase.Deployed == "Deployed"

    # ProfilingPhase — TitleCase suffix from Go const names
    assert ProfilingPhase.Initializing == "Initializing"
    assert ProfilingPhase.SweepingPrefill == "SweepingPrefill"

    # OptimizationType — TitleCase from Go const names
    assert OptimizationType.Latency == "latency"
    assert OptimizationType.Throughput == "throughput"

    # SearchStrategy — TitleCase from Go const names
    assert SearchStrategy.Rapid == "rapid"
    assert SearchStrategy.Thorough == "thorough"

    # BackendType — mixed case from Go const names
    assert BackendType.Auto == "auto"
    assert BackendType.Vllm == "vllm"

    # PlannerPreDeploymentSweepMode (None → None_ to avoid Python keyword clash)
    assert PlannerPreDeploymentSweepMode.None_ == "none"
    assert PlannerPreDeploymentSweepMode.Rapid == "rapid"

    print("✓ All enum values validated")


def test_status_models():
    """Test status model creation"""
    status = DynamoGraphDeploymentRequestStatus(
        phase=DGDRPhase.Profiling,
        profilingPhase=ProfilingPhase.SweepingPrefill,
        dgdName="test-dgd",
        profilingJobName="test-profiling-job",
        deploymentInfo=DeploymentInfoStatus(
            replicas=3,
            availableReplicas=2,
        ),
    )
    print("✓ Created DGDR status")

    assert status.phase == DGDRPhase.Profiling
    assert status.profilingPhase == ProfilingPhase.SweepingPrefill
    assert status.deploymentInfo.replicas == 3
    print("✓ DGDR status validation passed")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing v1beta1 Pydantic Models")
    print("=" * 60 + "\n")

    test_simple_dgdr()
    test_full_dgdr()
    test_sla_defaults_and_validation()
    test_workload_defaults()
    test_enums()
    test_status_models()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

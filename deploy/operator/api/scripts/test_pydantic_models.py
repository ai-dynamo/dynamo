#!/usr/bin/env python3
"""
Test script for v1beta1 Pydantic models.

Validates that the generated Pydantic models can be imported and used correctly.
"""

import sys
from pathlib import Path

# Add the components src to path so we can import the generated models
components_src = (
    Path(__file__).parent.parent.parent.parent.parent / "components" / "src"
)
sys.path.insert(0, str(components_src))

try:
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        BackendType,
        DeploymentInfoStatus,
        DGDRPhase,
        DynamoGraphDeploymentRequestSpec,
        DynamoGraphDeploymentRequestStatus,
        FeaturesSpec,
        KVRouterSpec,
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
except ImportError as e:
    print(f"✗ Failed to import Pydantic models: {e}")
    sys.exit(1)


def test_simple_dgdr():
    """Test creating a simple DGDR (minimal spec)"""
    try:
        spec = DynamoGraphDeploymentRequestSpec(
            model="Qwen/Qwen3-32B",
        )
        print("✓ Created simple DGDR spec")

        # Validate fields
        assert spec.model == "Qwen/Qwen3-32B"
        assert spec.backend is None  # Optional, defaults to None
        assert spec.autoApply is None  # Optional
        print("✓ Simple DGDR spec validation passed")
    except Exception as e:
        print(f"✗ Simple DGDR spec test failed: {e}")
        sys.exit(1)


def test_full_dgdr():
    """Test creating a full DGDR with all fields"""
    try:
        spec = DynamoGraphDeploymentRequestSpec(
            model="meta-llama/Llama-3.1-405B",
            backend=BackendType.VLLM,
            image="nvcr.io/nvidia/dynamo-runtime:latest",
            workload=WorkloadSpec(
                isl=1024,
                osl=512,
                concurrency=10.0,
            ),
            sla=SLASpec(
                optimizationType=OptimizationType.Latency,
                ttft=100.0,
                itl=10.0,
            ),
            modelCache=ModelCacheSpec(
                pvcName="model-cache",
                pvcPath="llama-3.1-405b",
            ),
            features=FeaturesSpec(
                planner=PlannerSpec(enabled=True),
                kvRouter=KVRouterSpec(enabled=True),
                mocker=MockerSpec(enabled=False),
            ),
            searchStrategy=SearchStrategy.Rapid,
            autoApply=True,
        )
        print("✓ Created full DGDR spec")

        # Validate nested fields
        assert spec.model == "meta-llama/Llama-3.1-405B"
        assert spec.backend == BackendType.VLLM
        assert spec.workload.isl == 1024
        assert spec.sla.optimizationType == OptimizationType.Latency
        assert spec.modelCache.pvcName == "model-cache"
        assert spec.features.planner.enabled is True
        assert spec.features.kvRouter.enabled is True
        assert spec.features.mocker.enabled is False
        print("✓ Full DGDR spec validation passed")
    except Exception as e:
        print(f"✗ Full DGDR spec test failed: {e}")
        sys.exit(1)


def test_enums():
    """Test enum values"""
    try:
        # DGDRPhase uses uppercase names
        assert DGDRPhase.PENDING == "Pending"
        assert DGDRPhase.PROFILING == "Profiling"
        assert DGDRPhase.DEPLOYED == "Deployed"

        # ProfilingPhase uses uppercase names
        assert ProfilingPhase.INITIALIZING == "Initializing"
        assert ProfilingPhase.SWEEPINGPREFILL == "SweepingPrefill"

        # OptimizationType uses titlecase names from Go
        assert OptimizationType.Latency == "latency"
        assert OptimizationType.Throughput == "throughput"

        # SearchStrategy uses titlecase names from Go
        assert SearchStrategy.Rapid == "rapid"
        assert SearchStrategy.Thorough == "thorough"

        # BackendType uses mixed case names from Go
        assert BackendType.Auto == "auto"
        assert BackendType.VLLM == "vllm"

        # PlannerPreDeploymentSweepMode (None is reserved, becomes None_)
        assert PlannerPreDeploymentSweepMode.None_ == "none"
        assert PlannerPreDeploymentSweepMode.Rapid == "rapid"

        print("✓ All enum values validated")
    except Exception as e:
        print(f"✗ Enum validation failed: {e}")
        sys.exit(1)


def test_status_models():
    """Test status model creation"""
    try:
        status = DynamoGraphDeploymentRequestStatus(
            phase=DGDRPhase.PROFILING,
            profilingPhase=ProfilingPhase.SWEEPINGPREFILL,
            dgdName="test-dgd",
            profilingJobName="test-profiling-job",
            deploymentInfo=DeploymentInfoStatus(
                replicas=3,
                availableReplicas=2,
            ),
        )
        print("✓ Created DGDR status")

        # Validate status fields
        assert status.phase == DGDRPhase.PROFILING
        assert status.profilingPhase == ProfilingPhase.SWEEPINGPREFILL
        assert status.deploymentInfo.replicas == 3
        print("✓ DGDR status validation passed")
    except Exception as e:
        print(f"✗ Status model test failed: {e}")
        sys.exit(1)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing v1beta1 Pydantic Models")
    print("=" * 60 + "\n")

    test_simple_dgdr()
    test_full_dgdr()
    test_enums()
    test_status_models()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

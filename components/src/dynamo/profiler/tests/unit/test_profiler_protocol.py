# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler config_modifiers/protocol helpers."""

import copy
import logging
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

try:
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
    from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
        PickedParallelConfig,
    )
    from dynamo.profiler.utils.config_modifiers.protocol import (
        BaseConfigModifier,
        apply_dgd_overrides,
    )
    from dynamo.profiler.utils.defaults import SearchStrategy
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        OverridesSpec,
    )
    from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def dgdr_name_env(monkeypatch):
    """Set DGDR_NAME so _validate_dgd_service_name_lengths runs in tests."""
    monkeypatch.setenv("DGDR_NAME", "test-dgdr")


def test_build_dgd_config_shapes_multinode_worker_resources() -> None:
    """DP-only workers keep per-node GPU shaping without multinode inflation."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-30B-A3B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0",
        prefill_cli_args=["--max-running-requests", "1"],
        prefill_replicas=1,
        prefill_gpus=1,
        decode_cli_args=["--data-parallel-size", "16"],
        decode_replicas=1,
        decode_gpus=16,
        num_gpus_per_node=8,
    )

    decode_service = next(
        service
        for service in dgd_config["spec"]["services"].values()
        if service.get("subComponentType") == "decode"
    )
    assert decode_service["resources"]["limits"]["gpu"] == "8"
    assert decode_service.get("multinode") is None


def test_build_dgd_config_multinode_when_tp_exceeds_node() -> None:
    """Single instances that exceed node capacity still get multinode config."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="meta-llama/Llama-3-70B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0",
        prefill_cli_args=["--max-running-requests", "1"],
        prefill_replicas=1,
        prefill_gpus=1,
        decode_cli_args=["--tp", "16"],
        decode_replicas=1,
        decode_gpus=16,
        num_gpus_per_node=8,
    )

    decode_service = next(
        service
        for service in dgd_config["spec"]["services"].values()
        if service.get("subComponentType") == "decode"
    )
    assert decode_service["resources"]["limits"]["gpu"] == "8"
    assert decode_service["multinode"] == {"nodeCount": 2}


def test_build_dgd_config_multinode_parses_shell_joined_parallelism_args() -> None:
    """Multinode detection should handle shell-joined CLI args from templates."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="meta-llama/Llama-3-70B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0",
        prefill_cli_args=["--max-running-requests", "1"],
        prefill_replicas=1,
        prefill_gpus=1,
        decode_cli_args=["--tp 16", "--pp 2"],
        decode_replicas=1,
        decode_gpus=32,
        num_gpus_per_node=8,
    )

    decode_service = next(
        service
        for service in dgd_config["spec"]["services"].values()
        if service.get("subComponentType") == "decode"
    )
    assert decode_service["resources"]["limits"]["gpu"] == "8"
    assert decode_service["multinode"] == {"nodeCount": 4}


def test_apply_dgd_overrides_strips_envelope() -> None:
    """Envelope fields are stripped; nested payload keys are deep-merged."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "my-deployment", "namespace": "default"},
        "spec": {
            "services": {
                "Frontend": {"replicas": 1},
            }
        },
    }
    overrides = {
        # Envelope fields — must be stripped entirely.
        "apiVersion": "dynamo.ai/v1beta1",
        "kind": "SomethingElse",
        # metadata identity keys must be stripped; labels/annotations kept.
        "metadata": {
            "name": "injected-name",
            "namespace": "injected-ns",
            "uid": "abc-123",
            "resourceVersion": "999",
            "labels": {"team": "infra"},
            "annotations": {"note": "perf-run"},
        },
        # Regular payload key — must be deep-merged.
        "spec": {
            "services": {
                "Frontend": {"replicas": 3},
            }
        },
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # apiVersion and kind must not be changed.
    assert result["apiVersion"] == "dynamo.ai/v1alpha1"
    assert result["kind"] == "DynamoGraphDeployment"

    # Identity metadata keys must not be overwritten.
    assert result["metadata"]["name"] == "my-deployment"
    assert result["metadata"]["namespace"] == "default"
    assert "uid" not in result["metadata"]
    assert "resourceVersion" not in result["metadata"]

    # Safe metadata keys must be merged in.
    assert result["metadata"]["labels"] == {"team": "infra"}
    assert result["metadata"]["annotations"] == {"note": "perf-run"}

    # Regular spec overrides must be applied.
    assert result["spec"]["services"]["Frontend"]["replicas"] == 3

    # Original dicts must not be mutated.
    assert dgd_config["apiVersion"] == "dynamo.ai/v1alpha1"
    assert dgd_config["spec"]["services"]["Frontend"]["replicas"] == 1


def test_apply_dgd_overrides_no_metadata_in_overrides() -> None:
    """When overrides contain no metadata key, existing metadata is untouched."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "svc", "namespace": "ns"},
        "spec": {"services": {"Backend": {"replicas": 2}}},
    }
    overrides = {"spec": {"services": {"Backend": {"replicas": 5}}}}

    result = apply_dgd_overrides(dgd_config, overrides)

    assert result["metadata"] == {"name": "svc", "namespace": "ns"}
    assert result["spec"]["services"]["Backend"]["replicas"] == 5


def test_apply_dgd_overrides_metadata_only_identity_keys_dropped_entirely() -> None:
    """If metadata override contains only identity keys, nothing is merged into metadata."""
    dgd_config = {
        "apiVersion": "dynamo.ai/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "svc"},
        "spec": {},
    }
    overrides = {
        "metadata": {"name": "other", "namespace": "other-ns", "uid": "x"},
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # Only original metadata should remain — no extra keys added.
    assert result["metadata"] == {"name": "svc"}


def test_apply_dgd_overrides_extrapodspec_tolerations() -> None:
    """extraPodSpec.tolerations from overrides are merged into existing services.

    Regression test for TC-5.2a: interpolation DGDs were deployed without
    tolerations because apply_dgd_overrides was called after run_interpolation.
    This test verifies the merge logic itself is correct.
    """
    toleration = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    dgd_config = {
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "componentType": "worker",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "my-image",
                            "args": ["--model", "Qwen3-32B"],
                        }
                    },
                    "replicas": 1,
                },
                "Frontend": {
                    "extraPodSpec": {},
                },
            }
        }
    }
    overrides = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "placeholder"},
        "spec": {
            "services": {
                "VllmDecodeWorker": {"extraPodSpec": {"tolerations": [toleration]}},
                "Frontend": {"extraPodSpec": {"tolerations": [toleration]}},
            }
        },
    }

    result = apply_dgd_overrides(dgd_config, overrides)

    # Tolerations must be present on both services.
    decode_eps = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    assert decode_eps["tolerations"] == [toleration]
    # mainContainer must be preserved (not overwritten).
    assert decode_eps["mainContainer"]["image"] == "my-image"

    frontend_eps = result["spec"]["services"]["Frontend"]["extraPodSpec"]
    assert frontend_eps["tolerations"] == [toleration]

    # Original must not be mutated.
    assert (
        "tolerations"
        not in dgd_config["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    )


def test_apply_dgd_overrides_missing_service_skipped_with_warning(caplog) -> None:
    """Overrides for services absent from the DGD are skipped with a warning."""
    dgd_config = {
        "spec": {
            "services": {
                "Frontend": {"replicas": 1},
            }
        }
    }
    overrides = {
        "spec": {
            "services": {
                "Frontend": {"replicas": 2},
                "NonExistentWorker": {
                    "extraPodSpec": {"tolerations": [{"key": "foo"}]}
                },
            }
        }
    }

    with caplog.at_level(
        logging.WARNING, logger="dynamo.profiler.utils.config_modifiers.protocol"
    ):
        result = apply_dgd_overrides(dgd_config, overrides)

    assert result["spec"]["services"]["Frontend"]["replicas"] == 2
    assert "NonExistentWorker" not in result["spec"]["services"]
    assert any(
        "NonExistentWorker" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), "Expected a WARNING mentioning 'NonExistentWorker'"


# ---------------------------------------------------------------------------
# Orchestration-level test: run_profile applies overrides before interpolation
# ---------------------------------------------------------------------------

_TOLERATION = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}

# Base DGD returned by the mocked strategy — no tolerations yet.
_BASE_DGD = {
    "spec": {
        "services": {
            "VllmDecodeWorker": {
                "extraPodSpec": {
                    "mainContainer": {"image": "my-image", "args": ["--model", "m"]},
                },
                "replicas": 1,
            },
        }
    }
}

# User-supplied DGD overrides: toleration for a real service + one ghost service.
_OVERRIDE_DGD = {
    "spec": {
        "services": {
            "VllmDecodeWorker": {"extraPodSpec": {"tolerations": [_TOLERATION]}},
            "GhostService": {"extraPodSpec": {"tolerations": [_TOLERATION]}},
        }
    }
}


async def test_run_profile_applies_dgd_overrides_before_interpolation(
    tmp_path, caplog
) -> None:
    """run_profile must apply DGD overrides to dgd_config before run_interpolation.

    Regression guard for TC-5.2a: without the fix, interpolation pods were
    deployed without extraPodSpec.tolerations, causing them to stay Pending on
    GPU nodes with nvidia.com/gpu:NoSchedule taints.
    """
    from dynamo.profiler.profile_sla import run_profile

    base_dgd = copy.deepcopy(_BASE_DGD)
    dgdr = DynamoGraphDeploymentRequestSpec(
        model="test/model",
        overrides=OverridesSpec(dgd=_OVERRIDE_DGD),
    )
    ops = ProfilerOperationalConfig(output_dir=str(tmp_path), dry_run=False)

    # Capture the disagg_config that run_interpolation receives.
    interpolation_kwargs: dict = {}

    async def _fake_interpolation(dgdr_arg, ops_arg, disagg_config, *args, **kwargs):
        interpolation_kwargs["disagg_config"] = copy.deepcopy(disagg_config)

    pick_result = {
        "dgd_config": base_dgd,
        "resolved_backend": "vllm",
        "chosen_exp": "disagg",
        "best_config_df": None,
        "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
    }

    with (
        patch("dynamo.profiler.profile_sla.valid_dgdr_spec"),
        patch("dynamo.profiler.profile_sla.validate_dgdr_dynamo_features"),
        patch(
            "dynamo.profiler.profile_sla.check_model_hardware_support",
            return_value=True,
        ),
        patch(
            "dynamo.profiler.profile_sla._extract_profiler_params",
            return_value=(
                "test/model",
                "vllm",
                "h100_sxm",
                8,
                4000,
                1000,
                None,
                2000.0,
                30.0,
                SearchStrategy.RAPID,
                "throughput",
            ),
        ),
        patch(
            "dynamo.profiler.profile_sla._execute_strategy",
            new=AsyncMock(
                return_value=(
                    pick_result,
                    PickedParallelConfig(),
                    PickedParallelConfig(),
                    2000.0,
                    30.0,
                )
            ),
        ),
        patch("dynamo.profiler.profile_sla.needs_profile_data", return_value=True),
        patch(
            "dynamo.profiler.profile_sla.run_interpolation",
            new=_fake_interpolation,
        ),
        patch(
            "dynamo.profiler.profile_sla.assemble_final_config",
            return_value=copy.deepcopy(base_dgd),
        ),
        patch("dynamo.profiler.profile_sla._write_final_output", return_value=True),
        patch("dynamo.profiler.profile_sla.write_profiler_status"),
        patch(
            "dynamo.profiler.profile_sla.cleanup_remaining_deployments",
            new=AsyncMock(),
        ),
    ):
        with caplog.at_level(
            logging.WARNING,
            logger="dynamo.profiler.utils.config_modifiers.protocol",
        ):
            await run_profile(dgdr, ops)

    assert interpolation_kwargs, "run_interpolation was never called"
    disagg_config = interpolation_kwargs["disagg_config"]

    # Tolerations must be present on VllmDecodeWorker before interpolation.
    eps = disagg_config["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    assert eps["tolerations"] == [_TOLERATION]

    # mainContainer must be preserved (not overwritten by the tolerations merge).
    assert eps["mainContainer"]["image"] == "my-image"

    # GhostService (absent from base DGD) must be silently skipped.
    assert "GhostService" not in disagg_config["spec"]["services"]

    # apply_dgd_overrides must emit a WARNING about the skipped service.
    assert any(
        "GhostService" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), "Expected a WARNING mentioning the skipped 'GhostService'"

    # apply_dgd_overrides must not mutate its input.
    assert (
        "tolerations"
        not in base_dgd["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]
    )


# ---------------------------------------------------------------------------
# Regression tests for #8568: pvc_name without pvcModelPath should NOT double
# the model path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_build_dgd_config_pvc_without_model_path_uses_hf_model_name(
    backend,
) -> None:
    """When pvc_name is set but model_path is None (no pvcModelPath), workers
    must receive the HF model ID — not the mount path — and the PVC must still
    be mounted on all services.

    Regression test for https://github.com/ai-dynamo/dynamo/issues/8568
    """
    modifier = CONFIG_MODIFIERS[backend]
    pvc_name = "model-cache"
    pvc_mount_path = "/opt/model-cache"
    model_name = "Qwen/Qwen3-32B"

    dgd_config = modifier.build_dgd_config(
        mode="agg",
        model_name=model_name,
        image=f"nvcr.io/nvidia/ai-dynamo/{backend}-runtime:1.1.0",
        agg_cli_args=["--tp", "4"],
        agg_replicas=1,
        agg_gpus=4,
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        # model_path is intentionally omitted (pvcModelPath not set)
    )

    services = dgd_config["spec"]["services"]

    # Workers must use HF model ID, NOT the mount path or a doubled path.
    for svc_name, svc in services.items():
        if svc_name in BaseConfigModifier._NON_WORKER_SERVICES:
            continue
        args = svc.get("extraPodSpec", {}).get("mainContainer", {}).get("args", [])
        flat_args = " ".join(args) if args else ""
        assert pvc_mount_path not in flat_args, (
            f"Worker '{svc_name}' model arg should be the HF model ID, "
            f"not the PVC mount path. args={args}"
        )

    # PVC must be declared in spec.pvcs
    pvcs = dgd_config["spec"].get("pvcs", [])
    pvc_names = [p["name"] for p in pvcs if isinstance(p, dict)]
    assert pvc_name in pvc_names, f"PVC '{pvc_name}' not found in spec.pvcs"

    # Every service must have a volumeMount for the PVC
    for svc_name, svc in services.items():
        vms = svc.get("volumeMounts", [])
        mount_names = [vm["name"] for vm in vms if isinstance(vm, dict)]
        assert (
            pvc_name in mount_names
        ), f"Service '{svc_name}' is missing volumeMount for PVC '{pvc_name}'"


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_build_dgd_config_pvc_with_model_path_uses_pvc_path(backend) -> None:
    """When both pvc_name and model_path are set (pvcModelPath provided),
    workers must receive the full PVC path — not the HF model ID.

    Ensures the explicit-pvcModelPath path still works after the fix.
    """
    modifier = CONFIG_MODIFIERS[backend]
    pvc_name = "model-cache"
    pvc_mount_path = "/opt/model-cache"
    model_name = "Qwen/Qwen3-32B"
    model_path = "/opt/model-cache/snapshots/abc123"

    dgd_config = modifier.build_dgd_config(
        mode="agg",
        model_name=model_name,
        image=f"nvcr.io/nvidia/ai-dynamo/{backend}-runtime:1.1.0",
        agg_cli_args=["--tp", "4"],
        agg_replicas=1,
        agg_gpus=4,
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        model_path=model_path,
    )

    services = dgd_config["spec"]["services"]

    # Workers must use the explicit PVC model path
    for svc_name, svc in services.items():
        if svc_name in BaseConfigModifier._NON_WORKER_SERVICES:
            continue
        args = svc.get("extraPodSpec", {}).get("mainContainer", {}).get("args", [])
        flat_args = " ".join(args) if args else ""
        assert model_path in flat_args, (
            f"Worker '{svc_name}' should use PVC model path '{model_path}'. "
            f"args={args}"
        )

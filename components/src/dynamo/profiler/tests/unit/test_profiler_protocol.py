# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler config_modifiers/protocol helpers."""

import copy
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
    from dynamo.profiler.utils.config_modifiers.protocol import BaseConfigModifier
    from dynamo.profiler.utils.defaults import EngineType, SearchStrategy
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


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_build_dgd_config_preserves_type_meta(backend: str, mode: str) -> None:
    dgd_config = CONFIG_MODIFIERS[backend].build_dgd_config(
        mode=mode,
        model_name="test/model",
        image=f"example/{backend}:test",
    )

    assert dgd_config["apiVersion"] in {
        "nvidia.com/v1alpha1",
        "nvidia.com/v1beta1",
    }
    assert dgd_config["kind"] == "DynamoGraphDeployment"


def test_build_dgd_config_vllm_disagg_restores_runtime_args() -> None:
    """AIC tuning args must not remove Dynamo's vLLM disaggregation contract."""
    modifier = CONFIG_MODIFIERS["vllm"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="test/model",
        image="example/vllm:test",
        prefill_cli_args=["--tensor-parallel-size", "2"],
        decode_cli_args=["--tensor-parallel-size", "4"],
    )

    services = dgd_config["spec"]["services"]
    prefill = next(
        service
        for service in services.values()
        if service.get("subComponentType") == "prefill"
    )
    decode = next(
        service
        for service in services.values()
        if service.get("subComponentType") == "decode"
    )
    prefill_args = prefill["extraPodSpec"]["mainContainer"]["args"]
    decode_args = decode["extraPodSpec"]["mainContainer"]["args"]

    assert prefill_args[prefill_args.index("--tensor-parallel-size") + 1] == "2"
    assert prefill_args[prefill_args.index("--disaggregation-mode") + 1] == "prefill"
    assert (
        prefill_args[prefill_args.index("--kv-transfer-config") + 1]
        == '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    )
    assert decode_args[decode_args.index("--tensor-parallel-size") + 1] == "4"
    assert decode_args[decode_args.index("--disaggregation-mode") + 1] == "decode"
    assert "--kv-transfer-config" not in decode_args


def test_build_dgd_config_vllm_disagg_preserves_explicit_kv_config() -> None:
    """An explicit connector remains authoritative while worker roles are canonical."""
    custom_kv_config = (
        '{"kv_connector":"NixlConnector","kv_role":"kv_both",'
        '"kv_buffer_device":"cpu"}'
    )
    modifier = CONFIG_MODIFIERS["vllm"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="test/model",
        image="example/vllm:test",
        prefill_cli_args=[
            "--disaggregation-mode=decode",
            f"--kv-transfer-config '{custom_kv_config}'",
        ],
        decode_cli_args=["--disaggregation-mode", "prefill"],
    )

    services = dgd_config["spec"]["services"]
    prefill_args = next(
        service["extraPodSpec"]["mainContainer"]["args"]
        for service in services.values()
        if service.get("subComponentType") == "prefill"
    )
    decode_args = next(
        service["extraPodSpec"]["mainContainer"]["args"]
        for service in services.values()
        if service.get("subComponentType") == "decode"
    )

    assert prefill_args.count("--disaggregation-mode") == 1
    assert prefill_args[prefill_args.index("--disaggregation-mode") + 1] == "prefill"
    assert prefill_args.count("--kv-transfer-config") == 1
    assert (
        prefill_args[prefill_args.index("--kv-transfer-config") + 1] == custom_kv_config
    )
    assert decode_args.count("--disaggregation-mode") == 1
    assert decode_args[decode_args.index("--disaggregation-mode") + 1] == "decode"


def test_build_dgd_config_vllm_disagg_removes_legacy_role_flags() -> None:
    """AIC legacy role flags must not reach the vLLM backend CLI."""
    modifier = CONFIG_MODIFIERS["vllm"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="test/model",
        image="example/vllm:test",
        prefill_cli_args=["--is-prefill-worker", "--is-decode-worker"],
        decode_cli_args=["--is-prefill-worker", "--is-decode-worker"],
    )

    services = dgd_config["spec"]["services"]
    prefill_args = next(
        service["extraPodSpec"]["mainContainer"]["args"]
        for service in services.values()
        if service.get("subComponentType") == "prefill"
    )
    decode_args = next(
        service["extraPodSpec"]["mainContainer"]["args"]
        for service in services.values()
        if service.get("subComponentType") == "decode"
    )

    for args, expected_mode in ((prefill_args, "prefill"), (decode_args, "decode")):
        assert "--is-prefill-worker" not in args
        assert "--is-decode-worker" not in args
        assert args.count("--disaggregation-mode") == 1
        assert args[args.index("--disaggregation-mode") + 1] == expected_mode


def test_build_dgd_config_shapes_multinode_worker_resources() -> None:
    """DP-only workers keep per-node GPU shaping without multinode inflation."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-30B-A3B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1",
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


def test_build_dgd_config_sglang_prefill_mrr_one_sets_dp_safe_cuda_graph_bs() -> None:
    """SGLang prefill capture bs must remain valid with DP attention."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-30B-A3B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-post.1",
        prefill_cli_args=[
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "2",
            "--max-running-requests",
            "1",
            "--max-prefill-tokens",
            "5500",
            "--enable-dp-attention",
        ],
        prefill_replicas=2,
        prefill_gpus=4,
        decode_cli_args=[
            "--max-running-requests",
            "512",
            "--cuda-graph-bs",
            "1",
        ],
        decode_replicas=2,
        decode_gpus=8,
        num_gpus_per_node=8,
    )

    prefill_service = next(
        service
        for service in dgd_config["spec"]["services"].values()
        if service.get("subComponentType") == "prefill"
    )
    prefill_args = prefill_service["extraPodSpec"]["mainContainer"]["args"]

    assert prefill_args.count("--cuda-graph-bs") == 1
    assert prefill_args[prefill_args.index("--cuda-graph-bs") + 1] == "2"


def test_build_dgd_config_sglang_prefill_keeps_existing_cuda_graph_bs() -> None:
    """Do not duplicate an explicit CUDA graph batch-size setting."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-30B-A3B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-post.1",
        prefill_cli_args=[
            "--max-running-requests",
            "1",
            "--cuda-graph-bs=1",
        ],
        prefill_replicas=2,
        prefill_gpus=4,
        decode_cli_args=["--max-running-requests", "512"],
        decode_replicas=2,
        decode_gpus=8,
        num_gpus_per_node=8,
    )

    prefill_service = next(
        service
        for service in dgd_config["spec"]["services"].values()
        if service.get("subComponentType") == "prefill"
    )
    prefill_args = prefill_service["extraPodSpec"]["mainContainer"]["args"]

    cuda_graph_bs_args = [
        arg
        for arg in prefill_args
        if arg == "--cuda-graph-bs" or arg.startswith("--cuda-graph-bs=")
    ]
    assert cuda_graph_bs_args == ["--cuda-graph-bs=1"]


def test_sglang_set_prefill_config_uses_effective_mrr_override() -> None:
    """Later MRR overrides must drive CUDA graph batch-size safety."""
    modifier = CONFIG_MODIFIERS["sglang"]
    config = modifier.convert_config(
        modifier.load_default_config(mode="disagg"),
        target=EngineType.PREFILL,
    )
    service = next(
        service
        for service in config["spec"]["services"].values()
        if service.get("subComponentType") == "decode"
    )
    service["extraPodSpec"]["mainContainer"]["args"] = [
        "--max-running-requests=512",
        "--dp=2",
    ]

    result = modifier.set_prefill_config(
        config,
        max_batch_size=1,
        max_num_tokens=5500,
    )
    worker = next(
        service
        for service in result["spec"]["services"].values()
        if service.get("subComponentType") == "decode"
    )
    args = worker["extraPodSpec"]["mainContainer"]["args"]

    assert args[args.index("--max-running-requests") + 1] == "1"
    assert args.count("--cuda-graph-bs") == 1
    assert args[args.index("--cuda-graph-bs") + 1] == "2"


def test_vllm_mamba_align_raises_max_num_batched_tokens() -> None:
    """vLLM Mamba align requires the scheduler token cap to cover block size."""
    modifier = CONFIG_MODIFIERS["vllm"]
    args = [
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--max-num-batched-tokens",
        "1024",
    ]

    with patch(
        "dynamo.profiler.utils.config_modifiers.vllm.get_mamba_cache_align_block_size",
        return_value=8320,
    ):
        result = modifier._apply_mamba_cache_align_token_floor(args, "nemotron")

    assert result[result.index("--max-num-batched-tokens") + 1] == "8320"


def test_vllm_mamba_align_skips_without_explicit_align_mode() -> None:
    """Do not probe model metadata for ordinary prefix-caching decode workers."""
    modifier = CONFIG_MODIFIERS["vllm"]
    args = [
        "--enable-prefix-caching",
        "--max-num-batched-tokens",
        "1024",
    ]

    with patch(
        "dynamo.profiler.utils.config_modifiers.vllm.get_mamba_cache_align_block_size"
    ) as mock_floor:
        result = modifier._apply_mamba_cache_align_token_floor(args, "llama")

    mock_floor.assert_not_called()
    assert result == args


def test_vllm_model_runtime_constraints_update_decode_config() -> None:
    """Candidate-level vLLM postprocessing fixes generated decode worker args."""
    modifier = CONFIG_MODIFIERS["vllm"]
    config = {
        "metadata": {"name": "test"},
        "spec": {
            "services": {
                "Frontend": {},
                "VllmDecodeWorker": {
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": [
                                "--mamba-cache-mode",
                                "align",
                                "--max-num-batched-tokens",
                                "1024",
                            ]
                        }
                    },
                },
            }
        },
    }

    with patch(
        "dynamo.profiler.utils.config_modifiers.vllm.get_mamba_cache_align_block_size",
        return_value=8320,
    ):
        result = modifier.apply_model_runtime_constraints(config, "nemotron")

    args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert args[args.index("--max-num-batched-tokens") + 1] == "8320"


def test_vllm_model_runtime_constraints_skip_partial_decode_config() -> None:
    """Final DGD postprocessing should tolerate partial mocked configs."""
    modifier = CONFIG_MODIFIERS["vllm"]
    config = {
        "metadata": {"name": "test"},
        "spec": {
            "services": {
                "Frontend": {},
                "VllmDecodeWorker": {"subComponentType": "decode"},
            }
        },
    }

    result = modifier.apply_model_runtime_constraints(config, "nemotron")

    decode_service = result["spec"]["services"]["VllmDecodeWorker"]
    assert decode_service["subComponentType"] == "decode"
    assert decode_service["extraPodSpec"] is None


def test_build_dgd_config_multinode_when_tp_exceeds_node() -> None:
    """Single instances that exceed node capacity still get multinode config."""
    modifier = CONFIG_MODIFIERS["sglang"]
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="meta-llama/Llama-3-70B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1",
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
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1",
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


# ---------------------------------------------------------------------------
# Orchestration-level test: each generated DGD receives the override once
# ---------------------------------------------------------------------------

_TOLERATION = {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}

# Base DGD returned by the mocked strategy — no tolerations yet.
_BASE_DGD = {
    "spec": {
        "services": {
            "decode": {
                "componentType": "worker",
                "subComponentType": "decode",
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
            "decode": {"extraPodSpec": {"tolerations": [_TOLERATION]}},
            "GhostService": {"extraPodSpec": {"tolerations": [_TOLERATION]}},
        }
    }
}


async def test_run_profile_applies_override_once_to_each_consumed_dgd(tmp_path) -> None:
    """Interpolation and final output each receive one independently merged DGD."""
    from dynamo.profiler.profile_sla import run_profile

    base_dgd = copy.deepcopy(_BASE_DGD)
    override_inputs: list[dict] = []

    def _fake_apply_dgd_overrides(dgd_config, overrides):
        override_inputs.append(copy.deepcopy(dgd_config))
        result = copy.deepcopy(dgd_config)
        services = result["spec"]["services"]
        for name, service_override in overrides["spec"]["services"].items():
            if name not in services:
                continue
            services[name].setdefault("extraPodSpec", {}).update(
                service_override["extraPodSpec"]
            )
        services["decode"]["extraPodSpec"]["mainContainer"]["args"].append(
            "--override-applied"
        )
        return result

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
        "resolved_backend": "trtllm",
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
                "trtllm",
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
            "dynamo.profiler.utils.dgd_materialization.apply_dgd_overrides",
            side_effect=_fake_apply_dgd_overrides,
        ),
        patch(
            "dynamo.profiler.profile_sla.assemble_final_config",
            return_value=copy.deepcopy(base_dgd),
        ) as assemble_final,
        patch(
            "dynamo.profiler.profile_sla._write_final_output", return_value=True
        ) as write_final,
        patch("dynamo.profiler.profile_sla.write_profiler_status"),
        patch(
            "dynamo.profiler.profile_sla.cleanup_remaining_deployments",
            new=AsyncMock(),
        ),
    ):
        await run_profile(dgdr, ops)

    assert interpolation_kwargs, "run_interpolation was never called"
    disagg_config = interpolation_kwargs["disagg_config"]

    # Tolerations and TRT-LLM runtime defaults must be present before interpolation.
    eps = disagg_config["spec"]["services"]["decode"]["extraPodSpec"]
    assert eps["tolerations"] == [_TOLERATION]

    # mainContainer must be preserved (not overwritten by the tolerations merge).
    assert eps["mainContainer"]["image"] == "my-image"
    assert eps["mainContainer"]["args"].count("--override-applied") == 1
    chunked_prefill_idx = eps["mainContainer"]["args"].index(
        "--trtllm.enable_chunked_prefill"
    )
    assert eps["mainContainer"]["args"][chunked_prefill_idx + 1] == "true"

    # GhostService (absent from base DGD) must be silently skipped.
    assert "GhostService" not in disagg_config["spec"]["services"]

    # The final assembly receives the clean picked DGD, not the interpolation copy.
    assert assemble_final.call_args.args[2] == base_dgd
    assert len(override_inputs) == 2
    for override_input in override_inputs:
        args = override_input["spec"]["services"]["decode"]["extraPodSpec"][
            "mainContainer"
        ]["args"]
        assert "--override-applied" not in args

    final_config = write_final.call_args.args[1]
    final_args = final_config["spec"]["services"]["decode"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert final_args.count("--override-applied") == 1

    # Neither merge mutates the clean picked DGD.
    assert "tolerations" not in base_dgd["spec"]["services"]["decode"]["extraPodSpec"]


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
        image=f"nvcr.io/nvidia/ai-dynamo/{backend}-runtime:1.1.1",
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
        image=f"nvcr.io/nvidia/ai-dynamo/{backend}-runtime:1.1.1",
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
        assert (
            model_path in flat_args
        ), f"Worker '{svc_name}' should use PVC model path '{model_path}'. args={args}"
        assert args[args.index("--served-model-name") + 1] == model_name


@pytest.mark.parametrize(
    "backend,model_arg",
    [("vllm", "--model"), ("sglang", "--model-path"), ("trtllm", "--model-path")],
)
def test_update_model_from_pvc_canonicalizes_duplicate_model_args(
    backend, model_arg
) -> None:
    """PVC model updates leave exactly one logical name and runtime path."""
    modifier = CONFIG_MODIFIERS[backend]
    model_name = "Qwen/Qwen3-32B"
    mount_path = "/opt/model-cache"
    model_path = f"{mount_path}/qwen3-32b"
    dgd_config = modifier.build_dgd_config(
        mode="agg",
        model_name="stale/model",
        image=f"example/{backend}:test",
        agg_cli_args=[],
        agg_replicas=1,
        agg_gpus=1,
    )

    services = dgd_config["spec"]["services"]
    worker = next(
        service
        for name, service in services.items()
        if name not in BaseConfigModifier._NON_WORKER_SERVICES
    )
    worker_args = worker["extraPodSpec"]["mainContainer"]["args"]
    worker_args.extend(
        [
            f"{model_arg}=/stale/equal-form",
            model_arg,
            "/stale/split-form",
            "--served-model-name=stale-equal",
            "--served-model-name",
            "stale-split",
        ]
    )

    frontend_container = services["Frontend"]["extraPodSpec"]["mainContainer"]
    frontend_container["args"] = frontend_container.get("args") or []
    frontend_args = frontend_container["args"]
    frontend_args.extend(
        [
            "--model-name=stale-equal",
            "--model-name",
            "stale-split",
            "--model-path=/stale/equal-form",
            "--model-path",
            "/stale/split-form",
        ]
    )

    result = modifier.update_model_from_pvc(
        dgd_config,
        model_name=model_name,
        pvc_name="model-cache",
        pvc_mount_path=mount_path,
        pvc_path="qwen3-32b",
    )

    result_services = result["spec"]["services"]
    result_worker = next(
        service
        for name, service in result_services.items()
        if name not in BaseConfigModifier._NON_WORKER_SERVICES
    )
    result_worker_args = result_worker["extraPodSpec"]["mainContainer"]["args"]
    assert [
        arg
        for arg in result_worker_args
        if arg == model_arg or arg.startswith(f"{model_arg}=")
    ] == [model_arg]
    assert result_worker_args[result_worker_args.index(model_arg) + 1] == model_path
    assert [
        arg
        for arg in result_worker_args
        if arg == "--served-model-name" or arg.startswith("--served-model-name=")
    ] == ["--served-model-name"]
    assert (
        result_worker_args[result_worker_args.index("--served-model-name") + 1]
        == model_name
    )

    result_frontend_args = result_services["Frontend"]["extraPodSpec"]["mainContainer"][
        "args"
    ]
    assert [
        arg
        for arg in result_frontend_args
        if arg == "--model-name" or arg.startswith("--model-name=")
    ] == ["--model-name"]
    assert [
        arg
        for arg in result_frontend_args
        if arg == "--model-path" or arg.startswith("--model-path=")
    ] == ["--model-path"]
    assert (
        result_frontend_args[result_frontend_args.index("--model-name") + 1]
        == model_name
    )
    assert (
        result_frontend_args[result_frontend_args.index("--model-path") + 1]
        == model_path
    )


def test_build_dgd_config_pvc_without_model_path_sets_hf_home() -> None:
    """When pvc_name is set but model_path doesn't point inside the PVC,
    HF_HOME must be set to pvc_mount_path so HuggingFace finds cached weights."""
    modifier = CONFIG_MODIFIERS["sglang"]
    mount = "/opt/model-cache"
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-32B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0",
        prefill_cli_args=["--max-running-requests", "1"],
        prefill_replicas=1,
        prefill_gpus=1,
        decode_cli_args=["--tp", "4"],
        decode_replicas=1,
        decode_gpus=4,
        pvc_name="model-cache",
        pvc_mount_path=mount,
    )

    for svc_name, svc in dgd_config["spec"]["services"].items():
        eps = svc.get("extraPodSpec", {})
        mc = eps.get("mainContainer", {})
        env_list = mc.get("env", [])
        hf_homes = [
            e for e in env_list if isinstance(e, dict) and e.get("name") == "HF_HOME"
        ]
        assert (
            len(hf_homes) == 1
        ), f"Expected exactly one HF_HOME env on {svc_name}, got {len(hf_homes)}"
        assert hf_homes[0]["value"] == mount, f"HF_HOME on {svc_name} should be {mount}"


def test_build_dgd_config_pvc_with_model_path_no_hf_home() -> None:
    """When pvc_name is set and model_path points inside the PVC,
    HF_HOME should NOT be injected — model is loaded by explicit path."""
    modifier = CONFIG_MODIFIERS["sglang"]
    mount = "/opt/model-cache"
    dgd_config = modifier.build_dgd_config(
        mode="disagg",
        model_name="Qwen/Qwen3-32B",
        image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0",
        prefill_cli_args=["--max-running-requests", "1"],
        prefill_replicas=1,
        prefill_gpus=1,
        decode_cli_args=["--tp", "4"],
        decode_replicas=1,
        decode_gpus=4,
        pvc_name="model-cache",
        pvc_mount_path=mount,
        model_path=f"{mount}/qwen3-32b",
    )

    for svc_name, svc in dgd_config["spec"]["services"].items():
        eps = svc.get("extraPodSpec", {})
        mc = eps.get("mainContainer", {})
        env_list = mc.get("env", [])
        hf_homes = [
            e for e in env_list if isinstance(e, dict) and e.get("name") == "HF_HOME"
        ]
        assert (
            len(hf_homes) == 0
        ), f"HF_HOME should not be set on {svc_name} when model_path is a PVC subpath"


# -----------------------------------------------------------------------------
# auto_inject_trust_remote_code / model_has_auto_map
# -----------------------------------------------------------------------------


def _make_dgd_with_workers(*worker_names: str) -> dict:
    """Build a minimal DGD dict with the given worker services + a Frontend."""
    services: dict = {
        "Frontend": {
            "extraPodSpec": {
                "mainContainer": {"args": ["--http-port", "8000"]},
            },
        },
    }
    for name in worker_names:
        services[name] = {
            "extraPodSpec": {
                "mainContainer": {"args": ["--model", "some/model", "--tp", "1"]},
            },
        }
    return {"spec": {"services": services}}


def test_model_has_auto_map_local_dir_with_auto_map(tmp_path) -> None:
    import json as _json

    from dynamo.profiler.utils.model_info import model_has_auto_map

    cfg = {
        "model_type": "nemotron_h",
        "auto_map": {
            "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
            "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
        },
    }
    (tmp_path / "config.json").write_text(_json.dumps(cfg))
    assert model_has_auto_map(tmp_path) is True


def test_model_has_auto_map_local_dir_without_auto_map(tmp_path) -> None:
    import json as _json

    from dynamo.profiler.utils.model_info import model_has_auto_map

    (tmp_path / "config.json").write_text(_json.dumps({"model_type": "llama"}))
    assert model_has_auto_map(tmp_path) is False


def test_model_has_auto_map_local_dir_missing_config_returns_false(tmp_path) -> None:
    from dynamo.profiler.utils.model_info import model_has_auto_map

    assert model_has_auto_map(tmp_path) is False


def test_materialize_dgd_injects_trust_remote_code_for_vllm() -> None:
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    decode_args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert decode_args[-1] == "--trust-remote-code"
    # Original args preserved.
    assert decode_args[:-1] == ["--model", "some/model", "--tp", "1"]
    # Frontend untouched.
    assert "--trust-remote-code" not in (
        result["spec"]["services"]["Frontend"]["extraPodSpec"]["mainContainer"]["args"]
    )


def test_materialize_dgd_injects_trust_remote_code_for_sglang() -> None:
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("SglangDecodeWorker", "SglangPrefillWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="sglang",
            model_name_or_path="some/model",
        )

    for svc in ("SglangDecodeWorker", "SglangPrefillWorker"):
        args = result["spec"]["services"][svc]["extraPodSpec"]["mainContainer"]["args"]
        assert args.count("--trust-remote-code") == 1


def test_materialize_dgd_skips_trust_when_no_auto_map() -> None:
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=False,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert "--trust-remote-code" not in args


def test_materialize_dgd_fails_closed_for_mutable_remote_ref() -> None:
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=False,
    ):
        with pytest.raises(
            RuntimeError, match="Refusing to auto-inject --trust-remote-code"
        ):
            materialize_dgd(
                cfg,
                purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
                runtime_backend="vllm",
                model_name_or_path="some/model",
            )


def test_materialize_dgd_skips_trust_for_trtllm() -> None:
    """TRT-LLM uses a YAML field, not the CLI flag."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("TRTLLMDecodeWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="trtllm",
            model_name_or_path="some/model",
        )

    args = result["spec"]["services"]["TRTLLMDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert "--trust-remote-code" not in args


def test_materialize_dgd_remote_ref_with_explicit_override_skips_error() -> None:
    """When the user already set --trust-remote-code via overrides, the
    mutable-remote-ref error must not fire — the manual escape hatch works."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    # Simulate user override having already appended the flag.
    cfg["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"][
        "args"
    ].append("--trust-remote-code")

    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=False,
    ):
        # Should NOT raise RuntimeError because the flag is already present.
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/remote-model",
        )

    args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert args.count("--trust-remote-code") == 1


def test_materialize_dgd_trust_injection_is_idempotent() -> None:
    """Running materialize_dgd twice must not duplicate the flag."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )
        result2 = materialize_dgd(
            result,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    args = result2["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert args.count("--trust-remote-code") == 1


def test_materialize_dgd_respects_existing_trust_flag() -> None:
    """An explicit --trust-remote-code already in args must not be duplicated."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    cfg["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"][
        "args"
    ].append("--trust-remote-code")

    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert args.count("--trust-remote-code") == 1


def test_materialize_dgd_excludes_frontend_and_planner() -> None:
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = _make_dgd_with_workers("VllmDecodeWorker")
    cfg["spec"]["services"]["Planner"] = {
        "extraPodSpec": {"mainContainer": {"args": ["--interval", "30"]}},
    }
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    assert "--trust-remote-code" not in (
        result["spec"]["services"]["Frontend"]["extraPodSpec"]["mainContainer"]["args"]
    )
    assert "--trust-remote-code" not in (
        result["spec"]["services"]["Planner"]["extraPodSpec"]["mainContainer"]["args"]
    )


def test_materialize_dgd_shell_form_worker() -> None:
    """Shell-form workers (command=['sh','-c'], args=['<single string>']) must
    have the flag appended inside the string, not as a second list element."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    cfg = {
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "extraPodSpec": {
                        "mainContainer": {
                            "command": ["sh", "-c"],
                            "args": [
                                "python3 -m vllm.entrypoints.openai.api_server "
                                "--model some/model --tp 1"
                            ],
                        },
                    },
                },
            }
        }
    }
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    result_args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    # Must still be a single-element list (shell form preserved).
    assert isinstance(result_args, list) and len(result_args) == 1
    assert result_args[0].endswith("--trust-remote-code")

    # Idempotency: materializing again must not duplicate the flag.
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result2 = materialize_dgd(
            result,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    result2_args = result2["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert len(result2_args) == 1
    assert result2_args[0].count("--trust-remote-code") == 1


def test_materialize_dgd_shell_form_preserves_syntax() -> None:
    """Shell-form args with shell operators (&&, |, etc.) must not be
    corrupted by shlex round-tripping."""
    from dynamo.profiler.utils.dgd_materialization import (
        DGDMaterializationPurpose,
        materialize_dgd,
    )

    original_cmd = (
        "export FOO=bar && python3 -m vllm.entrypoints.openai.api_server "
        "--model some/model --tp 1"
    )
    cfg = {
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "extraPodSpec": {
                        "mainContainer": {
                            "command": ["sh", "-c"],
                            "args": [original_cmd],
                        },
                    },
                },
            }
        }
    }
    with patch(
        "dynamo.profiler.utils.dgd_materialization.model_has_auto_map",
        return_value=True,
    ), patch(
        "dynamo.profiler.utils.dgd_materialization.model_ref_allows_implicit_trust_remote_code",
        return_value=True,
    ):
        result = materialize_dgd(
            cfg,
            purpose=DGDMaterializationPurpose.FINAL_OUTPUT,
            runtime_backend="vllm",
            model_name_or_path="some/model",
        )

    result_args = result["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert len(result_args) == 1
    # The original shell syntax (&&, export) must be preserved verbatim.
    assert result_args[0] == original_cmd + " --trust-remote-code"


def test_model_has_auto_map_returns_true_on_unexpected_error() -> None:
    """Unexpected errors (network, auth) must return True (conservative default)
    rather than silently returning False and risking a missed injection."""
    from dynamo.profiler.utils.model_info import model_has_auto_map

    with patch(
        "dynamo.profiler.utils.model_info.hf_hub_download",
        side_effect=OSError("simulated network failure"),
    ):
        result = model_has_auto_map("some/hf-model")

    assert result is True


def test_model_has_auto_map_returns_false_for_repo_not_found() -> None:
    """RepositoryNotFoundError means the model doesn't exist — no custom code.
    The detection uses type(e).__name__ so no huggingface_hub import is needed."""
    from dynamo.profiler.utils.model_info import model_has_auto_map

    class RepositoryNotFoundError(Exception):
        pass

    with patch(
        "dynamo.profiler.utils.model_info.hf_hub_download",
        side_effect=RepositoryNotFoundError("404"),
    ):
        result = model_has_auto_map("nonexistent/model")

    assert result is False


def test_model_has_auto_map_returns_false_for_malformed_json(tmp_path) -> None:
    """Malformed config.json must return False (can't parse, assume no auto_map)."""
    from dynamo.profiler.utils.model_info import model_has_auto_map

    (tmp_path / "config.json").write_text("{this is not valid json}")
    result = model_has_auto_map(tmp_path)
    assert result is False

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test: v1.1.0 image on A100-80GB (dynamo-aks-dev) before committing to
# the full N=3 head-to-head. Confirms image pulls, model loads, basic
# completions work, and vLLM metrics flow.
#
# N=1 topology (1 FE + 2 PF TP=2 + 1 DE TP=2 = 6 GPUs). A100 doesn't have
# native FP8 — the FP8 weights get dequantized to BF16 for compute (matches
# the customer's A100-40GB production code path). gpu-memory-utilization is
# capped at 0.5 to give ~40 GB usable on A100-80GB, matching prod envelope.
#
# Brief 3-minute c=24 load, then teardown. Pass criteria: load completes,
# decode KV reaches at least 0.20 (some actual pressure), zero panics.

import pytest

from tests.fault_tolerance.deploy.checks import (
    KvCacheUsagePeak,
    LoadApplied,
    LoadCompleted,
    RestartCountIncreased,
    WorkerPanics,
)
from tests.fault_tolerance.deploy.events import (
    StartLoad,
    WaitForLoadCompletion,
    WaitForModelReady,
)
from tests.fault_tolerance.deploy.reports import (
    ErrorBreakdownReport,
    FaultToleranceReport,
    PerWorkerLatencyReport,
)
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


_TEMPLATE = (
    "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
    "disagg_qwen3_30b_unit_prod.yaml"
)

_V110_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0"


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_aks_smoke_v110(runtime_env, request):
    spec = DeploymentSpec(_TEMPLATE)

    # Pin to v1.1.0; cap GPU memory util to match prod A100-40GB envelope on
    # A100-80GB host. 0.45 * 40 GB ≈ 18 GB per GPU in prod; 0.225 * 80 GB ≈
    # 18 GB on dev. Round up a bit since AKS A100 is slightly different SKU.
    for svc in ("Frontend", "VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].image = _V110_IMAGE
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        spec[svc].set_arg("--gpu-memory-utilization", "0.5")

    # aks-dev: pin Frontend to the A100 node pool (1.87 TB ephemeral).
    # The `aks-default-*` CPU nodes have only 120 GB and kubelet evicts
    # on disk pressure from other tenants' image layers. FE requests no
    # GPU, just borrows the A100 nodes' bigger root disk. The
    # nvidia.com/gpu:NoSchedule toleration is needed to land there.
    fe_pod = spec["Frontend"]._spec.setdefault("extraPodSpec", {})
    fe_pod["nodeSelector"] = {
        "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"
    }
    fe_pod["tolerations"] = [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
    ]

    # aks-dev: UCX/NIXL needs unlimited locked memory (default RLIMIT_MEMLOCK
    # is 64 KB, IB completion queues need MBs). The cluster's default
    # AppArmor profile (cri-containerd.apparmor.d) overrides
    # `privileged: true` at the kernel level. Per Hyunjae Woo's working
    # recipe on dynamo-aks-exp (OPS-4332):
    #   1. privileged: true (request all caps)
    #   2. capabilities.add: [IPC_LOCK] (belt-and-suspenders)
    #   3. runAsUser: 0  ← needed to actually raise ulimit -l
    #   4. appArmorProfile: Unconfined (bypass default profile)
    #   5. Call `ulimit -l unlimited` in the wrapper before exec'ing the
    #      worker — see dyn_tee.sh DYN_TEST_NOFILE_LIMIT analog.
    for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
        main = spec[svc]._spec.setdefault(
            "extraPodSpec", {}
        ).setdefault("mainContainer", {})
        secctx = main.setdefault("securityContext", {})
        secctx["privileged"] = True
        secctx["runAsUser"] = 0
        secctx["appArmorProfile"] = {"type": "Unconfined"}
        caps = secctx.setdefault("capabilities", {})
        caps.setdefault("add", []).append("IPC_LOCK")
        # Tell dyn_tee.sh to raise ulimit -l before exec'ing.
        spec[svc].set_env_var("DYN_TEST_MEMLOCK_UNLIMITED", "1")

    served_model = spec["VllmDecodeWorker"].model

    cfg = LoadConfig(
        model_name=served_model,
        tokenizer=served_model,
        input_tokens_mean=1600,
        input_tokens_stddev=400,
        output_tokens_mean=200,
        output_tokens_stddev=50,
        concurrency=24,
        duration_minutes=3.0,
        request_timeout_seconds=60,
        streaming=True,
        ignore_eos=True,
        warmup_requests=0,
        connection_reuse_strategy="never",
    )

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=2400),
            StartLoad(load_config=cfg, name="smoke"),
            WaitForLoadCompletion(name="smoke"),
        ],
        checks=[
            LoadApplied(load_name="smoke", min_requests=50),
            KvCacheUsagePeak(
                services=["VllmDecodeWorker"],
                threshold=0.10,
                within_seconds=600,
            ),
            RestartCountIncreased(
                services=["VllmDecodeWorker", "Frontend"],
                expect_min_increment=0,
            ),
            WorkerPanics(
                services=["VllmDecodeWorker", "VllmPrefillWorker", "Frontend"],
                acceptable=False,
            ),
            LoadCompleted(name="smoke"),
        ],
        reports=[
            FaultToleranceReport(),
            ErrorBreakdownReport(),
            PerWorkerLatencyReport(),
        ],
        test_name=request.node.name,
        runtime_env=runtime_env,
    )

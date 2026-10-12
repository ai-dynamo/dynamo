# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
EFA verification deploy test for SGLang.

The SGLang counterpart of test_deploy_efa.py: a disaggregated SGLang stack
deploys on an EFA-capable cluster, serves a chat completion, and the
prefill->decode KV-cache transfer rides NIXL -> LIBFABRIC -> EFA. The flow and
the assertions are shared with the vLLM lane via tests/deploy/efa_utils.py; only
the profile below is SGLang-specific.

Run it explicitly, e.g.:

    pytest tests/deploy/test_deploy_efa_sglang.py -m framework_with_efa \
        --image=<efa-sglang-runtime-image> --namespace=<ns> -v -s

The deployment name is fixed, so the namespace must be clear of a previous run
before starting a new one. CI is unaffected: every nightly gets a fresh
vCluster.
"""

import pytest

from tests.deploy.efa_utils import EfaFrameworkProfile, run_efa_deployment_check

# SGLang pushes: the prefill worker writes its KV into decode's registered
# blocks over NIXL (WRITE), the mirror image of vLLM's pull. So the counter that
# grows is tx on the *prefill* agent, and at the adapter it is writes issued by
# prefill and write-receives on decode. Asserting vLLM's read counters here
# would read permanently-zero series and fail a healthy deployment: decode's
# agent_rx_bytes stays at 0 for the whole run, as measured on aws-dev-02.
#
# Volume, for the EFA_MIN_TRANSFER_BYTES floor: Qwen3-0.6B holds 114,688 bytes of
# KV per token, and --page-size 16 makes one page (1,835,008 bytes) the smallest
# transfer possible, so even a one-token prompt clears the 1 MiB floor. The
# deploy test's own prompt moved 16,516,672 bytes.
SGLANG_EFA_PROFILE = EfaFrameworkProfile(
    name="SGLang",
    manifest_name="disagg-efa-sglang.yaml",
    prefill_service="prefill",
    decode_service="decode",
    backend_pin_hint=(
        "SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC is still set on both "
        "workers and --disaggregation-transfer-backend is still nixl."
    ),
    nixl_counter_service="prefill",
    nixl_counter_metric="agent_tx_bytes",
    efa_counter_by_role={
        "prefill": "node_amazonefa_rdma_write_bytes",
        "decode": "node_amazonefa_rdma_write_recv_bytes",
    },
)


@pytest.mark.framework_with_efa
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.nightly
@pytest.mark.e2e
# Two GPUs total: one prefill, one decode. No framework marker -- the nightly job
# selects with -m framework_with_efa, and carrying @pytest.mark.sglang would both
# require an exemption in tests/conftest.py's framework auto-skip and make this
# test collectable by the multi-GPU jobs, whose selectors are
# "sglang and gpu_2 and not h100" and "sglang and (gpu_1 or gpu_2)" with no
# lifecycle filter.
@pytest.mark.gpu_2
@pytest.mark.core
@pytest.mark.timeout(1200)
async def test_efa_deployment_sglang(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
) -> None:
    """Deploy a disaggregated SGLang stack with EFA enabled and verify it serves."""
    await run_efa_deployment_check(
        profile=SGLANG_EFA_PROFILE,
        image=image,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        request=request,
    )

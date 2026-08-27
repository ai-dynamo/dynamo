# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
EFA verification deploy test for vLLM.

Verifies that an EFA-tagged image built from the commit under test can run
Dynamo with Elastic Fabric Adapter (EFA) fully enabled: a disaggregated vLLM
stack deploys on an EFA-capable cluster, serves a chat completion, and the
prefill->decode KV-cache transfer rides NIXL -> LIBFABRIC -> EFA.

The deployment flow and every assertion live in tests/deploy/efa_utils.py and
are shared with the SGLang lane (test_deploy_efa_sglang.py); this file carries
only what is specific to vLLM.

This test is NOT part of the auto-discovered deploy-test matrix. It uses an
explicit manifest (tests/deploy/efa/disagg-efa.yaml) and the
``framework_with_efa`` marker, and only makes sense on a cluster with p5/EFA
nodes (the standard CI vCluster lacks RDMA/EFA, which is why the matrix test
skips vLLM disagg). Run it explicitly, e.g.:

    pytest tests/deploy/test_deploy_efa.py -m framework_with_efa \
        --image=<efa-vllm-runtime-image> --namespace=<ns> -v -s

The deployment name is fixed, so the namespace must be clear of a previous
run before starting a new one -- back-to-back manual runs collide with the
prior teardown. CI is unaffected: every nightly gets a fresh vCluster.

Runs nightly, against the nightly -efa image, rather than per-commit: EFA
changes are sparse and this needs a real EFA cluster, two GPUs and roughly three
minutes, which is far more than the per-commit risk warrants.
"""

import pytest

from tests.deploy.efa_utils import EfaFrameworkProfile, run_efa_deployment_check

# vLLM's NixlConnector pulls: decode issues RDMA READs against prefill's KV
# blocks (_read_blocks). The bytes are therefore counted as rx on the decode
# agent, and at the adapter as reads issued by decode and read-responses served
# by prefill. SGLang pushes instead, which is why the two profiles differ here.
VLLM_EFA_PROFILE = EfaFrameworkProfile(
    name="vLLM",
    manifest_name="disagg-efa.yaml",
    prefill_service="VllmPrefillWorker",
    decode_service="VllmDecodeWorker",
    backend_pin_hint=(
        "--kv-transfer-config still pins "
        "kv_connector_extra_config.backends=['LIBFABRIC']."
    ),
    nixl_counter_service="VllmDecodeWorker",
    nixl_counter_metric="agent_rx_bytes",
    efa_counter_by_role={
        "VllmDecodeWorker": "node_amazonefa_rdma_read_bytes",
        "VllmPrefillWorker": "node_amazonefa_rdma_read_resp_bytes",
    },
)


@pytest.mark.framework_with_efa
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.nightly
@pytest.mark.e2e
# Two GPUs total: one prefill, one decode. No framework marker -- the nightly job
# selects with -m framework_with_efa, and carrying @pytest.mark.vllm would both
# require an exemption in tests/conftest.py's framework auto-skip and make this
# test collectable by the multi-GPU jobs, whose selectors are
# "vllm and (gpu_2 or gpu_4)" with no lifecycle filter.
@pytest.mark.gpu_2
@pytest.mark.core
@pytest.mark.timeout(1200)
async def test_efa_deployment(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
) -> None:
    """Deploy a disaggregated vLLM stack with EFA enabled and verify it serves."""
    await run_efa_deployment_check(
        profile=VLLM_EFA_PROFILE,
        image=image,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        request=request,
    )

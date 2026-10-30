# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One CPU-only Request → Run → AISimulate/publisher → Candidate smoke test.

Run against a Helm installation with dynamo-operator.dgdrV1Beta2.enabled=true
and controllerManager.manager.dgdrDefaultImage set to the matching planner image.
The CI workflow owns the vCluster lifecycle, including cleanup after failures.
"""

import asyncio
import json
import logging
import time
from functools import partial
from uuid import uuid4

import aiohttp
import pytest
import yaml
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client.exceptions import ApiException

from tests.deploy.vcluster_utils import retry_vcluster_api_async

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.timeout(900),
]


async def _read_custom(
    custom: client.CustomObjectsApi, namespace: str, plural: str, name: str
) -> dict:
    return await retry_vcluster_api_async(
        f"reading {plural}/{name}",
        partial(
            custom.get_namespaced_custom_object,
            "nvidia.com",
            "v1beta2",
            namespace,
            plural,
            name,
            _request_timeout=30,
        ),
        (aiohttp.ClientError, asyncio.TimeoutError),
        logger,
    )


def _assert_owned_by(resource: dict, owner: dict) -> None:
    assert any(
        ref.get("controller") and ref["uid"] == owner["metadata"]["uid"]
        for ref in resource["metadata"].get("ownerReferences", [])
    ), f"{resource['metadata']['name']} is not owned by {owner['metadata']['name']}"


async def test_dgdr_v1beta2_publishes_candidate(namespace: str) -> None:
    """Exercise chart RBAC, admission, both controllers, and both Job containers."""
    assert namespace, "--namespace is required for the DGDR v1beta2 smoke test"
    name = f"dgdr-smoke-{uuid4().hex[:8]}"
    async with await config.new_client_from_config() as api:
        custom = client.CustomObjectsApi(api)
        core = client.CoreV1Api(api)
        batch = client.BatchV1Api(api)
        rbac = client.RbacAuthorizationV1Api(api)

        # Hardware is simulated; complete topology avoids DCGM and cluster GPU discovery.
        # Qwen3-32B has AIC performance data; this test never downloads its weights.
        request = await custom.create_namespaced_custom_object(
            "nvidia.com",
            "v1beta2",
            namespace,
            "dynamographdeploymentrequests",
            {
                "apiVersion": "nvidia.com/v1beta2",
                "kind": "DynamoGraphDeploymentRequest",
                "metadata": {"name": name},
                "spec": {
                    "model": {"name": "Qwen/Qwen3-32B"},
                    "backends": ["vllm"],
                    "hardware": {
                        "gpu": {"sku": "h200_sxm", "budget": 8},
                        "numGpusPerNode": 8,
                    },
                    "objective": {"sla": {}},
                    "aisimulate": {
                        "inputOverrides": {
                            "engine": {
                                "mode": "aggregated",
                                "context_length": 32768,
                                "workers": {
                                    # TODO(#17581): Remove after the unset block-size fix is pinned.
                                    "aggregated": {"kv_cache": {"block_size": 64}},
                                    "prefill": None,
                                    "decode": None,
                                },
                            },
                            "traffic": {"stop": {"requests": 8}},
                            "optimizer": {
                                "algorithm": "random",
                                "max_trials": 3,
                                "parallelism": 1,
                                "candidate_timeout_seconds": 120,
                            },
                            "dgdr_run": {"generator": "aic"},
                        }
                    },
                },
            },
            _request_timeout=30,
        )
        assert request["spec"]["workload"]["static"]["load"] == {"concurrency": 10}
        assert request["spec"]["objective"]["metric"] == "ThroughputPerGPU"

        # The installed operator creates the Run; the test never starts a manager or writes status.
        deadline = time.monotonic() + 600
        run = {}
        run_name = None
        while time.monotonic() < deadline:
            if run_name is None:
                request = await _read_custom(
                    custom, namespace, "dynamographdeploymentrequests", name
                )
                request_status = request.get("status", {})

                # A fast Run may already have moved from activeRunRef into history.
                refs = request_status.get("recentRunRefs", [])
                ref = request_status.get("activeRunRef") or (refs[0] if refs else {})
                run_name = ref.get("name")
            if run_name:
                # activeRunRef reserves the name before the controller creates the Run.
                try:
                    run = await _read_custom(
                        custom, namespace, "dynamographdeploymentruns", run_name
                    )
                except ApiException as error:
                    if error.status != 404:
                        raise
                else:
                    status = run.get("status", {})
                    assert status.get("phase") != "Failed", status
                    if any(
                        condition["type"] == "Completed"
                        and condition["status"] == "True"
                        for condition in status.get("conditions", [])
                    ):
                        break
            await asyncio.sleep(5)
        else:
            pytest.fail(f"Request did not complete: request={request}, run={run}")

        assert run["status"]["phase"] == "Succeeded", run["status"]
        _assert_owned_by(run, request)
        assert run["spec"] == request["spec"]
        logger.info("Completed Run: %s", json.dumps(run))

        # Completion requires both real containers and the generated publisher identity.
        job_name = run["status"]["jobRef"]["name"]
        job = await batch.read_namespaced_job(job_name, namespace, _request_timeout=30)
        _assert_owned_by(api.sanitize_for_serialization(job), run)
        assert any(
            c.type == "Complete" and c.status == "True" for c in job.status.conditions
        )
        pod_spec = job.spec.template.spec
        assert {c.name for c in pod_spec.containers} == {"sweeper", "publisher"}
        assert pod_spec.service_account_name != "default"
        binding = await rbac.read_namespaced_role_binding(
            job_name, namespace, _request_timeout=30
        )
        assert binding.role_ref.kind == "Role"
        assert any(
            subject.kind == "ServiceAccount"
            and subject.name == pod_spec.service_account_name
            and subject.namespace == namespace
            for subject in binding.subjects
        )
        pods = await core.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}", _request_timeout=30
        )
        assert len(pods.items) == 1
        assert {c.name for c in pods.items[0].status.container_statuses} == {
            "sweeper",
            "publisher",
        }
        for container in pods.items[0].status.container_statuses:
            assert container.state.terminated is not None, container
            assert container.state.terminated.exit_code == 0, container

        # The frozen input consumed by the Job must match the public Run status.
        config_map = await core.read_namespaced_config_map(
            job_name, namespace, _request_timeout=30
        )
        assert config_map.immutable is True
        _assert_owned_by(api.sanitize_for_serialization(config_map), run)
        assert (
            yaml.safe_load(config_map.data["input.yaml"])
            == run["status"]["aisimulate"]["input"]
        )

        refs = run["status"]["candidateRefs"]
        assert len(refs) == 1
        candidate = await _read_custom(
            custom, namespace, "dynamographdeploymentcandidates", refs[0]["name"]
        )
        _assert_owned_by(candidate, run)
        assert candidate["spec"]["components"]
        assert candidate["spec"]["parameters"]["backend"] == "vllm"
        assert 0 < candidate["status"]["metrics"]["usedGpus"] <= 8
        assert isinstance(candidate["status"]["metrics"]["score"], (int, float))
        logger.info("Published Candidate: %s", json.dumps(candidate))

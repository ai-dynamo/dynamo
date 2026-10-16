# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live acceptance for an explicitly supplied, two-group SGLang sidecar DGD.

Run against either provider and either template form with
--sidecar-multinode-manifest=<configured manifest>. The test requires four GPUs
on distinct nodes, KV routing, and two aggregated engine groups of two Pods.
"""

import asyncio
import copy
import logging
import time

import aiohttp
import pytest

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.router.helper import get_stored_kv_event_counts, send_router_chat_request
from tests.utils.router_nvext import require_router_worker_id


logger = logging.getLogger(__name__)


def engine_group(pod) -> str:
    """Use provider-owned group identity to distinguish failure boundaries."""
    labels = pod.metadata.labels
    if "grove.io/podcliquescalinggroup-replica-index" in labels:
        return labels["grove.io/podcliquescalinggroup-replica-index"]
    return labels["leaderworkerset.sigs.k8s.io/group-index"]


async def wait_for_serving_workers(session, frontend_url):
    # Pod readiness can precede model registration and frontend discovery.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        async with session.get(f"{frontend_url}/health") as response:
            response.raise_for_status()
            health = await response.json()
        workers = sorted(
            {
                instance["instance_id"]
                for instance in health["instances"]
                if instance["component"] in {"backend", "worker"}
                and instance["endpoint"] == "generate"
            }
        )
        if len(workers) == 2:
            return workers
        await asyncio.sleep(1)
    pytest.fail(f"Two serving sidecars did not register: {health}")


@pytest.mark.framework_only
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.e2e
@pytest.mark.post_merge
@pytest.mark.timeout(1800)
async def test_sglang_multinode_sidecar(request, namespace):
    manifest = request.config.getoption("--sidecar-multinode-manifest")
    if not manifest:
        pytest.skip("Supply --sidecar-multinode-manifest for live multinode acceptance")

    assert namespace, "Supply --namespace for live multinode acceptance"
    spec = DeploymentSpec(manifest)
    spec.namespace = namespace
    components = spec.spec()["spec"]["components"]
    worker = next(c for c in components if c["type"] == "worker")
    frontend = next(c for c in components if c["type"] == "frontend")
    assert worker["replicas"] == 2 and worker["multinode"]["nodeCount"] == 2
    template = worker.get("podTemplate") or next(
        r["podTemplate"] for r in worker["roles"] if r["name"] == "leader"
    )
    engine = next(c for c in template["spec"]["containers"] if c["name"] == "main")
    args = engine["args"]
    model = args[args.index("--model-path") + 1]
    ranks = (0, 1) if "--enable-dp-attention" in args else (0,)

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=spec,
        namespace=namespace,
        frontend_service_name=frontend["name"],
        skip_service_restart=True,
        readiness_timeout=900,
    ) as deployment:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            pods = deployment.get_pods([worker["name"]])[worker["name"]]
            if len(pods) == 4 and all(p.ready() for p in pods):
                break
            await asyncio.sleep(2)
        else:
            pytest.fail("Not all engine Pods became Ready")
        assert len({p.spec.nodeName for p in pods}) == 4
        runtime_pods = [
            p
            for p in pods
            if any(c.name == "runtime" for c in p.spec.get("initContainers", []))
        ]
        assert len(runtime_pods) == 2 * len(ranks)
        for pod in pods:
            main = next(c for c in pod.spec.containers if c.name == "main")
            assert main.command[:3] == ["python3", "-m", "sglang.launch_server"]
            assert "DYN_SYSTEM_ENABLED" not in {e.name for e in main.env}

        frontend_pod = deployment.get_pods([frontend["name"]])[frontend["name"]][0]
        http = deployment.port_forward(frontend_pod, spec.port)
        assert http is not None
        system_ports = []
        for pod in runtime_pods:
            runtime = next(c for c in pod.spec.initContainers if c.name == "runtime")
            port = next(p.containerPort for p in runtime.ports if p.name == "system")
            forward = deployment.port_forward(pod, port)
            assert forward is not None
            system_ports.append(forward.local_port)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            workers = await wait_for_serving_workers(
                session, f"http://localhost:{http.local_port}"
            )
            baselines = [
                await get_stored_kv_event_counts(session, p) for p in system_ports
            ]
            targets = [(worker_id, rank) for worker_id in workers for rank in ranks]
            url = f"http://localhost:{http.local_port}/v1/chat/completions"

            # Pin distinct prefixes to every rank, including ranks on follower Pods.
            payloads = []
            for i, (worker_id, rank) in enumerate(targets):
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Garden {i} has violet trees. " * 96,
                        }
                    ],
                    "max_tokens": 16,
                    "stream": True,
                    "temperature": 0,
                    "nvext": {"extra_fields": ["worker_id", "timing"]},
                }
                payloads.append(payload)
                nvext, text = await send_router_chat_request(
                    session,
                    url,
                    payload,
                    {
                        "x-dynamo-worker-instance-id": str(worker_id),
                        "x-dynamo-dp-rank": str(rank),
                    },
                )
                selected = require_router_worker_id({"nvext": nvext})
                assert text and (
                    selected["decode_worker_id"],
                    selected["decode_dp_rank"],
                ) == (worker_id, rank)

            # Require every node-local publisher to accept engine events.
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                counts = [
                    await get_stored_kv_event_counts(session, p) for p in system_ports
                ]
                if all(
                    all(c > b for c, b in zip(current, baseline))
                    for current, baseline in zip(counts, baselines)
                ):
                    break
                await asyncio.sleep(1)
            else:
                pytest.fail(f"KV publishers did not advance: {baselines=} {counts=}")

            logger.info(
                "KV publisher counters advanced on every runtime: %s -> %s",
                baselines,
                counts,
            )

            # Unpinned requests must route back to the independently warmed rank.
            for payload, target in zip(payloads, targets):
                query = copy.deepcopy(payload)
                query["nvext"]["annotations"] = ["query_instance_id:"]
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    nvext, text = await send_router_chat_request(session, url, query)
                    assert not text
                    selected = require_router_worker_id({"nvext": nvext})
                    if (
                        selected["decode_worker_id"],
                        selected["decode_dp_rank"],
                    ) == target:
                        break
                    await asyncio.sleep(1)
                else:
                    pytest.fail(f"Frontend did not observe cache on {target}: {nvext}")
                nvext, text = await send_router_chat_request(session, url, payload)
                assert text and nvext["timing"]["kv_hit_rate"] >= 0.5

            # Delete one follower and require only its complete engine group to restart.
            before = {str(p.metadata.uid): engine_group(p) for p in pods}
            victim = next(
                p
                for p in pods
                if "-wkr-" in p.name
                or p.metadata.labels.get("leaderworkerset.sigs.k8s.io/worker-index")
                == "1"
            )
            group = engine_group(victim)
            await asyncio.to_thread(victim.delete)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                current = deployment.get_pods([worker["name"]])[worker["name"]]
                uids = {str(p.metadata.uid) for p in current}
                unaffected = {uid for uid, g in before.items() if g != group}
                assert unaffected <= uids, "An unrelated engine group restarted"
                if (
                    len(current) == 4
                    and not ({uid for uid, g in before.items() if g == group} & uids)
                    and all(p.ready() for p in current)
                ):
                    break
                await asyncio.sleep(2)
            else:
                pytest.fail("The failed engine group did not recover")
            # Grove may also replace the frontend if it belongs to the failed base gang.
            frontend_pod = deployment.get_pods([frontend["name"]])[frontend["name"]][0]
            http = deployment.port_forward(frontend_pod, spec.port)
            assert http is not None
            url = f"http://localhost:{http.local_port}/v1/chat/completions"
            recovered_workers = await wait_for_serving_workers(
                session, f"http://localhost:{http.local_port}"
            )
            for worker_id in recovered_workers:
                for rank in ranks:
                    nvext, text = await send_router_chat_request(
                        session,
                        url,
                        payloads[0],
                        {
                            "x-dynamo-worker-instance-id": str(worker_id),
                            "x-dynamo-dp-rank": str(rank),
                        },
                    )
                    selected = require_router_worker_id({"nvext": nvext})
                    assert text and (
                        selected["decode_worker_id"],
                        selected["decode_dp_rank"],
                    ) == (worker_id, rank)
            logger.info(
                "Inference, rank-local KV routing, and group recovery passed for %s",
                targets,
            )

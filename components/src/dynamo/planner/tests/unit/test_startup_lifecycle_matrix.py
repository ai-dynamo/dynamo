# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner startup across shipped DGD shapes, startup paths, and worker registration.

Only Kubernetes and runtime I/O are faked; the environment, connector, DGD
classifiers, and FPM provider are real.
"""

import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from kubernetes import client

import dynamo.llm
from dynamo.planner.environment import base as environment_base
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.environment.metrics_provider.runtime_provider import (
    RuntimeFpmProvider,
)
from dynamo.planner.tests.unit.test_pending_startup_scaling import _config, _connector

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_EXAMPLES = Path(__file__).resolve().parents[6] / "examples" / "backends"
_RUNTIME_NAMESPACE = "test-ns"
_READINESS_TIMEOUT = "deployment not ready after 1800 seconds"

# One shipped DGD per distinct worker naming shape, plus a renamed disagg graph.
_SHAPES = {
    "vllm-agg-worker": ("vllm", "agg.yaml", {}),
    "sglang-agg-decode-typed-worker": ("sglang", "agg.yaml", {}),
    "trtllm-agg-pascalcase": ("trtllm", "agg.yaml", {}),
    "vllm-disagg-with-planner": ("vllm", "disagg_planner.yaml", {}),
    "vllm-disagg-renamed": ("vllm", "disagg.yaml", {"prefill": "p", "decode": "d"}),
}

# path -> (pending replica on the first worker, worker metadata published at
# startup, readiness wait outcome, Pod list forbidden, startup must succeed).
# Metadata missing at startup arrives on the second read when the wait blocks,
# and after startup otherwise.
_PATHS = {
    "ready": (False, True, "ready", False, True),
    "pending": (True, True, "blocks", False, True),
    "pending_before_metadata": (True, False, "blocks", False, True),
    "ready_before_metadata": (False, False, "ready", False, True),
    "timeout_without_serving_worker": (True, False, "timeout", False, False),
    "pods_forbidden": (False, True, "ready", True, True),
}


class _FakeSubscriber:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def start_tracking(self):
        pass

    def shutdown(self):
        pass

    def get_model_cards(self):
        return []

    def get_recent_stats(self):
        return {}


def _role(component):
    return "prefill" if component["type"] == "prefill" else "decode"


def _endpoint_name(runtime_component):
    parts = (runtime_component or "").split(".")
    return parts[1] if len(parts) > 1 else "generate"


def _cluster(backend, manifest, renames, runtime_component, path):
    pending, metadata_at_start, readiness, pods_forbidden, _ = _PATHS[path]
    source = (_EXAMPLES / backend / "deploy" / manifest).read_text()
    dgd = next(
        doc
        for doc in yaml.safe_load_all(source)
        if doc and doc.get("kind") == "DynamoGraphDeployment"
    )
    dgd["metadata"].update(name="qwen", uid="qwen-uid", generation=2)
    components = dgd["spec"]["components"]
    for component in components:
        component["name"] = renames.get(component["name"], component["name"])
    workers = [c for c in components if c["type"] not in ("frontend", "planner")]
    statuses, pods, cards = {}, [], {}
    for component in components:
        name, desired = component["name"], component.get("replicas", 1)
        ready = desired
        status = {}
        if component in workers:
            component["scalingAdapter"] = {}
            role = _role(component)
            served = (runtime_component or "").split(".")[0] or (
                "prefill" if role == "prefill" else "backend"
            )
            status = {
                "gpusPerEngine": 1,
                "gpusPerReplica": 1,
                "runtimeComponentName": served,
                "servedModelName": "test-model",
                "componentNames": [f"qwen-{name.lower()}"],
            }
            cards[role] = (f"qwen-{name.lower()}-0", served)
            if component is workers[0] and pending:
                desired = component["replicas"] = desired + 1
                ready = 0 if readiness == "timeout" else ready
        status.update(replicas=desired, updatedReplicas=desired, readyReplicas=ready)
        statuses[name] = status
        for index in range(desired):
            pods.append(
                client.V1Pod(
                    metadata=client.V1ObjectMeta(
                        name=f"{name.lower()}-{index}",
                        labels={"nvidia.com/dynamo-component": name},
                    ),
                    status=client.V1PodStatus(
                        phase="Running" if index < ready else "Pending"
                    ),
                )
            )
    dgd["status"] = {
        "observedGeneration": 2,
        "conditions": [{"type": "Ready", "status": "False" if pending else "True"}],
        "components": statuses,
    }

    connector = _connector(dgd, pods)
    connector.user_provided_model_name = None
    connector.parent_dgd_name = "qwen"
    api = connector.kube_api
    if pods_forbidden:
        api.list_pods_for_graph = Mock(side_effect=client.ApiException(status=403))

    async def wait_for_ready(*_args, **_kwargs):
        if readiness == "timeout":
            raise TimeoutError(_READINESS_TIMEOUT)
        if readiness == "blocks":
            await asyncio.Event().wait()

    api.wait_for_graph_deployment_ready = AsyncMock(side_effect=wait_for_ready)
    published = []

    def publish_metadata():
        published[:] = [
            {
                "metadata": {"name": cr_name},
                "spec": {
                    "data": {
                        "model_cards": {
                            "test-model": {
                                "type": "Model",
                                "component": served,
                                "endpoint": _endpoint_name(runtime_component),
                                "card_json": {
                                    "worker_type": role,
                                    "display_name": "test-model",
                                    "kv_cache_block_size": 16,
                                    "runtime_config": {
                                        "total_kv_blocks": 1024,
                                        "max_num_batched_tokens": 8192,
                                    },
                                },
                            }
                        }
                    }
                },
            }
            for role, (cr_name, served) in cards.items()
        ]

    reads = []

    def list_metadata(**_):
        if readiness == "blocks" and reads:
            publish_metadata()
        reads.append(len(published))
        return {"items": deepcopy(published)}

    api.custom_api = Mock()
    api.custom_api.list_namespaced_custom_object.side_effect = list_metadata
    if metadata_at_start:
        publish_metadata()
    expected = {
        _role(c): (c["name"], statuses[c["name"]]["readyReplicas"]) for c in workers
    }
    pending_by_name = {
        workers[0]["name"]: (
            statuses[workers[0]["name"]]["replicas"]
            - statuses[workers[0]["name"]]["readyReplicas"]
            if pending and not pods_forbidden
            else 0
        ),
    }
    return connector, publish_metadata, cards, expected, pending_by_name


@pytest.mark.asyncio
@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("runtime_component", [None, "custom", "custom.infer"])
@pytest.mark.parametrize("shape", list(_SHAPES))
async def test_startup_binds_metrics_to_registered_workers(
    shape, runtime_component, path, monkeypatch
):
    monkeypatch.setattr(environment_base, "STARTUP_PENDING_RECHECK_SECONDS", 0.01)
    monkeypatch.setattr(dynamo.llm, "FpmEventSubscriber", _FakeSubscriber)
    backend, manifest, renames = _SHAPES[shape]
    connector, publish_metadata, cards, expected, pending_by_name = _cluster(
        backend, manifest, renames, runtime_component, path
    )
    disagg = "prefill" in cards
    namespace = Mock()
    namespace.runtime_namespace.return_value = _RUNTIME_NAMESPACE
    namespace.refresh_runtime_namespace = AsyncMock(return_value=False)
    runtime = Mock()
    runtime.endpoint.side_effect = lambda endpoint: endpoint
    provider = RuntimeFpmProvider(
        require_prefill=disagg,
        require_decode=True,
        backend=backend,
        model_name="test-model",
        runtime=runtime,
    )
    environment = PlannerEnvironmentImpl(
        config=_config("disagg" if disagg else "agg", backend=backend),
        controller=connector,
        require_prefill=disagg,
        require_decode=True,
        fpm_provider=provider,
        runtime_namespace_source=namespace,
    )
    provider.bind_sources(state_source=environment, namespace_source=namespace)

    must_start = _PATHS[path][4]
    try:
        await asyncio.wait_for(environment.initialize(), timeout=5)
    except TimeoutError as exc:
        # Failing closed is acceptable only where startup is not required, only
        # from the readiness timeout (not a hang), and must not leave a
        # subscriber bound to a guessed endpoint.
        assert not must_start, "startup did not complete"
        assert _READINESS_TIMEOUT in str(exc), "startup hung"
        runtime.endpoint.assert_not_called()
        return

    state = environment.deployment_state()
    for role, (name, ready) in expected.items():
        replicas = getattr(state, role).replicas
        assert replicas.active == ready, role
        assert replicas.pending_startup == pending_by_name.get(name, 0), role

    publish_metadata()
    for _ in range(3):
        await environment.refresh()
    subscribers = {
        "prefill": provider._prefill_fpm_sub,
        "decode": provider._decode_fpm_sub,
    }
    for role, (_, served) in cards.items():
        assert (
            subscribers[role].endpoint
            == f"{_RUNTIME_NAMESPACE}.{served}.{_endpoint_name(runtime_component)}"
        )
        assert getattr(state, role).info.max_num_batched_tokens == 8192

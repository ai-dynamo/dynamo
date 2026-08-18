# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import power_agent
import pytest


def _pod(name: str, resource_version: str, uid: str | None = None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            namespace="dynamo",
            name=name,
            uid=uid or f"uid-{name}",
            resource_version=resource_version,
            annotations={},
        )
    )


def _agent(core_v1):
    agent = object.__new__(power_agent.PowerAgent)
    agent._core_v1 = core_v1
    agent.node_name = "node-a"
    agent.k8s_namespace = None
    agent._pod_cache = {}
    agent._pod_cache_initialized = False
    agent._pod_resource_version = ""
    agent.reconcile_once = MagicMock()
    return agent


class _Watch:
    def __init__(self, events=None, error=None):
        self.events = events or []
        self.error = error
        self.stream_args = None
        self.stream_kwargs = None
        self.stopped = False

    def stream(self, *args, **kwargs):
        self.stream_args = args
        self.stream_kwargs = kwargs
        if self.error is not None:
            raise self.error
        yield from self.events

    def stop(self):
        self.stopped = True


def test_startup_is_one_list_then_watch_cache_updates():
    initial = _pod("worker", "10")
    modified = _pod("worker", "11")
    core_v1 = MagicMock()
    core_v1.list_pod_for_all_namespaces.return_value = SimpleNamespace(
        items=[initial], metadata=SimpleNamespace(resource_version="10")
    )
    agent = _agent(core_v1)

    assert agent._list_pods_on_node() == [initial]
    assert agent._list_pods_on_node() == [initial]
    core_v1.list_pod_for_all_namespaces.assert_called_once()

    watcher = _Watch(events=[{"type": "MODIFIED", "object": modified}])
    with patch.object(power_agent, "k8s_watch", SimpleNamespace(Watch=lambda: watcher)):
        agent._watch_pods_once()

    assert watcher.stream_args == (core_v1.list_pod_for_all_namespaces,)
    assert watcher.stream_kwargs["resource_version"] == "10"
    assert watcher.stream_kwargs["field_selector"] == "spec.nodeName=node-a"
    assert agent._pod_resource_version == "11"
    assert agent._list_pods_on_node() == [modified]
    core_v1.list_pod_for_all_namespaces.assert_called_once()
    agent.reconcile_once.assert_called_once()


def test_disconnect_resumes_from_last_resource_version_without_relist():
    core_v1 = MagicMock()
    core_v1.list_pod_for_all_namespaces.return_value = SimpleNamespace(
        items=[_pod("worker", "20")],
        metadata=SimpleNamespace(resource_version="20"),
    )
    agent = _agent(core_v1)
    agent._list_pods_on_node()

    disconnected = _Watch(error=RuntimeError("connection reset"))
    with patch.object(
        power_agent, "k8s_watch", SimpleNamespace(Watch=lambda: disconnected)
    ):
        with pytest.raises(RuntimeError, match="connection reset"):
            agent._watch_pods_once()

    resumed = _Watch()
    with patch.object(power_agent, "k8s_watch", SimpleNamespace(Watch=lambda: resumed)):
        agent._watch_pods_once()

    assert disconnected.stream_kwargs["resource_version"] == "20"
    assert resumed.stream_kwargs["resource_version"] == "20"
    core_v1.list_pod_for_all_namespaces.assert_called_once()


def test_410_invalidates_cache_and_next_snapshot_relists():
    core_v1 = MagicMock()
    core_v1.list_pod_for_all_namespaces.side_effect = [
        SimpleNamespace(
            items=[_pod("old", "30")],
            metadata=SimpleNamespace(resource_version="30"),
        ),
        SimpleNamespace(
            items=[_pod("new", "40")],
            metadata=SimpleNamespace(resource_version="40"),
        ),
    ]
    agent = _agent(core_v1)
    agent._list_pods_on_node()

    expired = _Watch(events=[{"type": "ERROR", "object": {"code": 410}}])
    with patch.object(power_agent, "k8s_watch", SimpleNamespace(Watch=lambda: expired)):
        with pytest.raises(power_agent._PodWatchResourceVersionExpired):
            agent._watch_pods_once()

    agent._invalidate_pod_cache_for_relist()
    assert [pod.metadata.name for pod in agent._list_pods_on_node()] == ["new"]
    assert agent._pod_resource_version == "40"
    assert core_v1.list_pod_for_all_namespaces.call_count == 2


def test_watch_backoff_is_positive_and_bounded():
    delay = power_agent.K8S_WATCH_BACKOFF_INITIAL_S
    observed = []
    for _ in range(10):
        observed.append(delay)
        delay = min(delay * 2, power_agent.K8S_WATCH_BACKOFF_MAX_S)

    assert observed == sorted(observed)
    assert observed[0] > 0
    assert observed[-1] == power_agent.K8S_WATCH_BACKOFF_MAX_S


def test_deletion_then_same_name_new_uid_replaces_cached_identity():
    old = _pod("worker", "50", uid="old-uid")
    replacement = _pod("worker", "52", uid="new-uid")
    core_v1 = MagicMock()
    core_v1.list_pod_for_all_namespaces.return_value = SimpleNamespace(
        items=[old], metadata=SimpleNamespace(resource_version="50")
    )
    agent = _agent(core_v1)
    agent._list_pods_on_node()

    watcher = _Watch(
        events=[
            {"type": "DELETED", "object": _pod("worker", "51", uid="old-uid")},
            {"type": "ADDED", "object": replacement},
        ]
    )
    with patch.object(power_agent, "k8s_watch", SimpleNamespace(Watch=lambda: watcher)):
        agent._watch_pods_once()

    cached = agent._list_pods_on_node()
    assert len(cached) == 1
    assert cached[0].metadata.uid == "new-uid"
    assert agent._pod_resource_version == "52"
    assert agent._pod_replacement_barriers == {("dynamo", "worker")}
    assert agent.reconcile_once.call_count == 2

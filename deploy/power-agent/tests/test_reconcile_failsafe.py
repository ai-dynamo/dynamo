# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile fail-safe on pod-listing failure (PR #9682 review follow-up).

`_list_pods_on_node` uses an *explicit* result contract so the failure path
cannot silently drift into the empty-node path:

  * success  → a (possibly empty) list of pods
  * failure  → ``None``

`reconcile_once` keys its fail-safe off that ``None``: on a listing failure it
SKIPS the cycle, leaving every GPU at its last-known-good cap, rather than
proceeding with an empty pod view that would re-derive caps from a zero-pod
snapshot. An empty list (node genuinely has no pods) is NOT a failure and must
still drive a normal reconcile pass.
"""

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from actuator import _GpuIdentityMismatch
from power_agent import PowerAgent

from tests.actuator_double import actuator_double


def _make_agent(core_v1, device_count: int = 2) -> PowerAgent:
    """Build a PowerAgent without touching NVML / K8s (bypass __init__)."""
    agent = object.__new__(PowerAgent)
    agent._core_v1 = core_v1
    agent.node_name = "node-under-test"
    agent.k8s_namespace = None  # exercise the list_pod_for_all_namespaces path
    agent.device_count = device_count
    # reconcile_once re-snapshots the count from the actuator each cycle.
    actuator = MagicMock()
    actuator.device_count.return_value = device_count
    agent._actuator = actuator
    # reconcile_once ticks k8s_list_failures_total on the skip path.
    agent.metrics = MagicMock()
    return agent


class TestListPodsExplicitResult(unittest.TestCase):
    def test_returns_none_on_api_error(self):
        """An apiserver error yields the explicit ``None`` sentinel, never []."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("boom")
        agent = _make_agent(core_v1)

        self.assertIsNone(agent._list_pods_on_node())

    def test_returns_empty_list_when_node_has_no_pods(self):
        """A genuinely empty node returns ``[]`` — distinct from failure."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)

        result = agent._list_pods_on_node()
        self.assertIsNotNone(result)
        self.assertEqual(result, [])

    def test_cluster_scoped_list_served_from_watch_cache(self):
        """The cluster-scoped LIST passes resource_version="0" so the apiserver
        can serve it from its watch cache instead of an etcd read (PR #9682
        scale mitigation)."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_pod_for_all_namespaces.call_args
        self.assertEqual(kwargs.get("resource_version"), "0")
        core_v1.list_namespaced_pod.assert_not_called()

    def test_cluster_scoped_list_is_bounded_both_sides(self):
        """The cluster-scoped LIST must carry BOTH an apiserver-side
        ``timeout_seconds`` and a client-side ``_request_timeout`` so a
        throttled/stuck apiserver substantially limits how long it can delay
        SIGTERM cleanup (sttts P1). This intentionally supersedes the earlier
        "no client timeout" stance: transport retries are disabled on this
        client, so the amplification risk that motivated the old stance is
        controlled (see test_k8s_client_disables_transport_retries)."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_pod_for_all_namespaces.call_args
        self.assertEqual(
            kwargs.get("timeout_seconds"), power_agent.K8S_LIST_SERVER_TIMEOUT_S
        )
        self.assertEqual(
            kwargs.get("_request_timeout"), power_agent.K8S_LIST_CLIENT_TIMEOUT_S
        )

    def test_namespaced_list_served_from_watch_cache(self):
        """The namespace-scoped LIST also passes resource_version="0"."""
        core_v1 = MagicMock()
        core_v1.list_namespaced_pod.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)
        agent.k8s_namespace = "dynamo"

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_namespaced_pod.call_args
        self.assertEqual(kwargs.get("resource_version"), "0")
        self.assertEqual(kwargs.get("namespace"), "dynamo")
        core_v1.list_pod_for_all_namespaces.assert_not_called()

    def test_namespaced_list_is_bounded_both_sides(self):
        """The namespace-scoped LIST carries the same dual timeout bounds."""
        core_v1 = MagicMock()
        core_v1.list_namespaced_pod.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)
        agent.k8s_namespace = "dynamo"

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_namespaced_pod.call_args
        self.assertEqual(
            kwargs.get("timeout_seconds"), power_agent.K8S_LIST_SERVER_TIMEOUT_S
        )
        self.assertEqual(
            kwargs.get("_request_timeout"), power_agent.K8S_LIST_CLIENT_TIMEOUT_S
        )

    def test_client_timeout_at_least_server_timeout(self):
        """The client-side bound must be >= the server-side bound so the
        apiserver's own timeout is what normally fires; a client timeout below
        the server timeout would pre-empt graceful server-side abort on every
        slow-but-progressing LIST."""
        self.assertGreaterEqual(
            power_agent.K8S_LIST_CLIENT_TIMEOUT_S,
            power_agent.K8S_LIST_SERVER_TIMEOUT_S,
        )


class TestReconcileFailSafe(unittest.TestCase):
    def test_listing_failure_skips_reconcile_and_preserves_caps(self):
        """On listing failure, ``reconcile_once`` must not touch any GPU —
        no per-GPU reconcile runs, so existing caps are frozen in place."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("boom")
        agent = _make_agent(core_v1, device_count=4)
        agent._reconcile_gpu = MagicMock()

        agent.reconcile_once()

        agent._reconcile_gpu.assert_not_called()

    def test_empty_node_still_reconciles_every_gpu(self):
        """An empty (but successful) listing is NOT a failure: reconcile
        proceeds normally and still visits every GPU, instead of skipping the
        whole cycle as it does on a listing failure. (Each GPU then early-
        returns in ``_reconcile_gpu`` since no listed pod owns it — no
        safe-default is applied on a genuinely empty node.)"""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=4)
        agent._reconcile_gpu = MagicMock()

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 4)


class TestReconcileStopsOnShutdown(unittest.TestCase):
    """A SIGTERM arriving mid-cycle must stop the per-GPU loop promptly so
    run()'s finally can restore caps within the pod's termination grace period,
    rather than grinding through every remaining GPU's DCGM/NVML calls first."""

    def setUp(self):
        power_agent._shutdown.clear()

    def tearDown(self):
        power_agent._shutdown.clear()

    def test_gpu_loop_breaks_when_shutdown_requested_midcycle(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=4)

        processed = []

        def fake_reconcile(gpu_idx, uid_to_annotation):
            processed.append(gpu_idx)
            power_agent._shutdown.set()  # SIGTERM lands while handling GPU 0

        agent._reconcile_gpu = fake_reconcile

        agent.reconcile_once()

        # Only GPU 0 ran; the loop broke before GPUs 1-3 on the shutdown check.
        self.assertEqual(processed, [0])

    def test_no_gpu_processed_when_shutdown_already_set(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=3)
        agent._reconcile_gpu = MagicMock()
        power_agent._shutdown.set()

        agent.reconcile_once()

        agent._reconcile_gpu.assert_not_called()

    def test_early_guard_skips_pod_list_when_shutdown_already_set(self):
        """The top-of-cycle guard must return BEFORE issuing the pod LIST when
        SIGTERM has already landed. This is the crux of the P1 fix: starting a
        new LIST here (network I/O that can block under apiserver throttling)
        would delay run()'s finally cleanup and risk kubelet SIGKILL before caps
        are restored. Assert the guard short-circuits ahead of persistence
        flushes and the LIST."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=2)
        agent._reconcile_gpu = MagicMock()
        power_agent._shutdown.set()

        with patch.object(power_agent, "_flush_pending_retirements") as retirements:
            with patch.object(
                power_agent, "_flush_pending_acquisitions"
            ) as acquisitions:
                agent.reconcile_once()

        retirements.assert_not_called()
        acquisitions.assert_not_called()
        core_v1.list_pod_for_all_namespaces.assert_not_called()
        core_v1.list_namespaced_pod.assert_not_called()
        agent.metrics.k8s_list_failures_total.inc.assert_not_called()


class TestK8sClientTransport(unittest.TestCase):
    """The agent's CoreV1Api must be built on a client whose transport does NOT
    retry: a retried pod LIST amplifies apiserver load under P&F throttling
    (sttts's original concern) AND stretches a single LIST's wall-clock time,
    both of which the reconcile loop's own 15s application-level retry already
    covers. Disabling transport retries is what keeps the best-effort
    client-side LIST timeout from being silently multiplied."""

    def test_k8s_client_disables_transport_retries(self):
        fake_config = MagicMock()
        fake_client_mod = MagicMock()
        fake_client_mod.Configuration.get_default_copy.return_value = fake_config

        with patch.object(power_agent, "k8s_client", fake_client_mod):
            power_agent._build_k8s_core_v1()

        # retries set to 0 on the per-agent configuration copy...
        self.assertEqual(fake_config.retries, 0)
        # ...and CoreV1Api built on an ApiClient using THAT configuration, so
        # the choice is local to the agent and does not mutate global state.
        fake_client_mod.ApiClient.assert_called_once_with(fake_config)
        fake_client_mod.CoreV1Api.assert_called_once_with(
            fake_client_mod.ApiClient.return_value
        )


class TestReconcileDeviceCountRefresh(unittest.TestCase):
    """The reconcile loop must re-snapshot the
    GPU count from the actuator each cycle, not trust the value cached at
    startup. A DCGM hostengine reconnect can change the discovered-GPU set at
    runtime."""

    def test_grown_count_after_reconnect_is_picked_up(self):
        """Cached count is 2; a reconnect grew the set to 4. All 4 GPUs must be
        reconciled — a startup-frozen count would silently ignore the new
        GPUs."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=2)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.return_value = 4

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 4)
        self.assertEqual(agent.device_count, 4)

    def test_shrunk_count_after_reconnect_avoids_stale_indices(self):
        """Cached count is 8; a reconnect shrank the set to 3. Only the 3 live
        indices are iterated — iterating the stale larger range would raise
        per-index errors on the removed GPUs."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=8)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.return_value = 3

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 3)
        self.assertEqual(agent.device_count, 3)

    def test_count_refresh_failure_uses_last_known(self):
        """A transient device_count() read failure must not skip enforcement:
        fall back to the last-known count for this cycle."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=5)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.side_effect = RuntimeError("hostengine down")

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 5)
        self.assertEqual(agent.device_count, 5)


class _DcgmStyleActuator:
    """Minimal actuator exposing `managed_uuid_for_idx` on its TYPE.

    `_release_managed_gpu` gates the DCGM index-projection branches on
    `hasattr(type(actuator), "managed_uuid_for_idx")`. A `MagicMock` fails that
    check — its type is `MagicMock` — so those branches are unreachable with a
    plain double and a test using one would silently exercise the NVML path.
    """

    name = "dcgm"

    def __init__(self, current_uuid, managed_uuid):
        self._current_uuid = current_uuid
        self._managed_uuid = managed_uuid
        self.restore_default_by_uuid = MagicMock(return_value=True)

    def get_uuid(self, gpu_idx):
        return self._current_uuid

    def managed_uuid_for_idx(self, gpu_idx):
        if isinstance(self._managed_uuid, Exception):
            raise self._managed_uuid
        return self._managed_uuid


class TestCycleEnforcementBoolean(unittest.TestCase):
    """`reconcile_once` returns the whole-cycle enforcement boolean (DEP #14767).

    True iff every enforcement action the cycle was REQUIRED to take completed
    and the inputs it depended on were obtained. This is what `/readyz`
    publishes, so every path that swallows a failure to keep the loop running
    must drive it false — the agent's failure handling is deliberately
    non-fatal, and without this fold none of it is observable as state.
    """

    def setUp(self):
        power_agent._shutdown.clear()
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._shutdown.clear()
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _agent(self, device_count=2):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=device_count)
        agent.safe_default_watts = 500
        return agent

    def _gpu_agent(self, actuator, device_count=1):
        """An agent whose `_reconcile_gpu` runs for real against `actuator`."""
        agent = self._agent(device_count=device_count)
        actuator.device_count.return_value = device_count
        agent._actuator = actuator
        return agent

    # --- the happy path -------------------------------------------------

    def test_clean_cycle_returns_true(self):
        agent = self._agent(device_count=3)
        agent._reconcile_gpu = MagicMock(return_value=True)

        self.assertIs(agent.reconcile_once(), True)

    # --- inputs the cycle depended on -----------------------------------

    def test_pod_list_failure_returns_false_and_skips_the_gpu_loop(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("boom")
        agent = _make_agent(core_v1, device_count=4)
        agent._reconcile_gpu = MagicMock(return_value=True)

        self.assertIs(agent.reconcile_once(), False)
        agent._reconcile_gpu.assert_not_called()

    def test_device_count_refresh_failure_returns_false(self):
        """The cycle still reconciles against the last-known count, but it ran
        on a possibly-stale topology, so it does not count as enforcing."""
        agent = self._agent(device_count=5)
        agent._reconcile_gpu = MagicMock(return_value=True)
        agent._actuator.device_count.side_effect = RuntimeError("hostengine down")

        self.assertIs(agent.reconcile_once(), False)
        self.assertEqual(agent._reconcile_gpu.call_count, 5)

    # --- the fold must not short-circuit --------------------------------

    def test_failure_on_gpu_zero_still_reconciles_every_later_gpu(self):
        """Regression test for `ok and _reconcile_gpu(...)`, which would stop
        enforcing GPUs 1..N the moment GPU 0 failed."""
        agent = self._agent(device_count=4)
        visited = []

        def fake(gpu_idx, _uid_to_annotation):
            visited.append(gpu_idx)
            return gpu_idx != 0

        agent._reconcile_gpu = fake

        self.assertIs(agent.reconcile_once(), False)
        self.assertEqual(visited, [0, 1, 2, 3])

    def test_raising_gpu_marks_the_cycle_false_without_aborting_it(self):
        agent = self._agent(device_count=3)
        visited = []

        def fake(gpu_idx, _uid_to_annotation):
            visited.append(gpu_idx)
            if gpu_idx == 1:
                raise RuntimeError("NVML exploded")
            return True

        agent._reconcile_gpu = fake

        self.assertIs(agent.reconcile_once(), False)
        self.assertEqual(visited, [0, 1, 2])

    # --- per-GPU enforcement outcomes -----------------------------------

    def test_failed_cap_write_marks_the_cycle_false(self):
        """The core of the DEP: `apply_cap` returning normally is NOT success.
        Both actuators absorb their write errors and still return a plausible
        post-clamp wattage, so only `.ok` can tell the loop what happened."""
        actuator = actuator_double(effective_w=350, ok=False)
        actuator.list_running_pids.return_value = [1234]
        agent = self._gpu_agent(actuator)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="pod-uid-1"
        ):
            result = agent._reconcile_gpu(0, {"pod-uid-1": "350"})

        self.assertIs(result, False)

    def test_successful_cap_write_marks_the_gpu_enforced(self):
        actuator = actuator_double(effective_w=350, ok=True)
        actuator.list_running_pids.return_value = [1234]
        agent = self._gpu_agent(actuator)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="pod-uid-1"
        ):
            result = agent._reconcile_gpu(0, {"pod-uid-1": "350"})

        self.assertIs(result, True)

    def test_safe_default_fallback_is_not_a_failure(self):
        """An unparseable annotation falls back to `safe_default_watts`.
        Readiness tracks the cap the agent RESOLVED, not the one the user asked
        for, so writing that default is a successful cycle."""
        actuator = actuator_double(effective_w=500, ok=True)
        actuator.list_running_pids.return_value = [1234]
        agent = self._gpu_agent(actuator)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="pod-uid-1"
        ):
            result = agent._reconcile_gpu(0, {"pod-uid-1": "not-a-number"})

        self.assertIs(result, True)
        actuator.apply_cap.assert_called_once_with(
            0, 500, expected_uuid=actuator.get_uuid.return_value
        )

    def test_unconfirmed_gpu_identity_marks_the_cycle_false(self):
        actuator = actuator_double()
        actuator.get_uuid.side_effect = RuntimeError("identity unreadable")
        agent = self._gpu_agent(actuator)

        self.assertIs(agent._reconcile_gpu(0, {}), False)
        actuator.apply_cap.assert_not_called()

    def test_identity_mismatch_on_pid_snapshot_marks_the_cycle_false(self):
        actuator = actuator_double()
        actuator.list_running_pids.side_effect = _GpuIdentityMismatch("re-enumerated")
        agent = self._gpu_agent(actuator)

        self.assertIs(agent._reconcile_gpu(0, {}), False)
        actuator.apply_cap.assert_not_called()

    def test_idle_gpu_is_never_written_and_does_not_affect_the_boolean(self):
        """A GPU with no running workload is intentionally skipped: the claim
        covers the actions the cycle was REQUIRED to take, not the live state of
        every GPU on the node."""
        actuator = actuator_double()
        actuator.list_running_pids.return_value = []
        agent = self._gpu_agent(actuator)

        self.assertIs(agent._reconcile_gpu(0, {}), True)
        actuator.apply_cap.assert_not_called()

    # --- release outcomes ------------------------------------------------

    def test_release_failure_propagates_through_reconcile_gpu(self):
        """No opted-in pod owns the GPU, so the cycle must release our cap; a
        release that could not be confirmed leaves a cap possibly stranded."""
        actuator = actuator_double()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "GPU-A"
        actuator.list_running_pids.return_value = [1234]
        actuator.restore_default_by_uuid.return_value = False
        del actuator.managed_uuid_for_idx  # NVML-style: index-stable
        power_agent._previously_managed.add("GPU-A")
        agent = self._gpu_agent(actuator)

        with patch("power_agent._extract_pod_uid_from_cgroup", return_value=None):
            result = agent._reconcile_gpu(0, {})

        self.assertIs(result, False)

    def test_failed_uuid_read_during_release_returns_false(self):
        actuator = MagicMock()
        actuator.get_uuid.side_effect = RuntimeError("identity unreadable")

        self.assertIs(power_agent._release_managed_gpu(actuator, 0), False)

    def test_gpu_we_never_capped_is_a_successful_no_op(self):
        actuator = MagicMock()
        actuator.get_uuid.return_value = "GPU-NOT-OURS"

        self.assertIs(power_agent._release_managed_gpu(actuator, 0), True)

    def test_failed_managed_uuid_lookup_returns_false(self):
        actuator = _DcgmStyleActuator(
            current_uuid="GPU-A", managed_uuid=RuntimeError("lookup failed")
        )
        power_agent._managed_gpu_indices.add(0)

        self.assertIs(power_agent._release_managed_gpu(actuator, 0), False)
        actuator.restore_default_by_uuid.assert_not_called()

    def test_raising_restore_returns_false(self):
        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "GPU-A"
        actuator.managed_uuid_for_idx.return_value = "GPU-A"
        actuator.restore_default_by_uuid.side_effect = RuntimeError("restore failed")
        power_agent._previously_managed.add("GPU-A")

        self.assertIs(power_agent._release_managed_gpu(actuator, 0), False)

    def test_completed_release_returns_true(self):
        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "GPU-A"
        actuator.managed_uuid_for_idx.return_value = "GPU-A"
        actuator.restore_default_by_uuid.return_value = True
        power_agent._previously_managed.add("GPU-A")

        with patch.object(power_agent, "_commit_release"):
            self.assertIs(power_agent._release_managed_gpu(actuator, 0), True)

    def test_stale_projection_skip_returns_false(self):
        """A dcgm re-enumeration moved the GPU we capped off this index and put
        an unrelated GPU here. The displaced GPU is left unassessed, and the
        GPU loop only moves forward — so a cycle that skipped a managed GPU must
        not publish a fresh successful timestamp.

        Precondition matters: the occupant must be a GPU we have NEVER capped.
        An occupant already in `_previously_managed` takes the UUID-addressed
        release instead, and an index that is neither managed nor carrying a
        managed UUID returns at the "not ours" guard.
        """
        actuator = _DcgmStyleActuator(
            current_uuid="GPU-OCCUPANT",  # never capped by us
            managed_uuid="GPU-DISPLACED",
        )
        power_agent._managed_gpu_indices.add(0)

        self.assertIs(power_agent._release_managed_gpu(actuator, 0), False)
        # Skipped WITHOUT restoring or pruning — the displaced GPU is assessed
        # at its current index.
        actuator.restore_default_by_uuid.assert_not_called()


if __name__ == "__main__":
    unittest.main()

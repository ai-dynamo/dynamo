# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in scope: the agent only caps GPUs owned by annotated pods (PR #9682 review).

`_build_uid_to_annotation` is scope-by-annotation-key: a pod is in scope only
if it carries ``dynamo.nvidia.com/gpu-power-limit``. A GPU running only
unannotated pods — a co-located non-Dynamo workload, or a Dynamo worker the
planner has not yet annotated — must be left at its hardware default and never
written, instead of being silently capped to the safe default.

A pod that *does* carry the key but with a malformed value stays in scope so
the safe-default fail-safe still protects a genuinely-managed pod.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import POWER_ANNOTATION_KEY, PowerAgent

SAFE_DEFAULT = 500


def _pod(uid: str, annotations):
    """Fake K8s pod object exposing ``metadata.uid`` / ``metadata.annotations``."""
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(uid=uid, annotations=annotations)
    )


def _make_agent(device_count: int = 1) -> PowerAgent:
    """Build a PowerAgent without touching NVML / K8s (bypass __init__)."""
    agent = object.__new__(PowerAgent)
    agent.node_name = "node-under-test"
    agent.k8s_namespace = None
    agent.device_count = device_count
    agent.safe_default_watts = SAFE_DEFAULT
    agent.metrics = MagicMock()
    return agent


class TestBuildUidToAnnotationScope(unittest.TestCase):
    def test_unannotated_pods_are_omitted(self):
        agent = _make_agent()
        pods = [
            _pod("annotated", {POWER_ANNOTATION_KEY: "480"}),
            _pod("no-annotations-at-all", None),
            _pod("other-annotations", {"team.example.com/foo": "bar"}),
        ]
        mapping = agent._build_uid_to_annotation(pods)
        self.assertEqual(mapping, {"annotated": "480"})
        self.assertNotIn("no-annotations-at-all", mapping)
        self.assertNotIn("other-annotations", mapping)

    def test_malformed_value_stays_in_scope(self):
        """Key present but value broken/empty → kept, so the fail-safe still fires."""
        agent = _make_agent()
        pods = [
            _pod("bad", {POWER_ANNOTATION_KEY: "not-a-number"}),
            _pod("empty", {POWER_ANNOTATION_KEY: ""}),
        ]
        mapping = agent._build_uid_to_annotation(pods)
        self.assertEqual(mapping, {"bad": "not-a-number", "empty": ""})


class TestReconcileScope(unittest.TestCase):
    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_unannotated_gpu_active_pod_is_left_untouched(self):
        """A GPU whose only live process belongs to an unannotated pod gets
        NO cap write — not even the safe default. Reconcile is actuator-routed,
        so the assertion is on the actuator, not raw NVML."""
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ):
            agent._reconcile_gpu(0, uid_to_annotation)

        # Not opted-in and not previously managed → no cap, no release write.
        actuator.apply_cap.assert_not_called()
        actuator.restore_default.assert_not_called()

    def test_annotated_gpu_active_pod_is_capped(self):
        """Happy path still works: an annotated pod's GPU is capped to its
        value through the active actuator."""
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent._actuator = actuator
        pods = [_pod("worker", {POWER_ANNOTATION_KEY: "480"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch("power_agent._extract_pod_uid_from_cgroup", return_value="worker"):
            agent._reconcile_gpu(0, uid_to_annotation)

        # Cap write flows through the actuator at the annotated value.
        actuator.apply_cap.assert_called_once_with(
            0, 480, expected_uuid=actuator.get_uuid.return_value
        )


class TestReleaseOnReuse(unittest.TestCase):
    """A previously-managed GPU now running only unannotated work is released
    back to default, instead of stranding a stale cap on the new tenant."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _run_reconcile_with_unannotated_pod(self, current_w, default_w):
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        actuator.get_uuid.return_value = "GPU-A"
        actuator.default_w.return_value = default_w
        actuator.current_w.return_value = current_w
        # The runtime release delegates the restore decision to the actuator's
        # atomic by-UUID restore. Mirror its contract: True when a live cap was
        # restored (current < default), None when already at/above default.
        actuator.restore_default_by_uuid.return_value = (
            True if current_w < default_w else None
        )
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._persist_managed_gpus"):
            agent._reconcile_gpu(0, uid_to_annotation)

        return actuator

    def test_previously_managed_gpu_is_released_to_default(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        # Restored to default via the actuator's by-UUID path, never re-capped,
        # and unmanaged.
        actuator.restore_default_by_uuid.assert_called_once_with("GPU-A")
        actuator.apply_cap.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_previously_managed_across_restart_is_released(self):
        """After a restart `_managed_gpu_indices` is empty; the persisted UUID
        set is the only signal. A busy GPU we capped before the restart must
        still be released (startup orphan recovery skips busy GPUs)."""
        # No _managed_gpu_indices entry (cleared on restart); only persisted UUID.
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-A")
        actuator.apply_cap.assert_not_called()
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_never_managed_gpu_is_not_touched(self):
        # Neither in _managed_gpu_indices nor _previously_managed → not ours.
        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        actuator.restore_default.assert_not_called()
        actuator.restore_default_by_uuid.assert_not_called()
        actuator.apply_cap.assert_not_called()

    def test_release_unmanages_even_if_already_at_default(self):
        """If the cap was already cleared externally, still drop it from the
        managed set. The by-UUID restore returns None (already at default), so
        no cap write is issued, but ownership is still retired."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=700, default_w=700
        )

        # None result => nothing of ours to restore, but ownership is retired.
        self.assertIsNone(actuator.restore_default_by_uuid.return_value)
        actuator.restore_default.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)


class _FakeDcgmActuator:
    """Actuator exposing the dcgm-only ``managed_uuid_for_idx`` helper.

    The MagicMock-based cases above stay on the NVML-equivalent branch because
    ``MagicMock``'s *type* has no ``managed_uuid_for_idx`` attribute, so they
    cannot exercise re-enumeration-aware pruning or the ``restore_default`` ->
    ``False`` skip. This fake makes both branches reachable.
    """

    name = "dcgm"

    def __init__(
        self,
        *,
        current_uuid,
        managed_uuid,
        current_w,
        default_w,
        restore_result=True,
    ):
        self._current_uuid = current_uuid
        self._managed_uuid = managed_uuid
        self._current_w = current_w
        self._default_w = default_w
        self._restore_result = restore_result
        self.list_running_pids = MagicMock(return_value=[1234])
        self.apply_cap = MagicMock()
        # Index-addressed restore must NOT be used by the runtime release path
        # any more; the by-UUID list is what the same-GPU branch now drives.
        self.restore_calls = []
        self.restore_by_uuid_calls = []

    def get_uuid(self, gpu_idx):
        return self._current_uuid

    def managed_uuid_for_idx(self, gpu_idx):
        return self._managed_uuid

    def default_w(self, gpu_idx):
        return self._default_w

    def current_w(self, gpu_idx):
        return self._current_w

    def restore_default(self, gpu_idx):
        self.restore_calls.append(gpu_idx)
        return self._restore_result

    def restore_default_by_uuid(self, uuid):
        self.restore_by_uuid_calls.append(uuid)
        return self._restore_result


class TestReleaseDcgmReenumeration(unittest.TestCase):
    """dcgm-path release correctness: a hostengine re-enumeration must not let
    ``_release_managed_gpu`` prune the wrong UUID or drop retry state when the
    actuator could not conclusively restore the cap."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _run(self, actuator):
        agent = _make_agent(device_count=1)
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)
        persist = MagicMock()
        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._persist_managed_gpus", persist):
            agent._reconcile_gpu(0, uid_to_annotation)
        return persist

    def test_reenumerated_occupant_does_not_release_displaced_gpu(self):
        """Idx 0 now hosts GPU-B (re-enumerated), but we
        capped GPU-A which moved elsewhere. The unannotated workload observed
        here belongs to GPU-B, NOT to GPU-A — it is no evidence that GPU-A
        should be released (GPU-A may still be running its annotated workload
        at its new index). Release must SKIP the stale index→UUID projection:
        no restore, and GPU-A's ownership state is preserved so its own index
        reconcile (or UUID-keyed SIGTERM / startup recovery) decides its fate.
        """
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-B",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=True,
        )

        persist = self._run(actuator)

        # No relocation-restore of the displaced GPU, and state untouched.
        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_reenumerated_occupant_skip_is_independent_of_occupant_watts(self):
        """The skip on a re-enumerated occupant does not
        depend on the current occupant's watts. Even when GPU-B reads at
        default (which the pre-fix code used as a trigger to relocate a restore
        onto GPU-A), a managed_uuid≠occupant mismatch must skip: the evidence
        still belongs to GPU-B, not GPU-A."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-B",
            managed_uuid="GPU-A",
            current_w=700,
            default_w=700,
            restore_result=True,
        )

        persist = self._run(actuator)

        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_same_gpu_release_still_restores_and_prunes(self):
        """The guard must not regress the legitimate case.
        When the current occupant IS the GPU we capped here (no re-enumeration,
        managed_uuid == current UUID) and it now runs only unannotated work, we
        still restore and prune it — now via the atomic, identity-bound
        `restore_default_by_uuid` (which closes the A->B->A read hole), NOT the
        index-addressed `restore_default`."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-A",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=True,
        )

        persist = self._run(actuator)

        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])
        self.assertEqual(actuator.restore_calls, [])
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        persist.assert_called_once()

    def test_release_bails_when_occupant_differs_from_snapshot(self):
        """If the index re-enumerated between the pre-PID
        snapshot identity (`expected_uuid`) and the release check, the "no
        annotated pod" evidence no longer describes the current occupant, so
        release must bail without touching state — even when the current
        occupant is itself the GPU historically capped here."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-A",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=True,
        )

        with patch("power_agent._persist_managed_gpus") as persist:
            # Snapshot saw GPU-Z; the index now reads GPU-A → mismatch → bail.
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-Z")

        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_same_gpu_release_delegates_aba_decision_to_actuator(self):
        """The same-GPU release must NOT perform its
        own separate default_w()/current_w()/get_uuid() reads — that left an
        A->B->A hole where a reconnect mid-sequence paired GPU-A's default with
        GPU-B's 'already at default' current and pruned GPU-A's still-live cap.
        The decision is now delegated to the actuator's atomic, identity-bound
        `restore_default_by_uuid`, and when THAT returns False (it detected the
        re-enumeration and could not conclusively restore), release must retain
        ownership and must not touch the index-addressed `restore_default`."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        class _AtomicRestoreInconclusive:
            name = "dcgm"

            def __init__(self):
                self.restore_calls = []
                self.restore_by_uuid_calls = []

            def get_uuid(self, gpu_idx):
                return "GPU-A"

            def managed_uuid_for_idx(self, gpu_idx):
                return "GPU-A"

            def restore_default_by_uuid(self, uuid):
                # The actuator's atomic snapshot detected the mid-write
                # re-enumeration (or an unreadable identity) and refused to
                # prune a possibly-live cap.
                self.restore_by_uuid_calls.append(uuid)
                return False

            def restore_default(self, gpu_idx):  # pragma: no cover
                self.restore_calls.append(gpu_idx)
                return True

            def default_w(self, gpu_idx):  # pragma: no cover
                raise AssertionError(
                    "same-GPU release must delegate to restore_default_by_uuid, "
                    "not read default_w/current_w separately"
                )

            def current_w(self, gpu_idx):  # pragma: no cover
                raise AssertionError(
                    "same-GPU release must delegate to restore_default_by_uuid, "
                    "not read default_w/current_w separately"
                )

        actuator = _AtomicRestoreInconclusive()
        with patch("power_agent._persist_managed_gpus") as persist:
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-A")

        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])
        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_managed_uuid_lookup_failure_fails_closed_on_dcgm(self):
        """If the DCGM managed-identity lookup
        raises, release must NOT fall back to the current occupant. On a
        re-enumerated index that fallback would make managed_uuid == uuid, pass
        the stale-projection guard, and release/prune on stale integer
        membership alone. Fail closed: retain ownership and retry next cycle."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        class _LookupRaises:
            name = "dcgm"

            def __init__(self):
                self.restore_calls = []

            def get_uuid(self, gpu_idx):
                return "GPU-B"  # re-enumerated occupant now on index 0

            def managed_uuid_for_idx(self, gpu_idx):
                raise RuntimeError("hostengine reconnecting")

            def default_w(self, gpu_idx):
                return 700

            def current_w(self, gpu_idx):
                return 400

            def restore_default(self, gpu_idx):
                self.restore_calls.append(gpu_idx)
                return True

        actuator = _LookupRaises()
        with patch("power_agent._persist_managed_gpus") as persist:
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-B")

        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_index_swap_releases_both_managed_gpus_by_uuid(self):
        """Two managed GPUs that swap indices and
        both become unannotated must BOTH be released. After the swap each
        index's recorded map points at the OTHER GPU, so an index-based skip
        would block both releases until shutdown. Release must consider the
        current occupant's UUID independently and restore it via the
        UUID-addressed path."""
        power_agent._managed_gpu_indices.update({0, 1})
        power_agent._previously_managed.update({"GPU-A", "GPU-B"})

        class _SwapFake:
            name = "dcgm"

            def __init__(self):
                # Post-swap occupants vs the map recorded at cap time.
                self._current = {0: "GPU-B", 1: "GPU-A"}
                self._recorded = {0: "GPU-A", 1: "GPU-B"}
                self.restored_uuids = []

            def get_uuid(self, gpu_idx):
                return self._current[gpu_idx]

            def managed_uuid_for_idx(self, gpu_idx):
                return self._recorded[gpu_idx]

            def restore_default_by_uuid(self, uuid):
                self.restored_uuids.append(uuid)
                return True

            def restore_default(self, gpu_idx):  # pragma: no cover
                raise AssertionError(
                    "swap release must use restore_default_by_uuid (UUID-"
                    "addressed), not restore_default (index-addressed)"
                )

            def default_w(self, gpu_idx):
                return 700

            def current_w(self, gpu_idx):
                return 400

        actuator = _SwapFake()
        with patch("power_agent._persist_managed_gpus"):
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-B")
            power_agent._release_managed_gpu(actuator, 1, expected_uuid="GPU-A")

        # Each GPU restored by its OWN uuid (not the stale index map), and both
        # pruned from ownership state.
        self.assertEqual(sorted(actuator.restored_uuids), ["GPU-A", "GPU-B"])
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        self.assertNotIn("GPU-B", power_agent._previously_managed)
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn(1, power_agent._managed_gpu_indices)

    def test_reenumerated_unmanaged_occupant_by_uuid_path_still_skips(self):
        """The UUID-addressed release only fires when the CURRENT occupant is
        itself managed. If a re-enumeration dropped an UNRELATED GPU onto the
        index, release must still skip (no restore, state preserved) — the GPU
        we capped is handled at its own current index."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        class _UnrelatedOccupant:
            name = "dcgm"

            def __init__(self):
                self.restored_uuids = []
                self.restore_calls = []

            def get_uuid(self, gpu_idx):
                return "GPU-UNRELATED"

            def managed_uuid_for_idx(self, gpu_idx):
                return "GPU-A"

            def restore_default_by_uuid(self, uuid):  # pragma: no cover
                self.restored_uuids.append(uuid)
                return True

            def restore_default(self, gpu_idx):  # pragma: no cover
                self.restore_calls.append(gpu_idx)
                return True

            def default_w(self, gpu_idx):
                return 700

            def current_w(self, gpu_idx):
                return 400

        actuator = _UnrelatedOccupant()
        with patch("power_agent._persist_managed_gpus") as persist:
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-UNRELATED")

        self.assertEqual(actuator.restored_uuids, [])
        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()

    def test_restore_skipped_keeps_ownership_for_retry(self):
        """``restore_default_by_uuid`` returning False means the cap is still
        live but could not be located conclusively. State must be preserved (not
        pruned/persisted) so a later reconcile or startup orphan recovery
        retries."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-A",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=False,
        )

        persist = self._run(actuator)

        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])
        self.assertEqual(actuator.restore_calls, [])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()


class TestReleaseRetiresActuatorOwnership(unittest.TestCase):
    """A successful runtime release must tell the
    actuator to retire the UUID from its in-memory ownership set, so a later
    SIGTERM sweep no longer treats a cap we relinquished as ours (and cannot
    clobber a cap another workflow installs on that GPU afterwards)."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    class _RetiringActuator:
        name = "dcgm"

        def __init__(self):
            self.retired = []
            self.restore_calls = []
            self.restore_by_uuid_calls = []

        def get_uuid(self, gpu_idx):
            return "GPU-A"

        def managed_uuid_for_idx(self, gpu_idx):
            return "GPU-A"

        def default_w(self, gpu_idx):
            return 700

        def current_w(self, gpu_idx):
            return 400

        def restore_default(self, gpu_idx):  # pragma: no cover
            self.restore_calls.append(gpu_idx)
            return True

        def restore_default_by_uuid(self, uuid):
            self.restore_by_uuid_calls.append(uuid)
            return True

        def retire_managed_uuid(self, uuid):
            self.retired.append(uuid)

    def test_successful_release_retires_ownership(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = self._RetiringActuator()

        with patch("power_agent._persist_managed_gpus"):
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-A")

        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])
        self.assertEqual(actuator.retired, ["GPU-A"])
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_retire_failure_retains_ownership_and_does_not_prune(self):
        """Retirement must NOT be swallowed: if it fails, `_capped_uuids` still
        holds the released UUID, so pruning the persisted record anyway would
        re-arm the shutdown-sweep clobber. Instead the failure propagates
        (caught by `reconcile_once`'s per-GPU guard) and NOTHING is pruned, so
        a later cycle retries with ownership fully intact."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = self._RetiringActuator()
        actuator.retire_managed_uuid = MagicMock(side_effect=RuntimeError("boom"))

        with patch("power_agent._persist_managed_gpus") as persist:
            with self.assertRaises(RuntimeError):
                power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-A")

        # Retirement ran first and failed, so ownership state is untouched.
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()


class TestReleasePersistenceFailureIsRetryable(unittest.TestCase):
    """If the durable prune fails after a runtime release, the
    hardware is already at default and in-memory ownership is retired, so the
    GPU must NOT be eligible for another release (a repeated restore could erase
    a cap another workflow installed on it in the interim). The pending prune is
    retried — persistence ONLY — by `_flush_pending_retirements` next cycle."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()
        power_agent._pending_retirement.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()
        power_agent._pending_retirement.clear()

    class _CountingActuator:
        name = "dcgm"

        def __init__(self):
            self.restore_by_uuid_calls = []

        def get_uuid(self, gpu_idx):
            return "GPU-A"

        def managed_uuid_for_idx(self, gpu_idx):
            return "GPU-A"

        def restore_default_by_uuid(self, uuid):
            self.restore_by_uuid_calls.append(uuid)
            return True

    def test_persist_failure_defers_retirement_without_repeating_restore(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = self._CountingActuator()

        # Cycle 1: the release restores hardware, but the durable prune fails.
        with patch(
            "power_agent._persist_managed_gpus",
            side_effect=OSError("state volume unavailable"),
        ):
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-A")

        # Hardware restored exactly once; in-memory ownership retired so the GPU
        # is no longer eligible for release; the durable prune is pending.
        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        self.assertIn("GPU-A", power_agent._pending_retirement)

        # Another workflow now caps the SAME GPU after our release. The next
        # cycle must NOT touch hardware again: the flush retries ONLY the
        # persistence, and the release path skips (the GPU is no longer ours).
        with patch("power_agent._persist_managed_gpus") as persist:
            power_agent._flush_pending_retirements()
            power_agent._release_managed_gpu(actuator, 0, expected_uuid="GPU-A")

        persist.assert_called_once()  # the flush's retry only
        self.assertEqual(power_agent._pending_retirement, set())
        # No second hardware restore — the other workflow's cap is untouched.
        self.assertEqual(actuator.restore_by_uuid_calls, ["GPU-A"])


if __name__ == "__main__":
    unittest.main()

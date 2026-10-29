# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live retune reconciler (segment, window, temp-file and
DGDR Request garbage collection) and its ``SweepRegistry``."""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import pytest

from dynamo.planner.config.planner_config import DgdrExecutorConfig, RetuneConfig
from dynamo.planner.retune.live.contracts import (
    LABEL_DGD,
    LABEL_DGD_UID,
    LABEL_WINDOW,
    DgdRef,
    KubeError,
    PodInfo,
)
from dynamo.planner.retune.live.reconciler import (
    JOB_NAME_LABEL,
    RUN_GENERATION_ANNOTATION,
    Reconciler,
    window_file_name,
    window_relpath,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

DGD = DgdRef(name="dgd", namespace="ns", uid="uid-now", dynamo_namespace="dyn-ns")
NOW = 1_800_000_000.0
INTERVAL = 3600.0
GRACE = 120.0
TIMEOUT = 1800.0
MARGIN = 600.0
SEGMENT_AGE = INTERVAL + GRACE + TIMEOUT + MARGIN  # 6120
WINDOW_AGE = INTERVAL + TIMEOUT  # 5400
TEMP_AGE = 900.0 + 300.0  # cut_budget + temp margin
OLD = 100_000.0  # older than every threshold


def window_ms(k: int) -> int:
    """End of the k-th window before NOW's current one, in epoch ms."""

    return int((NOW - k * INTERVAL) * 1000)


def deadline_for(window_end_ms: int) -> float:
    return window_end_ms / 1000 + GRACE + TIMEOUT


# ------------------------------------------------------------------ fakes


class FakeKube:
    namespace = DGD.namespace

    def __init__(self) -> None:
        self.requests: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.job_pods: dict[str, list[PodInfo]] = {}
        self.deleted: list[str] = []
        self.selectors: list[str] = []
        self.get_run_calls: list[str] = []
        self.fail_list = False
        self.fail_get_run = False
        self.fail_delete: set[str] = set()
        self.on_delete: Optional[Callable[[str], None]] = None

    def list_requests(self, label_selector: str, *, timeout_s: float = 10.0):
        self.selectors.append(label_selector)
        if self.fail_list:
            raise KubeError("list failed")
        return [dict(obj) for obj in self.requests.values()]

    def delete_request(
        self, name: str, *, propagation: str = "Background", timeout_s: float = 10.0
    ):
        if name in self.fail_delete:
            raise KubeError("delete failed")
        if self.on_delete is not None:
            self.on_delete(name)
        self.deleted.append(name)
        self.requests.pop(name, None)

    def get_run(self, name: str, *, timeout_s: float = 10.0):
        self.get_run_calls.append(name)
        if self.fail_get_run:
            raise KubeError("get failed")
        return self.runs.get(name)

    def list_pods(self, label_selector: str, *, timeout_s: float = 10.0):
        key, _, value = label_selector.partition("=")
        assert key == JOB_NAME_LABEL
        return list(self.job_pods.get(value, []))

    # -- builders

    def add_request(
        self,
        name: str,
        window: int,
        *,
        dgd_uid: Optional[str] = DGD.uid,
        state: str = "completed",  # completed | running | creating | invalid
        job_pods: int = 0,
        job_pod_phase: Optional[str] = None,
        claim: str = "traces",
        path: Optional[str] = None,
        generation: int = 1,
    ) -> None:
        labels = {LABEL_DGD: DGD.name, LABEL_WINDOW: str(window)}
        if dgd_uid is not None:
            labels[LABEL_DGD_UID] = dgd_uid
        status: dict[str, Any] = {"conditions": []}
        run_name = f"{name}-{generation}"
        job = f"{run_name}-job"
        if state in ("completed", "running"):
            completed = state == "completed"
            self.runs[run_name] = {
                "metadata": {
                    "name": run_name,
                    "annotations": {RUN_GENERATION_ANNOTATION: str(generation)},
                },
                "status": {
                    "jobRef": {"name": job},
                    "conditions": [
                        {
                            "type": "Completed",
                            "status": "True" if completed else "False",
                            "reason": "SearchSucceeded" if completed else "JobRunning",
                        }
                    ],
                },
            }
            if completed:
                status["recentRunRefs"] = [{"name": run_name}]
            else:
                status["activeRunRef"] = {"name": run_name}
            self.job_pods[job] = [
                PodInfo(
                    name=f"{job}-{i}",
                    uid=str(i),
                    ip=None,
                    ready=False,
                    terminating=False,
                    phase=job_pod_phase,
                )
                for i in range(job_pods)
            ]
            accepted = ("True", "RunCreated")
        elif state == "invalid":
            accepted = ("False", "InvalidSpec")
        else:
            accepted = ("False", "RunCreating")
            status["activeRunRef"] = {"name": run_name}
        status["conditions"].append(
            {
                "type": "Accepted",
                "status": accepted[0],
                "reason": accepted[1],
                "observedGeneration": generation,
            }
        )
        self.requests[name] = {
            "metadata": {
                "name": name,
                "uid": f"{name}-uid",
                "generation": generation,
                "labels": labels,
            },
            "spec": {
                "workload": {
                    "trace": {
                        "format": "dynamo",
                        "source": {
                            "pvc": {
                                "claimName": claim,
                                "path": path or f"windows/dgd/{window}.jsonl.gz",
                            }
                        },
                    }
                }
            },
            "status": status,
        }

    def finish_job(self, name: str) -> None:
        for run in self.runs.values():
            if run["metadata"]["name"].startswith(name):
                self.job_pods[run["status"]["jobRef"]["name"]] = []


class RecordingMetrics:
    def __init__(self) -> None:
        self.sinks: list[tuple[str, bool]] = []
        self.fail_sinks = False

    def trace_sink_missing(self, pod: str, missing: bool) -> None:
        if self.fail_sinks:
            raise RuntimeError("metrics backend failed")
        self.sinks.append((pod, missing))

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def make(
    tmp_path: Path, *, keep_requests: int = 2
) -> tuple[Reconciler, FakeKube, RecordingMetrics]:
    cfg = DgdrExecutorConfig(
        claim_name="traces",
        model="Qwen/Qwen3-0.6B",
        sku="h200_sxm",
        engine={},
        cut_budget_seconds=900,
        keep_requests=keep_requests,
    )
    retune = RetuneConfig(
        interval_seconds=INTERVAL,
        grace_seconds=GRACE,
        timeout_seconds=TIMEOUT,
        executor="dgdr",
        dgdr=cfg,
    )
    kube = FakeKube()
    metrics = RecordingMetrics()
    reconciler = Reconciler(
        kube,
        DGD,
        cfg,
        retune,
        str(tmp_path),
        metrics,
        now=lambda: NOW,
        visibility_margin_s=MARGIN,
    )
    return reconciler, kube, metrics


def touch(path: Path, age_s: float, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (NOW - age_s, NOW - age_s))
    return path


def age_dir(path: Path, age_s: float) -> None:
    os.utime(path, (NOW - age_s, NOW - age_s))


def frontend(name: str, *, ready: bool = True, terminating: bool = False) -> PodInfo:
    return PodInfo(
        name=name,
        uid=f"{name}-uid",
        ip="10.0.0.1",
        ready=ready,
        terminating=terminating,
    )


def run_once(
    reconciler: Reconciler,
    *,
    last: Optional[int] = None,
    due: Optional[int] = None,
    pods: Optional[list[PodInfo]] = None,
    should_continue: Optional[Callable[[], bool]] = None,
):
    return reconciler.run_once(
        last_window_end_ms=last,
        due_window_end_ms=due,
        deadline_for=deadline_for,
        frontend_pods=pods,
        should_continue=should_continue,
    )


# ----------------------------------------------------------------- segments


def test_segments_of_a_live_pod_keep_the_newest_and_recent_ones(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    pod_dir = Path(reconciler.segments_root) / "fe-0"
    old0 = touch(pod_dir / "req.000000.jsonl.gz", OLD)
    recent = touch(pod_dir / "req.000001.jsonl.gz", SEGMENT_AGE - 60)
    old2 = touch(pod_dir / "req.000002.jsonl.gz", OLD)
    newest = touch(pod_dir / "req.000010.jsonl.gz", OLD)  # numeric, not lexical
    other = touch(pod_dir / "notes.txt", OLD)

    report = run_once(reconciler, pods=[frontend("fe-0")])

    assert report.deleted_segments == 2
    assert not old0.exists() and not old2.exists()
    assert recent.exists() and newest.exists() and other.exists()
    assert report.errors == {}


def test_directory_of_a_gone_pod_goes_only_when_everything_is_old(
    tmp_path: Path,
) -> None:
    reconciler, _, _ = make(tmp_path)
    root = Path(reconciler.segments_root)
    touch(root / "gone-old" / "req.000000.jsonl.gz", OLD)
    touch(root / "gone-old" / "req.000001.jsonl.gz", SEGMENT_AGE + 1)
    age_dir(root / "gone-old", OLD)
    touch(root / "gone-recent" / "req.000000.jsonl.gz", OLD)
    touch(root / "gone-recent" / "req.000001.jsonl.gz", 10)
    age_dir(root / "gone-recent", OLD)
    touch(root / "fe-0" / "req.000000.jsonl.gz", 10)

    report = run_once(reconciler, pods=[frontend("fe-0")])

    assert report.deleted_pod_dirs == ("gone-old",)
    assert not (root / "gone-old").exists()
    assert (root / "gone-recent").exists()
    assert (root / "fe-0").exists()


@pytest.mark.parametrize("pods", [None, []])
def test_no_segment_is_deleted_without_a_known_pod_set(tmp_path: Path, pods) -> None:
    reconciler, _, _ = make(tmp_path)
    root = Path(reconciler.segments_root)
    seg = touch(root / "fe-0" / "req.000000.jsonl.gz", OLD)
    touch(root / "fe-0" / "req.000001.jsonl.gz", OLD)
    age_dir(root / "fe-0", OLD)

    report = run_once(reconciler, pods=pods)

    assert report.deleted_segments == 0 and report.deleted_pod_dirs == ()
    assert seg.exists()


def test_trace_sink_missing_is_reported_and_cleared(tmp_path: Path) -> None:
    reconciler, _, metrics = make(tmp_path)
    root = Path(reconciler.segments_root)
    touch(root / "fe-0" / "req.000000.jsonl.gz", 0)
    pods = [
        frontend("fe-0"),
        frontend("fe-1"),
        frontend("starting", ready=False),
        frontend("leaving", terminating=True),
    ]

    report = run_once(reconciler, pods=pods)
    assert report.sink_missing == ("fe-1",)
    assert sorted(metrics.sinks) == [("fe-0", False), ("fe-1", True)]

    metrics.sinks.clear()
    (root / "fe-1").mkdir()
    report = run_once(reconciler, pods=pods)
    assert report.sink_missing == ()
    assert ("fe-1", False) in metrics.sinks

    # A pod that leaves has its gauge cleared.
    metrics.sinks.clear()
    run_once(reconciler, pods=[frontend("fe-0")])
    assert sorted(metrics.sinks) == [("fe-0", False), ("fe-1", False)]

    # An unknown pod set reports nothing.
    metrics.sinks.clear()
    assert run_once(reconciler, pods=None).sink_missing == ()
    assert metrics.sinks == []


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a directory whatever its mode",
)
def test_an_unreadable_pod_directory_does_not_stop_the_others(tmp_path: Path) -> None:
    reconciler, kube, metrics = make(tmp_path, keep_requests=0)
    root = Path(reconciler.segments_root)
    names = [f"fe-{i}" for i in range(6)]
    for name in names:
        touch(root / name / "req.000000.jsonl.gz", OLD)
        touch(root / name / "req.000001.jsonl.gz", OLD)
        touch(root / name / "req.000002.jsonl.gz", OLD)
    kube.add_request("rt-old", window_ms(6), state="completed")
    # Created by a frontend running as another uid, or with umask 077.
    locked = root / "fe-3"
    locked.chmod(0)
    try:
        report = run_once(
            reconciler, due=window_ms(0), pods=[frontend(name) for name in names]
        )
    finally:
        locked.chmod(stat.S_IRWXU)

    # Every other pod lost its two old segments, whatever the scandir order.
    assert report.deleted_segments == 2 * (len(names) - 1)
    assert len(list(locked.iterdir())) == 3
    assert report.errors == {"segments": 1}
    # The sink check and the Request GC still ran.
    assert sorted(metrics.sinks) == [(name, False) for name in names]
    assert report.deleted_requests == ("rt-old",)


def test_a_failing_sink_check_keeps_the_segment_counts(tmp_path: Path) -> None:
    reconciler, _, metrics = make(tmp_path)
    pod_dir = Path(reconciler.segments_root) / "fe-0"
    touch(pod_dir / "req.000000.jsonl.gz", OLD)
    touch(pod_dir / "req.000001.jsonl.gz", OLD)
    metrics.fail_sinks = True

    report = run_once(reconciler, pods=[frontend("fe-0")])

    assert report.deleted_segments == 1
    assert report.errors == {"sinks": 1}


def test_only_the_top_level_segments_of_a_live_pod_are_collected(
    tmp_path: Path,
) -> None:
    # P10 requires the sink's file prefix directly under the pod directory,
    # so the reconciler does not walk subdirectories of a live pod.
    reconciler, _, _ = make(tmp_path)
    pod_dir = Path(reconciler.segments_root) / "fe-0"
    nested = [touch(pod_dir / "sub" / f"req.00000{i}.jsonl.gz", OLD) for i in range(3)]
    top = [touch(pod_dir / f"req.00000{i}.jsonl.gz", OLD) for i in range(3)]

    report = run_once(reconciler, pods=[frontend("fe-0")])

    assert report.deleted_segments == 2
    assert [path.exists() for path in top] == [False, False, True]
    assert all(path.exists() for path in nested)


# ------------------------------------------------------------------ windows


def window_path(reconciler: Reconciler, window: int) -> Path:
    return Path(reconciler.windows_root) / window_file_name(window)


def test_protected_windows_are_never_deleted(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    last, due = window_ms(2), int((NOW - 60) * 1000)
    assert NOW >= deadline_for(last) and NOW < deadline_for(due)
    last_path = touch(window_path(reconciler, last), OLD)
    due_path = touch(window_path(reconciler, due), OLD)
    stale = window_ms(5)
    stale_path = touch(window_path(reconciler, stale), OLD)
    registry_window = window_ms(7)
    registry_path = touch(window_path(reconciler, registry_window), OLD)
    reconciler.protect(registry_window, [])

    report = run_once(reconciler, last=last, due=due, pods=[frontend("fe-0")])

    assert due_path.exists()  # undecided and before its deadline
    assert not last_path.exists()  # in the range, but past its deadline
    assert registry_path.exists()  # this process's sweep uses it
    assert not stale_path.exists()
    assert report.deleted_windows == tuple(sorted((last, stale)))

    reconciler.release(registry_window)
    report = run_once(reconciler, last=last, due=due)
    assert report.deleted_windows == (registry_window,)
    assert due_path.exists()


def test_range_protection_uses_the_window_deadline(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    # Two undecided windows, both before their deadline.
    young = int((NOW - 60) * 1000)
    older = int((NOW - 1000) * 1000)
    paths = [touch(window_path(reconciler, w), OLD) for w in (young, older)]

    run_once(reconciler, last=older, due=young)
    assert all(path.exists() for path in paths)

    # Without a persisted last window, every window before the due one with
    # a live deadline is protected too.
    run_once(reconciler, last=None, due=young)
    assert all(path.exists() for path in paths)


def test_referenced_window_goes_once_every_referrer_is_finished(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    # A window outside the protected range whose deadline has not passed,
    # so only the referrer rule decides.
    window = int((NOW - 60) * 1000)
    path = touch(window_path(reconciler, window), 10)  # young: no fallback
    kube.add_request("rt-a", window, state="completed")
    kube.add_request("rt-b", window, state="running", job_pods=1)

    def round_():
        return run_once(reconciler, last=window_ms(0), due=window_ms(0))

    round_()
    assert path.exists()  # rt-b's Run is still active

    # Completed, but the Job pod is still there.
    kube.add_request("rt-b", window, state="completed", job_pods=1)
    round_()
    assert path.exists()

    kube.finish_job("rt-b")
    report = round_()
    assert report.deleted_windows == (window,)
    assert not path.exists()
    assert kube.deleted == []  # the Requests stay (keep_requests)


@pytest.mark.parametrize(
    ("phase", "deleted"),
    [("Succeeded", True), ("Failed", True), ("Running", False), (None, False)],
)
def test_finished_job_pods_of_a_completed_run_release_its_window(
    tmp_path: Path, phase: Optional[str], deleted: bool
) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    window = int((NOW - 60) * 1000)
    path = touch(window_path(reconciler, window), 10)  # young: no fallback
    kube.add_request("rt-a", window, state="completed", job_pods=1, job_pod_phase=phase)

    report = run_once(reconciler, last=window_ms(0), due=window_ms(0))
    assert (report.deleted_windows == (window,)) == deleted
    assert path.exists() != deleted


def test_invalid_spec_request_does_not_hold_its_window(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    window = window_ms(4)
    path = touch(window_path(reconciler, window), 10)
    kube.add_request("rt-a", window, state="invalid")

    report = run_once(reconciler, due=window_ms(0))
    assert report.deleted_windows == (window,)
    assert not path.exists()


def test_request_waiting_for_its_run_holds_its_window(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    window = window_ms(4)
    path = touch(window_path(reconciler, window), OLD)
    kube.add_request("rt-a", window, state="creating")
    assert path.exists()
    # The deadline has passed, so the Request itself is deleted; the window
    # file then has no referrer and is old.
    report = run_once(reconciler, due=window_ms(0))
    assert kube.deleted == ["rt-a"]
    assert report.deleted_windows == (window,)
    assert not path.exists()

    reconciler2, kube2, _ = make(tmp_path / "other", keep_requests=10)
    young_window = int((NOW - 60) * 1000)
    young_path = touch(window_path(reconciler2, young_window), OLD)
    kube2.add_request("rt-b", young_window, state="creating")
    report = run_once(reconciler2, due=None, last=None)
    assert young_path.exists()
    assert kube2.deleted == []


def test_unreferenced_window_needs_the_fallback_age(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    young = touch(window_path(reconciler, window_ms(5)), WINDOW_AGE - 60)
    old = touch(window_path(reconciler, window_ms(6)), WINDOW_AGE + 60)
    unknown = touch(Path(reconciler.windows_root) / "README", OLD)

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_windows == (window_ms(6),)
    assert young.exists() and not old.exists() and unknown.exists()


def test_windows_are_kept_when_requests_cannot_be_listed(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path)
    path = touch(window_path(reconciler, window_ms(6)), OLD)
    temp = touch(Path(reconciler.windows_root) / f".tmp-{window_ms(6)}-abc", OLD)
    kube.fail_list = True

    report = run_once(reconciler, due=window_ms(0))

    assert path.exists()
    assert not temp.exists()  # temp files do not depend on references
    assert report.errors == {"list_requests": 1}


def test_a_request_of_another_claim_does_not_reference_the_window(
    tmp_path: Path,
) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    window = window_ms(6)
    path = touch(window_path(reconciler, window), OLD)
    kube.add_request("rt-a", window, state="running", claim="other-claim")
    kube.add_request(
        "rt-b", window, state="running", path=f"windows/other-dgd/{window}.jsonl.gz"
    )
    # Both Requests are past their deadline and get deleted anyway; the
    # window had no referrer and is old.
    report = run_once(reconciler, due=window_ms(0))
    assert report.deleted_windows == (window,)
    assert not path.exists()


def test_temp_files(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    root = Path(reconciler.windows_root)
    protected = window_ms(1)
    reconciler.protect(protected, [])
    old = touch(root / f".tmp-{window_ms(3)}-a1", TEMP_AGE + 1)
    young = touch(root / f".tmp-{window_ms(3)}-b2", TEMP_AGE - 1)
    in_use = touch(root / f".tmp-{protected}-c3", OLD)
    odd = touch(root / ".tmp-garbage", TEMP_AGE + 1)

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_temp_files == 2
    assert not old.exists() and not odd.exists()
    assert young.exists() and in_use.exists()


# ----------------------------------------------------------------- requests


def test_terminal_requests_outside_the_newest_kept_windows_go(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=2)
    windows = [window_ms(k) for k in (5, 4, 3)]
    for i, window in enumerate(windows):
        kube.add_request(f"rt-{i}-a", window, state="completed")
        kube.add_request(f"rt-{i}-b", window, state="invalid")

    report = run_once(reconciler, due=window_ms(0))

    assert kube.selectors == [f"{LABEL_DGD}=dgd"]
    assert report.deleted_requests == ("rt-0-a", "rt-0-b")
    assert set(kube.requests) == {"rt-1-a", "rt-1-b", "rt-2-a", "rt-2-b"}


def test_keep_zero_still_keeps_requests_of_an_undecided_window(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    due = int((NOW - 60) * 1000)
    kube.add_request("rt-due", due, state="completed")
    kube.add_request("rt-old", window_ms(4), state="completed")

    report = run_once(reconciler, last=window_ms(4), due=due)

    # rt-due's window is undecided and before its deadline: the next sweep
    # adopts its Succeeded Request instead of recreating it.
    assert report.deleted_requests == ("rt-old",)


def test_non_terminal_request_goes_after_its_deadline(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    current = int((NOW - 60) * 1000)
    kube.add_request("rt-live", current, state="running", job_pods=1)
    kube.add_request("rt-late", window_ms(2), state="running", job_pods=1)

    report = run_once(reconciler, due=current)

    assert report.deleted_requests == ("rt-late",)


def test_requests_of_a_recreated_dgd_are_deleted(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    current = int((NOW - 60) * 1000)
    kube.add_request("rt-old-uid", current, dgd_uid="uid-before", state="running")
    kube.add_request("rt-no-uid", current, dgd_uid=None, state="running")
    kube.add_request("rt-ours", current, state="running")

    report = run_once(reconciler, due=current)

    assert report.deleted_requests == ("rt-old-uid",)


def test_delete_queue(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    current = int((NOW - 60) * 1000)
    kube.add_request("rt-a", current, state="running")
    kube.add_request("rt-b", current, state="running")
    reconciler.enqueue_delete(["rt-a", "rt-b", "rt-missing"])
    kube.fail_delete = {"rt-b"}

    report = run_once(reconciler, due=current)

    assert report.deleted_requests == ("rt-a", "rt-missing")
    assert report.errors == {"delete_request": 1}
    assert reconciler.pending_deletes() == frozenset({"rt-b"})

    kube.fail_delete = set()
    report = run_once(reconciler, due=current)
    assert report.deleted_requests == ("rt-b",)
    assert reconciler.pending_deletes() == frozenset()


def test_the_delete_queue_is_processed_when_listing_fails(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path)
    reconciler.enqueue_delete(["rt-a"])
    kube.fail_list = True

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_requests == ("rt-a",)


def test_protected_requests_are_never_deleted(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    old = window_ms(6)
    kube.add_request("rt-terminal", old, state="completed")
    kube.add_request("rt-late", old, state="running")
    kube.add_request("rt-other-uid", old, dgd_uid="uid-before", state="running")
    reconciler.protect(old, ["rt-terminal", "rt-late", "rt-other-uid"])
    reconciler.enqueue_delete(["rt-late"])

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_requests == ()
    assert kube.deleted == []
    assert reconciler.pending_deletes() == frozenset({"rt-late"})
    assert reconciler.protected_requests() == frozenset(
        {"rt-terminal", "rt-late", "rt-other-uid"}
    )

    reconciler.release(old)
    report = run_once(reconciler, due=window_ms(0))
    assert report.deleted_requests == ("rt-late", "rt-other-uid", "rt-terminal")


def test_unreadable_run_state_keeps_the_request(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    kube.add_request("rt-a", window_ms(6), state="completed")
    window = touch(window_path(reconciler, window_ms(6)), 10)
    kube.fail_get_run = True

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_requests == ()
    assert report.deleted_windows == ()
    assert window.exists()
    assert report.errors == {"get_run": 1}


def test_terminal_and_finished_verdicts_are_cached(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    window = window_ms(6)
    kube.add_request("rt-a", window, state="completed")
    touch(window_path(reconciler, window), 10)

    run_once(reconciler, due=window_ms(0))
    calls = len(kube.get_run_calls)
    assert calls > 0
    touch(window_path(reconciler, window), 10)  # recreated: still finished
    report = run_once(reconciler, due=window_ms(0))
    assert len(kube.get_run_calls) == calls
    assert report.deleted_windows == (window,)


def test_deleted_requests_stop_referencing_their_window(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    old_window, young_window = window_ms(6), window_ms(7)
    old = touch(window_path(reconciler, old_window), WINDOW_AGE + 60)
    young = touch(window_path(reconciler, young_window), WINDOW_AGE - 60)
    kube.add_request("rt-old", old_window, state="running")
    kube.add_request("rt-young", young_window, state="running")

    report = run_once(reconciler, due=window_ms(0))

    assert report.deleted_requests == ("rt-old", "rt-young")
    assert report.deleted_windows == (old_window,)
    assert not old.exists() and young.exists()


# ------------------------------------------------------------ early stop


def one_of_each(tmp_path: Path) -> tuple[Reconciler, FakeKube]:
    """Six deletable things: 2 segments, a pod dir, a temp file, a Request
    and the window file it referenced."""

    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    root = Path(reconciler.segments_root)
    for i in range(3):
        touch(root / "fe-0" / f"req.00000{i}.jsonl.gz", OLD)
    touch(root / "gone" / "req.000000.jsonl.gz", OLD)
    age_dir(root / "gone", OLD)
    touch(Path(reconciler.windows_root) / f".tmp-{window_ms(9)}-a1", OLD)
    touch(window_path(reconciler, window_ms(6)), OLD)
    kube.add_request("rt-a", window_ms(6), state="completed")
    return reconciler, kube


def deletions(report) -> int:
    return (
        report.deleted_segments
        + len(report.deleted_pod_dirs)
        + len(report.deleted_windows)
        + report.deleted_temp_files
        + len(report.deleted_requests)
    )


def test_should_continue_is_asked_before_every_delete(tmp_path: Path) -> None:
    reconciler, _ = one_of_each(tmp_path)
    calls: list[int] = []

    def should_continue() -> bool:
        calls.append(1)
        return True

    report = run_once(
        reconciler,
        due=window_ms(0),
        pods=[frontend("fe-0")],
        should_continue=should_continue,
    )

    assert deletions(report) == 6
    assert len(calls) == 6
    assert not report.stopped


def test_a_no_from_should_continue_ends_the_round(tmp_path: Path) -> None:
    reconciler, kube = one_of_each(tmp_path)
    answers = [True, True]

    def should_continue() -> bool:
        return answers.pop(0) if answers else False

    report = run_once(
        reconciler,
        due=window_ms(0),
        pods=[frontend("fe-0")],
        should_continue=should_continue,
    )

    # Two segment deletions, then nothing: no Request was even listed.
    assert deletions(report) == 2
    assert report.stopped
    assert kube.selectors == [] and kube.deleted == []
    assert window_path(reconciler, window_ms(6)).exists()

    report = run_once(reconciler, due=window_ms(0), pods=[frontend("fe-0")])
    assert deletions(report) == 4
    assert not report.stopped


def test_a_failing_should_continue_ends_the_round(tmp_path: Path) -> None:
    reconciler, kube = one_of_each(tmp_path)

    def should_continue() -> bool:
        raise RuntimeError("lease state unknown")

    report = run_once(
        reconciler,
        due=window_ms(0),
        pods=[frontend("fe-0")],
        should_continue=should_continue,
    )

    assert deletions(report) == 0
    assert report.stopped
    assert kube.deleted == []


# --------------------------------------------------------------- set_dgd


RECREATED = DgdRef(
    name=DGD.name, namespace=DGD.namespace, uid="uid-new", dynamo_namespace="dyn-ns"
)


def test_set_dgd_follows_a_recreated_dgd(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=2)
    current = int((NOW - 60) * 1000)
    # The new DGD's Requests: one kept as a recent terminal window, one
    # still running before its deadline.
    kube.add_request("new-w1-done", window_ms(1), dgd_uid="uid-new")
    kube.add_request("new-w0-running", current, dgd_uid="uid-new", state="running")
    kube.add_request("old-w2", window_ms(2), dgd_uid=DGD.uid, state="running")

    reconciler.set_dgd(RECREATED)
    report = run_once(reconciler, due=current)

    assert reconciler.dgd == RECREATED
    assert report.deleted_requests == ("old-w2",)
    assert set(kube.requests) == {"new-w1-done", "new-w0-running"}


def test_set_dgd_clears_the_request_state_caches(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=10)
    kube.add_request("rt-a", window_ms(3), state="completed")

    run_once(reconciler, due=window_ms(0))
    calls = len(kube.get_run_calls)
    assert calls > 0
    run_once(reconciler, due=window_ms(0))
    assert len(kube.get_run_calls) == calls  # cached terminal verdict

    reconciler.set_dgd(DGD)
    run_once(reconciler, due=window_ms(0))
    assert len(kube.get_run_calls) > calls


def test_set_dgd_during_a_round_stops_it(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path)
    reconciler.enqueue_delete(["rt-a", "rt-b", "rt-c"])
    kube.on_delete = lambda name: reconciler.set_dgd(RECREATED)

    report = run_once(reconciler, due=window_ms(0))

    # It classified Requests by the old uid: nothing after the change.
    assert report.deleted_requests == ("rt-a",)
    assert report.stopped
    assert reconciler.pending_deletes() == frozenset({"rt-b", "rt-c"})


def test_set_dgd_with_a_new_name_moves_the_roots(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    renamed = DgdRef(name="other", namespace="ns", uid="u2", dynamo_namespace="d")

    reconciler.set_dgd(renamed)

    assert reconciler.segments_root == str(tmp_path / "segments" / "other")
    assert reconciler.windows_root == str(tmp_path / "windows" / "other")


# ------------------------------------------------------------------ registry


def test_registry_is_reference_counted_per_window(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    reconciler.protect(5, [])
    reconciler.protect(5, ["rt-a"])
    reconciler.protect(5, ["rt-b"])
    reconciler.protect(6, ["rt-c"])
    assert reconciler.protected_windows() == frozenset({5, 6})
    assert reconciler.protected_requests() == frozenset({"rt-a", "rt-b", "rt-c"})

    # Every protect call holds one reference; the names stay the union of
    # everything given to the window while any reference is left.
    reconciler.release(5)
    reconciler.release(5)
    assert reconciler.protected_windows() == frozenset({5, 6})
    assert reconciler.protected_requests() == frozenset({"rt-a", "rt-b", "rt-c"})
    reconciler.release(5)
    assert reconciler.protected_windows() == frozenset({6})
    assert reconciler.protected_requests() == frozenset({"rt-c"})

    reconciler.release(5)  # already released: no-op, never negative
    reconciler.release(99)  # unknown: no-op
    assert reconciler.protected_windows() == frozenset({6})
    reconciler.protect(5, ["rt-d"])
    assert reconciler.protected_requests() == frozenset({"rt-c", "rt-d"})
    reconciler.release(5)
    assert reconciler.protected_windows() == frozenset({6})
    assert reconciler.protected_requests() == frozenset({"rt-c"})


def test_an_old_sweeps_release_keeps_a_relaunched_sweeps_protection(
    tmp_path: Path,
) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    window = window_ms(6)
    path = touch(window_path(reconciler, window), OLD)
    kube.add_request("rt-old", window, state="running")
    kube.add_request("rt-new", window, state="running")
    # A rebuilt controller relaunches the window while the aborted sweep
    # of the previous generation is still winding down.
    reconciler.protect(window, ["rt-old"])
    reconciler.protect(window, ["rt-new"])
    reconciler.release(window)  # the aborted sweep's finally

    report = run_once(reconciler, due=window_ms(0))
    assert report.deleted_requests == ()
    assert report.deleted_windows == ()
    assert path.exists()

    reconciler.release(window)  # the relaunched sweep ends
    report = run_once(reconciler, due=window_ms(0))
    assert report.deleted_requests == ("rt-new", "rt-old")
    assert report.deleted_windows == (window,)


def test_registry_is_thread_safe_during_rounds(tmp_path: Path) -> None:
    reconciler, kube, _ = make(tmp_path, keep_requests=0)
    for k in range(10):
        kube.add_request(f"rt-{k}", window_ms(20 + k), state="completed")
    errors: list[BaseException] = []
    stop = threading.Event()

    def churn(offset: int) -> None:
        try:
            i = 0
            while not stop.is_set():
                window = 1_000 * offset + i % 50
                reconciler.protect(window, [f"x-{offset}-{i}"])
                reconciler.enqueue_delete([])
                reconciler.release(window)
                i += 1
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=churn, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(20):
            run_once(reconciler, due=window_ms(0))
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    assert errors == []
    assert reconciler.protected_windows() == frozenset()
    assert kube.requests == {}


# ------------------------------------------------------------------ helpers


def test_window_relpath_matches_the_reconciler_layout(tmp_path: Path) -> None:
    reconciler, _, _ = make(tmp_path)
    cfg = DgdrExecutorConfig(
        claim_name="traces", model="m", sku="s", engine={}, windows_dir="/w/x/"
    )
    assert window_relpath(cfg, "dgd", 1234) == "w/x/dgd/1234.jsonl.gz"
    assert reconciler.windows_root == str(tmp_path / "windows" / "dgd")
    assert reconciler.segments_root == str(tmp_path / "segments" / "dgd")

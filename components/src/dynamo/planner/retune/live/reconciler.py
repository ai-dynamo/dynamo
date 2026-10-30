# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Garbage collection of live retune files and DGDR Requests (design §4.4.5).

The supervisor runs ``Reconciler.run_once`` only while this process leads and
is initialized. It deletes, with ``fs_root`` the planner's mount of the trace
claim, ``T``/``G`` the retune interval/grace and ``timeout`` the sweep
timeout::

    <fs_root>/<segments_dir>/<dgd>/<pod>/<prefix>.NNNNNN.jsonl.gz   frontend segments
    <fs_root>/<windows_dir>/<dgd>/<window_end_ms>.jsonl.gz           window files
    <fs_root>/<windows_dir>/<dgd>/.tmp-<window_end_ms>-<nonce>       cut temp files

- Segments, only when the frontend pod set is known and nonempty: never the
  highest-numbered segment of a live pod; other segments of a live pod, and
  the whole directory of a pod that no longer exists, once every file is
  older than ``T + G + timeout + visibility margin``. A live pod's segments
  are the files directly in its directory: P10 requires the sink's file
  prefix to sit directly under ``$(DYN_POD_NAME)/``. A pod directory that
  cannot be read is skipped for the round; the other pods are still
  collected.
- Window files, never while protected: ``window_end_ms`` in
  ``[last_window_end_ms, due_window_end_ms]`` with ``now < D_k``, or in use
  by this process's sweep (``protect``). Otherwise deleted when at least one
  Request references the file and every referencing Request's current Run
  has ``Completed=True`` and no Job pod left, or when no Request references
  it and it is older than ``T + timeout``.
- Temp files older than ``cut_budget_seconds`` plus a margin, unless their
  window is in use by this process's sweep.
- Requests of this DGD name, never one this process's sweep protects: a
  terminal Request outside the newest ``keep_requests`` windows, a
  non-terminal Request whose window deadline has passed, a Request in the
  delete queue, and a Request of another DGD uid (the DGD was recreated).

``run_once(should_continue=...)`` asks the caller before every delete or
unlink and ends the round as soon as it says no (leadership or the term was
lost while the round ran on its worker thread). ``set_dgd`` follows the DGD
identity when the DGD is recreated under the same name.

``Reconciler`` is also the ``SweepRegistry`` the DGDR sweep uses to protect
its window and Requests and to queue deletions; those methods are thread-safe
(the sweep runs on the controller's executor thread). Protection is
reference-counted per window: each ``protect`` call takes one reference,
each ``release`` drops one, and the window with the union of every name
given to it stays protected while any reference is held. Two sweeps of the
same window (a rebuilt controller relaunching it while the old one winds
down) therefore cannot drop each other's protection. ``run_once`` blocks on
the filesystem and the Kubernetes API: call it with ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from dynamo.planner.config.planner_config import DgdrExecutorConfig, RetuneConfig
from dynamo.planner.retune.live.contracts import (
    LABEL_DGD,
    LABEL_DGD_UID,
    LABEL_WINDOW,
    DgdRef,
    KubeError,
    PodInfo,
    RetuneKube,
    RetuneMetricsSink,
)

logger = logging.getLogger(__name__)

# Allowance for RWX attribute caching and the frontend's flush interval on
# top of T + G + timeout before a segment counts as unused.
DEFAULT_VISIBILITY_MARGIN_S = 600.0
# Allowance on top of cut_budget_seconds before a temp file counts as
# abandoned (the cut subprocess is killed at its deadline).
DEFAULT_TEMP_MARGIN_S = 300.0

WINDOW_SUFFIX = ".jsonl.gz"
TEMP_PREFIX = ".tmp-"
# Label the Job controller puts on a Job's pods (batchv1.JobNameLabel).
JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_FINISHED_POD_PHASES = frozenset({"Succeeded", "Failed"})
# Annotation the DGDR v1beta2 Request controller puts on each Run: the
# Request generation it was created for.
RUN_GENERATION_ANNOTATION = "nvidia.com/dgdr-request-generation"
RUN_COMPLETED_CONDITION = "Completed"
REQUEST_ACCEPTED_CONDITION = "Accepted"
REQUEST_INVALID_SPEC_REASON = "InvalidSpec"

_WINDOW_RE = re.compile(r"^(\d+)" + re.escape(WINDOW_SUFFIX) + r"$")
_TEMP_RE = re.compile(r"^" + re.escape(TEMP_PREFIX) + r"(\d+)-")
# A Rust gzip JSONL sink segment: <prefix>.<index, at least 6 digits>.jsonl.gz
_SEGMENT_RE = re.compile(r"^(.*)\.(\d{6,})\.jsonl\.gz$")


def window_file_name(window_end_ms: int) -> str:
    return f"{int(window_end_ms)}{WINDOW_SUFFIX}"


def window_relpath(cfg: DgdrExecutorConfig, dgd_name: str, window_end_ms: int) -> str:
    """Claim-relative path of a window file (the Request's trace path)."""

    return "/".join(
        (cfg.windows_dir.strip("/"), dgd_name, window_file_name(window_end_ms))
    )


@dataclass(frozen=True)
class ReconcileReport:
    """What one ``run_once`` round did."""

    deleted_segments: int = 0
    deleted_pod_dirs: tuple[str, ...] = ()
    deleted_windows: tuple[int, ...] = ()
    deleted_temp_files: int = 0
    deleted_requests: tuple[str, ...] = ()
    # Ready frontend pods without a segment directory (trace_sink_missing).
    sink_missing: tuple[str, ...] = ()
    errors: dict[str, int] = field(default_factory=dict)
    # The round ended early: ``should_continue`` said no, or the DGD
    # identity changed (``set_dgd``) while it ran.
    stopped: bool = False


@dataclass
class _RequestView:
    name: str
    uid: str
    dgd_uid: Optional[str]
    window_end_ms: Optional[int]
    trace_file: Optional[str]  # window file name if it reads one of ours
    obj: dict[str, Any]


@dataclass
class _Protection:
    """One window's protection: references held and the names protected."""

    count: int = 0
    names: set[str] = field(default_factory=set)


@dataclass
class _Round:
    """Mutable bookkeeping of one ``run_once`` call. Counts accumulate as
    deletions happen, so a phase that fails half way still reports them."""

    now: float
    dgd: DgdRef
    segments_root: str
    windows_root: str
    should_continue: Optional[Callable[[], bool]]
    errors: dict[str, int] = field(default_factory=dict)
    stopped: bool = False
    deleted_segments: int = 0
    deleted_dirs: list[str] = field(default_factory=list)
    deleted_windows: list[int] = field(default_factory=list)
    deleted_temp: int = 0
    deleted_requests: list[str] = field(default_factory=list)
    sink_missing: list[str] = field(default_factory=list)

    def error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1


class Reconciler:
    """Deletes what live retune no longer needs; see the module docstring."""

    def __init__(
        self,
        kube: RetuneKube,
        dgd: DgdRef,
        cfg: DgdrExecutorConfig,
        retune: RetuneConfig,
        fs_root: str,
        metrics: RetuneMetricsSink,
        now: Callable[[], float] = time.time,
        *,
        visibility_margin_s: float = DEFAULT_VISIBILITY_MARGIN_S,
        temp_margin_s: float = DEFAULT_TEMP_MARGIN_S,
        kube_timeout_s: float = 10.0,
    ) -> None:
        self._kube = kube
        self._cfg = cfg
        self._fs_root = fs_root
        self._metrics = metrics
        self._now = now
        self._timeout_s = float(kube_timeout_s)
        interval_s = float(retune.interval_seconds)
        sweep_timeout_s = float(retune.effective_timeout_seconds)
        self._segment_max_age_s = (
            interval_s
            + float(retune.grace_seconds)
            + sweep_timeout_s
            + float(visibility_margin_s)
        )
        self._window_max_age_s = interval_s + sweep_timeout_s
        self._temp_max_age_s = float(cfg.cut_budget_seconds) + float(temp_margin_s)

        self._lock = threading.Lock()
        # Guarded by _lock (set_dgd may run while a round is on its thread).
        self._dgd = dgd
        self._segments_root = self._root(cfg.segments_dir, dgd)
        self._windows_root = self._root(cfg.windows_dir, dgd)
        # window_end_ms -> references and Request names of running sweeps.
        self._protected: dict[int, _Protection] = {}
        self._delete_queue: set[str] = set()
        # (request uid, generation) of Requests known terminal / finished;
        # neither can revert for one generation.
        self._terminal: set[tuple[str, int]] = set()
        self._finished: set[tuple[str, int]] = set()
        self._sink_reported: set[str] = set()

    def _root(self, directory: str, dgd: DgdRef) -> str:
        return os.path.join(self._fs_root, directory.strip("/"), dgd.name)

    # ------------------------------------------------------------- identity

    def set_dgd(self, dgd: DgdRef) -> None:
        """Follow the DGD identity (it was recreated under the same name).

        Requests labelled with the new uid become this DGD's; those of the
        old uid are then deleted as ``other_dgd_uid``. The Request-state
        caches are cleared. A round running on another thread ends at its
        next delete, since it classified Requests by the old identity.
        """

        with self._lock:
            if dgd != self._dgd:
                logger.info(
                    "retune reconciler: DGD %s uid %s -> %s %s",
                    self._dgd.name,
                    self._dgd.uid,
                    dgd.name,
                    dgd.uid,
                )
            self._dgd = dgd
            self._segments_root = self._root(self._cfg.segments_dir, dgd)
            self._windows_root = self._root(self._cfg.windows_dir, dgd)
            self._terminal.clear()
            self._finished.clear()

    @property
    def dgd(self) -> DgdRef:
        with self._lock:
            return self._dgd

    # --------------------------------------------------------- SweepRegistry

    def protect(self, window_end_ms: int, request_names: Iterable[str]) -> None:
        """Take one reference on the window and add ``request_names`` to
        its protected names; each call needs a matching ``release``."""

        names = set(request_names)
        with self._lock:
            entry = self._protected.setdefault(int(window_end_ms), _Protection())
            entry.count += 1
            entry.names.update(names)

    def release(self, window_end_ms: int) -> None:
        """Drop one reference; the window and its names are unprotected
        once no reference is left. Releasing an unknown window is a no-op."""

        with self._lock:
            window = int(window_end_ms)
            entry = self._protected.get(window)
            if entry is None:
                return
            entry.count -= 1
            if entry.count <= 0:
                del self._protected[window]

    def enqueue_delete(self, request_names: Iterable[str]) -> None:
        names = set(request_names)
        with self._lock:
            self._delete_queue.update(names)

    # ---------------------------------------------------------- inspection

    @property
    def segments_root(self) -> str:
        with self._lock:
            return self._segments_root

    @property
    def windows_root(self) -> str:
        with self._lock:
            return self._windows_root

    def protected_windows(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._protected)

    def protected_requests(self) -> frozenset[str]:
        with self._lock:
            return frozenset(
                name for entry in self._protected.values() for name in entry.names
            )

    def pending_deletes(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._delete_queue)

    # ---------------------------------------------------------------- round

    def run_once(
        self,
        *,
        last_window_end_ms: Optional[int],
        due_window_end_ms: Optional[int],
        deadline_for: Callable[[int], float],
        frontend_pods: Optional[Sequence[PodInfo]],
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> ReconcileReport:
        """One GC round. ``deadline_for(window_end_ms)`` is that window's
        sweep deadline ``D_k`` in epoch seconds; ``frontend_pods`` is the
        actuator's listing (None when it failed). ``should_continue`` is
        called before every delete or unlink; once it returns False (or
        raises) the round deletes nothing more and returns."""

        now = float(self._now())
        with self._lock:
            state = _Round(
                now=now,
                dgd=self._dgd,
                segments_root=self._segments_root,
                windows_root=self._windows_root,
                should_continue=should_continue,
            )

        def in_window_range(window_end_ms: int) -> bool:
            """The design's protected range: undecided and before D_k."""

            if last_window_end_ms is not None and window_end_ms < last_window_end_ms:
                return False
            if due_window_end_ms is not None and window_end_ms > due_window_end_ms:
                return False
            return now < deadline_for(window_end_ms)

        try:
            self._collect_segments(state, frontend_pods)
        except Exception:  # keep the sink check, Request and window GC running
            logger.exception("retune reconciler: segment GC failed")
            state.error("segments")
        if not state.stopped:
            try:
                state.sink_missing = self._report_sinks(state, frontend_pods)
            except Exception:
                logger.exception("retune reconciler: trace sink check failed")
                state.error("sinks")

        requests: Optional[list[_RequestView]] = None
        if not state.stopped:
            try:
                requests = self._list_requests(state.dgd)
            except KubeError as exc:
                logger.warning("retune reconciler: listing Requests failed: %s", exc)
                state.error("list_requests")
            except Exception:
                logger.exception("retune reconciler: listing Requests failed")
                state.error("list_requests")
            try:
                self._collect_requests(state, requests, in_window_range, deadline_for)
            except Exception:
                logger.exception("retune reconciler: Request GC failed")
                state.error("requests")

        if not state.stopped:
            try:
                remaining = (
                    None
                    if requests is None
                    else [
                        view
                        for view in requests
                        if view.name not in state.deleted_requests
                    ]
                )
                self._collect_windows(state, remaining, in_window_range)
            except Exception:
                logger.exception("retune reconciler: window GC failed")
                state.error("windows")

        report = ReconcileReport(
            deleted_segments=state.deleted_segments,
            deleted_pod_dirs=tuple(sorted(state.deleted_dirs)),
            deleted_windows=tuple(sorted(state.deleted_windows)),
            deleted_temp_files=state.deleted_temp,
            deleted_requests=tuple(sorted(state.deleted_requests)),
            sink_missing=tuple(sorted(state.sink_missing)),
            errors=state.errors,
            stopped=state.stopped,
        )
        if (
            report.deleted_segments
            or report.deleted_pod_dirs
            or report.deleted_windows
            or report.deleted_temp_files
            or report.deleted_requests
        ):
            logger.info(
                "retune reconciler: deleted %d segments, pod dirs %s, windows %s, "
                "%d temp files, Requests %s",
                report.deleted_segments,
                list(report.deleted_pod_dirs),
                list(report.deleted_windows),
                report.deleted_temp_files,
                list(report.deleted_requests),
            )
        return report

    def _proceed(self, state: _Round) -> bool:
        """Whether the round may delete one more thing; asked right before
        every delete or unlink. A no is final for the round."""

        if state.stopped:
            return False
        with self._lock:
            current = self._dgd
        if current != state.dgd:
            logger.info(
                "retune reconciler: the DGD identity changed during the round; "
                "stopping it"
            )
            state.stopped = True
            return False
        if state.should_continue is None:
            return True
        try:
            proceed = bool(state.should_continue())
        except Exception:
            logger.exception("retune reconciler: should_continue failed")
            proceed = False
        if not proceed:
            logger.info("retune reconciler: told to stop; ending the round early")
            state.stopped = True
        return proceed

    # ------------------------------------------------------------- segments

    def _collect_segments(
        self, state: _Round, frontend_pods: Optional[Sequence[PodInfo]]
    ) -> None:
        if not frontend_pods:
            # Unknown or empty pod set: every directory would look orphaned.
            return
        live = {pod.name for pod in frontend_pods}
        cutoff = state.now - self._segment_max_age_s
        for entry in _scandir(state.segments_root):
            if state.stopped:
                return
            # One pod directory that cannot be read (another uid, umask 077)
            # must not starve the pods after it in scandir order.
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if entry.name in live:
                    self._collect_pod_segments(state, entry.path, cutoff)
                    continue
                if _newest_mtime(entry.path) >= cutoff:
                    continue
                if not self._proceed(state):
                    return
                shutil.rmtree(entry.path)
                state.deleted_dirs.append(entry.name)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "retune reconciler: segment GC of %s failed: %s", entry.path, exc
                )
                state.error("segments")

    def _collect_pod_segments(
        self, state: _Round, directory: str, cutoff: float
    ) -> None:
        # Top level only: P10 puts the sink's file prefix directly under the
        # pod directory, so there is no subdirectory to walk.
        segments: dict[str, list[tuple[int, os.DirEntry[str]]]] = {}
        for entry in _scandir(directory):
            match = _SEGMENT_RE.match(entry.name)
            if match is None or not entry.is_file(follow_symlinks=False):
                continue
            segments.setdefault(match.group(1), []).append((int(match.group(2)), entry))
        for entries in segments.values():
            entries.sort(key=lambda item: item[0])
            # The highest-numbered segment is the one the frontend appends to.
            for _, entry in entries[:-1]:
                try:
                    if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                        continue
                    if not self._proceed(state):
                        return
                    os.unlink(entry.path)
                    state.deleted_segments += 1
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning(
                        "retune reconciler: removing %s failed: %s", entry.path, exc
                    )
                    state.error("segments")

    def _report_sinks(
        self, state: _Round, frontend_pods: Optional[Sequence[PodInfo]]
    ) -> list[str]:
        if frontend_pods is None:
            return []
        missing: list[str] = []
        current: set[str] = set()
        for pod in frontend_pods:
            if not pod.ready or pod.terminating:
                continue
            current.add(pod.name)
            absent = not os.path.isdir(os.path.join(state.segments_root, pod.name))
            if absent:
                missing.append(pod.name)
            self._metrics.trace_sink_missing(pod.name, absent)
        for name in self._sink_reported - current:
            self._metrics.trace_sink_missing(name, False)
        self._sink_reported = current
        if missing:
            logger.warning(
                "retune reconciler: Ready frontends without a trace directory "
                "under %s: %s",
                state.segments_root,
                missing,
            )
        return missing

    # -------------------------------------------------------------- windows

    def _collect_windows(
        self,
        state: _Round,
        requests: Optional[list[_RequestView]],
        in_window_range: Callable[[int], bool],
    ) -> None:
        protected = self.protected_windows()
        references: dict[str, list[_RequestView]] = {}
        for view in requests or ():
            if view.trace_file is not None:
                references.setdefault(view.trace_file, []).append(view)

        for entry in _scandir(state.windows_root):
            if state.stopped:
                return
            try:
                self._collect_window_entry(
                    state, entry, requests, references, protected, in_window_range
                )
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "retune reconciler: window GC of %s failed: %s", entry.path, exc
                )
                state.error("windows")

    def _collect_window_entry(
        self,
        state: _Round,
        entry: os.DirEntry[str],
        requests: Optional[list[_RequestView]],
        references: Mapping[str, list[_RequestView]],
        protected: frozenset[int],
        in_window_range: Callable[[int], bool],
    ) -> None:
        if not entry.is_file(follow_symlinks=False):
            return
        if entry.name.startswith(TEMP_PREFIX):
            temp = _TEMP_RE.match(entry.name)
            window = int(temp.group(1)) if temp is not None else None
            # A temp file of a window this process's sweep is cutting
            # belongs to the running cut subprocess.
            if window is not None and window in self.protected_windows():
                return
            if self._older_than(entry, state.now, self._temp_max_age_s):
                if self._unlink(state, entry.path):
                    state.deleted_temp += 1
            return
        match = _WINDOW_RE.match(entry.name)
        if match is None or requests is None:
            # Unknown file, or references unknown: keep it.
            return
        window = int(match.group(1))
        if window in protected or in_window_range(window):
            return
        referrers = references.get(entry.name, [])
        if referrers:
            if not all(self._finished_with(state, view) for view in referrers):
                return
        elif not self._older_than(entry, state.now, self._window_max_age_s):
            return
        # The sweep may have protected it since the snapshot above.
        if window in self.protected_windows():
            return
        if self._unlink(state, entry.path):
            state.deleted_windows.append(window)

    # ------------------------------------------------------------- requests

    def _list_requests(self, dgd: DgdRef) -> list[_RequestView]:
        items = self._kube.list_requests(
            f"{LABEL_DGD}={dgd.name}", timeout_s=self._timeout_s
        )
        views = []
        prefix = self._cfg.windows_dir.strip("/") + "/" + dgd.name + "/"
        for obj in items:
            metadata = obj.get("metadata") or {}
            labels = metadata.get("labels") or {}
            pvc = _dig(obj, "spec", "workload", "trace", "source", "pvc") or {}
            path = pvc.get("path")
            trace_file = None
            if (
                pvc.get("claimName") == self._cfg.claim_name
                and isinstance(path, str)
                and path.startswith(prefix)
                and "/" not in path[len(prefix) :]
            ):
                trace_file = path[len(prefix) :]
            views.append(
                _RequestView(
                    name=str(metadata.get("name", "")),
                    uid=str(metadata.get("uid", "")),
                    dgd_uid=labels.get(LABEL_DGD_UID),
                    window_end_ms=_int_or_none(labels.get(LABEL_WINDOW)),
                    trace_file=trace_file,
                    obj=obj,
                )
            )
        return [view for view in views if view.name]

    def _collect_requests(
        self,
        state: _Round,
        requests: Optional[list[_RequestView]],
        in_window_range: Callable[[int], bool],
        deadline_for: Callable[[int], float],
    ) -> None:
        now = state.now
        dgd_uid = state.dgd.uid
        reasons: dict[str, str] = {name: "queued" for name in self.pending_deletes()}
        if requests is not None:
            listed = {view.uid for view in requests}
            with self._lock:
                for cache in (self._terminal, self._finished):
                    cache.difference_update(
                        [key for key in cache if key[0] not in listed]
                    )
            ours = [view for view in requests if view.dgd_uid == dgd_uid]
            for view in requests:
                if view.dgd_uid is None:
                    logger.warning(
                        "retune reconciler: Request %s has no %s label",
                        view.name,
                        LABEL_DGD_UID,
                    )
                elif view.dgd_uid != dgd_uid:
                    # The DGD was deleted and recreated under the same name.
                    reasons.setdefault(view.name, "other_dgd_uid")
            states = {view.name: self._is_terminal(state, view) for view in ours}
            terminal = {name for name, value in states.items() if value is True}
            kept_windows = sorted(
                {
                    view.window_end_ms
                    for view in ours
                    if view.name in terminal and view.window_end_ms is not None
                },
                reverse=True,
            )[: self._cfg.keep_requests]
            for view in ours:
                window = view.window_end_ms
                if window is None:
                    logger.warning(
                        "retune reconciler: Request %s has no %s label",
                        view.name,
                        LABEL_WINDOW,
                    )
                    continue
                value = states[view.name]
                if value is None:
                    continue  # Run state unknown this round
                if value:
                    if window not in kept_windows and not in_window_range(window):
                        reasons.setdefault(view.name, "terminal")
                elif now >= deadline_for(window):
                    # Not terminal and past D_k: no sweep can adopt it anymore.
                    reasons.setdefault(view.name, "deadline_passed")

        for name in sorted(reasons):
            if name in self.protected_requests():
                continue
            if not self._proceed(state):
                return
            try:
                self._kube.delete_request(name, timeout_s=self._timeout_s)
            except KubeError as exc:
                logger.warning(
                    "retune reconciler: deleting Request %s (%s) failed: %s",
                    name,
                    reasons[name],
                    exc,
                )
                state.error("delete_request")
                continue
            logger.info(
                "retune reconciler: deleted Request %s (%s)", name, reasons[name]
            )
            state.deleted_requests.append(name)
            with self._lock:
                self._delete_queue.discard(name)

    # ------------------------------------------------- Request / Run state

    def _cached(self, cache: set[tuple[str, int]], key: tuple[str, int]) -> bool:
        with self._lock:
            return key in cache

    def _remember(
        self, state: _Round, cache: set[tuple[str, int]], key: tuple[str, int]
    ) -> None:
        """Cache a verdict unless ``set_dgd`` cleared the caches meanwhile
        (the caches are cleared in place, never replaced)."""

        with self._lock:
            if self._dgd == state.dgd:
                cache.add(key)

    def _is_terminal(self, state: _Round, view: _RequestView) -> Optional[bool]:
        """Whether the Request's current-generation Run completed, or the
        operator rejected the current generation (InvalidSpec); None when
        the Run could not be read."""

        generation = _int_or_none(_dig(view.obj, "metadata", "generation")) or 0
        key = (view.uid, generation)
        if self._cached(self._terminal, key):
            return True
        try:
            run = self._current_run(view, generation)
        except KubeError as exc:
            logger.warning(
                "retune reconciler: reading the Run of Request %s failed: %s",
                view.name,
                exc,
            )
            state.error("get_run")
            return None
        terminal = (
            _condition_true(run, RUN_COMPLETED_CONDITION)
            if run is not None
            else _invalid_spec(view.obj, generation)
        )
        if terminal:
            self._remember(state, self._terminal, key)
        return terminal

    def _finished_with(self, state: _Round, view: _RequestView) -> bool:
        """The Request no longer reads its window file: terminal, no active
        Run, and no pod of the current Run's Job left."""

        generation = _int_or_none(_dig(view.obj, "metadata", "generation")) or 0
        key = (view.uid, generation)
        if self._cached(self._finished, key):
            return True
        if _dig(view.obj, "status", "activeRunRef", "name"):
            return False
        try:
            run = self._current_run(view, generation)
            if run is None:
                finished = _invalid_spec(view.obj, generation)
            elif not _condition_true(run, RUN_COMPLETED_CONDITION):
                finished = False
            else:
                job = _dig(run, "status", "jobRef", "name")
                # A Succeeded or Failed pod no longer reads the trace; one
                # with no phase reported is treated as still running.
                finished = job is None or not any(
                    pod.phase not in _FINISHED_POD_PHASES
                    for pod in self._kube.list_pods(
                        f"{JOB_NAME_LABEL}={job}", timeout_s=self._timeout_s
                    )
                )
        except KubeError as exc:
            logger.warning(
                "retune reconciler: reading Run state of Request %s failed: %s",
                view.name,
                exc,
            )
            return False
        if finished:
            self._remember(state, self._finished, key)
        return finished

    def _current_run(
        self, view: _RequestView, generation: int
    ) -> Optional[dict[str, Any]]:
        """The Run created for the Request's current generation, if any.

        The Run's generation annotation identifies it, whatever its name;
        the Request status names the active and the retained terminal Runs.
        """

        status = view.obj.get("status") or {}
        names = []
        active = (status.get("activeRunRef") or {}).get("name")
        if active:
            names.append(active)
        names.extend(
            ref.get("name")
            for ref in status.get("recentRunRefs") or ()
            if ref.get("name")
        )
        for name in names:
            run = self._kube.get_run(name, timeout_s=self._timeout_s)
            if run is None:
                continue
            annotations = _dig(run, "metadata", "annotations") or {}
            if _int_or_none(annotations.get(RUN_GENERATION_ANNOTATION)) == generation:
                return run
        return None

    # ---------------------------------------------------------- filesystem

    @staticmethod
    def _older_than(entry: os.DirEntry[str], now: float, age_s: float) -> bool:
        try:
            return entry.stat(follow_symlinks=False).st_mtime < now - age_s
        except FileNotFoundError:
            return False

    def _unlink(self, state: _Round, path: str) -> bool:
        if not self._proceed(state):
            return False
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("retune reconciler: removing %s failed: %s", path, exc)
            state.error("windows")
            return False


# ---------------------------------------------------------------- helpers


def _scandir(path: str) -> list[os.DirEntry[str]]:
    """Entries of ``path``; none when it is missing. Other ``OSError``s
    (e.g. ``PermissionError``) propagate to the caller's per-entry handler."""

    try:
        with os.scandir(path) as entries:
            return list(entries)
    except (FileNotFoundError, NotADirectoryError):
        return []


def _newest_mtime(path: str) -> float:
    """Newest mtime of ``path`` and everything under it."""

    newest = os.lstat(path).st_mtime
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
            except FileNotFoundError:
                continue
    return newest


def _dig(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, Mapping):
            return None
        obj = obj.get(key)
    return obj


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _conditions(obj: Optional[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    conditions = _dig(obj, "status", "conditions")
    return [c for c in conditions or () if isinstance(c, Mapping)]


def _condition_true(obj: Optional[Mapping[str, Any]], kind: str) -> bool:
    return any(
        c.get("type") == kind and c.get("status") == "True" for c in _conditions(obj)
    )


def _invalid_spec(request: Mapping[str, Any], generation: int) -> bool:
    """The operator rejected this generation; only a new one changes that."""

    for condition in _conditions(request):
        if (
            condition.get("type") == REQUEST_ACCEPTED_CONDITION
            and condition.get("status") == "False"
            and condition.get("reason") == REQUEST_INVALID_SPEC_REASON
            and _int_or_none(condition.get("observedGeneration")) == generation
        ):
            return True
    return False

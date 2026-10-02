# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-cluster (DGDR v2) Sweeper status adapter.

Hooks both ``Sweeper.run``'s real callbacks -- ``on_candidate`` (fired once
per evaluated candidate, as soon as its outcome is known) and ``on_round``
(fired once per round, with the round's evaluated candidates) -- and reports
through the status-snapshot mechanism in ``kube_status``.

Both callbacks run synchronously and unguarded inside Sweeper's main search
loop (confirmed against the real ``Sweeper.run`` implementation): neither may
block on rendering or file I/O. Every callback here only enqueues; a single
background thread owns rendering and the actual (atomic) status write.

Scope for this first pass: scalar goals only. A Pareto-goal config raises
``NotImplementedError`` with a clear message rather than guessing at an
incremental crowding/front-eviction policy -- that needs its own design pass,
not a silent approximation.

Complementary to, not a replacement for, ``dynamo.aisimulate.output.dgd.adapter``
(DEP #14282): that adapter is AISimulate's own output-adapter plugin, called
once after a sweep finishes with the final selected candidate(s) -- it has
no way to report anything if the Sweeper process dies mid-run, since it
simply never gets called. This module hooks the search loop directly instead,
so there is always a last-written, durable status even on a crash.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dynamo.aisimulate.output.dgd.kube_status import (
    CandidateOutcome,
    CandidateStatusEntry,
    SweeperStatusSnapshot,
    SweepRunStatus,
    write_sweeper_status,
)
from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    DGDRenderer,
    render_dgd,
)
from dynamo.aisimulate.output.dgd.writers.atomic import replace_text

logger = logging.getLogger(__name__)

_OVERFLOW_WARNING = (
    "dgdr adapter status queue is full; dropping oldest pending update "
    "(consumer -- the background writer thread -- is falling behind)"
)

# Hard ceiling on top_n, independent of whatever a DGDC's maxCandidates field
# allows. Candidate manifests are carried inline (not as a path reference --
# see kube_status.py's module docstring for why), so the dominant cost per
# retained slot is a real rendered DGD manifest, not a scalar. Measured
# against real manifests pulled from this repo (1.3KiB for a single-node agg
# template up to 11.5KiB for the largest observed multi-node disaggregated
# recipe), the worst case at this ceiling -- every one of 20 retained slots
# the largest observed manifest shape -- sits at ~232KiB, about 22% of the
# 1MiB ConfigMap limit, leaving real headroom for etcd's own per-revision
# overhead and future schema growth. (At top_n=50 the same worst case would
# reach ~55%, too close to the limit to leave a safety margin.) This is
# deliberately an error, not a silent clamp: a config asking to retain more
# than this is a request this adapter cannot safely satisfy and that is a
# backend-level bug or misconfiguration, not a case to paper over.
_MAX_TOP_N = 20


class CandidateRecordLike(Protocol):
    """The subset of aisimulate's real CandidateRecord this adapter needs."""

    candidate_id: str
    status: Any  # aisimulate.sweeper.result.CandidateStatus
    score: float | None
    used_gpus: int | None
    config: dict[str, Any]


def _is_feasible(record: CandidateRecordLike) -> bool:
    # Compared by value, not identity: callers may pass aisimulate's real
    # CandidateStatus enum, a plain string, or a test double -- any of which
    # compare equal to "feasible" via the real enum's str mixin.
    return str(getattr(record.status, "value", record.status)) == "feasible"


def _retention_key(record: CandidateRecordLike) -> tuple[float, int]:
    """Match Sweeper's own scalar ranking: highest score, then fewer GPUs.

    Mirrors ``dynamo.profiler.sweeper.__main__._candidate_key``'s ordering
    (score descending, used_gpus ascending as the tie-break), kept
    independent of that CLI module so this adapter has no import-time
    dependency on it.
    """
    assert record.score is not None and record.used_gpus is not None
    return (record.score, -record.used_gpus)


@dataclass(frozen=True)
class _CandidateUpdate:
    record_id: str
    feasible: bool
    config: dict[str, Any] | None
    score: float | None
    used_gpus: int | None


@dataclass(frozen=True)
class _RoundUpdate:
    round_no: int
    cumulative_evaluated: int


@dataclass(frozen=True)
class _TerminalUpdate:
    status: SweepRunStatus
    message: str = ""
    error: str = ""


_QueueItem = _CandidateUpdate | _RoundUpdate | _TerminalUpdate


@dataclass
class _ScoredRecord:
    """Minimal candidate snapshot retained past a callback's return.

    The real ``CandidateRecord`` the Sweeper hands ``on_candidate`` is
    already a detached deep copy (see aisimulate's ``on_candidate``
    dispatch), but this adapter only needs a handful of its fields and
    holds them on the background thread for as long as the candidate stays
    in the retained set -- so it copies just those fields into its own
    small, explicitly-owned type rather than holding a reference to
    aisimulate's richer record for the run's duration.
    """

    candidate_id: str
    score: float | None
    used_gpus: int | None
    config: dict[str, Any]


class DGDRAdapter:
    """Reports Sweeper progress and candidates via the DGDR status-ConfigMap path.

    One instance is created per Sweeper run. Pass its bound methods as
    ``Sweeper.run(..., on_round=adapter.on_round, on_candidate=adapter.on_candidate)``,
    then call ``start()`` before ``run()`` and ``close()`` (or use as a
    context manager) after, so the final state is always flushed even if the
    run raises.
    """

    def __init__(
        self,
        output_dir: Path,
        options: DGDGenerationOptions,
        workload: Any,
        *,
        dgd_name_prefix: str = "sweeper-dgd",
        renderer: DGDRenderer = "aic",
        top_n: int = 5,
        queue_size: int = 64,
        is_pareto: bool = False,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if top_n > _MAX_TOP_N:
            raise ValueError(
                f"top_n={top_n} exceeds the ConfigMap size budget "
                f"(hard ceiling is {_MAX_TOP_N}); see the comment on "
                f"_MAX_TOP_N for the sizing math"
            )
        if is_pareto:
            raise NotImplementedError(
                "DGDRAdapter only supports scalar goals so far; "
                "Pareto retention (which candidates to keep as the front "
                "changes shape round to round) needs its own design, not an "
                "incremental approximation of top-N-by-score eviction"
            )
        self._output_dir = output_dir
        self._options = options
        self._workload = workload
        self._dgd_name_prefix = dgd_name_prefix
        self._renderer = renderer
        self._top_n = top_n

        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._drain_loop, name="dgdr-adapter-writer", daemon=True
        )
        self._stop = threading.Event()

        # Owned exclusively by the background thread -- never touched from
        # the Sweeper callbacks, which only ever enqueue.
        self._retained: dict[str, _ScoredRecord] = {}
        self._round_no = 0
        self._cumulative_evaluated = 0
        self._final_status: SweepRunStatus | None = None
        self._final_message = ""
        self._final_error = ""
        self._started = False

    # -- Sweeper callbacks: enqueue only, never block, never raise ---------

    def on_candidate(self, record: CandidateRecordLike) -> None:
        feasible = _is_feasible(record)
        self._enqueue(
            _CandidateUpdate(
                record_id=record.candidate_id,
                feasible=feasible,
                config=dict(record.config) if feasible else None,
                score=record.score if feasible else None,
                used_gpus=record.used_gpus if feasible else None,
            )
        )

    def on_round(self, round_number: int, candidates: list[Any]) -> None:
        self._enqueue(
            _RoundUpdate(
                round_no=round_number,
                cumulative_evaluated=len(candidates),
            )
        )

    def _enqueue(self, item: _QueueItem) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                logger.warning(_OVERFLOW_WARNING)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Someone else refilled it between our get and put; the
                # dropped update is not worth blocking the search loop over.
                pass

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def close(
        self, *, status: SweepRunStatus, message: str = "", error: str = ""
    ) -> None:
        """Flush final state and stop the background thread.

        Always call this, including when the run raised -- the caller is
        responsible for choosing ``status=FAILED`` in that case so the
        terminal snapshot reflects it (mirrors v1's run.completed guarantee).

        Safe to call even if ``start()`` was never called (e.g. setup failed
        before the run began): there is no background thread to join in that
        case, so the terminal snapshot is applied synchronously on the
        caller's own thread instead of enqueued.
        """
        if not self._started:
            self._apply(_TerminalUpdate(status=status, message=message, error=error))
            return
        self._enqueue(_TerminalUpdate(status=status, message=message, error=error))
        self._stop.set()
        self._thread.join()

    def __enter__(self) -> DGDRAdapter:
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: Any) -> None:
        if exc_type is not None:
            self.close(status=SweepRunStatus.FAILED, error=str(exc_type.__name__))
        else:
            self.close(status=SweepRunStatus.SUCCESS)

    # -- background thread: all rendering and I/O happens here only ---------

    def _drain_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            self._apply(item)
            if isinstance(item, _TerminalUpdate):
                return

    def _apply(self, item: _QueueItem) -> None:
        if isinstance(item, _CandidateUpdate):
            self._apply_candidate(item)
        elif isinstance(item, _RoundUpdate):
            self._round_no = item.round_no
            self._cumulative_evaluated += item.cumulative_evaluated
        elif isinstance(item, _TerminalUpdate):
            self._final_status = item.status
            self._final_message = item.message
            self._final_error = item.error
        self._write_snapshot()

    def _apply_candidate(self, item: _CandidateUpdate) -> None:
        if not item.feasible:
            return  # non-feasible candidates aren't retained or rendered
        self._retained[item.record_id] = _ScoredRecord(
            candidate_id=item.record_id,
            score=item.score,
            used_gpus=item.used_gpus,
            config=item.config or {},
        )
        if len(self._retained) > self._top_n:
            worst_id = min(
                self._retained, key=lambda rid: _retention_key(self._retained[rid])
            )
            del self._retained[worst_id]

    def _write_snapshot(self) -> None:
        entries: list[CandidateStatusEntry] = []
        for record in self._retained.values():
            try:
                rendered = render_dgd(
                    record,
                    self._workload,
                    self._options,
                    dgd_name=f"{self._dgd_name_prefix}-{record.candidate_id}",
                    renderer=self._renderer,
                )
            except CandidateMaterializationError as exc:
                entries.append(
                    CandidateStatusEntry(
                        candidate_id=record.candidate_id,
                        outcome=CandidateOutcome.MATERIALIZATION_FAILED,
                        error=str(exc),
                    )
                )
                continue
            # Inlined so the controller can build the DGDC's Spec straight
            # from the relayed ConfigMap; also written to disk at `path`
            # purely as a human-readable artifact -- the controller never
            # reads it.
            path = self._write_candidate_manifest(record.candidate_id, rendered)
            entries.append(
                CandidateStatusEntry(
                    candidate_id=record.candidate_id,
                    outcome=CandidateOutcome.MATERIALIZED,
                    manifest=rendered,
                    path=path,
                )
            )

        status = self._final_status or SweepRunStatus.RUNNING
        write_sweeper_status(
            self._output_dir,
            SweeperStatusSnapshot(
                status=status,
                round_no=self._round_no,
                cumulative_evaluated=self._cumulative_evaluated,
                candidates=tuple(entries),
                message=self._final_message,
                error=self._final_error,
            ),
        )

    def _write_candidate_manifest(self, candidate_id: str, rendered: str) -> str:
        relative = f"candidates/{candidate_id}.yaml"
        replace_text(self._output_dir / relative, rendered)
        return relative

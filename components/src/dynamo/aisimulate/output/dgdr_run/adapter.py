# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live DGDRRun output adapter (``--output dgdr_run``).

Independent of the standalone ``dgd`` adapter, which is called once with the
final recommendation. This adapter subscribes to Sweeper's ``on_round`` and
``on_candidate`` callbacks for the whole search, keeps the *current bounded
projection* of the run in memory, and periodically publishes it as an
atomically replaced ``dgdr_run_snapshot.yaml`` (see ``snapshot.py``) that the
publisher sidecar in the same pod consumes.

Both adapters share one Candidate-to-DGD implementation through
``dynamo.aisimulate.output.dgd.renderers`` (its public
CandidateMaterializationError / DGDGenerationOptions / render_dgd surface);
nothing else is shared.

Design points (all adapter-specific, none of them shared with ``dgd``):

* **Callbacks never block and never lose state.** They only update the latest
  in-memory projection under a short lock and wake the writer. Nothing is
  queued, so nothing can be dropped under pressure: only *file writes* are
  coalesced, never logical state changes.
* **Bounded.** At most ``max_candidates`` points are retained and published,
  no matter how many are evaluated. The bound comes from the run's
  ``recommendation.maxCandidates``; it is not derived from any transport.
* **Rate limited.** The writer publishes the latest complete state at most once
  per ``snapshot_interval_seconds``. The initial snapshot (``start()``) and the
  terminal snapshot (``close()``) bypass the limit.
* **Materialize once.** A point's DGD is rendered the first time it enters the
  projection and cached by its stable id, so progress or ordering changes never
  re-render retained points.
* **Terminal snapshot is a completion barrier.** ``close()`` stops accepting
  updates, folds in everything accepted so far, and synchronously writes the
  terminal snapshot before returning. A failure to write it is raised, never
  swallowed, so the Sweeper container cannot exit successfully without it.

Scalar searches only for now (best-first ordering by score); a Pareto
selection policy is a separate piece of work and is rejected explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from aisimulate.output_adapter import (
    OUTPUT_ADAPTER_API_VERSION,
    RecommendationOutputCallbacks,
)
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.result import SweepResult
from pydantic import BaseModel, ConfigDict, Field

from dynamo.aisimulate.output.dgd.renderers import (
    CandidateMaterializationError,
    DGDGenerationOptions,
    render_dgd,
)
from dynamo.aisimulate.output.dgdr_run.snapshot import (
    SNAPSHOT_FILE_NAME,
    CandidateOutcome,
    DGDRRunSnapshot,
    RunPhase,
    SnapshotCandidate,
    write_snapshot,
)

logger = logging.getLogger(__name__)

_ID_PREFIX = "evaluated-point-"
_MAX_RENDER_ATTEMPTS = 3
_MAX_ERROR_MESSAGE = 500
_WRITE_RETRY_BACKOFF = 1.0
_ID_HEX_LENGTH = 12


class CandidateRecordLike(Protocol):
    """The subset of aisimulate's ``CandidateRecord`` this adapter reads."""

    candidate_id: str
    status: Any  # aisimulate.sweeper.result.CandidateStatus
    score: float | None
    used_gpus: int | None
    config: dict[str, Any]


def _plain(value: Any) -> Any:
    """Copy ``value`` into plain, YAML-safe containers (detached from the caller)."""
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _bounded(text: str) -> str:
    return text[:_MAX_ERROR_MESSAGE]


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the snapshot stays valid JSON/YAML."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


class _RendererUnavailable(Exception):
    """Rendering failed for a reason that is not specific to one candidate."""


def candidate_id_for(parameters: Mapping[str, Any]) -> str:
    """Stable identity of an evaluated point.

    Depends only on the point's resolved parameters: never on rank, list
    position, or the order in which points were observed.
    """
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_ID_PREFIX}{digest[:_ID_HEX_LENGTH]}"


def _is_feasible(record: CandidateRecordLike) -> bool:
    # By value, not identity: callers may pass aisimulate's CandidateStatus
    # enum, a plain string, or a test double.
    return str(getattr(record.status, "value", record.status)) == "feasible"


@dataclass(frozen=True)
class _Retained:
    """One retained point: immutable evaluation facts, no rendering."""

    id: str
    score: float
    used_gpus: int
    parameters: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def rank_key(self) -> tuple[float, int, str]:
        """Ascending sort key: best first.

        Mirrors Sweeper's scalar ``rank()``: highest score, then fewer GPUs,
        then the canonical config JSON. It is only used to bound the live
        retained set; the final snapshot keeps Sweeper's own selection order.
        """
        return (
            -self.score,
            self.used_gpus,
            json.dumps(self.parameters, sort_keys=True, separators=(",", ":")),
        )


@dataclass(frozen=True)
class _Terminal:
    phase: RunPhase
    message: str
    error: str


class DGDRRunOutputConfig(BaseModel):
    """Configuration accepted for ``--output dgdr_run``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
    snapshot_dir: str | None = Field(default=None, min_length=1)
    namespace: str | None = None
    generator: Literal["aic", "direct"] = "aic"
    runtime_image: str = Field(min_length=1)
    runtime_version_override: str | None = None
    num_gpus_per_node: int = Field(gt=0)
    max_candidates: int = Field(default=5, ge=1)
    snapshot_interval_seconds: float = Field(default=5.0, ge=0)

    def generation_options(self) -> DGDGenerationOptions:
        return DGDGenerationOptions(
            runtime_image=self.runtime_image,
            runtime_version_override=self.runtime_version_override,
            namespace=self.namespace,
            num_gpus_per_node=self.num_gpus_per_node,
        )


class DGDRRunOutputAdapter:
    """Publishes the current bounded projection of one Sweeper run.

    One instance per run. Pass the bound methods to ``Sweeper.run`` and use the
    adapter as a context manager (or call ``start()``/``close()``) so the
    terminal snapshot is always written, including when the run raises::

        with DGDRRunOutputAdapter(config, workload, snapshot_dir=path) as adapter:
            sweeper.run(on_round=adapter.on_round, on_candidate=adapter.on_candidate)
    """

    def __init__(
        self,
        config: DGDRRunOutputConfig,
        workload: Any,
        *,
        snapshot_dir: Path | None = None,
        is_pareto: bool = False,
        keep_arrival_order: bool = False,
    ) -> None:
        if is_pareto:
            raise NotImplementedError(
                "dgdr_run only supports scalar objectives so far; Pareto "
                "retention needs its own selection policy rather than an "
                "approximation of top-N-by-score"
            )
        directory = snapshot_dir if snapshot_dir is not None else config.snapshot_dir
        if directory is None:
            raise ValueError("a snapshot directory is required")
        self._directory = Path(directory)
        self._config = config
        self._options = config.generation_options()
        self._workload = workload
        self._max_candidates = config.max_candidates
        self._interval = config.snapshot_interval_seconds

        # Latest in-memory desired state. Callbacks mutate it under _lock.
        self._lock = threading.Lock()
        self._retained: dict[str, _Retained] = {}
        self._round_no = 0
        self._evaluated = 0
        self._terminal: _Terminal | None = None
        self._accepting = True

        # Writer coordination.
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_allowed = float("-inf")

        # Materialization cache, keyed by stable id. Touched only while
        # holding _publish_lock. A failure is retried on later writes up to
        # _MAX_RENDER_ATTEMPTS in total, because a first render can fail
        # transiently (cold start); after that it is final.
        # The final result is already ranked by Sweeper (goal and SLA aware);
        # publishing it must not re-rank it.
        self._keep_arrival_order = keep_arrival_order
        self._publish_lock = threading.Lock()
        self._materialized: dict[str, SnapshotCandidate] = {}
        self._render_attempts: dict[str, int] = {}
        self._writes = 0

    # -- observability ---------------------------------------------------

    @property
    def snapshot_writes(self) -> int:
        """Number of snapshots written so far (initial and terminal included)."""
        return self._writes

    @property
    def snapshot_path(self) -> Path:
        return self._directory / SNAPSHOT_FILE_NAME

    # -- Sweeper callbacks: update latest state, never block, never raise --

    def on_candidate(self, record: CandidateRecordLike) -> None:
        try:
            retained = self._retain(record) if _is_feasible(record) else None
        except Exception:  # a callback must not break the search
            logger.exception("dgdr_run: ignoring candidate that could not be read")
            retained = None
        with self._lock:
            if not self._accepting:
                return
            self._evaluated += 1
            # First observation wins: an evaluated point is immutable, so a
            # re-evaluation of the same point never changes what was published.
            if retained is not None and retained.id not in self._retained:
                self._retained[retained.id] = retained
                if len(self._retained) > self._max_candidates:
                    worst = max(self._retained.values(), key=lambda r: r.rank_key)
                    del self._retained[worst.id]
            self._wake.set()

    def on_round(self, round_number: int, candidates: list[Any]) -> None:
        del candidates  # cumulative feasible list: the adapter tracks its own set
        with self._lock:
            if not self._accepting:
                return
            self._round_no = max(self._round_no, round_number)
            self._wake.set()

    def report_progress(self, *, evaluated: int, round_no: int = 0) -> None:
        """Raise the evaluated count and round to authoritative values (never lowers them)."""
        with self._lock:
            if not self._accepting:
                return
            self._evaluated = max(self._evaluated, evaluated)
            self._round_no = max(self._round_no, round_no)
            self._wake.set()

    @staticmethod
    def _retain(record: CandidateRecordLike) -> _Retained | None:
        if record.score is None or record.used_gpus is None:
            logger.warning(
                "dgdr_run: feasible candidate %s has no score/used_gpus; skipped",
                record.candidate_id,
            )
            return None
        parameters = _plain(dict(record.config))
        score = float(record.score)
        metrics: dict[str, Any] = {
            "score": score if math.isfinite(score) else None,
            "usedGpus": int(record.used_gpus),
        }
        objectives = getattr(record, "objectives", None)
        if objectives:
            metrics["objectives"] = _finite(_plain(dict(objectives)))
        return _Retained(
            id=candidate_id_for(parameters),
            # A NaN score has no place in an ordering; it ranks as the worst.
            score=-math.inf if math.isnan(score) else score,
            used_gpus=int(record.used_gpus),
            parameters=parameters,
            metrics=metrics,
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Write the initial snapshot immediately, then start the writer."""
        if self._thread is not None:
            raise RuntimeError("DGDRRunOutputAdapter already started")
        self._publish()
        self._next_allowed = time.monotonic() + self._interval
        self._thread = threading.Thread(
            target=self._writer_loop, name="dgdr-run-snapshot-writer", daemon=True
        )
        self._thread.start()

    def close(
        self,
        *,
        phase: RunPhase = RunPhase.SUCCEEDED,
        message: str = "",
        error: str = "",
    ) -> None:
        """Write the terminal snapshot; returns only once it is on disk.

        Stops accepting updates, folds in every update accepted before this
        call, bypasses the rate limit, and replaces the file synchronously.
        Raises if the terminal snapshot cannot be written. Safe to call without
        ``start()`` and safe to call twice (the second call is a no-op).
        """
        if not phase.terminal:
            raise ValueError("close() requires a terminal phase")
        with self._lock:
            if self._terminal is not None:
                return
            self._accepting = False
            self._terminal = _Terminal(phase=phase, message=message, error=error)
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        self._publish_or_record_failure()

    def _publish_or_record_failure(self) -> None:
        """Publish; if rendering itself is broken, record that as a terminal failure.

        A broken renderer is a run-level problem, not a candidate-specific one, so it
        must not become ``MATERIALIZATION_FAILED``. The terminal ``Failed`` snapshot
        carries the candidates that were already rendered, and the original exception
        is re-raised. Failures to write the file keep propagating unchanged.
        """
        try:
            self._publish()
        except _RendererUnavailable as unavailable:
            cause = unavailable.__cause__ or unavailable
            logger.error("dgdr_run: renderer unavailable", exc_info=cause)
            with self._lock:
                self._accepting = False
                self._terminal = _Terminal(
                    phase=RunPhase.FAILED,
                    message=_bounded(str(cause)),
                    error=type(cause).__name__,
                )
            self._publish(render=False)
            raise cause

    def finish_search(self) -> None:
        """The search ended normally: flush the last live state and stop the writer.

        Not a terminal state by itself: the terminal ``Succeeded`` snapshot is written
        afterwards by ``write()``, which sees the final result. A search that found no
        feasible candidate never reaches ``write()``, so that case is recorded here as
        ``Failed`` instead of being left looking in progress.
        """
        with self._lock:
            if self._terminal is not None:
                return
            nothing_found = not self._retained
        if nothing_found:
            self.close(phase=RunPhase.FAILED, message="no feasible candidate found")
            return
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        self._publish_or_record_failure()

    def fail(self, error: BaseException) -> None:
        """The search raised: write the terminal ``Failed`` snapshot."""
        self.close(
            phase=RunPhase.FAILED,
            error=type(error).__name__,
            message=_bounded(str(error)),
        )

    def __enter__(self) -> DGDRRunOutputAdapter:
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: Any) -> None:
        if exc_type is None:
            self.close(phase=RunPhase.SUCCEEDED)
            return
        try:
            self.close(phase=RunPhase.FAILED, error=exc_type.__name__)
        except Exception:  # never mask the run's own exception
            logger.exception("dgdr_run: could not write the terminal snapshot")

    # -- writer ----------------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            self._wake.wait()
            if self._stop.is_set():
                return
            delay = self._next_allowed - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                return
            wait = self._interval
            try:
                self._publish()
            except Exception:  # keep the writer alive; retry
                logger.exception("dgdr_run: snapshot write failed; will retry")
                self._wake.set()
                # Even with a zero interval, a persistent failure must not spin.
                wait = max(wait, _WRITE_RETRY_BACKOFF)
            self._next_allowed = time.monotonic() + wait

    def _publish(self, *, render: bool = True) -> None:
        with self._publish_lock:
            with self._lock:
                # Cleared together with the copy so no update is ever lost: an
                # update after this point sets the event again.
                self._wake.clear()
                ordered = (
                    list(self._retained.values())
                    if self._keep_arrival_order
                    else sorted(self._retained.values(), key=lambda r: r.rank_key)
                )
                round_no = self._round_no
                evaluated = self._evaluated
                terminal = self._terminal
            if render:
                candidates = tuple(self._materialize(item) for item in ordered)
            else:
                candidates = tuple(
                    self._materialized[item.id]
                    for item in ordered
                    if item.id in self._materialized
                )
            live_ids = {item.id for item in ordered}
            for stale in set(self._materialized) - live_ids:
                del self._materialized[stale]
                self._render_attempts.pop(stale, None)
            write_snapshot(
                self._directory,
                DGDRRunSnapshot(
                    phase=terminal.phase if terminal else RunPhase.RUNNING,
                    round_no=round_no,
                    evaluated=evaluated,
                    candidates=candidates,
                    message=terminal.message if terminal else "",
                    error=terminal.error if terminal else "",
                ),
            )
            self._writes += 1

    def _materialize(self, item: _Retained) -> SnapshotCandidate:
        cached = self._materialized.get(item.id)
        if cached is not None and (
            cached.outcome is CandidateOutcome.MATERIALIZED
            or self._render_attempts.get(item.id, 0) >= _MAX_RENDER_ATTEMPTS
        ):
            return cached
        self._render_attempts[item.id] = self._render_attempts.get(item.id, 0) + 1
        try:
            manifest = render_dgd(
                _RenderInput(config=item.parameters, used_gpus=item.used_gpus),
                self._workload,
                self._options,
                dgd_name=f"{self._config.name}-{item.id.removeprefix(_ID_PREFIX)}",
                renderer=self._config.generator,
            )
            entry = SnapshotCandidate(
                id=item.id,
                outcome=CandidateOutcome.MATERIALIZED,
                parameters=item.parameters,
                metrics=item.metrics,
                manifest=manifest,
            )
        except CandidateMaterializationError as exc:
            entry = SnapshotCandidate(
                id=item.id,
                outcome=CandidateOutcome.MATERIALIZATION_FAILED,
                parameters=item.parameters,
                metrics=item.metrics,
                error=_bounded(str(exc)),
            )
        except Exception as exc:
            raise _RendererUnavailable(str(exc)) from exc
        self._materialized[item.id] = entry
        return entry


@dataclass(frozen=True)
class _RenderInput:
    """The evaluated candidate as the renderer sees it: its config and GPU count."""

    config: dict[str, Any]
    used_gpus: int


class DGDRRunOutputPlugin:
    """The object AISimulate discovers for ``--output dgdr_run``.

    AISimulate calls ``subscribe(config, context=...)`` once, in the process that
    runs the search, and wires the returned callbacks into the Sweeper: that is
    the live path (initial snapshot immediately, then coalesced snapshots).
    ``write()`` runs afterwards in the CLI process, only when the search
    succeeded with a selection, and publishes the authoritative terminal
    snapshot from the final result. A search that fails or finds nothing never
    reaches ``write()``, so it leaves no terminal snapshot; the publisher
    detects that from the Sweeper container's exit.
    """

    name = "dgdr_run"
    api_version = OUTPUT_ADAPTER_API_VERSION

    def subscribe(
        self, config: Mapping[str, Any], context: Any = None
    ) -> RecommendationOutputCallbacks | None:
        """Validate the configuration and, when possible, start live publishing.

        ``context`` (``RecommendationOutputContext``) carries the Sweeper workload,
        which rendering needs. Without a context or without ``snapshot_dir`` there
        is nothing to publish live, so only the config is validated and the final
        snapshot is still produced by ``write()``.
        """
        resolved = DGDRRunOutputConfig.model_validate(dict(config))
        resolved.generation_options()
        workload = getattr(context, "workload", None)
        if workload is None or resolved.snapshot_dir is None:
            return None
        # Live publishing needs both lifecycle hooks: without on_complete nothing stops
        # the live writer before write() publishes the terminal snapshot, which a pending
        # live write could then overwrite. Older AISimulate releases lack them; the
        # terminal snapshot then comes from write() alone.
        supported = {field.name for field in fields(RecommendationOutputCallbacks)}
        if not {"on_complete", "on_failure"} <= supported:
            logger.info(
                "dgdr_run: AISimulate has no lifecycle callbacks; not publishing live"
            )
            return None
        adapter = DGDRRunOutputAdapter(resolved, workload)
        adapter.start()
        return RecommendationOutputCallbacks(
            on_candidate=adapter.on_candidate,
            on_round=adapter.on_round,
            on_complete=adapter.finish_search,
            on_failure=adapter.fail,
        )

    def live(
        self,
        config: Mapping[str, Any],
        *,
        workload: Any,
        is_pareto: bool = False,
    ) -> DGDRRunOutputAdapter:
        resolved = DGDRRunOutputConfig.model_validate(dict(config))
        return DGDRRunOutputAdapter(resolved, workload, is_pareto=is_pareto)

    def write(
        self,
        config: Mapping[str, Any],
        *,
        result: SweepResult,
        output_dir: Path,
    ) -> Sequence[str | Path]:
        resolved = DGDRRunOutputConfig.model_validate(dict(config))
        if result.views.pareto_front:
            raise NotImplementedError("dgdr_run does not support Pareto results yet")
        selected = list(result.selected_candidates)[: resolved.max_candidates]
        if not selected:
            raise CandidateMaterializationError("no feasible candidate found")
        workload = SmartSearchConfig.model_validate(result.provenance.config).workload
        directory = Path(resolved.snapshot_dir or output_dir)
        adapter = DGDRRunOutputAdapter(
            resolved, workload, snapshot_dir=directory, keep_arrival_order=True
        )
        for candidate in selected:
            adapter.on_candidate(_as_feasible(candidate))
        counts = getattr(result, "counts", None)
        evaluated = getattr(counts, "evaluated", None)
        adapter.report_progress(
            evaluated=evaluated if isinstance(evaluated, int) else 0,
            # The final result does not carry the round, but the live snapshot
            # written by the search process does; progress never goes backwards.
            round_no=_last_published_round(directory),
        )
        adapter.close(phase=RunPhase.SUCCEEDED)
        # AISimulate requires reported artifacts to live under output_dir; a snapshot
        # on the shared volume does not, and is consumed by the publisher instead.
        if directory.resolve() == Path(output_dir).resolve():
            return [Path(SNAPSHOT_FILE_NAME)]
        return []


def _last_published_round(directory: Path) -> int:
    """The round recorded by the live snapshot already in ``directory``, else 0."""
    try:
        loaded = yaml.safe_load((directory / SNAPSHOT_FILE_NAME).read_text())
        round_no = loaded["progress"]["round"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return 0
    return round_no if isinstance(round_no, int) and round_no > 0 else 0


@dataclass(frozen=True)
class _FinalCandidate:
    candidate_id: str
    config: dict[str, Any]
    score: float | None
    used_gpus: int | None
    objectives: dict[str, float] | None = None
    status: str = "feasible"


def _as_feasible(candidate: Any) -> _FinalCandidate:
    score = getattr(candidate, "score", None)
    used_gpus = getattr(candidate, "used_gpus", None)
    if score is None or used_gpus is None:
        raise CandidateMaterializationError(
            "selected candidate has no score/used_gpus, so it cannot be ranked"
        )
    return _FinalCandidate(
        candidate_id=str(getattr(candidate, "candidate_id", "")),
        config=dict(candidate.config),
        score=score,
        used_gpus=used_gpus,
        objectives=getattr(candidate, "objectives", None),
    )


def create_adapter() -> DGDRRunOutputPlugin:
    """Create the Dynamo plugin discovered by AISimulate."""
    return DGDRRunOutputPlugin()


__all__ = [
    "CandidateRecordLike",
    "DGDRRunOutputAdapter",
    "DGDRRunOutputConfig",
    "DGDRRunOutputPlugin",
    "candidate_id_for",
    "create_adapter",
]

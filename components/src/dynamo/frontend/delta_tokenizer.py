# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, process-local state for certified incremental prompt tokenization.

The session provider owns the exact repair algorithm. This module owns admission,
isolation, memory reservations and publication. It never guesses a repair tail.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import secrets
import time
import weakref
from array import array
from collections import Counter, OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from threading import Event, Lock, Thread
from typing import Any, Protocol

from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_client import Counter as MetricCounter
from prometheus_client import Gauge, Histogram
from typing_extensions import Self

logger = logging.getLogger(__name__)
# Queue length is bounded by the shared process entry/byte reservations.
_SEED_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="delta-tokenizer-seed"
)


class UnsupportedTransition(Exception):
    """The provider could not certify this input or state transition."""


class PatchSession(Protocol):
    def append(self, text: str) -> tuple[int, list[int]]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class TokenizerIdentity:
    model: str
    revision: str
    artifact: str
    loader: str
    configuration: str
    backend: str


@dataclass(frozen=True)
class DeltaConfig:
    max_entries: int = 16
    max_bytes: int = 1024 * 1024 * 1024
    ttl_secs: float = 300
    verify_rate: float = 0

    def __post_init__(self) -> None:
        if self.max_entries <= 0 or self.max_bytes <= 0:
            raise ValueError("delta tokenizer entry and byte limits must be positive")
        if not math.isfinite(self.ttl_secs) or self.ttl_secs <= 0:
            raise ValueError("delta tokenizer TTL must be finite and positive")
        if not math.isfinite(self.verify_rate) or not 0 <= self.verify_rate <= 1:
            raise ValueError("delta tokenizer verification rate must be in [0, 1]")


class DeltaBudget:
    """Shared process budget across model-specific managers.

    A manager can evict its own idle entries. If another model owns the remaining
    budget, use the reference until those reservations expire or are released.
    """

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = 0
        self._bytes = 0
        self._lock = Lock()

    def adjust(self, entries: int, size: int) -> bool:
        with self._lock:
            if (
                self._entries + entries > self.max_entries
                or self._bytes + size > self.max_bytes
            ):
                return False
            self._entries += entries
            self._bytes += size
            return True


class DeltaMetrics:
    """Fixed-label metrics shared by managers, with no session or model labels."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.registry = registry
        self.events = MetricCounter(
            "dynamo_frontend_delta_tokenizer_events",
            "Experimental delta tokenizer decisions by fixed outcome.",
            ["outcome"],
            registry=registry,
        )
        self.duration = Histogram(
            "dynamo_frontend_delta_tokenizer_seconds",
            "Tokenization duration by execution path.",
            ["path"],
            registry=registry,
        )
        self.processed = MetricCounter(
            "dynamo_frontend_delta_tokenizer_incremental_bytes",
            "Bytes submitted to certified incremental repair.",
            registry=registry,
        )
        self.avoided = MetricCounter(
            "dynamo_frontend_delta_tokenizer_avoided_bytes",
            "Full-input bytes avoided on unsampled incremental requests.",
            registry=registry,
        )
        self.entries = Gauge(
            "dynamo_frontend_delta_tokenizer_entries",
            "Live entries including in-flight reservations.",
            registry=registry,
        )
        self.retained = Gauge(
            "dynamo_frontend_delta_tokenizer_retained_bytes",
            "Conservative approximate bytes including in-flight reservations.",
            registry=registry,
        )


DEFAULT_METRICS = DeltaMetrics(CollectorRegistry())


@dataclass
class _Entry:
    template: str
    created: float
    accessed: float
    reserved: int
    generation: int
    text: str = ""
    tokens: tuple[int, ...] = ()
    text_digest: bytes = b""
    tokens_digest: bytes = b""
    reference_seeded: bool = False
    session: PatchSession | None = None
    busy: bool = True
    retired: bool = False
    released: bool = False


def _cleanup_loop(
    owner: weakref.ReferenceType[DeltaTokenizer], stop: Event, interval: float
) -> None:
    while not stop.wait(interval):
        instance = owner()
        if instance is None:
            return
        instance.expire()
        del instance


class DeltaTokenizer:
    """One immutable reference identity, with one writer per session.

    A busy entry delegates competing requests to the reference without publishing.
    Independent entries never hold the index lock while encoding. Retirement keeps
    in-flight bytes reserved until the writer exits, preventing eviction races.
    """

    def __init__(
        self,
        reference: Callable[[str], list[int]],
        identity: TokenizerIdentity,
        session_factory: Callable[[], PatchSession],
        config: DeltaConfig,
        *,
        base_bytes: int = 0,
        fixed_bytes: int = 0,
        reference_after_close: Callable[[str], list[int]] | None = None,
        background_seed: bool = False,
        min_prompt_bytes: int = 0,
        budget: DeltaBudget | None = None,
        metrics: DeltaMetrics = DEFAULT_METRICS,
        clock: Callable[[], float] = time.monotonic,
        start_cleanup: bool = True,
    ) -> None:
        self._initialized = False
        self.reference = reference
        self._reference_after_close = reference_after_close or reference
        self.identity = identity
        self._factory = session_factory
        self.config = config
        self._background_seed = background_seed
        self._min_prompt_bytes = min_prompt_bytes
        self._jobs: set[Future[None]] = set()
        self._base_bytes = base_bytes
        self._budget = budget or DeltaBudget(config.max_entries, config.max_bytes)
        self.metrics = metrics
        self.clock = clock
        self._lock = Lock()
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self._counts: Counter[str] = Counter()
        self._bytes = 0
        self._fixed_bytes = fixed_bytes
        if not self._budget.adjust(0, fixed_bytes):
            raise ValueError(
                "delta tokenizer process budget cannot admit reference snapshot"
            )
        self._bytes = fixed_bytes
        self.metrics.retained.inc(fixed_bytes)
        self._slots = 0
        self._generation = 0
        self._secret = secrets.token_bytes(32)
        self._closed = False
        self._disabled = False
        self._stop = Event()
        self._finalizer = weakref.finalize(self, self._stop.set)
        self._initialized = True
        if start_cleanup:
            Thread(
                target=_cleanup_loop,
                args=(weakref.ref(self), self._stop, min(config.ttl_secs, 1.0)),
                name="delta-tokenizer-expiry",
                daemon=True,
            ).start()

    def __del__(self) -> None:
        # Frontend model retirement may release the owning engine without an
        # explicit close hook. Pending workers retain self until they exit.
        if self._initialized:
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _record(self, outcome: str) -> None:
        # All callers pass constants, never user data or upstream exception text.
        with self._lock:
            self._counts[outcome] += 1
        self.metrics.events.labels(outcome).inc()
        logger.debug("delta tokenizer decision", extra={"delta_outcome": outcome})

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._counts,
                "entries": self._slots,
                "retained_bytes": self._bytes,
                "mismatches": self._counts["mismatches"],
            }

    def _charge(self, slots: int, size: int) -> bool:
        # Called only while holding the index lock.
        if not self._budget.adjust(slots, size):
            return False
        self._slots += slots
        self._bytes += size
        self.metrics.entries.inc(slots)
        self.metrics.retained.inc(size)
        return True

    def _retire(self, key: bytes, entry: _Entry) -> bool:
        # Return whether the caller owns disposal. A running writer disposes itself.
        if self._entries.get(key) is entry:
            del self._entries[key]
        entry.retired = True
        if not entry.busy and not entry.released:
            entry.released = True
            self._charge(-1, -entry.reserved)
            self._release_fixed_if_closed()
            return True
        return False

    def _release_fixed_if_closed(self) -> None:
        if self._closed and self._slots == 0 and self._fixed_bytes:
            self._charge(0, -self._fixed_bytes)
            self._fixed_bytes = 0

    @staticmethod
    def _dispose(entries: list[_Entry]) -> None:
        for entry in entries:
            if entry.session is not None:
                entry.session.close()
                entry.session = None
            entry.text = ""
            entry.tokens = ()

    def expire(self) -> None:
        now = self.clock()
        disposed = []
        expired = 0
        with self._lock:
            # Access order also orders idle deadlines. Bound each cleanup pass.
            for key, entry in list(islice(self._entries.items(), 16)):
                if now - entry.accessed < self.config.ttl_secs:
                    break
                expired += 1
                if self._retire(key, entry):
                    disposed.append(entry)
        self._dispose(disposed)
        for _ in range(expired):
            self._record("expired")

    def close(self) -> None:
        self._stop.set()
        disposed = []
        with self._lock:
            self._closed = True
            self.reference = self._reference_after_close
            for key, entry in list(self._entries.items()):
                if self._retire(key, entry):
                    disposed.append(entry)
            self._release_fixed_if_closed()
        self._dispose(disposed)

    def _full(self, reason: str, prompt: str) -> list[int]:
        self._record("fallback_" + reason)
        with self.metrics.duration.labels("reference").time():
            return self.reference(prompt)

    def tokenize(
        self,
        session_id: str | None,
        rendered_prompt: str,
        *,
        scope: str | None,
        template_identity: str,
        tokenizer_identity: TokenizerIdentity | None = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> list[int]:
        if tokenizer_identity is not None and tokenizer_identity != self.identity:
            self.close()
            return self._full("identity", rendered_prompt)
        if not session_id or not scope or len(session_id) > 1024 or len(scope) > 1024:
            return self._full("missing_session", rendered_prompt)
        if cancelled():
            return self._full("cancelled", rendered_prompt)
        try:
            raw_size = len(rendered_prompt.encode("utf-8"))
        except UnicodeEncodeError:
            return self._full("unicode", rendered_prompt)
        if raw_size < self._min_prompt_bytes:
            return self._full("short", rendered_prompt)
        # Includes several Python token/text copies, native repair state and buffers.
        # The provider contributes a conservative loaded-model estimate separately.
        needed = self._base_bytes + 2048 + 192 * raw_size
        if needed + self._fixed_bytes > self.config.max_bytes:
            return self._full("budget", rendered_prompt)
        key = hmac.digest(
            self._secret,
            len(scope).to_bytes(4, "big") + scope.encode() + session_id.encode(),
            "sha256",
        )
        self.expire()
        disposed: list[_Entry] = []
        decisions: list[str] = []
        entry: _Entry | None = None
        reason: str | None = None
        with self._lock:
            if self._closed or self._disabled:
                reason = "unavailable"
            else:
                entry = self._entries.get(key)
                if entry is not None and entry.busy:
                    reason = "busy"
                elif entry is not None and (
                    entry.template != template_identity
                    or not rendered_prompt.startswith(entry.text)
                ):
                    decisions.append(
                        "invalidated_template"
                        if entry.template != template_identity
                        else "invalidated_non_append"
                    )
                    if self._retire(key, entry):
                        disposed.append(entry)
                    entry = None

                if reason is None:
                    growth = needed - entry.reserved if entry else needed
                    while self._bytes + growth > self.config.max_bytes or (
                        entry is None and self._slots >= self.config.max_entries
                    ):
                        victim = next(
                            (
                                (k, v)
                                for k, v in self._entries.items()
                                if v is not entry and not v.busy
                            ),
                            None,
                        )
                        if victim is None:
                            reason = "budget"
                            break
                        victim_key, victim_entry = victim
                        if self._retire(victim_key, victim_entry):
                            disposed.append(victim_entry)
                        decisions.append("evicted")
                    if reason is None:
                        now = self.clock()
                        if entry is None:
                            if self._charge(1, needed):
                                self._generation += 1
                                entry = _Entry(
                                    template_identity,
                                    now,
                                    now,
                                    needed,
                                    self._generation,
                                )
                                self._entries[key] = entry
                            else:
                                reason = "budget"
                        else:
                            if self._charge(0, growth):
                                entry.reserved = needed
                                entry.accessed = now
                                entry.busy = True
                                self._entries.move_to_end(key)
                            else:
                                reason = "budget"
        self._dispose(disposed)
        for decision in decisions:
            self._record(decision)
        if reason is not None:
            return self._full(reason, rendered_prompt)
        assert entry is not None
        generation = entry.generation
        keep = False
        deferred = False
        start_gate: Event | None = None
        try:
            with self.metrics.duration.labels("candidate").time():
                cold = entry.session is None
                duplicate = entry.reference_seeded and entry.text == rendered_prompt
                if entry.reference_seeded and not self._valid_snapshot(entry):
                    return self._full("corrupt", rendered_prompt)
                if cold and not duplicate:
                    expected = self.reference(rendered_prompt)
                    if self._background_seed:
                        self._record("cold")
                        if not entry.reference_seeded:
                            # First observe a reference-only snapshot. Only a
                            # confirmed second append justifies costly admission.
                            if not cancelled():
                                keep = self._publish(
                                    key, entry, rendered_prompt, expected, generation
                                )
                            return expected
                        # The request owns expected. The worker owns a separate
                        # buffer, so downstream mutation cannot alter cached IDs.
                        start_gate = Event()
                        job = _SEED_EXECUTOR.submit(
                            self._seed,
                            start_gate,
                            key,
                            entry,
                            generation,
                            rendered_prompt,
                            expected.copy(),
                            cancelled,
                        )
                        with self._lock:
                            self._jobs.add(job)
                        job.add_done_callback(self._job_finished)
                        deferred = True
                        return expected
                    entry.session = self._factory()
                    cut, suffix = entry.session.append(rendered_prompt)
                    candidate = list(suffix) if cut == 0 else []
                    self._record("verified")
                    if candidate != expected:
                        return self._mismatch(expected, len(candidate))
                    self._record("cold")
                elif duplicate:
                    candidate = list(entry.tokens)
                    self._record("duplicate")
                else:
                    appended = rendered_prompt[len(entry.text) :]
                    assert entry.session is not None
                    cut, suffix = entry.session.append(appended)
                    if not 0 <= cut <= len(entry.tokens):
                        return self._full("patch", rendered_prompt)
                    candidate = list(entry.tokens[:cut])
                    candidate.extend(suffix)
                    sampled = self.config.verify_rate > 0 and (
                        secrets.randbelow(1_000_000)
                        < self.config.verify_rate * 1_000_000
                    )
                    if sampled:
                        self._record("verified")
                        expected = self.reference(rendered_prompt)
                        if candidate != expected:
                            return self._mismatch(expected, len(candidate))
                if cancelled():
                    return self._full("cancelled", rendered_prompt)
                if not cold and not duplicate:
                    if not sampled:
                        self.metrics.avoided.inc(len(entry.text.encode("utf-8")))
                    self.metrics.processed.inc(len(appended.encode("utf-8")))
                    self._record("incremental")
                keep = self._publish(key, entry, rendered_prompt, candidate, generation)
                return candidate
        except UnsupportedTransition:
            return self._full("uncertified", rendered_prompt)
        finally:
            if deferred:
                assert start_gate is not None
                start_gate.set()
            else:
                self._finish(key, entry, keep)

    @staticmethod
    def _valid_snapshot(entry: _Entry) -> bool:
        try:
            return (
                hashlib.sha256(entry.text.encode("utf-8")).digest() == entry.text_digest
                and hashlib.sha256(array("I", entry.tokens)).digest()
                == entry.tokens_digest
            )
        except (UnicodeEncodeError, OverflowError, TypeError):
            return False

    def _publish(
        self,
        key: bytes,
        entry: _Entry,
        prompt: str,
        candidate: list[int],
        generation: int,
    ) -> bool:
        tokens = tuple(candidate)
        tokens_digest = hashlib.sha256(array("I", tokens)).digest()
        text_digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        with self._lock:
            if (
                self._entries.get(key) is not entry
                or entry.generation != generation
                or entry.retired
                or self._closed
                or self._disabled
            ):
                return False
            entry.text = prompt
            entry.tokens = tokens
            entry.tokens_digest = tokens_digest
            entry.text_digest = text_digest
            entry.reference_seeded = True
            entry.generation += 1
            return True

    def _finish(self, key: bytes, entry: _Entry, keep: bool) -> None:
        dispose = False
        with self._lock:
            entry.busy = False
            if not keep:
                dispose = self._retire(key, entry)
        if dispose:
            self._dispose([entry])

    def _job_finished(self, job: Future[None]) -> None:
        with self._lock:
            self._jobs.discard(job)
        # _seed reports provider refusals normally. Unexpected implementation
        # failures remain observable and are never converted to successful state.
        if job.exception() is not None:
            self._record("seed_error")
            logger.error("delta tokenizer background seed failed")

    def wait_for_idle(self, timeout: float = 60) -> None:
        """Wait for already admitted seeds, for deterministic validation only."""
        with self._lock:
            jobs = list(self._jobs)
        for job in jobs:
            job.result(timeout=timeout)

    def _seed(
        self,
        start_gate: Event,
        key: bytes,
        entry: _Entry,
        generation: int,
        prompt: str,
        expected: list[int],
        cancelled: Callable[[], bool],
    ) -> None:
        start_gate.wait()
        keep = False
        started = time.perf_counter()
        try:
            if cancelled() or entry.retired or self._closed:
                return
            entry.session = self._factory()
            if cancelled() or entry.retired or self._closed:
                return
            cut, suffix = entry.session.append(prompt)
            self._record("verified")
            if cut != 0 or suffix != expected:
                self._mismatch(expected, len(suffix))
                return
            if cancelled():
                self._record("invalidated_cancelled")
                return
            keep = self._publish(key, entry, prompt, expected, generation)
        except UnsupportedTransition:
            self._record("fallback_uncertified")
        finally:
            self.metrics.duration.labels("seed").observe(time.perf_counter() - started)
            self._finish(key, entry, keep)

    def _mismatch(self, expected: list[int], observed_count: int) -> list[int]:
        self._record("mismatches")
        logger.warning(
            "delta tokenizer verification mismatch; reuse disabled",
            extra={
                "delta_reason": "token_mismatch",
                "reference_token_count": len(expected),
                "observed_token_count": observed_count,
            },
        )
        disposed = []
        with self._lock:
            self._disabled = True
            for key, entry in list(self._entries.items()):
                if self._retire(key, entry):
                    disposed.append(entry)
        self._dispose(disposed)
        return expected


def delta_request_context(
    manager: DeltaTokenizer | None,
    context: Any | None,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Eligibility at the Python engine boundary; metadata is never authentication."""
    if manager is None:
        if enabled:
            DEFAULT_METRICS.events.labels("fallback_loader").inc()
            logger.debug(
                "delta tokenizer decision", extra={"delta_outcome": "fallback_loader"}
            )
        return {}
    if (
        context is None
        or context.metadata.get("dynamo.frontend.api") != "chat_completions"
    ):
        manager._record("fallback_api")
        return {}
    session = context.metadata.get("delta-tokenizer-session")
    scope = context.metadata.get("delta-tokenizer-scope")
    if not session or not scope:
        manager._record("fallback_missing_session")
        return {}
    return {
        "delta_tokenizer": manager,
        "delta_session_id": session,
        "delta_scope": scope,
        "delta_cancelled": lambda: context.is_stopped() or context.is_killed(),
    }

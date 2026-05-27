# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario checks for fault tolerance testing.

Checks validate results after all events complete. Each check has:
- validate(ctx): Assert conditions, raises AssertionError on failure
- description: Human-readable description for logging
- get_load(ctx, name): Helper to find StartLoad events by name

To create a custom check, subclass Check and implement validate() and description.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

__all__ = [
    "Check",
    "ZeroErrors",
    "MaxErrors",
    "MinRequests",
    "LoadStopped",
    "LoadCompleted",
    "ServiceLogContains",
    "ServiceLogNotContains",
    "RankProcessCount",
    "WorkerPanics",
    # 2026-05-03 disagg-cascade repro suite (PR #8254 panic, KV cascade, etc.)
    "ServiceLogPatternRate",
    "KvCacheUsagePeak",
    "PodMemoryGrowth",
    "CascadeAfterFault",
    "EngineDeathDetected",
    "RestartCountIncreased",
]

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.events import StartLoad
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Check Base Class
# =============================================================================


@dataclass
class Check(ABC):
    """Base class for result validation.

    Checks receive ScenarioContext and can access:
    - self.get_load(ctx, name) to find StartLoad and get results
    - ctx.deployment.collect_service_logs() for service logs
    """

    @abstractmethod
    def validate(self, ctx: "ScenarioContext") -> None:
        """Validate results. Raises AssertionError on failure."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the check."""
        pass

    def get_load(self, ctx: "ScenarioContext", name: str) -> "StartLoad | None":
        """Find StartLoad event by name."""
        from tests.fault_tolerance.deploy.events import StartLoad

        for event in ctx.events:
            if isinstance(event, StartLoad) and event.name == name:
                return event
        return None


# =============================================================================
# Check Implementations
# =============================================================================


@dataclass
class ZeroErrors(Check):
    """Assert zero errors in load results."""

    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        error_result = load.results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        ctx.logger.info(f"ZeroErrors: error_count = {error_count}")
        assert error_count == 0, f"Expected 0 errors, got {error_count}"

    @property
    def description(self) -> str:
        return f"Zero errors ('{self.name}')"


@dataclass
class MaxErrors(Check):
    """Assert errors below a threshold."""

    max_errors: int
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        error_result = load.results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        ctx.logger.info(
            f"MaxErrors: error_count = {error_count}, max = {self.max_errors}"
        )
        assert (
            error_count <= self.max_errors
        ), f"Expected at most {self.max_errors} errors, got {error_count}"

    @property
    def description(self) -> str:
        return f"Max {self.max_errors} errors ('{self.name}')"


@dataclass
class MinRequests(Check):
    """Assert minimum number of successful requests."""

    min_count: int
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        request_count = load.results.get("request_count", {}).get("avg", 0)

        ctx.logger.info(
            f"MinRequests: request_count = {request_count}, min = {self.min_count}"
        )
        assert (
            request_count >= self.min_count
        ), f"Expected at least {self.min_count} requests, got {request_count}"

    @property
    def description(self) -> str:
        return f"Min {self.min_count} requests ('{self.name}')"


@dataclass
class LoadStopped(Check):
    """Assert the load was stopped early (e.g. by a ``StopLoad`` event).

    Pairs with ``LoadCompleted`` for the inverse case. Both read aiperf's
    ``was_cancelled`` flag — True when the load job was terminated mid-run,
    False when it ran to its configured request count or duration.
    """

    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"
        stopped = load.results.get("was_cancelled", False)
        ctx.logger.info(f"LoadStopped: stopped = {stopped}")
        assert stopped, "Expected load to be stopped early, but it completed naturally"

    @property
    def description(self) -> str:
        return f"Load stopped early ('{self.name}')"


@dataclass
class LoadCompleted(Check):
    """Assert the load ran to completion (was NOT stopped early).

    Pairs with ``LoadStopped``. Use after a ``WaitForLoadCompletion`` event
    or whenever the scenario expects aiperf to finish on its own.
    """

    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"
        stopped = load.results.get("was_cancelled", False)
        ctx.logger.info(f"LoadCompleted: stopped = {stopped}")
        assert (
            not stopped
        ), "Expected load to complete naturally, but it was stopped early"

    @property
    def description(self) -> str:
        return f"Load completed naturally ('{self.name}')"


@dataclass
class ServiceLogContains(Check):
    """Assert a service log contains a pattern."""

    service: str
    pattern: str

    def validate(self, ctx: "ScenarioContext") -> None:
        logs = _get_service_logs(ctx)
        log = logs.get(self.service, "")
        ctx.logger.info(
            f"ServiceLogContains: checking '{self.pattern}' in {self.service}"
        )
        assert self.pattern in log, (
            f"Pattern '{self.pattern}' not found in {self.service} logs "
            f"(log length: {len(log)} chars)"
        )

    @property
    def description(self) -> str:
        return f"Service '{self.service}' logs contain '{self.pattern}'"


@dataclass
class WorkerPanics(Check):
    """Scan collected service logs for Rust panic strings and grade per
    policy.

    Designed as a generic detector for unhandled Rust panics in any
    Dynamo service (frontend, workers). Originally added for PR #8254
    (TCP panic -> warn+break), where the precondition is high-concurrency
    load + induced TCP RST and the assertion is that no
    ``tcp/client.rs`` / ``tcp/server.rs`` panic landed in the logs.

    Policy:
      - ``acceptable=False`` (default): assert no panic lines match.
      - ``acceptable=True``: log the count, no assertion.
      - ``max_count``: optional ceiling; only enforced when set. Useful
        for "tolerate up to N panics" sweeps.

    The default ``patterns`` match the standard Rust panic preamble plus
    the two PR #8254 sites; pass your own list to narrow or broaden.
    """

    services: list
    acceptable: bool = False
    max_count: int | None = None
    patterns: list = field(
        default_factory=lambda: [
            r"thread '.*' panicked at",
            r"panicked at .*tcp/client\.rs",
            r"panicked at .*tcp/server\.rs",
            r"RUST_BACKTRACE",
        ]
    )

    def validate(self, ctx) -> None:
        logs = _get_service_logs(ctx)
        compiled = [re.compile(p) for p in self.patterns]
        hits: list[str] = []
        per_service_counts: dict[str, int] = {}
        for svc in self.services:
            text = logs.get(svc, "") or ""
            svc_hits = 0
            for line in text.splitlines():
                if any(c.search(line) for c in compiled):
                    hits.append(f"{svc}: {line.strip()}")
                    svc_hits += 1
            per_service_counts[svc] = svc_hits
        total = len(hits)
        ctx.logger.info(
            f"WorkerPanics: total={total} per-service={per_service_counts} "
            f"(acceptable={self.acceptable}, max_count={self.max_count})"
        )
        # Log the first ~20 distinct hits so we can see what we got.
        for h in hits[:20]:
            ctx.logger.info(f"  panic-line: {h}")

        if self.acceptable:
            return  # report-only mode
        if self.max_count is not None:
            assert total <= self.max_count, (
                f"WorkerPanics: {total} > max_count={self.max_count} in "
                f"services {self.services}; first hits:\n"
                + "\n".join(f"  - {h}" for h in hits[:20])
            )
        else:
            assert total == 0, (
                f"WorkerPanics: {total} panic line(s) found in "
                f"services {self.services}; first hits:\n"
                + "\n".join(f"  - {h}" for h in hits[:20])
            )

    @property
    def description(self) -> str:
        policy = (
            "acceptable=yes"
            if self.acceptable
            else (
                f"max={self.max_count}"
                if self.max_count is not None
                else "must be zero"
            )
        )
        return f"WorkerPanics ({policy}) in {', '.join(self.services)}"


@dataclass
class RankProcessCount(Check):
    """Assert that each pod in the named services runs exactly the expected
    number of processes matching ``process_name``.

    For a vLLM worker pod at TP=N the expected pattern is ``1 launcher
    + N rank workers``; with ``process_name="dynamo.vllm"`` (or another
    substring that matches both launcher and ranks) ``expected=N+1``.
    If your matcher targets only the ranks (e.g. a tighter regex / a
    PPID filter applied elsewhere), pass ``expected=N``.

    The check exec's ``ps`` inside each pod and counts processes whose
    command line contains ``process_name``. Per-pod actual counts are
    logged so a mismatch shows you exactly which pod is off and by how
    many.

    Example::

        RankProcessCount(
            services=["VllmDecodeWorker", "VllmPrefillWorker"],
            process_name="dynamo.vllm",
            expected=3,  # launcher + 2 ranks for TP=2
        )
    """

    services: list
    process_name: str
    expected: int

    def validate(self, ctx) -> None:
        service_pod_dict = ctx.deployment.get_pods(self.services)
        failures: list[str] = []
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                processes = ctx.deployment.get_processes(pod)
                matches = [p for p in processes if self.process_name in p.command]
                actual = len(matches)
                ctx.logger.info(
                    f"RankProcessCount: {service_name}/{pod.name}: "
                    f"{actual} processes matching '{self.process_name}' "
                    f"(expected {self.expected})"
                )
                if actual != self.expected:
                    failures.append(
                        f"{service_name}/{pod.name}: got {actual}, "
                        f"expected {self.expected}"
                    )
        assert not failures, (
            f"RankProcessCount mismatches for process '{self.process_name}':\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    @property
    def description(self) -> str:
        return (
            f"Each pod in {', '.join(self.services)} runs exactly "
            f"{self.expected} '{self.process_name}' processes"
        )


@dataclass
class ServiceLogNotContains(Check):
    """Assert a service log does NOT contain a pattern."""

    service: str
    pattern: str

    def validate(self, ctx: "ScenarioContext") -> None:
        logs = _get_service_logs(ctx)
        log = logs.get(self.service, "")
        ctx.logger.info(
            f"ServiceLogNotContains: checking '{self.pattern}' NOT in {self.service}"
        )
        assert (
            self.pattern not in log
        ), f"Pattern '{self.pattern}' should NOT be in {self.service} logs but was found"

    @property
    def description(self) -> str:
        return f"Service '{self.service}' logs do NOT contain '{self.pattern}'"


# =============================================================================
# 2026-05-03 disagg-cascade repro suite
# =============================================================================
#
# Six checks added 2026-05-12 for the e2e DGD scenarios that reproduce the
# PR #8254 TCP panic, the kv-router radix_tree warning storm under
# AGG-enum-decode, KV-saturation cascades, and the engine-hang /
# liveness-restart signature. Each follows the same Check.validate(ctx)
# contract as the older checks above.
#
# Design notes:
# - Reading time-series data: prefer `server_metrics_export.jsonl` from any
#   StartLoad's load-dir under `ctx.log_dir/load/load-<name>-*/` (aiperf
#   captures one record per 1 Hz scrape per scraped endpoint). Helpers below.
# - Reading log text: `ctx.deployment.collect_service_logs()` returns
#   `{service_name: str}` (the catted pod logs).
# - Reading pod status: `ctx.deployment.get_pods([service_name])` returns
#   live Pod objects with `.raw['status']['containerStatuses']`.

import json as _json  # noqa: E402
import os as _os  # noqa: E402
from glob import escape as _glob_escape  # noqa: E402
from glob import glob as _glob  # noqa: E402


def _get_service_logs(ctx) -> dict:
    """Return per-service catted pod logs.

    Prefer the pre-teardown snapshot on ``ctx.service_logs`` when
    populated. Otherwise read from disk: the framework extracts each
    pod's log via the PVC-extractor sidecar into
    ``<log_dir>/<service-name-lowercase>/<podname>_<ts>.log``.
    """
    snap = getattr(ctx, "service_logs", None) or {}
    if snap:
        return snap

    log_dir = getattr(ctx, "log_dir", None)
    if not log_dir or not _os.path.isdir(log_dir):
        return {}

    out: dict = {}
    # One sub-directory per service (lowercased name).
    for entry in _os.listdir(log_dir):
        sub = _os.path.join(log_dir, entry)
        if not _os.path.isdir(sub) or entry in ("load",):
            continue
        # Concatenate every <pod>*.log under this service dir.
        catted_parts = []
        log_files = sorted(_glob(_os.path.join(_glob_escape(sub), "*.log")))
        for f in log_files:
            try:
                with open(f, "r", errors="replace") as fh:
                    catted_parts.append(fh.read())
            except OSError:
                continue
        if catted_parts:
            text = "\n".join(catted_parts)
            # Disk dir is lowercase ("frontend", "vllmdecodeworker") but
            # check consumers refer to services by PascalCase
            # ("Frontend", "VllmDecodeWorker"). Map both keys to the
            # same text so either casing resolves.
            out[entry] = text
            for cased in ("Frontend", "VllmDecodeWorker", "VllmPrefillWorker"):
                if cased.lower() == entry:
                    out[cased] = text
                    break
    return out


def _find_load_dirs(ctx) -> list:
    """Return all load-*/ subdirs of the test's log_dir (one per StartLoad).

    The log_dir embeds the pytest parametrize id (``test_name[arm-id]``)
    which contains ``[`` ``]`` — glob metacharacters that get parsed as
    character classes and silently match nothing. Use ``glob.escape``
    on the parent path so the bracketed test name survives intact.
    """
    log_dir = getattr(ctx, "log_dir", None)
    if not log_dir:
        return []
    base = _os.path.join(_glob_escape(log_dir), "load", "load-*")
    out = sorted(_glob(base))
    return [p for p in out if _os.path.isdir(p)]


def _iter_jsonl(path: str):
    """Yield parsed JSON records from a JSONL file, skipping malformed rows."""
    if not _os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield _json.loads(line)
            except _json.JSONDecodeError:
                continue


def _resolve_pod_names(ctx, services: list) -> dict:
    """Snapshot ``{service: [pod_name, ...]}`` for the given services."""
    out = {}
    for svc in services:
        pods_by_svc = ctx.deployment.get_pods([svc])
        out[svc] = sorted(p.name for p in (pods_by_svc.get(svc) or []))
    return out


@dataclass
class ServiceLogPatternRate(Check):
    """Assert a regex pattern fires at ≥ ``min_rate_per_sec`` (or == 0) in
    the collected service logs over the entire run window.

    Time-source: log lines are scanned for the first / last ISO-8601-ish
    timestamp encountered; rate = count / (last - first).seconds. Falls
    back to a conservative 1-second floor on the denominator if log
    timestamps can't be parsed.

    Examples::

        # radix_tree storm on AGG-enum decode pod
        ServiceLogPatternRate(
            services=["VllmDecodeWorker"],
            pattern=r"radix_tree\\.rs:(341|431)",
            min_rate_per_sec=100.0,
        )
        # panic-burst proof on bug-image; same check with floor=0.0
        # asserts ZERO occurrences (use ``expect_zero=True`` for clarity).
        ServiceLogPatternRate(
            services=["Frontend", "VllmDecodeWorker"],
            pattern=r"panicked at .*tcp/(client|server)\\.rs",
            min_rate_per_sec=1.0,
        )
    """

    services: list
    pattern: str
    min_rate_per_sec: float = 1.0
    expect_zero: bool = False  # invert: assert rate == 0

    def validate(self, ctx) -> None:
        compiled = re.compile(self.pattern)
        ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
        logs = _get_service_logs(ctx)
        total_matches = 0
        total_span_s = 0.0
        per_service: dict = {}
        for svc in self.services:
            text = logs.get(svc, "") or ""
            n = 0
            first_ts = None
            last_ts = None
            for line in text.splitlines():
                m = compiled.search(line)
                if m:
                    n += 1
                t = ts_re.search(line)
                if t:
                    iso = t.group(1).replace(" ", "T")
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(iso)
                        first_ts = dt if first_ts is None else min(first_ts, dt)
                        last_ts = dt if last_ts is None else max(last_ts, dt)
                    except ValueError:
                        pass
            span = (
                max(1.0, (last_ts - first_ts).total_seconds())
                if (first_ts and last_ts)
                else 1.0
            )
            per_service[svc] = {"matches": n, "span_s": span, "rate": n / span}
            total_matches += n
            total_span_s = max(total_span_s, span)

        ctx.logger.info(
            f"ServiceLogPatternRate(pattern={self.pattern!r}): "
            f"per-service={per_service} total_matches={total_matches} "
            f"total_span_s={total_span_s:.1f}"
        )
        # Aggregate rate = sum(matches) / max(span). Could change to
        # per-service strictest, but aggregate is reasonable for "did
        # this happen somewhere".
        aggregate_rate = total_matches / max(1.0, total_span_s)

        if self.expect_zero:
            assert total_matches == 0, (
                f"ServiceLogPatternRate(expect_zero=True) but found "
                f"{total_matches} matches for {self.pattern!r} in "
                f"services {self.services}: {per_service}"
            )
            return

        assert aggregate_rate >= self.min_rate_per_sec, (
            f"ServiceLogPatternRate: aggregate rate "
            f"{aggregate_rate:.2f}/s < floor {self.min_rate_per_sec}/s "
            f"for pattern {self.pattern!r} in services {self.services}: "
            f"{per_service}"
        )

    @property
    def description(self) -> str:
        if self.expect_zero:
            return (
                f"Pattern {self.pattern!r} matches NONE in "
                f"{', '.join(self.services)}"
            )
        return (
            f"Pattern {self.pattern!r} fires ≥ {self.min_rate_per_sec}/s in "
            f"{', '.join(self.services)}"
        )


@dataclass
class KvCacheUsagePeak(Check):
    """Assert at least one pod's ``vllm:kv_cache_usage_perc`` reaches
    ``threshold`` within ``within_seconds`` of the **specified load's**
    first-sample timestamp.

    ``load_name``: anchor for ``within_seconds``. When set (e.g. "cliff"),
    the elapsed-time clock starts at the named load's first metric
    sample for that pod — not the FIRST load's first sample. This is
    the right semantic for cliff-rung tests where the warmup rung
    runs first and would otherwise push the cliff KV peak past the
    ``within_seconds`` ceiling even when the cliff itself pegged in
    seconds. Per observe-agent NEXT_STEPS_FOR_TEST_AGENT P1.2 (prior
    sanity-suite-1) — S0b false-failed without this anchor.

    Default behaviour (``load_name=None``) preserves the original
    R4-replacement-KV semantics: anchor on each pod's first sample
    across all loads.
    """

    services: list
    threshold: float = 0.9
    within_seconds: float = 240.0
    load_name: Optional[str] = None

    def validate(self, ctx) -> None:
        load_dirs = _find_load_dirs(ctx)
        if self.load_name:
            load_dirs = [d for d in load_dirs if self.load_name in _os.path.basename(d)]
        observed: dict = {}
        for ldir in load_dirs:
            jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
            for rec in _iter_jsonl(jsonl):
                ts_ns = rec.get("timestamp_ns")
                metrics = rec.get("metrics", {}) or {}
                kv = metrics.get("vllm:kv_cache_usage_perc") or []
                for sample in kv:
                    labels = sample.get("labels", {}) or {}
                    pod = labels.get("pod") or rec.get("endpoint_url", "")
                    val = sample.get("value")
                    if val is None or ts_ns is None:
                        continue
                    if pod not in observed:
                        observed[pod] = {
                            "first_ts_ns": ts_ns,
                            "max": float(val),
                            # Time of FIRST crossing of threshold — what
                            # within_seconds measures against. Stays
                            # None until threshold is first reached.
                            "first_cross_ts_ns": (
                                ts_ns if float(val) >= self.threshold else None
                            ),
                        }
                    else:
                        observed[pod]["max"] = max(observed[pod]["max"], float(val))
                        if (
                            float(val) >= self.threshold
                            and observed[pod]["first_cross_ts_ns"] is None
                        ):
                            observed[pod]["first_cross_ts_ns"] = ts_ns

        anchor_desc = (
            f"load '{self.load_name}' start"
            if self.load_name
            else "first sample across all loads"
        )
        ctx.logger.info(
            f"KvCacheUsagePeak: per-pod observations={observed} "
            f"threshold={self.threshold} within={self.within_seconds}s "
            f"anchor={anchor_desc}"
        )
        # Pass condition: at least one pod's FIRST crossing of the
        # threshold happened within ``within_seconds`` of the anchor's
        # first sample. The prior implementation tracked the LAST sample
        # ≥ threshold (effectively the end of the sustained cliff
        # plateau) → for any sustained cliff, elapsed = full duration →
        # false-fail. Per observe-agent NEXT_STEPS P1.2-v2 (round 2).
        winners = []
        for pod, rec in observed.items():
            if rec["first_cross_ts_ns"] is None:
                continue
            elapsed_s = (rec["first_cross_ts_ns"] - rec["first_ts_ns"]) / 1e9
            if elapsed_s <= self.within_seconds:
                winners.append((pod, rec["max"], elapsed_s))
        assert winners, (
            f"KvCacheUsagePeak: no pod in {self.services} reached "
            f"kv_cache_usage_perc >= {self.threshold} within "
            f"{self.within_seconds}s of {anchor_desc}. Observed: {observed}"
        )

    @property
    def description(self) -> str:
        anchor = f" (within '{self.load_name}')" if self.load_name else ""
        return (
            f"KV cache usage peak ≥ {self.threshold} within "
            f"{self.within_seconds}s on some {', '.join(self.services)} pod"
            f"{anchor}"
        )


@dataclass
class PodMemoryGrowth(Check):
    """Assert at least one pod's ``/proc/1/status:VmRSS`` grows by
    ``growth_bytes_per_min`` over a ``window_seconds`` window during the
    scenario.

    Implementation: a background asyncio task started by the check at
    scenario-start exec's `cat /proc/1/status` on each pod every
    ``poll_interval_s``, parses VmRSS, and writes a TSV to
    ``ctx.log_dir/pod_memory_growth.tsv``. validate() reads the TSV.

    The background-task plumbing is in the test file (StartLoad event
    can't host it cleanly); this check class is the assertion only.
    Tests that need it install a ``MemoryPoller`` event into the
    scenario events list before calling run_scenario.
    """

    services: list
    growth_bytes_per_min: float = 1_000_000_000.0
    window_seconds: float = 600.0
    tsv_filename: str = "pod_memory_growth.tsv"
    # Which column to assert on. "working_set" = kubelet cgroup view
    # (OOMKill-correlated). "pid1_rss" = per-process attribution.
    # Default to working_set since that's the kubelet-side signal.
    source: str = "working_set"
    # "min": assert ``max_rate >= growth_bytes_per_min`` (leak detection —
    # original behaviour, used by fault-tolerance leak-shape tests).
    # "max": assert ``max_rate <= growth_bytes_per_min`` (bounded-growth
    # assertion — used by memory-stability tests to prove a recommended
    # config doesn't leak; the field name still says ``growth`` but
    # functions as a ceiling).
    assert_mode: str = "min"

    def validate(self, ctx) -> None:
        log_dir = getattr(ctx, "log_dir", None)
        path = _os.path.join(log_dir, self.tsv_filename) if log_dir else None
        if not path or not _os.path.isfile(path):
            assert False, (
                f"PodMemoryGrowth: {self.tsv_filename} not found at "
                f"{path!r}; the test must install a MemoryPoller event "
                f"before running the scenario."
            )
        # TSV columns:
        #   v3 (current, 6 fields): epoch_s, svc, pod, container,
        #       working_set_bytes, pid1_rss_bytes
        #   v2 (5 fields): epoch_s, svc, pod, container, working_set_bytes
        #   v1 (4 fields): epoch_s, svc, pod, vm_rss_kb (legacy)
        # self.source ("working_set"|"pid1_rss") picks the column.
        per_pod = {}
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if not parts or parts[0] == "epoch_s":
                    continue
                try:
                    t = float(parts[0])
                    svc = parts[1]
                    pod = parts[2]
                    if len(parts) >= 6:
                        bytes_ = float(
                            parts[5] if self.source == "pid1_rss" else parts[4]
                        )
                    elif len(parts) >= 5:
                        bytes_ = float(parts[4])
                    elif len(parts) >= 4:
                        bytes_ = float(parts[3]) * 1024.0
                    else:
                        continue
                except ValueError:
                    continue
                if svc not in self.services:
                    continue
                per_pod.setdefault(pod, []).append((t, bytes_))

        max_rate_pod = None
        max_rate = 0.0
        for pod, series in per_pod.items():
            series.sort()
            # Sliding window: for each pair (t0, t1) in window, compute
            # (rss_t1 - rss_t0) / (t1 - t0) in seconds, then × 60.
            for i, (t0, r0) in enumerate(series):
                for j in range(len(series) - 1, i, -1):
                    t1, r1 = series[j]
                    if t1 - t0 > self.window_seconds:
                        continue
                    span = t1 - t0
                    if span < 30.0:
                        continue
                    rate_per_min = (r1 - r0) / span * 60.0
                    if rate_per_min > max_rate:
                        max_rate = rate_per_min
                        max_rate_pod = pod
                    break  # take the longest span; faster
        threshold_label = "floor" if self.assert_mode == "min" else "ceiling"
        ctx.logger.info(
            f"PodMemoryGrowth({self.assert_mode}): max_rate={max_rate:.0f} bytes/min "
            f"on pod={max_rate_pod}; {threshold_label}={self.growth_bytes_per_min}"
        )
        if self.assert_mode == "min":
            assert max_rate >= self.growth_bytes_per_min, (
                f"PodMemoryGrowth: peak RSS growth {max_rate:.0f}/min < "
                f"{self.growth_bytes_per_min:.0f}/min floor on services "
                f"{self.services}"
            )
        elif self.assert_mode == "max":
            assert max_rate <= self.growth_bytes_per_min, (
                f"PodMemoryGrowth: peak RSS growth {max_rate:.0f}/min > "
                f"{self.growth_bytes_per_min:.0f}/min ceiling on services "
                f"{self.services} (worst pod: {max_rate_pod})"
            )
        else:
            assert False, (
                f"PodMemoryGrowth: invalid assert_mode {self.assert_mode!r} "
                f"(want 'min' or 'max')"
            )

    @property
    def description(self) -> str:
        return (
            f"At least one {', '.join(self.services)} pod's RSS grows ≥ "
            f"{self.growth_bytes_per_min:.0g} bytes/min "
            f"over a {self.window_seconds}s window"
        )


@dataclass
class CascadeAfterFault(Check):
    """Assert that on at least one surviving pod, after a named fault event,
    ``vllm:num_requests_waiting`` increases by ``waiting_min_delta`` AND
    ``vllm:request_latency_seconds`` p99 grows by ``latency_p99_min_multiplier``×.

    "Surviving" = any pod in ``services`` that was NOT the named fault's
    target. Pre-window: ``pre_seconds`` immediately before the fault.
    Post-window: starting ``settle_seconds`` after the fault to allow the
    cascade to develop, ending ``post_seconds`` later.
    """

    services: list
    fault_event_name: str
    waiting_min_delta: float = 5.0
    latency_p99_min_multiplier: float = 2.0
    pre_seconds: float = 120.0
    post_seconds: float = 120.0
    settle_seconds: float = 30.0

    def _fault_ts_ns(self, ctx) -> int | None:
        for event in ctx.events:
            if getattr(event, "name", None) == self.fault_event_name:
                started = getattr(event, "started_at", None)
                if started is not None:
                    return int(started.timestamp() * 1e9)
        return None

    def validate(self, ctx) -> None:
        fault_ts = self._fault_ts_ns(ctx)
        assert fault_ts is not None, (
            f"CascadeAfterFault: could not find started_at on event "
            f"named {self.fault_event_name!r}"
        )
        pre_lo = fault_ts - int(self.pre_seconds * 1e9)
        pre_hi = fault_ts
        post_lo = fault_ts + int(self.settle_seconds * 1e9)
        post_hi = post_lo + int(self.post_seconds * 1e9)

        per_pod = (
            {}
        )  # pod -> {"pre_waiting":[], "post_waiting":[], "pre_p99":[], "post_p99":[]}
        for ldir in _find_load_dirs(ctx):
            jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
            for rec in _iter_jsonl(jsonl):
                ts = rec.get("timestamp_ns")
                if ts is None:
                    continue
                bucket = None
                if pre_lo <= ts <= pre_hi:
                    bucket = "pre"
                elif post_lo <= ts <= post_hi:
                    bucket = "post"
                if bucket is None:
                    continue
                metrics = rec.get("metrics", {}) or {}
                for m in ("vllm:num_requests_waiting",):
                    for s in metrics.get(m, []) or []:
                        labels = s.get("labels", {}) or {}
                        pod = labels.get("pod") or rec.get("endpoint_url", "")
                        v = s.get("value")
                        if v is None:
                            continue
                        per_pod.setdefault(pod, {}).setdefault(
                            f"{bucket}_waiting", []
                        ).append(float(v))
                for s in metrics.get("vllm:request_latency_seconds", []) or []:
                    labels = s.get("labels", {}) or {}
                    pod = labels.get("pod") or rec.get("endpoint_url", "")
                    buckets = s.get("buckets") or {}
                    count = s.get("count") or 0
                    # estimate p99 by walking buckets
                    if not buckets or not count:
                        continue
                    target = 0.99 * count
                    le_sorted = sorted(
                        [
                            (
                                float(k.rstrip("Inf").rstrip("+"))
                                if k != "+Inf"
                                else float("inf"),
                                float(v),
                            )
                            for k, v in buckets.items()
                        ],
                        key=lambda x: x[0],
                    )
                    p99 = None
                    for le, cum in le_sorted:
                        if cum >= target:
                            p99 = le
                            break
                    if p99 is None:
                        continue
                    per_pod.setdefault(pod, {}).setdefault(f"{bucket}_p99", []).append(
                        p99
                    )

        # Try to identify the fault target pod from the event so we exclude it.
        fault_target_name = None
        for event in ctx.events:
            if getattr(event, "name", None) == self.fault_event_name:
                names = (
                    getattr(event, "_targeted_pod_names", None)
                    or getattr(event, "_deleted_names", None)
                    or [p for _, p in getattr(event, "_stalled_pids", []) or []]
                )
                if names:
                    fault_target_name = names[0] if isinstance(names, list) else names
                break

        cascade_evidence = []
        for pod, slices in per_pod.items():
            if fault_target_name and fault_target_name in pod:
                continue
            pre_w = max(slices.get("pre_waiting") or [0.0])
            post_w = max(slices.get("post_waiting") or [0.0])
            pre_p99 = max(slices.get("pre_p99") or [0.0]) or 0.001
            post_p99 = max(slices.get("post_p99") or [0.0])
            waiting_d = post_w - pre_w
            latency_mult = post_p99 / pre_p99 if pre_p99 > 0 else float("inf")
            if (
                waiting_d >= self.waiting_min_delta
                and latency_mult >= self.latency_p99_min_multiplier
            ):
                cascade_evidence.append((pod, pre_w, post_w, pre_p99, post_p99))
        ctx.logger.info(
            f"CascadeAfterFault({self.fault_event_name}): cascade survivors="
            f"{cascade_evidence} (need waiting_d ≥ {self.waiting_min_delta} "
            f"AND latency × {self.latency_p99_min_multiplier})"
        )
        assert cascade_evidence, (
            f"CascadeAfterFault: no surviving pod in {self.services} "
            f"showed both waiting growth ≥ {self.waiting_min_delta} AND "
            f"latency p99 growth ≥ {self.latency_p99_min_multiplier}× "
            f"after fault {self.fault_event_name!r}"
        )

    @property
    def description(self) -> str:
        return (
            f"After fault {self.fault_event_name!r}: some survivor in "
            f"{', '.join(self.services)} sees waiting ≥ +{self.waiting_min_delta} "
            f"AND p99 × ≥ {self.latency_p99_min_multiplier}"
        )


@dataclass
class EngineDeathDetected(Check):
    """Detect 'engine hung but kubelet hasn't killed it yet': a window of
    ``idle_seconds`` or more where ``vllm:request_success_total`` rate is
    zero on a pod while aiperf load was still admitting requests.

    Use this as the 'RPS collapse minutes before container restart'
    signature when investigating whether an indirect fault drives a pod
    into a silent-stalled state.

    ``expect_zero=False`` (default) → assert ≥ 1 such window is found
    on any pod (proves the engine-hung state was reached somewhere).
    ``expect_zero=True`` → assert NO pod entered a sustained idle window
    (used by R2 / sanity tests).
    """

    services: list
    idle_seconds: float = 30.0
    expect_zero: bool = False

    def validate(self, ctx) -> None:
        per_pod = {}
        for ldir in _find_load_dirs(ctx):
            jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
            for rec in _iter_jsonl(jsonl):
                ts = rec.get("timestamp_ns")
                if ts is None:
                    continue
                for s in (rec.get("metrics", {}) or {}).get(
                    "vllm:request_success_total", []
                ) or []:
                    labels = s.get("labels", {}) or {}
                    pod = labels.get("pod") or rec.get("endpoint_url", "")
                    v = s.get("value")
                    if v is None:
                        continue
                    per_pod.setdefault(pod, []).append((ts, float(v)))

        idle_windows = []  # (pod, idle_start_ts, idle_end_ts, duration_s)
        for pod, series in per_pod.items():
            series.sort()
            run_start_ts = None
            run_start_val = None
            for ts, v in series:
                if run_start_ts is None:
                    run_start_ts = ts
                    run_start_val = v
                    continue
                if v == run_start_val:
                    dur_s = (ts - run_start_ts) / 1e9
                    if dur_s >= self.idle_seconds:
                        idle_windows.append((pod, run_start_ts, ts, dur_s))
                else:
                    run_start_ts = ts
                    run_start_val = v

        ctx.logger.info(
            f"EngineDeathDetected: idle_windows ≥ {self.idle_seconds}s: "
            f"{len(idle_windows)} found; samples={idle_windows[:3]}"
        )
        if self.expect_zero:
            assert not idle_windows, (
                f"EngineDeathDetected(expect_zero=True): "
                f"{len(idle_windows)} sustained-zero-RPS windows ≥ "
                f"{self.idle_seconds}s found on pods in {self.services}"
            )
            return
        assert idle_windows, (
            f"EngineDeathDetected: no pod in {self.services} entered a "
            f"≥ {self.idle_seconds}s zero-RPS window during the run"
        )

    @property
    def description(self) -> str:
        suffix = (
            "no idle windows"
            if self.expect_zero
            else f"≥ 1 idle window of ≥ {self.idle_seconds}s"
        )
        return f"{', '.join(self.services)}: {suffix}"


@dataclass
class RestartCountIncreased(Check):
    """Assert that at least one pod in ``services`` had its
    ``containerStatuses[0].restartCount`` increment during the scenario.

    Reads the live pod manifests at validate() time. The ManagedDeployment
    spec snapshot at start should be captured by tests; for simplicity
    this check just looks for any non-zero restartCount on any matching
    pod (since fresh DGD starts at 0).
    """

    services: list
    expect_min_increment: int = 1
    expect_zero: bool = False

    def validate(self, ctx) -> None:
        seen = {}
        snapshot = getattr(ctx, "pod_restart_state", None) or {}
        for svc in self.services:
            if ctx.deployment is not None:
                # Live mode — useful for in-scenario checks.
                pods = ctx.deployment.get_pods([svc]).get(svc) or []
                for pod in pods:
                    statuses = (pod.raw.get("status", {}) or {}).get(
                        "containerStatuses"
                    ) or []
                    if not statuses:
                        continue
                    rc = int(statuses[0].get("restartCount", 0))
                    reason = (
                        (statuses[0].get("lastState") or {}).get("terminated") or {}
                    ).get("reason")
                    seen[pod.name] = {"restartCount": rc, "lastReason": reason}
            else:
                # Post-teardown — read from the scenario-runner snapshot.
                seen.update(snapshot.get(svc, {}))

        ctx.logger.info(f"RestartCountIncreased: per-pod={seen}")
        total = sum(v["restartCount"] for v in seen.values())
        if self.expect_zero:
            assert total == 0, (
                f"RestartCountIncreased(expect_zero=True): sum of "
                f"restartCounts = {total} across services {self.services}: "
                f"{seen}"
            )
            return
        assert total >= self.expect_min_increment, (
            f"RestartCountIncreased: sum of restartCounts = {total} < "
            f"{self.expect_min_increment} across services {self.services}: "
            f"{seen}"
        )

    @property
    def description(self) -> str:
        if self.expect_zero:
            return f"No container in {', '.join(self.services)} restarted"
        return (
            f"At least {self.expect_min_increment} container restart(s) "
            f"across {', '.join(self.services)}"
        )


# =============================================================================
# decode-overload-disagg repro verifiers
# =============================================================================
#
# Five checks added for the decode-worker overload campaign that reproduces
# the canonical disagg decode-side cascade signature shape.
# Each fails fast if the test didn't actually drive the conditions it claims
# to drive — they are *verifiers*, not gates, so the test author can read
# the report and confirm load was applied, pressure was reached, and the
# expected symptoms occurred.
#
# Source signals come from two places:
#  - AIPerf-side: ``profile_export_aiperf.json`` (request_count, error_summary,
#    latency percentiles, goodput).
#  - Server-side: ``server_metrics_export.jsonl`` (vllm:* gauges + dynamo
#    counters, 1Hz scrape).


@dataclass
class LoadApplied(Check):
    """Assert aiperf actually issued at least ``min_requests`` requests
    during the scenario. Fails if the load harness misbehaved — zero
    traffic (broken endpoint, port-forward failure), job created but
    cancelled before warmup, or the AIPerf job pod failed before
    reporting any results.
    """

    min_requests: int = 100
    load_name: Optional[str] = None  # filter to a specific StartLoad name

    def validate(self, ctx) -> None:
        load_dirs = _find_load_dirs(ctx)
        observed = []
        for ldir in load_dirs:
            if self.load_name and self.load_name not in _os.path.basename(ldir):
                continue
            # AIPerf 0.7.x writes profile_export_aiperf.json directly in
            # the load dir (no nested attempt subdirectory).
            jpath = _os.path.join(ldir, "profile_export_aiperf.json")
            if not _os.path.isfile(jpath):
                continue
            try:
                with open(jpath) as fh:
                    data = _json.load(fh)
            except (_json.JSONDecodeError, OSError):
                continue
            # AIPerf summary schema:
            #   request_count.avg        — successful requests
            #   error_request_count.avg  — errored requests (AIPerf 0.8+)
            # We want the TOTAL requests issued (success + error) — that's
            # what "did the load actually reach the cluster?" boils down to.
            # AIPerf 0.7.x omits error_request_count; AIPerf 0.8 omits
            # request_count when all-error. Sum both nodes so either layout
            # is counted correctly.
            rc_node = data.get("request_count") or {}
            ec_node = data.get("error_request_count") or {}
            rc = int(rc_node.get("avg") or rc_node.get("count") or 0)
            ec = int(ec_node.get("avg") or ec_node.get("count") or 0)
            observed.append((ldir, rc + ec))
        total = sum(rc for _, rc in observed)
        ctx.logger.info(
            f"LoadApplied: per-load-dir={observed} total={total} "
            f"min_required={self.min_requests}"
        )
        assert total >= self.min_requests, (
            f"LoadApplied: only {total} requests issued across "
            f"{len(observed)} load runs; expected at least "
            f"{self.min_requests}. Was the endpoint reachable?"
        )

    @property
    def description(self) -> str:
        if self.load_name:
            return f"Load '{self.load_name}' issued ≥ {self.min_requests} requests"
        return f"Total requests issued ≥ {self.min_requests}"


def _iter_server_metric(ctx, metric: str):
    """Yield (ts_ns, pod, value) for every sample of ``metric`` across all
    load dirs and rungs of the current scenario.

    Pod identifier is taken from (in order): ``labels.pod``, the
    ``dynamo_component`` label combined with the endpoint host
    (so multiple workers of the same role stay distinguishable),
    or just the endpoint_url. The vLLM Prometheus surface in
    Dynamo 1.0.x labels samples with ``dynamo_component`` =
    ``prefill`` | ``decode`` | ``frontend`` but does NOT include
    a ``pod`` label — without this fallback every prefill rep
    aliases to "prefill" and we lose the pod-level distinction.
    """
    for ldir in _find_load_dirs(ctx):
        jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
        for rec in _iter_jsonl(jsonl):
            ts_ns = rec.get("timestamp_ns")
            endpoint = rec.get("endpoint_url", "")
            metrics = rec.get("metrics", {}) or {}
            samples = metrics.get(metric) or []
            for s in samples:
                labels = s.get("labels", {}) or {}
                pod = labels.get("pod")
                if not pod:
                    comp = labels.get("dynamo_component") or ""
                    # Endpoint URL embeds the pod IP — use it as the
                    # tie-breaker so two prefills don't collapse.
                    host = endpoint
                    pod = f"{comp}@{host}" if comp else host
                val = s.get("value")
                if val is None or ts_ns is None:
                    continue
                yield int(ts_ns), str(pod), float(val)


@dataclass
class RequestsRunningPeak(Check):
    """Assert at least one decode pod's ``vllm:num_requests_running``
    spiked to ``threshold`` and stayed there for ``sustained_seconds``.
    Mirrors a production-class running-set spike (e.g. 40 → 246 in one minute).

    Implementation: for each pod, find the longest contiguous run of
    samples ≥ threshold. Pass if any pod's longest run ≥ sustained_seconds.
    """

    threshold: int = 200
    sustained_seconds: float = 30.0

    def validate(self, ctx) -> None:
        per_pod: dict = {}  # pod -> [(ts_ns, val), ...]
        for ts_ns, pod, val in _iter_server_metric(ctx, "vllm:num_requests_running"):
            per_pod.setdefault(pod, []).append((ts_ns, val))
        runs: dict = {}
        for pod, series in per_pod.items():
            series.sort()
            best_s = 0.0
            cur_start = None
            cur_last = None
            for ts_ns, v in series:
                if v >= self.threshold:
                    if cur_start is None:
                        cur_start = ts_ns
                    cur_last = ts_ns
                else:
                    if cur_start is not None and cur_last is not None:
                        best_s = max(best_s, (cur_last - cur_start) / 1e9)
                    cur_start = None
                    cur_last = None
            if cur_start is not None and cur_last is not None:
                best_s = max(best_s, (cur_last - cur_start) / 1e9)
            runs[pod] = best_s
        ctx.logger.info(
            f"RequestsRunningPeak: per-pod longest sustained run (s) "
            f"at >= {self.threshold} = {runs}"
        )
        winners = [p for p, s in runs.items() if s >= self.sustained_seconds]
        assert winners, (
            f"RequestsRunningPeak: no pod sustained num_requests_running "
            f">= {self.threshold} for >= {self.sustained_seconds}s. "
            f"Longest run per pod: {runs}"
        )

    @property
    def description(self) -> str:
        return (
            f"At least one pod's num_requests_running ≥ {self.threshold} "
            f"for ≥ {self.sustained_seconds}s"
        )


@dataclass
class RequestCancellationOccurred(Check):
    """Assert AIPerf observed at least ``min_count`` request
    cancellations across the run. Counts both:
      - explicit ``--request-cancellation-rate`` aborts (error tag
        ``RequestCancellationError``)
      - SLA-timeout aborts (``--request-timeout-seconds``, surfaces as
        timeout-style errors in error_summary).

    Validates the disconnect/timeout pressure path is exercised — the
    point of the campaign. If this is zero, AIPerf wasn't producing
    the SO_LINGER=0-trigger pattern even under sustained load.
    """

    min_count: int = 10
    load_name: Optional[str] = None

    def validate(self, ctx) -> None:
        per_run: list = []
        for ldir in _find_load_dirs(ctx):
            if self.load_name and self.load_name not in _os.path.basename(ldir):
                continue
            jpath = _os.path.join(ldir, "profile_export_aiperf.json")
            if not _os.path.isfile(jpath):
                continue
            try:
                with open(jpath) as fh:
                    data = _json.load(fh)
            except (_json.JSONDecodeError, OSError):
                continue
            err_summary = data.get("error_summary") or []
            if isinstance(err_summary, dict):
                err_summary = err_summary.get("summary") or []
            cancelled = 0
            for entry in err_summary:
                # AIPerf 0.7.x schema:
                #   {"error_details": {"type": "TimeoutError", ...}, "count": N}
                # earlier versions used top-level "type" / "error".
                details = entry.get("error_details") or {}
                tag = str(
                    details.get("type") or entry.get("error") or entry.get("type") or ""
                ).lower()
                if (
                    "cancel" in tag
                    or "timeout" in tag
                    or entry.get("code") in (408, 499, 504)
                ):
                    cancelled += int(entry.get("count") or 0)
            # SLA-timeout cancellations are emergent: when goodput SLO is
            # set (e.g. request_latency:30000ms), request_count counts
            # all completions while good_request_count counts only those
            # within SLO. The difference is the SLA-breach count, which
            # in our 30s-timeout regime equals AIPerf-side cancellations.
            rc = int((data.get("request_count") or {}).get("avg") or 0)
            gc = int((data.get("good_request_count") or {}).get("avg") or 0)
            slo_breached = max(0, rc - gc) if gc and rc else 0
            per_run.append((ldir, cancelled + slo_breached))
        total = sum(c for _, c in per_run)
        ctx.logger.info(
            f"RequestCancellationOccurred: per-run={per_run} "
            f"total={total} min_required={self.min_count}"
        )
        assert total >= self.min_count, (
            f"RequestCancellationOccurred: only {total} "
            f"cancellations/timeouts observed; expected ≥ {self.min_count}. "
            f"Disconnect/timeout pressure path was not exercised."
        )

    @property
    def description(self) -> str:
        return f"AIPerf observed ≥ {self.min_count} cancellations/timeouts"


@dataclass
class ThroughputCollapse(Check):
    """Assert at least one decode pod's ``vllm:num_requests_running``
    dropped to 0 for ``collapse_seconds`` while frontend
    ``vllm:num_requests_waiting`` (or aiperf-side inflight) remained > 0.

    Signature of the "engine alive but unproductive" mode seen in
    disagg cascades (engine idle for minutes while sockets pile up).

    Implementation: walk per-pod num_requests_running. Find the longest
    contiguous run of 0-samples. To require coincident upstream demand,
    require that *some* frontend num_requests_waiting sample > 0 falls
    inside the same window.
    """

    collapse_seconds: float = 30.0
    require_upstream_demand: bool = True

    def validate(self, ctx) -> None:
        running: dict = {}
        for ts_ns, pod, val in _iter_server_metric(ctx, "vllm:num_requests_running"):
            running.setdefault(pod, []).append((ts_ns, val))
        waiting_ts = [
            ts_ns
            for ts_ns, _pod, val in _iter_server_metric(
                ctx, "vllm:num_requests_waiting"
            )
            if val > 0
        ]
        waiting_set = set(waiting_ts)
        results: dict = {}
        for pod, series in running.items():
            series.sort()
            best_s = 0.0
            best_window = None
            cur_start = None
            cur_last = None
            cur_has_demand = False
            for ts_ns, v in series:
                if v == 0:
                    if cur_start is None:
                        cur_start = ts_ns
                    cur_last = ts_ns
                    if ts_ns in waiting_set:
                        cur_has_demand = True
                else:
                    if cur_start is not None and cur_last is not None:
                        dur = (cur_last - cur_start) / 1e9
                        if dur > best_s and (
                            not self.require_upstream_demand or cur_has_demand
                        ):
                            best_s = dur
                            best_window = (cur_start, cur_last)
                    cur_start = None
                    cur_last = None
                    cur_has_demand = False
            if cur_start is not None and cur_last is not None:
                dur = (cur_last - cur_start) / 1e9
                if dur > best_s and (
                    not self.require_upstream_demand or cur_has_demand
                ):
                    best_s = dur
                    best_window = (cur_start, cur_last)
            results[pod] = (best_s, best_window)
        ctx.logger.info(
            f"ThroughputCollapse: per-pod longest 0-running window (s) "
            f"= { {p: r[0] for p, r in results.items()} }"
        )
        winners = [p for p, r in results.items() if r[0] >= self.collapse_seconds]
        assert winners, (
            f"ThroughputCollapse: no pod sustained num_requests_running == 0 "
            f"for >= {self.collapse_seconds}s "
            f"({'with' if self.require_upstream_demand else 'without'} "
            f"upstream demand). Longest collapse per pod: "
            f"{ {p: r[0] for p, r in results.items()} }"
        )

    @property
    def description(self) -> str:
        return (
            f"At least one decode pod went idle (running=0) for ≥ "
            f"{self.collapse_seconds}s while frontend had pending work"
        )


@dataclass
class SlaViolation(Check):
    """Assert that AIPerf observed at least one SLA breach: e2e
    request_latency p99 > ``e2e_p99_ms`` OR time_to_first_token p99
    > ``ttft_p99_ms``. End-user impact signal — the point of driving
    overload is that someone notices.

    AIPerf 0.7.x summary has flat top-level keys; ``request_latency``
    and ``time_to_first_token`` are dicts with percentile keys directly
    (``p50``, ``p99``, ...) holding the value in milliseconds.
    """

    e2e_p99_ms: float = 30000.0
    ttft_p99_ms: float = 10000.0
    load_name: Optional[str] = None

    def validate(self, ctx) -> None:
        observed: list = []
        for ldir in _find_load_dirs(ctx):
            if self.load_name and self.load_name not in _os.path.basename(ldir):
                continue
            jpath = _os.path.join(ldir, "profile_export_aiperf.json")
            if not _os.path.isfile(jpath):
                continue
            try:
                with open(jpath) as fh:
                    data = _json.load(fh)
            except (_json.JSONDecodeError, OSError):
                continue
            e2e = (data.get("request_latency") or {}).get("p99")
            ttft = (data.get("time_to_first_token") or {}).get("p99")
            observed.append((ldir, e2e, ttft))
        ctx.logger.info(
            f"SlaViolation: per-run (e2e_p99_ms, ttft_p99_ms) = {observed} "
            f"thresholds=(e2e>{self.e2e_p99_ms}, ttft>{self.ttft_p99_ms})"
        )
        violated = any(
            (e2e is not None and float(e2e) > self.e2e_p99_ms)
            or (ttft is not None and float(ttft) > self.ttft_p99_ms)
            for _, e2e, ttft in observed
        )
        assert violated, (
            f"SlaViolation: no run breached e2e p99 > {self.e2e_p99_ms}ms "
            f"or TTFT p99 > {self.ttft_p99_ms}ms. Observed: {observed}"
        )

    @property
    def description(self) -> str:
        return (
            f"At least one run breaches e2e p99 > {self.e2e_p99_ms}ms "
            f"or TTFT p99 > {self.ttft_p99_ms}ms"
        )


# =============================================================================
# Cascade-signature checks (prior cascade-repro suite)
# =============================================================================
#
# Each check below reads server_metrics_export.jsonl from one named load
# rung and asserts one piece of the cascade signature:
#
#   1. KvCacheUsagePeak (already above)                — KV peg
#   2. WaitingForKVTransferExceeds                     — smoking gun (leads collapse ~60s)
#   3. NixlXferTimeMultiplied                          — transport blow-up
#   4. GenerationThroughputDropped                     — throughput collapse
#   5. CliffContained                                  — S0a-only: pin stays on target
#
# Full mechanism context lives in the FRAMEWORK_TASK / PLAN docs in
# dynamo-observe (the cascade-small-pool reproducer scenarios).


def _load_dirs_for(ctx, load_name: Optional[str]) -> list:
    """Like ``_find_load_dirs`` but optionally filters to a single load
    name (matches the AIPerf load-dir naming convention ``load-<name>-*``).
    """
    dirs = _find_load_dirs(ctx)
    if not load_name:
        return dirs
    return [d for d in dirs if load_name in _os.path.basename(d)]


def _iter_named_metric(
    ctx,
    metric: str,
    load_name: Optional[str] = None,
):
    """Yield ``(ts_ns, pod, val)`` for ``metric`` from the JSONL of one
    named load (or all loads if ``load_name`` is None).

    Pod identity follows the same convention as ``_iter_server_metric``:
    prefer ``labels.pod``, fall back to ``<dynamo_component>@<endpoint_url>``
    so two workers of the same role stay distinguishable when the vLLM
    surface omits the ``pod`` label.
    """
    for ldir in _load_dirs_for(ctx, load_name):
        jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
        for rec in _iter_jsonl(jsonl):
            ts_ns = rec.get("timestamp_ns")
            endpoint = rec.get("endpoint_url", "")
            metrics = rec.get("metrics", {}) or {}
            samples = metrics.get(metric) or []
            for s in samples:
                labels = s.get("labels", {}) or {}
                pod = labels.get("pod")
                if not pod:
                    comp = labels.get("dynamo_component") or ""
                    pod = f"{comp}@{endpoint}" if comp else endpoint
                val = s.get("value")
                if val is None or ts_ns is None:
                    continue
                yield int(ts_ns), str(pod), float(val)


def _per_pod_timeseries(ctx, metric: str, load_name: Optional[str] = None) -> dict:
    """Return ``{pod: [(ts_ns, val), ...sorted]}`` for one metric."""
    series: dict = {}
    for ts_ns, pod, val in _iter_named_metric(ctx, metric, load_name=load_name):
        series.setdefault(pod, []).append((ts_ns, val))
    for pod in series:
        series[pod].sort()
    return series


def _iter_histogram(ctx, metric: str, load_name: Optional[str] = None):
    """Yield ``(ts_ns, pod, sum_val, count_val)`` for a Prometheus histogram
    stored single-key in the server_metrics_export.

    The export schema is:
      ``metrics[<base_name>] = [{"labels": {...}, "buckets": {...},
                                 "sum": <float>, "count": <float>}, ...]``

    NOT the on-the-wire Prometheus form with ``_sum`` / ``_count`` suffix
    keys — the Dynamo metric exporter writes a single key carrying both
    sum and count fields per sample. Use this iterator for histograms;
    use ``_iter_named_metric`` for counters/gauges with a ``value`` field.
    """
    for ldir in _load_dirs_for(ctx, load_name):
        jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
        for rec in _iter_jsonl(jsonl):
            ts_ns = rec.get("timestamp_ns")
            endpoint = rec.get("endpoint_url", "")
            metrics = rec.get("metrics", {}) or {}
            samples = metrics.get(metric) or []
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if "sum" not in s or "count" not in s:
                    continue
                labels = s.get("labels", {}) or {}
                pod = labels.get("pod")
                if not pod:
                    comp = labels.get("dynamo_component") or ""
                    pod = f"{comp}@{endpoint}" if comp else endpoint
                try:
                    sum_val = float(s["sum"])
                    count_val = float(s["count"])
                except (TypeError, ValueError):
                    continue
                if ts_ns is None:
                    continue
                yield int(ts_ns), str(pod), sum_val, count_val


def _window_average_histogram(
    ctx,
    metric: str,
    load_name: Optional[str],
    window_start_s: float,
    window_end_s: Optional[float],
) -> dict:
    """Per-pod average of a Prometheus histogram (sum / count delta) over
    a wall-clock window measured from the first sample of the named load.
    ``window_end_s=None`` means "to end of load".

    Returns ``{pod: avg_seconds}`` for pods that recorded ≥ 2 samples in
    the window with non-zero count delta. Pods without enough data are
    omitted, not zero-defaulted, so the caller can decide policy.

    Reads via ``_iter_histogram`` which expects the single-key histogram
    schema (``{labels, buckets, sum, count}`` per sample under one
    metric key). The on-the-wire ``_sum`` / ``_count`` split that
    Prometheus uses is NOT how Dynamo's exporter writes JSONL.
    """
    series: dict = {}
    for ts_ns, pod, sum_val, count_val in _iter_histogram(
        ctx, metric, load_name=load_name
    ):
        series.setdefault(pod, []).append((ts_ns, sum_val, count_val))
    for pod in series:
        series[pod].sort()

    out: dict = {}
    for pod, samples in series.items():
        if not samples:
            continue
        t0 = samples[0][0]
        win_lo = t0 + int(window_start_s * 1e9)
        win_hi = t0 + int(window_end_s * 1e9) if window_end_s is not None else None
        lo = hi = None
        for ts, sm, cnt in samples:
            if ts < win_lo:
                continue
            if win_hi is not None and ts > win_hi:
                break
            if lo is None:
                lo = (sm, cnt)
            hi = (sm, cnt)
        if lo is None or hi is None:
            continue
        d_sum = hi[0] - lo[0]
        d_cnt = hi[1] - lo[1]
        if d_cnt <= 0:
            continue
        out[pod] = d_sum / d_cnt
    return out


def _counter_rate(
    ctx,
    metric: str,
    load_name: Optional[str],
    window_start_s: float,
    window_end_s: Optional[float],
) -> float:
    """Aggregate per-second rate of a counter across all pods over the
    named load's wall-clock window. Returns 0.0 if no data.
    """
    per_pod = _per_pod_timeseries(ctx, metric, load_name=load_name)
    total_delta = 0.0
    total_span_s = 0.0
    for pod, series in per_pod.items():
        if len(series) < 2:
            continue
        t0 = series[0][0]
        lo = t0 + int(window_start_s * 1e9)
        hi = t0 + int(window_end_s * 1e9) if window_end_s is not None else None
        clipped = [(ts, v) for ts, v in series if ts >= lo and (hi is None or ts <= hi)]
        if len(clipped) < 2:
            continue
        d_val = clipped[-1][1] - clipped[0][1]
        d_t = (clipped[-1][0] - clipped[0][0]) / 1e9
        if d_t <= 0:
            continue
        total_delta += max(0.0, d_val)
        total_span_s = max(total_span_s, d_t)
    if total_span_s <= 0:
        return 0.0
    return total_delta / total_span_s


@dataclass
class WaitingForKVTransferExceeds(Check):
    """Assert the derived gauge ``inflight − running − waiting`` exceeds
    ``threshold`` on at least one decode pod during the named load.

    This is the **smoking-gun** cascade signal — it climbs from baseline
    single-digits to hundreds-or-thousands ~60s before throughput
    collapse, so it leads every other signature metric.
    """

    services: list
    threshold: int = 100
    load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        # Join three metrics on (pod, ts_ns). Each JSONL record carries
        # all three at the same timestamp, so per-record arithmetic
        # works without alignment heuristics.
        per_pod_max: dict = {}
        for ldir in _load_dirs_for(ctx, self.load_name):
            jsonl = _os.path.join(ldir, "server_metrics_export.jsonl")
            for rec in _iter_jsonl(jsonl):
                metrics = rec.get("metrics", {}) or {}
                inflight = metrics.get("dynamo_component_inflight_requests") or []
                running = metrics.get("vllm:num_requests_running") or []
                waiting = metrics.get("vllm:num_requests_waiting") or []

                # Index by pod for this record. ``dynamo_component_inflight_requests``
                # has one sample per (component × ``dynamo_endpoint``) — for a
                # single prefill worker that's ``generate`` + ``clear_kv_blocks``
                # + ``get_perf_metrics`` (3 samples), all with the same
                # dynamo_component label. Sum across those sub-dimensions to
                # produce one value per pod (non-``generate`` endpoints are
                # always 0 in practice so sum equals the ``generate``-only value).
                def _by_pod(samples):
                    out = {}
                    for s in samples:
                        labels = s.get("labels", {}) or {}
                        pod = labels.get("pod") or (
                            (labels.get("dynamo_component") or "")
                            + "@"
                            + rec.get("endpoint_url", "")
                        )
                        v = s.get("value")
                        if v is not None:
                            out[pod] = out.get(pod, 0.0) + float(v)
                    return out

                infl_by_pod = _by_pod(inflight)
                run_by_pod = _by_pod(running)
                wait_by_pod = _by_pod(waiting)
                # Compute derived for every pod that has all three
                for pod in infl_by_pod.keys() & run_by_pod.keys() & wait_by_pod.keys():
                    derived = infl_by_pod[pod] - run_by_pod[pod] - wait_by_pod[pod]
                    cur = per_pod_max.get(pod, float("-inf"))
                    if derived > cur:
                        per_pod_max[pod] = derived

        winners = {p: v for p, v in per_pod_max.items() if v >= self.threshold}
        ctx.logger.info(
            f"WaitingForKVTransferExceeds: per-pod max={per_pod_max} "
            f"winners={winners} threshold={self.threshold}"
        )
        assert winners, (
            f"WaitingForKVTransferExceeds: no pod reached "
            f"derived (inflight − running − waiting) ≥ {self.threshold} "
            f"during load {self.load_name!r}; observed per-pod max: {per_pod_max}"
        )

    @property
    def description(self) -> str:
        return (
            f"WaitingForKVTransfer derived gauge ≥ {self.threshold} on some "
            f"{','.join(self.services)} pod during {self.load_name!r}"
        )


@dataclass
class NixlXferTimeMultiplied(Check):
    """Assert ``vllm:nixl_xfer_time_seconds`` per-handle average climbs
    by ≥ ``min_multiplier``× from the warmup window to the cliff window,
    on at least one pod.

    Healthy ~50ms → cliff ~250ms+ → 5×; sanity tests run with 3× because
    the compressed warmup window is shorter and noisier.
    """

    services: list
    min_multiplier: float = 5.0
    warmup_seconds: float = 300.0
    cliff_load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        warmup_avg = _window_average_histogram(
            ctx,
            "vllm:nixl_xfer_time_seconds",
            self.cliff_load_name,
            window_start_s=0.0,
            window_end_s=self.warmup_seconds,
        )
        cliff_avg = _window_average_histogram(
            ctx,
            "vllm:nixl_xfer_time_seconds",
            self.cliff_load_name,
            window_start_s=self.warmup_seconds,
            window_end_s=None,
        )
        max_ratio = 0.0
        per_pod_ratio: dict = {}
        for pod, w in warmup_avg.items():
            c = cliff_avg.get(pod)
            if c is None or w <= 0:
                continue
            ratio = c / w
            per_pod_ratio[pod] = ratio
            if ratio > max_ratio:
                max_ratio = ratio
        ctx.logger.info(
            f"NixlXferTimeMultiplied: warmup_avg={warmup_avg} cliff_avg={cliff_avg} "
            f"per_pod_ratio={per_pod_ratio} max_ratio={max_ratio:.2f}× "
            f"threshold={self.min_multiplier:.2f}×"
        )
        assert max_ratio >= self.min_multiplier, (
            f"NixlXferTimeMultiplied: max per-pod ratio {max_ratio:.2f}× < "
            f"required {self.min_multiplier:.2f}× on load {self.cliff_load_name!r}; "
            f"per-pod ratios: {per_pod_ratio}"
        )

    @property
    def description(self) -> str:
        return (
            f"NIXL xfer_time avg grows ≥ {self.min_multiplier}× on some "
            f"{','.join(self.services)} pod during {self.cliff_load_name!r}"
        )


@dataclass
class GenerationThroughputDropped(Check):
    """Assert ``vllm:generation_tokens`` aggregate rate drops by
    ≥ ``min_drop_frac`` from the warmup window to the last
    ``cliff_window_seconds`` of the named load.
    """

    services: list
    min_drop_frac: float = 0.30
    warmup_seconds: float = 300.0
    cliff_window_seconds: float = 300.0
    cliff_load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        # Find the total load span to compute "last N seconds" window.
        per_pod = _per_pod_timeseries(
            ctx, "vllm:generation_tokens", load_name=self.cliff_load_name
        )
        if not per_pod:
            assert False, (
                f"GenerationThroughputDropped: no samples for "
                f"vllm:generation_tokens on load {self.cliff_load_name!r}"
            )
        # Earliest first-sample and latest last-sample across pods.
        first_ts = min(s[0][0] for s in per_pod.values() if s)
        last_ts = max(s[-1][0] for s in per_pod.values() if s)
        total_span_s = (last_ts - first_ts) / 1e9

        warmup_rate = _counter_rate(
            ctx,
            "vllm:generation_tokens",
            self.cliff_load_name,
            window_start_s=0.0,
            window_end_s=self.warmup_seconds,
        )
        cliff_start = max(self.warmup_seconds, total_span_s - self.cliff_window_seconds)
        cliff_rate = _counter_rate(
            ctx,
            "vllm:generation_tokens",
            self.cliff_load_name,
            window_start_s=cliff_start,
            window_end_s=None,
        )
        drop_frac = (warmup_rate - cliff_rate) / warmup_rate if warmup_rate > 0 else 0.0
        ctx.logger.info(
            f"GenerationThroughputDropped: warmup_rate={warmup_rate:.1f} tok/s "
            f"cliff_rate={cliff_rate:.1f} tok/s drop={drop_frac:.1%} "
            f"threshold={self.min_drop_frac:.0%} span={total_span_s:.0f}s"
        )
        assert drop_frac >= self.min_drop_frac, (
            f"GenerationThroughputDropped: only {drop_frac:.1%} drop "
            f"(warmup {warmup_rate:.1f} → cliff {cliff_rate:.1f} tok/s); "
            f"required ≥ {self.min_drop_frac:.0%}"
        )

    @property
    def description(self) -> str:
        return (
            f"Generation throughput drops ≥ {self.min_drop_frac:.0%} on "
            f"load {self.cliff_load_name!r}"
        )


@dataclass
class CliffPropagated(Check):
    """Assert the cliff spreads to ≥ ``min_pods`` decode pods.

    Counterpart to ``CliffContained``: where containment asserts the
    pinned target alone hit the cliff, propagation asserts that under
    natural LeastLoaded routing the cliff peer-infects multiple decodes
    within the load window — the signature of the cross-FE blind spot.

    Reads ``vllm:kv_cache_usage_perc`` per pod and counts pods whose
    peak reached ``kv_threshold`` during the named load.
    """

    services: list
    min_pods: int = 2
    kv_threshold: float = 0.85
    load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        per_pod_max: dict = {}
        for _ts, pod, val in _iter_named_metric(
            ctx, "vllm:kv_cache_usage_perc", load_name=self.load_name
        ):
            cur = per_pod_max.get(pod, float("-inf"))
            if val > cur:
                per_pod_max[pod] = val
        saturated = [p for p, v in per_pod_max.items() if v >= self.kv_threshold]
        ctx.logger.info(
            f"CliffPropagated: per_pod_max={per_pod_max} "
            f"saturated_pods={saturated} min_pods={self.min_pods} "
            f"kv_threshold={self.kv_threshold}"
        )
        assert len(saturated) >= self.min_pods, (
            f"CliffPropagated: only {len(saturated)} pods reached KV ≥ "
            f"{self.kv_threshold}; required ≥ {self.min_pods}. "
            f"Per-pod max: {per_pod_max}"
        )

    @property
    def description(self) -> str:
        return (
            f"≥ {self.min_pods} pods reach KV ≥ {self.kv_threshold} "
            f"during {self.load_name!r}"
        )


@dataclass
class CliffContained(Check):
    """Assert the cliff stays on the nvext-pinned target replica during
    the named load.

    Looks up the pod for ``(pinned_service, pinned_replica_index)`` via
    the same name-sorted ordering used by
    ``ManagedDeployment.get_instance_id``, then asserts:
      - the pinned pod's max KV ≥ ``kv_threshold``
      - every other pod in ``services`` stays ≤ ``peer_max_kv``

    Proves the WorkerPin flow actually concentrated load on the intended
    pod (no leakage from FE fallback / non-honored nvext).
    """

    services: list
    pinned_service: str
    pinned_replica_index: int
    kv_threshold: float = 0.85
    peer_max_kv: float = 0.50
    load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        # Resolve the pinned pod name the same way get_instance_id does:
        # sort by name, pick replica_index. Use only the pinned_service's
        # pods — the cliff metric is per-pod, so we identify the pinned
        # one by its pod name suffix appearing in the metric's pod label.
        pods_by_svc = ctx.deployment.get_pods([self.pinned_service])
        pods = pods_by_svc.get(self.pinned_service) or []
        if not pods:
            assert False, f"CliffContained: no pods for {self.pinned_service!r}"
        pods_sorted = sorted(pods, key=lambda p: p.name)
        if self.pinned_replica_index >= len(pods_sorted):
            assert False, (
                f"CliffContained: pinned_replica_index="
                f"{self.pinned_replica_index} >= {len(pods_sorted)} pods"
            )
        pinned_pod_name = pods_sorted[self.pinned_replica_index].name

        # Compute per-pod max KV from the metric stream. The metric's
        # pod label sometimes carries the full pod name; sometimes it's
        # `decode@<endpoint>`. Match the pinned name as a substring to
        # cover both cases.
        per_pod_max: dict = {}
        for _ts, pod, val in _iter_named_metric(
            ctx, "vllm:kv_cache_usage_perc", load_name=self.load_name
        ):
            cur = per_pod_max.get(pod, float("-inf"))
            if val > cur:
                per_pod_max[pod] = val

        target_pods = [p for p in per_pod_max if pinned_pod_name in p]
        target_hit = any(per_pod_max[p] >= self.kv_threshold for p in target_pods)
        peer_violations = [
            (p, v)
            for p, v in per_pod_max.items()
            if (pinned_pod_name not in p) and v > self.peer_max_kv
        ]

        ctx.logger.info(
            f"CliffContained: pinned_pod={pinned_pod_name!r} "
            f"target_pods_in_metrics={target_pods} per_pod_max={per_pod_max} "
            f"target_hit={target_hit} peer_violations={peer_violations}"
        )
        assert target_hit, (
            f"CliffContained: pinned pod {pinned_pod_name!r} did not reach "
            f"KV ≥ {self.kv_threshold} (max per pod: {per_pod_max})"
        )
        assert not peer_violations, (
            f"CliffContained: peer pod(s) exceeded peer_max_kv "
            f"{self.peer_max_kv} — {peer_violations}. Pinning leaked."
        )

    @property
    def description(self) -> str:
        return (
            f"Cliff stays on {self.pinned_service}#{self.pinned_replica_index} "
            f"(KV ≥ {self.kv_threshold}); peers ≤ {self.peer_max_kv}"
        )


@dataclass
class PinningContained(Check):
    """Assert request-routing pinning kept the load on one replica only.

    Generalization of ``CliffContained`` for arbitrary activity metrics
    (not just KV pressure). Resolves the pinned pod the same way —
    name-sorted, ``replica_index``-th — then asserts:

      - peak(``metric``) on the pinned pod ≥ ``active_threshold``
        (proves the pinned pod actually saw traffic)
      - peak(``metric``) on every other pod of ``pinned_service``
        ≤ ``peer_max`` (proves nvext.worker_id actually concentrated
        the load, no leakage via FE fallback or non-honored nvext)

    For prefill containment use ``vllm:num_requests_running`` —
    a running > 0 anywhere on a prefill replica is proof it served
    at least one prefill, so an idle peer's peak is genuinely 0.
    """

    services: list
    pinned_service: str
    pinned_replica_index: int
    metric: str = "vllm:num_requests_running"
    active_threshold: float = 1.0
    peer_max: float = 0.0
    load_name: Optional[str] = "default"

    def validate(self, ctx) -> None:
        pods_by_svc = ctx.deployment.get_pods([self.pinned_service])
        pods = pods_by_svc.get(self.pinned_service) or []
        if not pods:
            assert False, f"PinningContained: no pods for {self.pinned_service!r}"
        pods_sorted = sorted(pods, key=lambda p: p.name)
        if self.pinned_replica_index >= len(pods_sorted):
            assert False, (
                f"PinningContained: pinned_replica_index="
                f"{self.pinned_replica_index} >= {len(pods_sorted)} pods"
            )
        pinned_pod_name = pods_sorted[self.pinned_replica_index].name

        per_pod_peak: dict = {}
        for _ts, pod, val in _iter_named_metric(
            ctx, self.metric, load_name=self.load_name
        ):
            cur = per_pod_peak.get(pod, float("-inf"))
            if val > cur:
                per_pod_peak[pod] = val

        target_pods = [p for p in per_pod_peak if pinned_pod_name in p]
        peer_pods = {p: v for p, v in per_pod_peak.items() if pinned_pod_name not in p}
        target_active = any(
            per_pod_peak[p] >= self.active_threshold for p in target_pods
        )
        peer_violations = [(p, v) for p, v in peer_pods.items() if v > self.peer_max]
        ctx.logger.info(
            f"PinningContained({self.metric}): pinned_pod={pinned_pod_name!r} "
            f"per_pod_peak={per_pod_peak} active_threshold={self.active_threshold} "
            f"peer_max={self.peer_max} target_active={target_active} "
            f"peer_violations={peer_violations}"
        )
        assert target_active, (
            f"PinningContained: pinned pod {pinned_pod_name!r} never reached "
            f"{self.metric} ≥ {self.active_threshold} — pinning hit a dead "
            f"replica or AIPerf isn't scraping it (peaks: {per_pod_peak})"
        )
        assert not peer_violations, (
            f"PinningContained: peer pod(s) saw {self.metric} > "
            f"{self.peer_max} — pinning leaked. Violations: {peer_violations}"
        )

    @property
    def description(self) -> str:
        return (
            f"{self.metric} concentrated on "
            f"{self.pinned_service}#{self.pinned_replica_index} "
            f"(≥ {self.active_threshold}); peers ≤ {self.peer_max}"
        )


from tests.fault_tolerance.deploy._checks_rejection import (  # noqa: E402, F401
    RejectionFired,
)

# Re-export sub-module check classes so they're visible to the auto-discovered
# Check registry in scenario_lib/_runtime.py (which walks Check.__subclasses__()
# after importing this module).
from tests.fault_tolerance.deploy._checks_workload_shape import (  # noqa: E402, F401
    WorkloadShapeVerified,
)

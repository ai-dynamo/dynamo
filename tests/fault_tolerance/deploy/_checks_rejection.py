# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cascade fail-fast rejection check. Subclassed Check rather than added inline to
# checks.py to keep the cascade-rejection signal sources (AIPerf 503
# tally + worker ``TrySendError::Full`` + Frontend ``Status code: 503``)
# co-located in one file the next round can iterate on without churning
# the main checks module.
#
# Pass criterion: ANY one of three signals fires above ``min_503_count``
# during the named load's wall-clock window. The window comes from the
# matching ``StartLoad`` event's ``started_at`` / ``ended_at`` so that
# noisy bootstrap-time 503s from the Frontend's
# ``system_status_server.rs`` health probe handler (the first ~2 min
# after pod start, before kube readiness flips) are excluded.

from __future__ import annotations

import json as _json
import os as _os
import re as _re
from dataclasses import dataclass
from datetime import datetime
from glob import escape as _glob_escape
from glob import glob as _glob
from typing import Optional

from tests.fault_tolerance.deploy.checks import (
    Check,
    _get_service_logs,
    _iter_jsonl,
    _load_dirs_for,
)

__all__ = ["RejectionFired"]


_TIME_RE = _re.compile(r'"time"\s*:\s*"([^"]+)"')


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (Z-suffixed or with +00:00 offset).

    Returns None on parse failure so the caller can skip malformed lines
    without aborting the whole scan.
    """
    if not ts:
        return None
    s = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _find_503_in_error(rec_error) -> bool:
    """Return True if an AIPerf jsonl ``error`` record looks like an
    HTTP 503. AIPerf wraps the underlying exception under ``type`` /
    ``message`` / ``cause`` / ``cause_chain`` — match a literal '503' or
    'Service Unavailable' anywhere in those fields.
    """
    if not isinstance(rec_error, dict):
        return False
    blob_parts = []
    for k in ("type", "message", "cause"):
        v = rec_error.get(k)
        if isinstance(v, str):
            blob_parts.append(v)
    chain = rec_error.get("cause_chain")
    if isinstance(chain, list):
        blob_parts.extend(str(x) for x in chain)
    blob = " ".join(blob_parts)
    if "503" in blob or "Service Unavailable" in blob:
        return True
    return False


@dataclass
class RejectionFired(Check):
    """Assert the worker-side fail-fast path fired during ``load_name``.

    Cascade fail-fast pass criterion: ANY of three signals shows up
    ≥ ``min_503_count`` times across the load's window:

      (a) AIPerf-side HTTP 503 — counted from the load's
          ``profile_export.jsonl`` ``error`` records (any record whose
          type / message / cause mentions 503 or "Service Unavailable")
          plus a fall-back tally of ``error_summary`` entries in
          ``profile_export_aiperf.json``.
      (b) Worker log line ``TrySendError::Full`` — emitted by the
          worker-side TCP work-queue rejection / backpressure path's
          ``work_tx.try_send()`` rejection path. Scans
          every ``VllmPrefillWorker`` / ``VllmDecodeWorker`` log file.
      (c) Frontend log line ``Status code: 503``, **filtered to the
          load's wall-clock window** (``StartLoad.started_at`` →
          ``StartLoad.ended_at``). Bootstrap-time 503s from the FE
          ``system_status_server.rs`` health probe (first ~2 min after
          pod start, before readiness flips) are NOISE and excluded.

    Logs all three counts at INFO so the report shows the breakdown
    even when only one source fires.
    """

    load_name: str
    min_503_count: int = 1

    # ------------------------------------------------------------------
    # Signal (a): AIPerf-side 503 tally
    # ------------------------------------------------------------------
    def _count_aiperf_503(self, ctx) -> int:
        total = 0
        for ldir in _load_dirs_for(ctx, self.load_name):
            # Per-record errors from the streaming jsonl.
            jsonl = _os.path.join(ldir, "profile_export.jsonl")
            for rec in _iter_jsonl(jsonl):
                if _find_503_in_error(rec.get("error")):
                    total += 1
            # Summary fall-back — AIPerf 0.7.x rolls some error
            # categories up into ``error_summary`` only.
            summary_path = _os.path.join(ldir, "profile_export_aiperf.json")
            if _os.path.isfile(summary_path):
                try:
                    with open(summary_path) as fh:
                        data = _json.load(fh)
                except (OSError, _json.JSONDecodeError):
                    data = {}
                es = data.get("error_summary") or []
                if isinstance(es, dict):
                    es = es.get("summary") or []
                # Avoid double counting when a 503 entry's count would
                # also appear in the per-record jsonl loop above: the
                # jsonl is the source of truth when present; the summary
                # is only consulted when the jsonl is missing.
                if not _os.path.isfile(jsonl):
                    for entry in es:
                        details = entry.get("error_details") or entry.get("error") or {}
                        if isinstance(details, dict):
                            blob = " ".join(
                                str(v)
                                for v in details.values()
                                if isinstance(v, (str, int))
                            )
                        else:
                            blob = str(details)
                        if "503" in blob or "Service Unavailable" in blob:
                            total += int(entry.get("count") or 0)
        return total

    # ------------------------------------------------------------------
    # Signal (b): worker TrySendError::Full
    # ------------------------------------------------------------------
    def _count_worker_try_send_full(self, ctx) -> int:
        """Walk per-worker on-disk log files directly (one file per
        pod). Cannot use ``_get_service_logs`` here because that
        concatenates everything under one key — we want a count, not a
        keyword search across a service-flat blob, but the count is
        also fine via the catted blob.
        """
        logs = _get_service_logs(ctx)
        total = 0
        for svc in ("VllmPrefillWorker", "VllmDecodeWorker"):
            text = logs.get(svc) or logs.get(svc.lower()) or ""
            total += text.count("TrySendError::Full")
        return total

    # ------------------------------------------------------------------
    # Signal (c): Frontend "Status code: 503", window-filtered
    # ------------------------------------------------------------------
    def _count_frontend_503_in_window(self, ctx) -> tuple[int, int]:
        """Return ``(in_window, total)``. ``total`` is the raw 503 count
        across all FE log files (debug breadcrumb so we can see how much
        of the FE noise we're discarding).
        """
        load = self.get_load(ctx, self.load_name)
        win_lo = getattr(load, "started_at", None) if load else None
        win_hi = getattr(load, "ended_at", None) if load else None

        in_window = 0
        total = 0
        log_dir = getattr(ctx, "log_dir", None)
        if not log_dir or not _os.path.isdir(log_dir):
            return 0, 0
        # FE logs live under <log_dir>/frontend/*.log (lowercased
        # service name, one file per pod per launch).
        fe_dir = _os.path.join(log_dir, "frontend")
        if not _os.path.isdir(fe_dir):
            return 0, 0
        for path in sorted(_glob(_os.path.join(_glob_escape(fe_dir), "*.log"))):
            try:
                fh = open(path, "r", errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    # Cheap pre-filter — most FE lines are not 503s.
                    if "503" not in line:
                        continue
                    if "Status code: 503" not in line and '"status":"503"' not in line:
                        continue
                    total += 1
                    if win_lo is None or win_hi is None:
                        continue
                    tm = _TIME_RE.search(line)
                    if not tm:
                        continue
                    dt = _parse_iso(tm.group(1))
                    if dt is None:
                        continue
                    if win_lo <= dt <= win_hi:
                        in_window += 1
        return in_window, total

    # ------------------------------------------------------------------
    # Check entry point
    # ------------------------------------------------------------------
    def validate(self, ctx) -> None:
        aiperf_503 = self._count_aiperf_503(ctx)
        worker_full = self._count_worker_try_send_full(ctx)
        fe_503_in_window, fe_503_total = self._count_frontend_503_in_window(ctx)

        observed = aiperf_503 + worker_full + fe_503_in_window
        ctx.logger.info(
            f"RejectionFired(load={self.load_name!r}): "
            f"aiperf_503={aiperf_503} "
            f"worker_TrySendError_Full={worker_full} "
            f"fe_503_in_window={fe_503_in_window} "
            f"(fe_503_total_unfiltered={fe_503_total}) "
            f"observed_total={observed} min_required={self.min_503_count}"
        )
        assert observed >= self.min_503_count, (
            f"RejectionFired: only {observed} rejection signals observed "
            f"(aiperf_503={aiperf_503}, worker_TrySendError_Full={worker_full}, "
            f"fe_503_in_window={fe_503_in_window}); expected ≥ "
            f"{self.min_503_count}. The worker fail-fast path did not "
            f"engage during load {self.load_name!r}."
        )

    @property
    def description(self) -> str:
        return (
            f"≥ {self.min_503_count} rejection signal(s) (AIPerf 503 / "
            f"worker TrySendError::Full / FE Status code:503 in-window) "
            f"during {self.load_name!r}"
        )

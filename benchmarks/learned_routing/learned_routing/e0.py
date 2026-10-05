# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E0(ISL, OSL): a request's policy-independent uncontended latency (CONTRACT Amendment A2).

E0 is the AIS-timed end-to-end latency of the request alone on an idle worker with no prefix
reuse: full chunked prefill of ISL, then OSL - 1 decode steps at batch 1. It is used only to score
requests; no policy ever sees it.

Method ``ais-chunked-estimator-v2`` (default):

- prefill = sum over chunks of ``max_num_batched_tokens`` of
  ``AisSession.predict_prefill(1, new, done)``. Replay TTFT equals this sum exactly (setup audit
  r1). A single unchunked call underestimates by 1.5% at 120K, so it is never used.
- decode = sum over ``j = 0 .. OSL - 2`` of the batch-1 step time at KV context ``ISL + j + 2``.
  This is replay's convention: the step that produces output token ``j + 2`` runs at
  ``sequence.len() = ISL + j + 1`` (aisimulate-core vLLM scheduler), and the mocker passes that
  length to the AIS callback as ``predict_decode(batch, len, 2)``, a single step at ``len + 1``
  (``lib/mocker/src/common/perf_model.rs``, ``lib/bindings/python/rust/llm/ais_callback.rs``).

v1 used context ``ISL + j + 1``, so it sat 6e-8 to 1.6e-5 below every isolated replay and an
uncontended no-reuse request (``e2e == E0``) failed ``S = 1`` (build audit goodput r2 F1). v2
matches 17,233 isolated single-request replays to within 2e-8 relative (fixer r2). The engine is the nominal
``engine.json`` identity; per-cell ``engine_overrides`` (timing perturbations) do not change E0.
:mod:`learned_routing.rescore_timing` scores timing-perturbed records a second time against the
perturbation-consistent E0' = prefill / s + decode / (s d) (phase2-mid sim-exploitation F2).

Values are cached per engine identity and method in ``CR/runs/cache/e0/<engine_sha>-<method>.json``
(``{"ISL,OSL": ms}``); concurrent writers merge under an ``flock``.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterable
from pathlib import Path

from learned_routing.canon import atomic_write_text, sha256_json

METHOD = "ais-chunked-estimator-v2"


class E0Table:
    def __init__(self, engine: dict, cache_dir: Path, method: str = METHOD):
        if method != METHOD:
            raise ValueError(f"unsupported E0 method {method!r}; available: {METHOD}")
        self.engine = engine
        self.method = method
        self.chunk = int(engine["mock_engine_args"]["max_num_batched_tokens"])
        identity = {"ais_perf_config": engine["ais_perf_config"], "chunk": self.chunk}
        self.engine_sha = sha256_json(identity)
        self.path = Path(cache_dir) / f"{self.engine_sha[:16]}-{method}.json"
        self.values: dict[str, float] = {}
        self._session = None
        self._step: dict[int, float] = {}
        self._prefill: dict[int, float] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self.values.update(json.loads(self.path.read_text()))

    @property
    def session(self):
        if self._session is None:
            from dynamo._internal.ais import create_session

            self._session = create_session(
                self.engine["ais_perf_config"], worker_type="aggregated"
            )
        return self._session

    def _prefill_ms(self, isl: int) -> float:
        if isl not in self._prefill:
            done, total = 0, 0.0
            while done < isl:
                new = min(self.chunk, isl - done)
                total += self.session.predict_prefill(1, new, done)
                done += new
            self._prefill[isl] = total
        return self._prefill[isl]

    def prefill_ms(self, isl: int) -> float:
        """The prefill part of E0 (chunked, no reuse) for ``isl`` input tokens."""
        return self._prefill_ms(max(int(isl), 1))

    def _step_ms(self, ctx: int) -> float:
        if ctx not in self._step:
            self._step[ctx] = self.session._estimate(
                {"num_decode_requests": 1, "sum_decode_kv_tokens": ctx}
            )
        return self._step[ctx]

    def compute(self, isl: int, osl: int) -> float:
        isl = max(int(isl), 1)
        total = self._prefill_ms(isl)
        for j in range(max(int(osl) - 1, 0)):
            total += self._step_ms(isl + j + 2)
        return total

    def __call__(self, isl: int, osl: int) -> float:
        key = f"{int(isl)},{int(osl)}"
        value = self.values.get(key)
        if value is None:
            value = self.compute(isl, osl)
            self.values[key] = value
            self._dirty = True
        return value

    def ensure(self, pairs: Iterable[tuple[int, int]]) -> None:
        for isl, osl in pairs:
            self(isl, osl)

    def persist(self) -> None:
        """Merge new values into the shared cache file (no-op when nothing was computed)."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            merged = json.loads(self.path.read_text()) if self.path.exists() else {}
            merged.update(self.values)
            atomic_write_text(self.path, json.dumps(merged, sort_keys=True))
            self.values = merged
            self._dirty = False
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
v2-only agg determinism matrix: G1 registration path × G2 block layout.

Companion to `test_determinism_agg.py` (which keeps the v1+v2+MLA coverage).
This file pins the matrix that the new connector dimensions opened up:

    G1 = LayerSeparate (LW)        G1 = FullyContiguous (FC)
    G2 = Operational      ✓            ✓
    G2 = Universal        ✓            ✓

Four specs, all on Qwen3-0.6B (small fast iter target; MLA is excluded
because FC does not support 3-dim fused K/V). Single onboard mode (inter)
on first pass — extend to {intra, inter} cross when the matrix proves stable.

The reuse-base determinism check (`base_test_determinism_with_cache_reset`)
catches output divergence across the cache reset. On top of that, this
module asserts:

  1. KVBM actually exercised offload+onboard during the run. If it did
     not, the determinism check would silently pass with KVBM disabled.
     Asserted via metrics-endpoint deltas across phase 1 / cache reset /
     phase 2.

  2. The G1 path was the one the spec requested. Read back from the
     server stdout log (the connector logs
     `[KVBM] prefer_cross_layer_blocks={True,False}` and
     `[KVBM] {Cross-layer KV cache,KV caches} registered`).

  3. G2 utilization is surfaced in the test output so we can iterate on
     `cpu_blocks` sizing. The cell with the largest G2 footprint pins
     the cluster-wide minimum.
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import pytest

from .common import TestDeterminism as BaseTestDeterminism
from .common import fetch_kvbm_metrics
from .fixtures import KvbmModelConfig, KvbmServerSpec

# Iteration count — overridable via KVBM_MATRIX_MAX_ITERATIONS only.
#
# Default 40: deliberately conservative calibration value. We do NOT yet
# have real G2 utilization numbers from this matrix (a previous draft
# extrapolated off synthetic adversarial-test data and landed on 200,
# which was unsubstantiated). 40 gives each unique Shakespeare prompt
# one recorded issuance, each control prompt ~4 issuances per phase
# (control_interval=10), and each random prompt ~5-6 issuances
# (random_interval=7) — enough to surface routing regressions without
# blowing G2 capacity. After the first real run reports per-cell
# utilization, ratchet up to push the heaviest cell toward 75%.
#
# Set `KVBM_MATRIX_MAX_ITERATIONS=10` for fast iteration cycles
# (structural assertions still fire; determinism signal is weaker).
#
# This is resolved ONCE at import time and applied to each per-test
# `kvbm_tester` instance inside the test body. We deliberately do NOT
# mutate `os.environ["KVBM_MAX_ITERATIONS"]` here — pytest imports all
# test modules during collection, and a module-level env mutation
# would leak into sibling test modules (e.g.
# `test_determinism_agg.py` reads `KVBM_MAX_ITERATIONS` at its OWN
# module load to compute timeouts, and `DeterminismTester.__init__`
# reads it again per-instance). Setting the value on the tester
# instance keeps the iteration override scoped to this test.
_DEFAULT_MATRIX_MAX_ITER = int(os.environ.get("KVBM_MATRIX_MAX_ITERATIONS", "40"))

# Server startup timeout (Qwen3-0.6B should come up in <60s but pad for
# slow CI machines).
_SERVER_START_TIMEOUT = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))

# Test body timeout: warmup + 2 phases of N iterations × ~4s/iter + 60s teardown.
_PER_TEST_TIMEOUT = _SERVER_START_TIMEOUT + 2 * (_DEFAULT_MATRIX_MAX_ITER * 4 + 60)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.kvbm,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Spec matrix
# ---------------------------------------------------------------------------

_MATRIX_MODEL = KvbmModelConfig(
    model_id="Qwen/Qwen3-0.6B",
    block_size=16,
    attention_backend="FLASH_ATTN",
)

# Onboard mode for the matrix. inter = production default; intra exists
# but is exercised separately by test_determinism_agg.py.
_MATRIX_ONBOARD_MODE = os.environ.get("KVBM_MATRIX_ONBOARD_MODE", "inter")

# cpu_blocks chosen large enough to fit the test workload without G2
# eviction (which would mask determinism bugs in onboard). We log actual
# G2 usage at end of phase 1 so we can ratchet this number down.
_MATRIX_CPU_BLOCKS = int(os.environ.get("KVBM_MATRIX_CPU_BLOCKS", "4000"))
_MATRIX_GPU_BLOCKS = int(os.environ.get("KVBM_MATRIX_GPU_BLOCKS", "2048"))


def _matrix_specs() -> List[KvbmServerSpec]:
    """Build the 4-cell matrix.

    Order: (block_layout, prefer_fc) — sorted so test ids appear in a
    predictable matrix order when pytest lists parametrizations.
    """
    cells = []
    for block_layout in ("operational", "universal"):
        for prefer_fc in (False, True):
            cells.append(
                KvbmServerSpec(
                    kvbm_version="v2",
                    model_config=_MATRIX_MODEL,
                    cpu_blocks=_MATRIX_CPU_BLOCKS,
                    gpu_blocks=_MATRIX_GPU_BLOCKS,
                    onboard_mode=_MATRIX_ONBOARD_MODE,
                    block_layout=block_layout,
                    prefer_fc=prefer_fc,
                )
            )
    return cells


_MATRIX_SPECS = _matrix_specs()
_MATRIX_PARAMS = [pytest.param(s, id=s.id) for s in _MATRIX_SPECS]


# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------


def _metric_delta(after: dict, before: dict, name: str) -> int:
    return int(after.get(name, 0)) - int(before.get(name, 0))


def _assert_phase1_offload(m_start: dict, m_phase1_end: dict) -> None:
    """Phase 1 (warmup + iterations on cold G2): G2 must get filled.

    A zero delta here means no offload happened — KVBM was effectively
    disabled. Without this guard the determinism check below passes
    trivially (vLLM produces the same output regardless of KVBM state).
    """
    offload_delta = _metric_delta(m_phase1_end, m_start, "kvbm_offload_blocks_d2h")
    print(f"[matrix] phase1: offload Δ={offload_delta} blocks")
    assert offload_delta > 0, (
        "KVBM did NOT offload any blocks during phase 1. "
        "Determinism test would silently pass with KVBM disabled — check "
        "connector init logs for fallback / disabled paths."
    )


def _assert_phase2_onboard(m_phase1_end: dict, m_end: dict) -> None:
    """Phase 2 (post-cache-reset, no warmup, G2 prewarmed): G2 hits must
    onboard back into G1.

    Phase 1 fills G2; the cache reset clears G1's prefix cache; phase 2
    re-issues the same prompts and should see G2 hits flowing through
    `onboard_blocks_h2d`. If onboard is zero in phase 2, either the
    cache reset evicted G2 too (regression in reset_prefix_cache scope)
    or the onboard path silently fell back to recompute. Either is a
    real failure that won't show up in the determinism string-equality
    check (recompute still produces the same output for temperature=0).
    """
    onboard_delta = _metric_delta(m_end, m_phase1_end, "kvbm_onboard_blocks_h2d")
    print(f"[matrix] phase2: onboard Δ={onboard_delta} blocks")
    assert onboard_delta > 0, (
        "KVBM did NOT onboard any blocks during phase 2. "
        "Either the cache reset cleared G2 (eviction-scope regression) "
        "or the onboard path silently fell back to recompute "
        "(determinism would still pass at temperature=0)."
    )


def _read_connector_log(log_dir: Path) -> str:
    """Concatenate the vLLM server log files written by KvbmServerManager.

    KvbmServerManager writes server stdout/stderr under `log_dir`; the
    connector's `print(...)` lines show up there. We grep them to verify
    the prefer_cross_layer_blocks branch the connector actually took.
    """
    if not log_dir.exists():
        return ""
    text = []
    for f in sorted(log_dir.iterdir()):
        if f.is_file() and f.suffix in (".log", ".txt"):
            try:
                text.append(f.read_text(errors="replace"))
            except Exception:
                pass
    return "\n".join(text)


_RE_PREFER_TRUE = re.compile(r"\[KVBM\] prefer_cross_layer_blocks=True")
_RE_PREFER_FALSE = re.compile(r"\[KVBM\] prefer_cross_layer_blocks=False")
_RE_FC_REGISTER = re.compile(r"\[KVBM\] Cross-layer KV cache registered")
_RE_LW_REGISTER = re.compile(r"\[KVBM\] KV caches registered \(deferred mode\)")


def _assert_g1_path_matches_spec(server_log: str, spec: KvbmServerSpec) -> None:
    """Verify the connector took the G1 registration path the spec asked for.

    A failure here means the JSON config plumb to
    `kv_connector_extra_config.default.prefer_fully_contiguous_blocks` did
    not flow through, or the connector ignored it. Either way we want a
    loud failure instead of a determinism-test pass that masked the
    routing regression.
    """
    if spec.prefer_fc is True:
        assert _RE_PREFER_TRUE.search(server_log), (
            "spec.prefer_fc=True but connector did not log "
            "'prefer_cross_layer_blocks=True'. Check JSON-config plumb."
        )
        assert _RE_FC_REGISTER.search(server_log), (
            "spec.prefer_fc=True but connector did not register the "
            "cross-layer FC tensor."
        )
    elif spec.prefer_fc is False:
        assert _RE_PREFER_FALSE.search(server_log), (
            "spec.prefer_fc=False but connector did not log "
            "'prefer_cross_layer_blocks=False'."
        )
        assert _RE_LW_REGISTER.search(server_log), (
            "spec.prefer_fc=False but connector did not register per-layer "
            "KV caches (LW path)."
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterminismAggV2Matrix(BaseTestDeterminism):
    """v2 agg determinism over the (G1, G2) routing matrix.

    Inherits `base_test_determinism_with_cache_reset` from `TestDeterminism`
    and wraps it with structural assertions specific to this matrix.
    """

    @pytest.mark.parametrize("kvbm_server_spec", _MATRIX_PARAMS, indirect=True)
    @pytest.mark.timeout(_PER_TEST_TIMEOUT)
    def test_determinism_v2_matrix(self, kvbm_tester, kvbm_server):
        """Determinism across cache reset, asserted to actually exercise KVBM
        on BOTH phases (phase 1 offload, phase 2 onboard).
        """
        # Pin iteration count on the per-test (function-scoped) tester
        # instance. Setting the env var would leak into sibling test
        # modules imported in the same pytest collection — see the
        # _DEFAULT_MATRIX_MAX_ITER comment for why this is on the
        # instance, not in os.environ.
        kvbm_tester.max_iterations = _DEFAULT_MATRIX_MAX_ITER

        metrics_port = kvbm_server.metrics_port
        spec = kvbm_server.spec
        print(
            f"\n[matrix] spec={spec.id} iterations/phase={kvbm_tester.max_iterations}"
        )
        print(
            f"[matrix] block_layout={spec.block_layout}, "
            f"prefer_fc={spec.prefer_fc}, onboard_mode={spec.onboard_mode}"
        )

        # Snapshot 0: pre-warmup.
        time.sleep(2)  # let metrics endpoint settle
        m0 = fetch_kvbm_metrics(port=metrics_port)
        print(f"[matrix] m0 (pre-warmup): {m0}")

        # Capture metrics at the phase 1/2 boundary by wrapping
        # `tester.reset_prefix_cache` — the base method calls it exactly
        # once, right between the two phases. Without this snapshot, we
        # can only assert "KVBM did work at some point" (which the
        # determinism check could mask) instead of "phase 1 filled G2
        # AND phase 2 onboarded from it" (the actual coverage claim).
        boundary: Dict[str, dict] = {}
        original_reset = kvbm_tester.reset_prefix_cache

        def _reset_with_snapshot() -> None:
            boundary["m_phase1_end"] = fetch_kvbm_metrics(port=metrics_port)
            print(f"[matrix] m_phase1_end (pre-reset): {boundary['m_phase1_end']}")
            original_reset()

        kvbm_tester.reset_prefix_cache = _reset_with_snapshot

        try:
            super().base_test_determinism_with_cache_reset(
                kvbm_tester, kvbm_server, None
            )
        finally:
            kvbm_tester.reset_prefix_cache = original_reset

        # Snapshot end-of-test.
        m_end = fetch_kvbm_metrics(port=metrics_port)
        print(f"[matrix] m_end (post-test): {m_end}")

        # Defensive: the boundary callback MUST have fired. If it didn't,
        # the test silently regressed to single-shot coverage.
        if "m_phase1_end" not in boundary:
            pytest.fail(
                "phase 1/2 boundary snapshot was never taken — "
                "tester.reset_prefix_cache was not called by the base "
                "method? This guard exists so the matrix gate cannot "
                "silently regress to single-shot coverage."
            )
        m_phase1_end = boundary["m_phase1_end"]

        # Structural assertions: KVBM must have done work in BOTH phases.
        _assert_phase1_offload(m0, m_phase1_end)
        _assert_phase2_onboard(m_phase1_end, m_end)

        # Verify the G1 routing path matched the spec.
        log_dir = getattr(kvbm_server, "log_dir", None)
        if log_dir is None:
            print("[matrix] warning: server has no log_dir; skipping G1-path assert")
        else:
            log_text = _read_connector_log(Path(log_dir))
            _assert_g1_path_matches_spec(log_text, spec)
            print(f"[matrix] G1 path assertion ok (prefer_fc={spec.prefer_fc})")

        # Visibility: how much G2 was actually used? Future iteration of
        # this skill / test should use the largest cell's value to size
        # KVBM_MATRIX_CPU_BLOCKS more tightly.
        offload_total = m_end.get("kvbm_offload_blocks_d2h", 0)
        print(
            f"[matrix] {spec.id}: G2 offload total = {offload_total} blocks "
            f"(of {spec.cpu_blocks} configured) — "
            f"utilization = {100.0 * offload_total / max(1, spec.cpu_blocks):.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

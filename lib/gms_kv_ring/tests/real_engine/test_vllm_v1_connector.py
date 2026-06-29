# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-engine validation for GMSKVCacheConnectorV1.

HEAVILY env-gated; never runs in the default suite. Brings up a
real vLLM + GMSWorker + GMSKVCacheConnectorV1 stack and validates:

  1. Smoke: vLLM constructs; the connector's `register_kv_caches`
     fires; one generation completes without crashing.
  2. Output-equality: same prompt twice produces byte-identical
     tokens (the connector's restore-on-hit path must not perturb
     KV bytes).
  3. Connector telemetry: the four new Prometheus counters are
     readable post-run.

Parametrized over `tensor_parallel_size`. TP=1 on any single GPU;
TP=2 requires 2+ GPUs (auto-skipped otherwise).

RUN:

    GMS_KVR_REAL_VLLM=1 \\
    GMS_KVR_REAL_VLLM_TP=1 \\
    GMS_KVR_REAL_VLLM_MODEL=facebook/opt-125m \\
    LD_LIBRARY_PATH=/path/to/cuda/libs:... \\
    PYTHONPATH=/path/to/dynamo/lib \\
    pytest -v -s lib/gms_kv_ring/tests/real_engine/test_vllm_v1_connector.py

ENV vars (all optional unless noted):
  GMS_KVR_REAL_VLLM=1          required gate (default skip)
  GMS_KVR_REAL_VLLM_TP=N       1 (default) or 2
  GMS_KVR_REAL_VLLM_MODEL      HF id; default facebook/opt-125m
  GMS_KVR_REAL_VLLM_DAEMON     daemon socket path
  GMS_KVR_REAL_VLLM_PROMPT     prompt; default short factual

Known concerns surfaced during initial bring-up are commented
inline. If a test fails on first run on new hardware, the
traceback typically points at one of:
  - vLLM version mismatch (KVConnectorBase_V1 API drift)
  - kv_transfer_config field rename
  - worker_cls forwarding through EngineArgs
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

# vLLM 0.20.x triggers a tcgen05.OperandMajorMode DeprecationWarning
# from nvidia_cutlass_dsl during EngineCore init — repo-wide
# pytest filterwarnings=error escalates it to fatal, killing the
# worker subprocess. The warning is third-party + not actionable
# on our side; scope-ignore it for this real-engine harness only.
pytestmark = pytest.mark.filterwarnings(
    "ignore:tcgen05.*is deprecated:DeprecationWarning",
)

if os.environ.get("GMS_KVR_REAL_VLLM") != "1":
    pytest.skip(
        "Set GMS_KVR_REAL_VLLM=1 to run real-engine tests "
        "(needs vLLM + GPU + meaningful runtime).",
        allow_module_level=True,
    )

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


TP = int(os.environ.get("GMS_KVR_REAL_VLLM_TP", "1"))
MODEL = os.environ.get("GMS_KVR_REAL_VLLM_MODEL", "facebook/opt-125m")
DAEMON_SOCK = os.environ.get(
    "GMS_KVR_REAL_VLLM_DAEMON",
    "/tmp/gms-real-vllm.sock",
)
PROMPT = os.environ.get(
    "GMS_KVR_REAL_VLLM_PROMPT",
    "The capital of France is the city of",
)

if TP > torch.cuda.device_count():
    pytest.skip(
        f"GMS_KVR_REAL_VLLM_TP={TP} but only "
        f"{torch.cuda.device_count()} CUDA device(s) available",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------
# In-process daemon fixture (module-scoped).
# Production deploys a per-GPU daemon as a separate process; in this
# smoke test we co-locate it to keep the harness self-contained.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def gms_daemon():
    """Start an in-process GMS daemon serving on DAEMON_SOCK."""
    from gms_kv_ring.daemon.server import Daemon

    if os.path.exists(DAEMON_SOCK):
        os.unlink(DAEMON_SOCK)

    d = Daemon(DAEMON_SOCK, supervise_backend=False)
    holder: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True, name="gms-real-vllm")
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(DAEMON_SOCK):
        time.sleep(0.05)
    assert os.path.exists(DAEMON_SOCK), "daemon socket did not appear"

    yield d

    holder["loop"].call_soon_threadsafe(d.stop)
    t.join(timeout=5)


# ---------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------


def _build_llm():
    """Construct a vLLM `LLM` wired to GMSWorker + the V1 connector.

    Notes:
      - `worker_cls` is forwarded to EngineArgs.worker_cls.
      - `kv_transfer_config` carries the connector's engine_id +
        daemon socket. Connector reads
        `kv_transfer_config.engine_id` and
        `kv_connector_extra_config.gms_daemon_socket`.
      - `enforce_eager=True` skips CUDA Graph init (faster smoke).
      - `enable_prefix_caching=True` is REQUIRED — without it vLLM
        never calls `get_num_new_matched_tokens` and our cache-hit
        path is never exercised.
      - `gpu_memory_utilization=0.4` keeps room for two engines
        sharing a single GPU during failover testing.
    """
    from vllm import LLM
    from vllm.config import KVTransferConfig

    cfg = KVTransferConfig(
        kv_connector="GMSKVCacheConnectorV1",
        # External module path takes priority over internal registry
        # in vLLM's KVConnectorFactory (see factory._get_connector_
        # class_with_compat). This lets us load the connector
        # without needing GMSWorker registration — the legacy
        # GMSWorker requires a separate `weights` GMS server which
        # is an orthogonal feature (model-weight sharing). For
        # connector-only smoke tests, use vLLM's default worker
        # and import our class by full module path.
        kv_connector_module_path=(
            "gpu_memory_service.integrations.vllm.gds_connector_v1"
        ),
        # Our connector both saves (evicts) and loads (restores)
        # KV bytes — role "kv_both". vLLM requires this field on
        # any non-trivial connector config.
        kv_role="kv_both",
        engine_id="0",
        kv_connector_extra_config={"gms_daemon_socket": DAEMON_SOCK},
    )
    return LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enforce_eager=True,
        gpu_memory_utilization=float(
            os.environ.get(
                "GMS_KVR_REAL_VLLM_GPU_MEM_UTIL",
                "0.4",
            )
        ),
        enable_prefix_caching=True,
        kv_transfer_config=cfg,
        max_model_len=256,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm(gms_daemon):
    """One LLM per module (heavy init). Reused across tests."""
    obj = _build_llm()
    yield obj


# NOTE: a separate "adversarial" LLM fixture with low
# gpu_memory_utilization was attempted but cannot coexist with the
# main `llm` fixture in the same pytest process — vLLM V1 spawns
# the EngineCore as a CUDA-using subprocess, and a second
# `LLM(...)` in the same parent process fails CUDA init in its
# subprocess. The adversarial test therefore shares the main
# fixture and softly reports whether eviction actually fired
# (depends on workload + HBM budget).


def test_smoke_one_generation(llm):
    """Single request: connector registered, generation produces
    non-empty output. If this fails the integration is broken at
    one of: connector registration, GMSWorker init, daemon socket
    connect, or per-rank GMSKvRing build."""
    from vllm import SamplingParams

    out = llm.generate(
        PROMPT,
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    assert len(out) == 1
    gen = out[0]
    assert gen.outputs and gen.outputs[0].token_ids, f"empty generation: {gen!r}"
    print(
        f"[smoke TP={TP}] prompt={PROMPT!r} → tokens="
        f"{gen.outputs[0].token_ids} text={gen.outputs[0].text!r}"
    )


def test_output_equality_on_cache_hit(llm):
    """Two identical requests; output tokens must be byte-identical.

    With temperature=0.0 vLLM is deterministic, so the test
    would pass even WITHOUT a cache hit — that's by design. The
    point is that the cache-hit path must ALSO be byte-correct:
    if the connector restores wrong KV bytes, the model would
    still sample but tokens would diverge from the cold-compute
    run."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=8)

    out1 = llm.generate(PROMPT, sp)
    tokens1 = tuple(out1[0].outputs[0].token_ids)
    text1 = out1[0].outputs[0].text

    out2 = llm.generate(PROMPT, sp)
    tokens2 = tuple(out2[0].outputs[0].token_ids)
    text2 = out2[0].outputs[0].text

    assert tokens1 == tokens2, (
        f"[TP={TP}] non-deterministic across identical prompts: "
        f"first={tokens1} second={tokens2}\n"
        f"first_text={text1!r}\nsecond_text={text2!r}\n"
        f"(this means the restore-on-hit path corrupted KV bytes — "
        f"production-blocking)"
    )
    print(f"[equality TP={TP}] tokens match across two calls: {tokens1}")


def _read_counter(name: str, engine_id: str = "0") -> int:
    """Cheap helper for adversarial-test metric snapshots."""
    from gms_kv_ring.common import metrics

    c = getattr(metrics, name, None)
    if c is None:
        return 0
    try:
        return int(c.get(engine_id=engine_id))
    except Exception:  # noqa: BLE001
        return 0


def test_adversarial_prefix_cache_byte_correctness(llm):
    """End-to-end adversarial validation of the connector under
    realistic conditions:

      1. Build a long SHARED prefix (~512 prompt tokens).
      2. Generate a "cold" output for a (shared_prefix + cats)
         request. Captures the ground-truth tokens.
      3. Run many DISTINCT decoy prompts to push the vLLM
         scheduler into actually evicting cached blocks — the
         connector's `request_finished` populates the prefix
         index AND the worker-side eviction fires for blocks
         that vLLM is about to free.
      4. Re-issue (shared_prefix + cats). The hot HBM blocks
         from step 2 are gone; vLLM's V1 prefix cache asks the
         connector via `get_num_new_matched_tokens` and gets
         the daemon-stored prefix back. Output must be
         byte-identical to step 2 — proves restore-on-hit is
         byte-correct under realistic flow.
      5. Report metric deltas so an operator running the test
         can see what fired (evicts, restores, ring-full
         fallbacks, conflict-sync, daemon-epoch changes,
         prefix-invalidations).

    Failure of this test ≠ a connector bug in isolation; it
    means at least one of the Phase A-D mechanisms is broken
    under realistic conditions. Output equality is the
    production-blocking assert.
    """
    from vllm import SamplingParams

    # ~96 prompt tokens of TinyLlama-tokenized text — fits in
    # the fixture's max_model_len=256 with room for output and
    # a Q/A suffix. Repetitive but block-aligned enough that the
    # prefix-hash machinery records a consistent digest.
    SHARED_PREFIX = (
        "You are a careful, methodical assistant. "
        "Read context carefully and answer concisely. " * 7
    )
    eid = "0"

    # Step 2: cold reference.
    sp = SamplingParams(temperature=0.0, max_tokens=12)
    cold_prompt = SHARED_PREFIX + " Q: Name a domestic feline. A:"
    out_cold = llm.generate(cold_prompt, sp)
    cold_tokens = tuple(out_cold[0].outputs[0].token_ids)
    cold_text = out_cold[0].outputs[0].text
    print(
        f"[adversarial TP={TP}] cold prompt → tokens="
        f"{cold_tokens} text={cold_text!r}"
    )

    # Snapshot counters before the pressure phase.
    metrics_before = {
        name: _read_counter(name, eid)
        for name in (
            "evict_records",
            "restore_records",
            "connector_evict_failures",
            "connector_restore_failures",
            "connector_restore_ring_full",
            "connector_restore_conflict_sync",
            "connector_daemon_epoch_changes",
            "connector_prefix_invalidated_on_failure",
        )
    }

    # Step 3: decoy pressure. Each decoy has a DIFFERENT prefix
    # so it doesn't share with our cold prompt. Generating these
    # in sequence pushes blocks out of vLLM's HBM cache (the
    # connector's prefix index, being scheduler-side, retains
    # the cold prompt's hash → block_id mapping; vLLM's HBM
    # cache might lose it under pressure).
    decoy_count = 16
    for i in range(decoy_count):
        # Each decoy starts with a DIFFERENT preamble so it
        # never shares a prefix with the cold prompt — that
        # would defeat the purpose of measuring eviction.
        decoy = f"Topic {i} stands alone. " * 4 + f" Continue topic {i}:"
        llm.generate(decoy, sp)

    # Step 4: warm re-issue. Must produce IDENTICAL tokens.
    out_warm = llm.generate(cold_prompt, sp)
    warm_tokens = tuple(out_warm[0].outputs[0].token_ids)
    warm_text = out_warm[0].outputs[0].text

    metrics_after = {name: _read_counter(name, eid) for name in metrics_before}
    deltas = {
        name: metrics_after[name] - metrics_before[name] for name in metrics_before
    }

    # Surface what fired regardless of pass/fail.
    print(
        f"[adversarial TP={TP}] metric deltas across "
        f"{decoy_count} decoys + warm reissue: {deltas}"
    )
    print(f"[adversarial TP={TP}] warm tokens={warm_tokens}")

    # PRODUCTION-BLOCKING ASSERT: cold == warm. If this fires,
    # the restore-on-hit path corrupted KV bytes under load —
    # the kind of bug that only surfaces in realistic flow.
    assert cold_tokens == warm_tokens, (
        f"[TP={TP}] byte-correctness FAILED under adversarial "
        f"flow.\n"
        f"  cold  = {cold_tokens}\n  warm  = {warm_tokens}\n"
        f"  cold_text = {cold_text!r}\n  warm_text = {warm_text!r}\n"
        f"  metric deltas: {deltas}\n"
        f"This indicates a bug in the eviction → daemon → "
        f"restore round-trip surfaced ONLY under memory "
        f"pressure. Investigate before shipping."
    )

    # PRODUCTION-BLOCKING: any failure-path metric incrementing
    # means a Phase A-D mechanism fired during a workload that
    # SHOULD be entirely on the happy path. Treat as bug.
    for name in (
        "connector_evict_failures",
        "connector_restore_failures",
        "connector_prefix_invalidated_on_failure",
    ):
        assert deltas[name] == 0, (
            f"[TP={TP}] {name} incremented by {deltas[name]} "
            f"during adversarial run. Full deltas: {deltas}"
        )

    # SOFT signal: did we actually exercise the daemon round-
    # trip? In a tight-memory configuration this should be
    # non-zero; in a loose-memory configuration vLLM's HBM
    # prefix cache holds everything and the connector never
    # spills. Print a clear warning either way so the operator
    # running this test knows what was actually validated.
    if deltas["evict_records"] == 0 and deltas["restore_records"] == 0:
        print(
            f"\n[adversarial TP={TP}] WARNING: connector daemon "
            f"round-trip did NOT fire — vLLM's in-HBM prefix "
            f"cache absorbed every request. The byte-correctness "
            f"assert above is still valid (proves the connector "
            f"didn't corrupt cache-hit reads served by vLLM's own "
            f"cache), but Phase A-D mechanisms are exercised only "
            f"by unit tests. To exercise the real daemon path, "
            f"tighten gpu_memory_utilization further or run a "
            f"longer-prompt workload.\n"
        )
    else:
        print(
            f"\n[adversarial TP={TP}] connector daemon round-"
            f"trip exercised: evicts={deltas['evict_records']} "
            f"restores={deltas['restore_records']}. Phase A-D "
            f"validated end-to-end.\n"
        )


def test_connector_telemetry_visible(llm):
    """The 4 connector counters must be readable. Even zero is OK
    on a fresh process; we only assert the labels schema."""
    from gms_kv_ring.common import metrics

    for name in (
        "connector_evict_failures",
        "connector_restore_failures",
        "connector_restore_ring_full",
        "connector_restore_conflict_sync",
    ):
        c = getattr(metrics, name)
        v = c.get(engine_id="0")
        assert v >= 0, f"{name} has negative count {v}"
    print(f"[metrics TP={TP}] all 4 connector counters readable")

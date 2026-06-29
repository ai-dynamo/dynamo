# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-engine eviction validation — env-gated harness.

PROBLEM THIS FILE TRACKS
========================

The TinyLlama harness in `test_vllm_v1_connector.py` validates
that the connector doesn't BREAK byte-correctness under load, but
cannot force the daemon round-trip to actually fire: vLLM's HBM
prefix cache absorbs every cache hit before it reaches our
connector.

A Qwen3-8B harness with `gpu_memory_utilization` tight enough to
overflow vLLM's HBM cache DOES exist in this file (commented out
below), and it correctly triggers vLLM's LRU eviction. However,
when the connector's eviction path fires and the daemon tries to
read the engine's HBM bytes, we hit a SEGFAULT in cuMemcpyDtoH:

  vLLM V1 architecture
    ├── parent Python process: scheduler-side connector + daemon
    └── EngineCore subprocess: worker-side connector + KV tensors

  The engine's HBM pointers are valid only in the EngineCore
  subprocess. The daemon, running in the parent process, cannot
  cudaMemcpy() from those pointers — that's the segfault.

WHY UNIT TESTS DON'T HIT THIS
-----------------------------

The unit tests (`test_demote_hbm_to_storage.py`) plug in a
`_FakeGpuDirectBackend` AND run vLLM-shaped mocks entirely
in-process: the "engine" is a torch.Tensor in the test's own
process, so cudaMemcpy works. Real vLLM V1 has a hard process
boundary the daemon can't reach with naive memcpy.

WHAT MAKES IT WORK IN PRODUCTION
--------------------------------

Real GDS-capable backends (NIXL with cuFile, or Mooncake with
its transfer engine) cross the process boundary through CUDA
driver IPC primitives + cuFile direct-DMA. They map the engine
subprocess's GPU memory into the daemon process via the kernel,
or push bytes engine→storage on the engine's own CUDA stream.
Either way, no naive parent-process cudaMemcpy is involved.

WHAT VALIDATES OUR CODE TODAY
-----------------------------

  - 78 unit tests, including:
      * Evict ring + restore ring shaped exactly like production
      * Drop-on-failure (Phase B)
      * Host-tier scrub (Phase C)
      * Backend scrub (Phase D)
      * Daemon-epoch invalidation (Phase A)
  - Real-vLLM TinyLlama smoke + equality + adversarial tests:
      * Validate that the connector doesn't break byte-correctness
        under load (Phase A-D wiring would catch failures here
        even though the daemon round-trip doesn't fire)

WHAT DOES NOT YET HAVE END-TO-END VALIDATION
--------------------------------------------

  - Real-vLLM + real-NIXL-GDS + real eviction. Requires:
      * cuFile-capable storage mount on the test host
      * NIXL built with GDS plugin
      * `nvidia-fs` kernel module loaded
    The end-to-end test would otherwise be straightforward —
    same workload as below, daemon configured with NixlBackend
    (GDS or GDS_MT plugin), pre-flight gate via
    `scripts/gds_preflight.sh`.

When that env is available, set:
    GMS_KVR_REAL_VLLM=1
    GMS_KVR_REAL_VLLM_BIG=1
    GMS_KVR_REAL_VLLM_BIG_NIXL_GDS=1   # opt-in to real-GDS
    GMS_KVR_NIXL_GDS_PATH=/cufile/mount/path

Until that env exists on this host, this file skips
unconditionally — the harness body is documented inline below
for the eventual operator to validate against real GDS.
"""

from __future__ import annotations

import os

import pytest

# Hard skip until real GDS hardware is set up. The "BIG" gate is
# preserved for forward compat — when an operator turns it on
# WITHOUT the GDS opt-in, surface a clear message instead of
# silently passing.
if os.environ.get("GMS_KVR_REAL_VLLM") != "1":
    pytest.skip(
        "Set GMS_KVR_REAL_VLLM=1 to enable the real-engine harness.",
        allow_module_level=True,
    )
if os.environ.get("GMS_KVR_REAL_VLLM_BIG") != "1":
    pytest.skip(
        "Set GMS_KVR_REAL_VLLM_BIG=1 (in addition to "
        "GMS_KVR_REAL_VLLM=1) to enable the 8B eviction harness.",
        allow_module_level=True,
    )
if os.environ.get("GMS_KVR_REAL_VLLM_BIG_NIXL_GDS") != "1":
    pytest.skip(
        "Set GMS_KVR_REAL_VLLM_BIG_NIXL_GDS=1 to run the real-"
        "eviction harness against a NIXL-GDS backend. Without "
        "real GDS, the daemon cannot cross the EngineCore "
        "subprocess boundary to read engine HBM — see this "
        "file's module docstring.",
        allow_module_level=True,
    )

# Real-GDS harness body — preserved for the future operator with
# the right environment. NOT validated on this host (no cuFile
# mount). When you enable this, also build NixlBackend with the
# GDS plugin and pass it via `storage_backend=NixlBackend(...)`
# to the Daemon constructor below.

import asyncio  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:tcgen05.*is deprecated:DeprecationWarning",
)
vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

TP = int(os.environ.get("GMS_KVR_REAL_VLLM_TP", "1"))
MODEL = os.environ.get(
    "GMS_KVR_REAL_VLLM_BIG_MODEL",
    "Qwen/Qwen3-8B",
)
GMU = float(os.environ.get("GMS_KVR_REAL_VLLM_BIG_GMU", "0.38"))
MAX_LEN = int(os.environ.get("GMS_KVR_REAL_VLLM_BIG_MAX_LEN", "2048"))
DAEMON_SOCK = os.environ.get(
    "GMS_KVR_REAL_VLLM_DAEMON",
    "/tmp/gms-real-vllm-big.sock",
)
NIXL_GDS_PATH = os.environ.get(
    "GMS_KVR_NIXL_GDS_PATH",
    "",
)
if not NIXL_GDS_PATH:
    pytest.skip(
        "GMS_KVR_NIXL_GDS_PATH must be set to a cuFile-mount "
        "directory for the GDS data plane.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def gms_daemon():
    from gms_kv_ring.daemon.backends_nixl import NixlBackend
    from gms_kv_ring.daemon.server import Daemon

    if os.path.exists(DAEMON_SOCK):
        os.unlink(DAEMON_SOCK)

    # NixlBackend with GDS plugin (cuFile). Caller is responsible
    # for ensuring nvidia-fs is loaded + the mount is cuFile-
    # capable; gds_preflight.sh covers that.
    backend = NixlBackend(
        storage_dir=NIXL_GDS_PATH,
        plugin="GDS",
    )
    d = Daemon(
        DAEMON_SOCK,
        storage_backend=backend,
        supervise_backend=False,
    )
    holder: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True, name="gms-big-d")
    t.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not os.path.exists(DAEMON_SOCK):
        time.sleep(0.05)
    assert os.path.exists(DAEMON_SOCK), "daemon socket did not appear"

    yield d

    holder["loop"].call_soon_threadsafe(d.stop)
    t.join(timeout=5)


def _build_llm():
    from vllm import LLM
    from vllm.config import KVTransferConfig

    cfg = KVTransferConfig(
        kv_connector="GMSKVCacheConnectorV1",
        kv_connector_module_path=(
            "gpu_memory_service.integrations.vllm.gds_connector_v1"
        ),
        kv_role="kv_both",
        engine_id="0",
        kv_connector_extra_config={"gms_daemon_socket": DAEMON_SOCK},
    )
    return LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enforce_eager=True,
        gpu_memory_utilization=GMU,
        enable_prefix_caching=True,
        kv_transfer_config=cfg,
        max_model_len=MAX_LEN,
    )


@pytest.fixture(scope="module")
def llm(gms_daemon):
    obj = _build_llm()
    yield obj


def _read_counter(name: str, engine_id: str = "0") -> int:
    from gms_kv_ring.common import metrics

    c = getattr(metrics, name, None)
    if c is None:
        return 0
    try:
        return int(c.get(engine_id=engine_id))
    except Exception:  # noqa: BLE001
        return 0


def _snapshot_metrics(engine_id: str = "0") -> dict:
    return {
        name: _read_counter(name, engine_id)
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


def test_real_eviction_byte_correctness(llm):
    """Forces actual eviction → daemon round-trip → restore.
    Validated against a real NIXL-GDS backend. See module
    docstring for why this requires real GDS rather than a fake."""
    from vllm import SamplingParams

    SHARED_PREFIX = (
        "You are a thoughtful research assistant. Read context "
        "carefully. Answer concisely. Be precise. "
        "Cite sources when relevant. Avoid speculation. " * 16
    )
    eid = "0"
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    cold_prompt = SHARED_PREFIX + " Q: Name one common house pet. A:"
    out_cold = llm.generate(cold_prompt, sp)
    cold_tokens = tuple(out_cold[0].outputs[0].token_ids)

    metrics_before = _snapshot_metrics(eid)

    decoy_count = 16
    for i in range(decoy_count):
        decoy = (
            f"Topic {i}: This is a self-contained essay on subject "
            f"number {i}, with content unique to topic {i}. " * 50
            + f" Continue subject {i}:"
        )
        llm.generate(decoy, sp)

    out_warm = llm.generate(cold_prompt, sp)
    warm_tokens = tuple(out_warm[0].outputs[0].token_ids)

    metrics_after = _snapshot_metrics(eid)
    deltas = {
        name: metrics_after[name] - metrics_before[name] for name in metrics_before
    }

    print(f"[real-evict TP={TP}] metric deltas: {deltas}")

    assert cold_tokens == warm_tokens, (
        f"[TP={TP}] byte-correctness FAILED under real eviction.\n"
        f"  cold={cold_tokens} warm={warm_tokens}\n"
        f"  deltas={deltas}"
    )
    for name in (
        "connector_evict_failures",
        "connector_restore_failures",
        "connector_prefix_invalidated_on_failure",
    ):
        assert (
            deltas[name] == 0
        ), f"[TP={TP}] {name} fired during happy-path run: {deltas}"
    assert deltas["evict_records"] > 0, (
        f"[TP={TP}] eviction did not fire; tighten "
        f"GMS_KVR_REAL_VLLM_BIG_GMU (currently {GMU}) or "
        f"increase decoy workload. deltas={deltas}"
    )

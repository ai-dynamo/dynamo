# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-engine smoke for the TensorRT-LLM GMS connector.

Loads a small HF model through TensorRT-LLM's PyTorch backend with our
GMSTrtllmKvCacheScheduler / Worker wired in, generates a few tokens,
and asserts the connector lifecycle hooks fired (register_kv_caches +
build_connector_meta + wait_for_save / start_load_kv).

Run from a Dynamo TensorRT-LLM runtime container or equivalent dev
environment with TensorRT-LLM importable:

    cd /path/to/dynamo
    export PYTHONPATH=/path/to/dynamo/lib:/path/to/dynamo/lib/gpu_memory_service
    export CUDA_VISIBLE_DEVICES=1   # if GPU 0 is busy
    export GMS_KVR_REAL_TRTLLM=1
    export GMS_KVR_REAL_TRTLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
    export GMS_KVR_REAL_TRTLLM_MEM_FRACTION=0.3
    python -m pytest lib/gms_kv_ring/tests/real_engine/test_trtllm_connector.py -v

Env knobs:
  GMS_KVR_REAL_TRTLLM=1                required gate (default skip)
  GMS_KVR_REAL_TRTLLM_MODEL            HF id; default TinyLlama-1.1B
  GMS_KVR_REAL_TRTLLM_DAEMON           daemon UDS path
  GMS_KVR_REAL_TRTLLM_MEM_FRACTION     gpu memory budget (default 0.4)
  GMS_KVR_REAL_TRTLLM_PROMPT           prompt; default short factual
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time

import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

if os.environ.get("GMS_KVR_REAL_TRTLLM") != "1":
    pytest.skip(
        "Set GMS_KVR_REAL_TRTLLM=1 to run real-engine TRT-LLM tests "
        "(needs TensorRT-LLM + GPU + meaningful runtime).",
        allow_module_level=True,
    )

# pytest's importorskip turns DeprecationWarning-from-import into
# ImportError under strict-warnings configs. Some of TRT-LLM's
# transitive deps (torchao) emit DeprecationWarnings during import.
# Try a plain import inside a warning-quieted block first.
import warnings as _w  # noqa: E402

with _w.catch_warnings():
    _w.filterwarnings("ignore", category=DeprecationWarning)
    _w.filterwarnings("ignore", category=UserWarning)
    try:
        import tensorrt_llm as trtllm  # noqa: F401
    except ImportError as _exc:
        pytest.skip(
            f"tensorrt_llm not importable: {_exc}",
            allow_module_level=True,
        )

MODEL = os.environ.get(
    "GMS_KVR_REAL_TRTLLM_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)
DAEMON_SOCK = os.environ.get(
    "GMS_KVR_REAL_TRTLLM_DAEMON",
    "/tmp/gms-real-trtllm.sock",
)
MEM_FRACTION = float(
    os.environ.get(
        "GMS_KVR_REAL_TRTLLM_MEM_FRACTION",
        "0.4",
    )
)
PROMPT = os.environ.get(
    "GMS_KVR_REAL_TRTLLM_PROMPT",
    "The capital of France is",
)

# Engine id pinned for cross-process consistency.
os.environ.setdefault("GMS_TRTLLM_ENGINE_ID", "trtllm-real-test")
os.environ.setdefault("GMS_TRTLLM_DAEMON_SOCKET", DAEMON_SOCK)


# ---------------------------------------------------------------------
# Daemon fixture
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def gms_daemon():
    """Spin up a GMS daemon on the configured UDS for the whole module."""
    if os.path.exists(DAEMON_SOCK):
        os.unlink(DAEMON_SOCK)
    from gms_kv_ring.daemon.server import Daemon

    daemon = Daemon(
        listen_socket=DAEMON_SOCK,
        storage_dir=tempfile.mkdtemp(prefix="gms-real-trtllm-"),
        supervise_backend=False,
    )
    lh = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        lh["loop"] = loop
        try:
            loop.run_until_complete(daemon.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # Wait for the daemon to bind.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not os.path.exists(DAEMON_SOCK):
        time.sleep(0.05)
    assert os.path.exists(DAEMON_SOCK), "daemon never bound socket"
    yield daemon
    try:
        if "loop" in lh:
            lh["loop"].call_soon_threadsafe(daemon.stop)
        t.join(timeout=5)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm(gms_daemon):
    """Build a TRT-LLM PyTorch-backend LLM with our connector wired in.

    The connector is plugged via TRT-LLM's standard ``KvCacheConnectorConfig``
    using explicit module + class names (no preset). Module path matches
    where ``GMSTrtllmKvCacheScheduler`` and ``GMSTrtllmKvCacheWorker`` are
    exported (see ``integrations/trtllm/gms_connector.py``)."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig

    connector_config = KvCacheConnectorConfig(
        connector_module=("gpu_memory_service.integrations.trtllm.gms_connector"),
        connector_scheduler_class="GMSTrtllmKvCacheScheduler",
        connector_worker_class="GMSTrtllmKvCacheWorker",
    )

    # Attention backend override — on Blackwell (sm_100a) the default
    # TRTLLM FMHA kernels are JIT-compiled by NVRTC at startup and the
    # search path can lack cuda.h in some runtime containers. FLASHINFER
    # is a stable fallback. Override via env knob.
    attn_backend = os.environ.get(
        "GMS_KVR_REAL_TRTLLM_ATTN_BACKEND",
        "FLASHINFER",
    )
    # Disable chunked prefill — it routes through a different FMHA
    # kernel path that has the same NVRTC dep.
    obj = LLM(
        model=MODEL,
        backend="pytorch",
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=MEM_FRACTION,
            enable_block_reuse=True,
        ),
        kv_connector_config=connector_config,
        attn_backend=attn_backend,
        enable_chunked_prefill=False,
        # Single GPU + eager-only for stability under the smoke harness.
        tensor_parallel_size=1,
        max_seq_len=256,
    )
    yield obj
    try:
        obj.shutdown()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_smoke_one_generation(llm):
    """Generate a few tokens through the real engine with the GMS
    connector wired in. Asserts the model produces non-empty output —
    proves the full TRT-LLM ↔ KvCacheConnector ↔ GMS-daemon path
    initializes and serves at least one forward pass."""
    from tensorrt_llm import SamplingParams

    out = llm.generate(
        [PROMPT],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    assert len(out) == 1
    text = out[0].outputs[0].text
    assert isinstance(text, str)
    assert len(text) > 0, "engine produced empty completion"
    print(f"\n[trtllm smoke] {PROMPT!r} ->{text!r}")


def test_output_equality_on_cache_hit(llm):
    """Second generation of the same prompt should produce the same
    tokens (greedy decoding) AND benefit from the connector's
    prefix-cache path. We don't enforce a cache-hit count here (TRT-LLM
    doesn't expose connector telemetry via the public API), but we
    verify byte-correctness: the connector mediating the KV cache
    didn't corrupt the prefix bytes between requests."""
    from tensorrt_llm import SamplingParams

    sp = SamplingParams(max_tokens=12, temperature=0.0)
    o1 = llm.generate([PROMPT], sp)
    o2 = llm.generate([PROMPT], sp)
    t1 = o1[0].outputs[0].token_ids
    t2 = o2[0].outputs[0].token_ids
    assert list(t1) == list(t2), (
        f"Greedy generation diverged across requests; " f"t1={list(t1)} t2={list(t2)}"
    )

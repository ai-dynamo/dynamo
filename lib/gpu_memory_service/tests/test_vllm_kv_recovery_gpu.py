# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real model proof of KV remapping and native prefix-index reconstruction.

This deliberately simulates loss of host cache state inside one engine. Real
process/container replacement is qualified separately through Snapshot.
"""

import threading
from contextlib import ExitStack

import pytest
from _deps import HAS_CUDA, HAS_GMS

if not HAS_GMS or not HAS_CUDA:
    pytest.skip("requires GMS and CUDA", allow_module_level=True)

pytest.importorskip("vllm")

from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.v1.checkpoint import GMSCheckpointLifecycle
from gpu_memory_service.v1.device import get_device_uuid, get_socket_path
from gpu_memory_service.v1.integrations.vllm.recovery import register
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager
from vllm import LLM, SamplingParams

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.requested_vllm_kv_cache_bytes(268435456),
    pytest.mark.profiled_vram_gib(6.4),
]


@pytest.fixture
def gms(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("DYN_GMS_USE_V1", "true")
    monkeypatch.setenv("DYN_KV_RECOVERY", "true")
    monkeypatch.setenv("DYN_KV_RECOVERY_PATH", str(tmp_path / "metadata"))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    vmm = get_vmm()
    vmm.ensure_initialized()
    lifecycle = GMSCheckpointLifecycle()
    managers = {
        domain: GMSServerMemoryManager(
            get_device_uuid(0),
            vmm,
            0,
            checkpoint_lifecycle=lifecycle,
            allow_retention=domain == "kv_cache",
        )
        for domain in ("weights", "kv_cache")
    }
    lifecycle.bind_domains(managers)
    with ExitStack() as stack:
        for domain, manager in managers.items():
            server = stack.enter_context(
                GMSRPCServer(get_socket_path(0, domain), manager)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def stop(server=server, thread=thread, manager=manager):
                server.shutdown()
                thread.join(timeout=5)
                assert not thread.is_alive()
                manager._clear_allocations()

            stack.callback(stop)
        yield managers


@pytest.mark.timeout(240)
def test_qwen_prefix_is_reused_after_kv_remap_and_index_loss(gms):
    register()
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
        worker_cls="gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker",
        enable_sleep_mode=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=2,
        kv_cache_memory_bytes=268435456,
        gpu_memory_utilization=0.05,
    )
    core = llm.llm_engine.engine_core.engine_core
    try:
        core.sleep(1)
        adapter = core._gms_kv_recovery_adapter
        assert not gms["kv_cache"].allocation_snapshot()
        core.wake_up()
        prompt = "The quick brown fox jumps over the lazy dog. " * 20
        sampling = SamplingParams(temperature=0, max_tokens=8)
        warm = llm.generate(prompt, sampling, use_tqdm=False)[0]
        retained = gms["kv_cache"].allocation_snapshot()
        assert retained

        # Simulate host-state loss while preserving the recovery journal. The
        # normal semantic reset is tested separately and must clear that journal.
        core.pause_scheduler(mode="abort", clear_cache=False)
        adapter.session.close()
        adapter.session = None
        adapter.pool.cache_observer = None
        assert core.reset_prefix_cache()
        core.model_executor.sleep(1)
        core.model_executor.wake_up(None)
        adapter.pool.cache_observer = adapter
        core.resume_scheduler()
        assert len(adapter.session.blocks()) > 0
        assert gms["kv_cache"].allocation_snapshot() == retained
        recovered = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert recovered.num_cached_tokens > 0
        assert recovered.outputs[0].token_ids == warm.outputs[0].token_ids
        print(
            {
                "recovered_blocks": len(adapter.session.blocks()),
                "cached_tokens": recovered.num_cached_tokens,
                "output_token_ids": recovered.outputs[0].token_ids,
            }
        )
    finally:
        core.sleep(1)
        if adapter.session is not None:
            adapter.session.close()
        llm.llm_engine.engine_core.shutdown()

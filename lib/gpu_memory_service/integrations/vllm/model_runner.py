# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS model runner subclass for shadow mode.

Allows for kv cache to be skipped for a shadow engine init.
During failover scenarios, multiple engines will be running on the same device.
They should only allocate on their cache when they are the active/leader engine.
"""

from __future__ import annotations

import logging
import time

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = logging.getLogger(__name__)


class GMSShadowModelRunner(GPUModelRunner):
    """GPUModelRunner subclass for shadow mode overrides.

    Injected via __class__ swap in GMSWorker.init_device()
    """

    @property
    def in_shadow_init(self) -> bool:
        """True while shadow engine is in init phase (KV cache skipped)."""
        return getattr(self, "_shadow_init_phase", False)

    def enter_shadow_init(self) -> None:
        """Enter shadow init phase — KV cache allocation will be skipped."""
        self._shadow_init_phase = True
        logger.info("[Shadow] Entered shadow init phase")

    def exit_shadow_init(self) -> None:
        """Exit shadow init phase — KV cache allocation will proceed normally."""
        self._shadow_init_phase = False
        logger.info("[Shadow] Exited shadow init phase")

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """No-op during shadow init; store config for later allocation on wake."""
        if self.in_shadow_init:
            self._shadow_kv_cache_config = kv_cache_config
            self._shadow_kernel_block_sizes = kernel_block_sizes
            logger.info(
                "[Shadow] Init phase: stored config, skipping KV cache allocation"
            )
            return {}
        return super().initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)

    def _get_slot_mappings(self, *args, **kwargs):
        """Return (None, None) when KV caches are empty.

        _dummy_run() calls this unconditionally during warmup. Without KV
        tensors there is nothing to index into. This coerces a graceful no-op.
        """
        if not self.kv_caches:
            return None, None
        return super()._get_slot_mappings(*args, **kwargs)

    def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
        """Force PIECEWISE (or keep NONE for enforce_eager) and skip backend resolution.

        vLLM's default resolution may escalate to FULL_AND_PIECEWISE. We
        intercept to clamp back to a shadow-compatible mode.
        """
        from vllm.config import CUDAGraphMode

        mode = self.compilation_config.cudagraph_mode
        if mode == CUDAGraphMode.NONE:
            # enforce_eager — keep NONE, just init keys
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.NONE, self.uniform_decode_query_len
            )
        else:
            # Default shadow path — force PIECEWISE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.PIECEWISE, self.uniform_decode_query_len
            )

    def allocate_kv_cache_on_wake(self) -> dict:
        """Allocate KV cache on wake using config stored during shadow init.

        Called by GMSWorker.wake_up() after shadow init phase is exited.
        Waits up to 60s for GPU memory to be freed.
        """
        assert hasattr(
            self, "_shadow_kv_cache_config"
        ), "_shadow_kv_cache_config not set — was enter_shadow_init() called?"
        assert hasattr(
            self, "_shadow_kernel_block_sizes"
        ), "_shadow_kernel_block_sizes not set — was enter_shadow_init() called?"

        # OOM remediation during failover: wait for the dying engine to release memory.
        # TODO: This will be replaced with a barrier in GMS when we manage kv cache there instead
        config = self._shadow_kv_cache_config
        kv_cache_bytes = sum(t.size for t in config.kv_cache_tensors)

        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < kv_cache_bytes:
            logger.info(
                "[Shadow] Waiting for GPU memory (need %.2f GiB, free %.2f GiB)",
                kv_cache_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            deadline = time.monotonic() + 60.0
            last_log = time.monotonic()
            while free_bytes < kv_cache_bytes:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out waiting for GPU memory: "
                        f"need {kv_cache_bytes / (1 << 30):.2f} GiB, "
                        f"free {free_bytes / (1 << 30):.2f} GiB"
                    )
                now = time.monotonic()
                if now - last_log >= 5.0:
                    elapsed = now - (deadline - 60.0)
                    remaining = deadline - now
                    logger.info(
                        "[Shadow] Still waiting for GPU memory: "
                        "need %.2f GiB, free %.2f GiB "
                        "(%.0fs elapsed, %.0fs remaining)",
                        kv_cache_bytes / (1 << 30),
                        free_bytes / (1 << 30),
                        elapsed,
                        remaining,
                    )
                    last_log = now
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
            logger.info(
                "[Shadow] GPU memory available (free %.2f GiB), proceeding",
                free_bytes / (1 << 30),
            )

        logger.info("[Shadow] Allocating KV cache on wake")

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            kv_caches = self.initialize_kv_cache_tensors(
                config,
                self._shadow_kernel_block_sizes,
            )

        # Re-register with KV transfer group (skipped at init since kv_caches was {}).
        # Mirrors GPUModelRunner.initialize_kv_cache() — update if upstream changes.
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if has_kv_transfer_group() and kv_caches:
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.debug("[Shadow] Registered KV caches with transfer group")
        except ImportError:
            logger.debug("[Shadow] KV transfer group not available")

        total_bytes = sum(t.numel() * t.element_size() for t in kv_caches.values())
        logger.info(
            "[Shadow] Allocated KV cache on wake: %.2f GiB (%d tensors)",
            total_bytes / (1 << 30),
            len(kv_caches),
        )

        return kv_caches

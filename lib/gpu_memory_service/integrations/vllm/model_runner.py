# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS model runner subclass for shadow mode.

Allows for kv cache to be skipped for a shadow engine init.
During failover scenarios, multiple engines will be running on the same device.
They should only allocate on their cache when they are the active/leader engine.
"""

from __future__ import annotations

import logging

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
            print(
                "[Shadow] Init phase: stored config, skipping KV cache allocation",
                flush=True,
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
        GMS kv_cache RW lock acquisition serves as the memory barrier — the
        dying engine's abort() releases the lock and frees memory before we
        can connect.
        """
        assert hasattr(
            self, "_shadow_kv_cache_config"
        ), "_shadow_kv_cache_config not set — was enter_shadow_init() called?"
        assert hasattr(
            self, "_shadow_kernel_block_sizes"
        ), "_shadow_kernel_block_sizes not set — was enter_shadow_init() called?"

        config = self._shadow_kv_cache_config

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
        msg = "[Shadow] Allocated KV cache on wake: %.2f GiB (%d tensors)" % (
            total_bytes / (1 << 30),
            len(kv_caches),
        )
        logger.info(msg)
        print(msg, flush=True)

        return kv_caches

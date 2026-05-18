# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelExpress v2 refit receiver, as a vLLM v1 ``worker_extension_cls``.

Registered into the vLLM ``Worker`` class via ``parallel_config.worker_extension_cls``
so its methods become callable through ``AsyncLLM.collective_rpc``. Each method
runs **inside the worker process**, where ``self.model_runner.model`` and
``self.device`` are available.

Lifetime model (mirrors ``VllmInternalWorkerExtension``):

  * ``prepare_refit_info(state_dict_info)`` is called once per worker before the
    first refit. Stores per-tensor (shape, dtype) for asserts and FP8 paths.
  * ``update_weights_via_mx(version, mx_config)`` is called every refit cycle.
    Lazy-initializes an :class:`modelexpress.MxV2RefitReceiver` on first call,
    registers ``model.named_parameters()`` as NIXL receive buffers, then for
    every subsequent cycle: discover same-rank source → RDMA receive → call
    ``_load_weights`` → optionally republish as inference_replica for tree
    fan-out.

The Dynamo handler in ``components/src/dynamo/vllm/handlers.py`` invokes this
via ``await self.engine_client.collective_rpc("update_weights_via_mx", kwargs=...)``.
"""

from __future__ import annotations

import gc
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# MxConfig — wire-compatible with nemo_rl.distributed.mx_helpers.MxConfig
# =============================================================================


@dataclass
class MxConfig:
    """Subset of nemo-rl's MxConfig needed on the receiver side.

    The trainer sends this as a dict over the Dynamo Endpoint RPC; we parse it
    with :meth:`from_dict`. Field names and defaults must stay in sync with
    nemo_rl.distributed.mx_helpers.MxConfig.
    """

    enabled: bool = True
    mx_server_url: str = "modelexpress-server:8001"
    timeout_seconds: float = 300.0
    same_rank_only: bool = True
    tree_scale_out: bool = True
    moe_expert_filter: bool = True
    register_self_buffers: list[str] = field(default_factory=list)
    nic_pin: str = "auto"
    retain_latest_k: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MxConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", True)),
            mx_server_url=str(d.get("mx_server_url", "modelexpress-server:8001")),
            timeout_seconds=float(d.get("timeout_seconds", 300.0)),
            same_rank_only=bool(d.get("same_rank_only", True)),
            tree_scale_out=bool(d.get("tree_scale_out", True)),
            moe_expert_filter=bool(d.get("moe_expert_filter", True)),
            register_self_buffers=list(d.get("register_self_buffers", []) or []),
            nic_pin=str(d.get("nic_pin", "auto")),
            retain_latest_k=int(d.get("retain_latest_k", 1)),
        )


# =============================================================================
# NIC pinning (port of nemo_rl.distributed.mx_helpers.pin_local_nic)
# =============================================================================


def _pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    """Best-effort NUMA-local NIC pinning before NIXL initializes.

    On multi-NIC RDMA fabrics (e.g. GB200/GCP four-subnet RoCE) each rank's
    NIXL agent must bind to the NIC NUMA-closest to its GPU; cross-NIC writes
    are unrouted. Delegated to modelexpress's helper.
    """
    if mode == "off":
        return
    try:
        from modelexpress.ucx_utils import apply_nic_pin_for_device

        if mode == "auto":
            apply_nic_pin_for_device(device_id=device_id)
            logger.info("[mx] pinned NIC for device %d (auto)", device_id)
        else:
            os.environ["UCX_NET_DEVICES"] = mode
            os.environ["MX_RDMA_NIC_PIN"] = "off"
            logger.info("[mx] pinned NIC explicitly: %s", mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[mx] NIC pin failed (mode=%s): %s", mode, exc)


# =============================================================================
# Worker extension class — injected into vLLM Worker via worker_extension_cls
# =============================================================================


class MxRefitWorkerExtension:
    """Methods added to vLLM's ``Worker`` class via ``worker_extension_cls``.

    Has no ``__init__``: vLLM merges this class's methods into the existing
    ``Worker`` via ``__bases__``, and any state we need is stashed on ``self``
    lazily inside the methods themselves (``self._mx_receiver``,
    ``self._mx_recv_buffers``, ``self._mx_state_dict_info``). No conflict
    checks fire because all our attribute names use the ``_mx_`` prefix.
    """

    # ------------------------------------------------------------------ #
    # Refit-info preparation
    # ------------------------------------------------------------------ #
    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Record per-tensor (shape, dtype) info from the trainer.

        Called once by the trainer driver before the first refit cycle.
        Mirrors :py:meth:`VllmInternalWorkerExtension.prepare_refit_info` on
        the NeMo-RL side — assigns an instance attribute used by the FP8 path
        and for assertion checks. No-op for now if the worker class already
        had this attribute (the IPC-ZMQ path also stores it).
        """
        self._mx_state_dict_info = state_dict_info  # noqa: SLF001
        if not hasattr(self, "state_dict_info"):
            self.state_dict_info = state_dict_info

    # ------------------------------------------------------------------ #
    # Weight loading (minimal — adds GPT-OSS / FP8 / draft handling later)
    # ------------------------------------------------------------------ #
    def _mx_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Push refitted weights into the running vLLM model.

        Initial implementation covers the dense BF16/FP16 path only. GPT-OSS
        transpose, FP8 KV-cache post-processing, and Eagle3 draft-weight
        splitting are TODOs — see the NeMo-RL reference
        (``nemo_rl/models/generation/vllm/vllm_backend.py:_load_weights``)
        for those branches.
        """
        # The model parameters are the same buffers we registered with NIXL,
        # so by the time we get here the bytes are already in place. The call
        # below still runs vLLM's model-specific weight loader so any
        # per-tensor renaming / reshape / quant-state updates happen
        # consistently across paths.
        self.model_runner.model.load_weights(weights=weights)

    def _mx_maybe_process_fp8_kv_cache(self) -> None:
        """If the model uses FP8 KV cache, re-run vLLM's weight-loading hook.

        Static FP8 KV scales are computed in ``process_weights_after_loading``;
        they need to be recomputed after every refit so the scales match the
        new weights. Skipped silently for non-FP8 KV cache configurations.
        """
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        target_device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    # ------------------------------------------------------------------ #
    # The refit RPC entry point
    # ------------------------------------------------------------------ #
    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any = None,
    ) -> bool:
        """Receive weights via NIXL RDMA from the MX server (v2 path).

        Mirrors ``VllmInternalWorkerExtension.update_weights_via_mx`` in
        NeMo-RL. The lazy-init path runs on first call per worker; subsequent
        calls reuse the registered NIXL buffers.

        Args:
            version: monotonically-increasing training-step counter; the
                receiver picks sources whose ``training_step >= version``.
            mx_config: an ``MxConfig`` instance or a dict matching its fields.
                The trainer typically sends a dict over the Dynamo Endpoint
                RPC and we parse it here.

        Returns:
            True on successful refit, False on recoverable failures
            (no source found, source doesn't cover required experts).
            Unrecoverable errors are logged with a traceback and return False
            so the caller can decide whether to retry.
        """
        try:
            # Allow dict input — the handler may pass through unparsed JSON
            if not isinstance(mx_config, MxConfig):
                mx_config = MxConfig.from_dict(mx_config or {})

            # ---- Lazy-init receiver (no pre-registered buffers; scratch path) ----
            # We use ``MxRefitReceiver.receive_weights_scratch`` rather than the
            # pre-registered-buffer path because the trainer publishes HF
            # state_dict names (``q_proj``, ``k_proj``, ``v_proj``) but vLLM's
            # internal params are fused (``qkv_proj``). Registering vLLM's
            # ``named_parameters()`` as receive buffers gives a name mismatch
            # that breaks ``model.load_weights`` (it does the HF→fused merge
            # itself, and if you feed it ``qkv_proj`` it produces ``qkqkv_proj``
            # via its stacked_params_mapping). The scratch path allocates temp
            # CUDA buffers sized to the publisher's tensor list, RDMA-pulls
            # into them, and yields ``(hf_name, tensor)`` pairs that
            # ``load_weights`` consumes correctly. Extra GPU memory cost:
            # ~1× model size briefly per refit, freed at end.
            if not getattr(self, "_mx_receiver", None):
                # Import here so workers that never refit via MX don't pay
                # the modelexpress import cost.
                from modelexpress import MxV2RefitReceiver

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                _pin_local_nic(
                    device_id=self.device.index, mode=mx_config.nic_pin
                )
                self._mx_receiver = MxV2RefitReceiver(  # noqa: SLF001
                    agent_name=f"dynamo-vllm-r{rank}",
                    device_id=self.device.index,
                    mx_server_url=mx_config.mx_server_url,
                    worker_rank=rank,
                )
                self._mx_receiver.initialize(model_tensors=None)
                logger.info(
                    "[mx] receiver initialized (scratch path): rank=%d device=%d",
                    rank,
                    self.device.index,
                )

            # ---- Discover, pick source, RDMA pull ----
            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=model_name,
                min_version=int(version),
                same_rank_only=mx_config.same_rank_only,
                include_replicas=mx_config.tree_scale_out,
            )
            if not candidates:
                logger.warning(
                    "[mx] no v2 source available for version>=%d on rank %d",
                    version,
                    self._mx_receiver.worker_rank,
                )
                return False

            chosen = self._mx_receiver.pick_best_source(candidates)
            if chosen is None:
                logger.warning(
                    "[mx] no candidate covers required experts on rank %d",
                    self._mx_receiver.worker_rank,
                )
                return False
            logger.info(
                "[mx] rank=%d chosen role=%s src_rank=%d version=%s",
                self._mx_receiver.worker_rank,
                chosen.role,
                chosen.worker_rank,
                chosen.ref.training_step,
            )

            # Scratch path: allocate temp buffers, RDMA-pull, collect HF-named
            # tensors. ``receive_weights_scratch`` lives on the inner
            # ``MxRefitReceiver``; ``MxV2RefitReceiver.receive_from`` wraps the
            # non-scratch ``receive_weights`` which assumes name parity.
            #
            # Build a tensor_shapes dict from the v2 candidate's registry so
            # the yielded tensors come back with their original shape (not
            # flat 1D). vLLM's ``load_weights`` calls ``.copy_(t)`` into the
            # model param, so shape must match (or .view() must succeed).
            tensor_shapes: dict[str, tuple[int, ...]] = {}
            registry = getattr(chosen, "registry", None)
            if registry:
                for td in registry.get("tensors", []):
                    tensor_shapes[td.name] = tuple(int(s) for s in td.global_shape)
            weights: list[tuple[str, torch.Tensor]] = list(
                self._mx_receiver._receiver.receive_weights_scratch(
                    chosen.ref,
                    timeout_seconds=mx_config.timeout_seconds,
                    tensor_shapes=tensor_shapes or None,
                )
            )

            # ---- vLLM's load_weights handles HF→fused merge ----
            self._mx_load_weights(weights)
            torch.cuda.current_stream().synchronize()
            self._mx_maybe_process_fp8_kv_cache()

            # ---- Tree fan-out: republish self as inference_replica ----
            if mx_config.tree_scale_out:
                try:
                    self._mx_receiver.publish_self_as_source(
                        version=int(version),
                        model_name=model_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Non-fatal: refit succeeded, we just can't serve as a
                    # source for downstream receivers this cycle.
                    logger.warning(
                        "[mx] tree-scale-out republish failed: %s", exc
                    )

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[mx] update_weights_via_mx failed on rank=%d: %s\n%s",
                getattr(
                    getattr(self, "_mx_receiver", None), "worker_rank", -1
                ),
                exc,
                traceback.format_exc(),
            )
            return False

    # ------------------------------------------------------------------ #
    # Receiver-side polling — the trainer publishes to MX without sending
    # any trigger RPC, so the worker watches the MX server itself and
    # refits whenever a newer version appears for its model_name.
    # ------------------------------------------------------------------ #
    def start_mx_refit_poller(
        self,
        *,
        mx_config: Any = None,
        poll_interval_s: float = 5.0,
    ) -> bool:
        """Spawn a background thread that watches MX for new versions.

        Called once per worker at startup (or on first publish-detection),
        from the dynamo handler. The thread:
          1. Builds an MxConfig (defaults are fine for the smoke).
          2. Loops on ``discover_v2_sources(min_version=last_seen+1)``.
          3. When a new version appears, calls ``update_weights_via_mx``
             with the new version. That method's lazy-init flow registers
             NIXL buffers on first call.
          4. Sleeps ``poll_interval_s`` between polls.

        Returns True if the thread was started (or was already running).
        Idempotent — repeated calls are no-ops.
        """
        if getattr(self, "_mx_poller_thread", None) is not None:
            return True

        import threading

        cfg = (
            mx_config
            if isinstance(mx_config, MxConfig)
            else MxConfig.from_dict(mx_config or {})
        )
        self._mx_poller_stop = threading.Event()
        self._mx_poller_last_version: int = 0
        self._mx_poller_cfg = cfg
        self._mx_poller_interval = float(poll_interval_s)

        def _poll_loop() -> None:
            from modelexpress import MxV2RefitReceiver

            model_name = getattr(
                self.model_runner.vllm_config.model_config, "model", "unknown"
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                "[mx-poller] started: rank=%d model=%s interval=%.1fs",
                rank,
                model_name,
                self._mx_poller_interval,
            )
            # Lazy receiver just for discovery — the refit path lazy-inits
            # its own receiver against the same MX server. We call
            # ``initialize(model_tensors=None)`` to wire the NIXL agent +
            # gRPC client without registering receive buffers (we don't
            # use this receiver to pull; only to poll for new versions).
            discover_only = MxV2RefitReceiver(
                agent_name=f"dynamo-vllm-poller-r{rank}",
                device_id=self.device.index,
                mx_server_url=cfg.mx_server_url,
                worker_rank=rank,
            )
            try:
                discover_only.initialize(model_tensors=None)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[mx-poller] discover-receiver initialize() failed: %s; "
                    "polling thread will not start", exc,
                )
                return
            while not self._mx_poller_stop.is_set():
                try:
                    candidates = discover_only.discover_v2_sources(
                        model_name=model_name,
                        min_version=int(self._mx_poller_last_version) + 1,
                        same_rank_only=cfg.same_rank_only,
                        include_replicas=cfg.tree_scale_out,
                    )
                    if candidates:
                        latest = max(
                            int(c.ref.training_step) for c in candidates
                        )
                        logger.info(
                            "[mx-poller] new version detected: %d (last=%d)",
                            latest,
                            self._mx_poller_last_version,
                        )
                        ok = self.update_weights_via_mx(
                            version=latest, mx_config=cfg
                        )
                        if ok:
                            self._mx_poller_last_version = latest
                            logger.info(
                                "[mx-poller] refit OK to version %d", latest
                            )
                        else:
                            logger.warning(
                                "[mx-poller] refit failed for version %d; will retry",
                                latest,
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[mx-poller] poll error: %s", exc)
                self._mx_poller_stop.wait(self._mx_poller_interval)
            logger.info("[mx-poller] stopped on rank=%d", rank)

        self._mx_poller_thread = threading.Thread(
            target=_poll_loop, name="mx-refit-poller", daemon=True
        )
        self._mx_poller_thread.start()
        return True

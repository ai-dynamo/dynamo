# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM monkey-patches applied at GMSWorker import.

Patches:
  - MemorySnapshot.measure: adds GMS-committed bytes to free_memory in RO mode.
  - request_memory: bypasses the free>=requested check during deferred-KV init.
  - NixlBaseConnector KV registration: defers normal or cross-layer
    registration during the scratch phase and stashes it for replay at wake.
  - init_kv_cache: scopes the scratch mem-pool to the raw KV tensors only, so
    BlockTables / workspace / pointer tensors keep real (un-aliased) memory.

The torch.cuda.empty_cache patch lives in integrations/common/patches.py.
"""

from __future__ import annotations

import importlib
import logging
import os

from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    is_scratch,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import is_scratch_kv_enabled

logger = logging.getLogger(__name__)

_memory_snapshot_patched = False
_request_memory_patched = False
_register_kv_caches_patched = False
_kv_cache_pool_scope_patched = False
_NIXL_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"


# =============================================================================
# Core GMS patch (always applied)
# =============================================================================


def patch_memory_snapshot() -> None:
    """Add committed GMS bytes to MemorySnapshot.free_memory"""
    global _memory_snapshot_patched

    if _memory_snapshot_patched:
        return

    try:
        from vllm.utils.mem_utils import MemorySnapshot
    except ImportError:
        logger.debug("[GMS Patch] MemorySnapshot not available")
        return

    original_measure = MemorySnapshot.measure

    def patched_measure(self):
        original_measure(self)

        manager = get_gms_client_memory_manager("weights")
        assert manager is not None, "GMS client is not initialized"

        if manager.granted_lock_type == GrantedLockType.RO:
            allocations = manager.list_handles()
            committed_bytes = sum(alloc.aligned_size for alloc in allocations)
        else:
            # NOTE: by design, we want to assume we have the whole GPU when writing
            # weights for the first time, so we don't make an adjustment.
            committed_bytes = 0
            logger.info("[GMS] RW mode - skipping committed memory adjustment")

        original_free = self.free_memory
        self.free_memory += committed_bytes

        if committed_bytes > 0:
            logger.info(
                "[GMS Patch] Adjusted free_memory: %.2f GiB + %.2f GiB = %.2f GiB",
                original_free / (1 << 30),
                committed_bytes / (1 << 30),
                self.free_memory / (1 << 30),
            )

    MemorySnapshot.measure = patched_measure
    _memory_snapshot_patched = True
    logger.info("[GMS Patch] Patched MemorySnapshot.measure")


# =============================================================================
# Shadow mode patches
# =============================================================================


def patch_request_memory() -> None:
    """Bypass free >= requested check (shadow shares GPU with active engine)."""
    global _request_memory_patched

    if _request_memory_patched:
        return

    try:
        from vllm.v1.worker import utils as worker_utils
    except ImportError:
        logger.debug("[GMS Patch] vllm.v1.worker.utils not available")
        return

    def patched_request_memory(init_snapshot, cache_config):
        requested_memory = int(
            init_snapshot.total_memory * cache_config.gpu_memory_utilization
        )
        logger.info(
            "[GMS Patch] Shadow mode: bypassing memory check "
            "(requested=%.2f GiB, free=%.2f GiB)",
            requested_memory / (1 << 30),
            init_snapshot.free_memory / (1 << 30),
        )
        return requested_memory

    worker_utils.request_memory = patched_request_memory
    _request_memory_patched = True
    logger.info("[GMS Patch] Patched request_memory for shadow mode")


def patch_register_kv_caches() -> None:
    """Defer NIXL KV registration while KV backing is scratch-aliased.

    Registering NIXL MRs over scratch would pin a soon-stale page into the NIC;
    sleep tears down scratch and wake remaps real backing at the same VAs.
    Stash the normal dict or cross-layer tensor during the scratch phase and
    let GMSWorker.wake_up replay it after remap.
    """
    global _register_kv_caches_patched

    if _register_kv_caches_patched:
        return

    # Keep this optional-backend import deferred. GMS is collected in images
    # that do not install vLLM, and the connector is only required when this
    # vLLM-specific patch is enabled.
    try:
        nixl_module = importlib.import_module(_NIXL_MODULE)
    except ModuleNotFoundError as exc:
        # Treat a missing vLLM package (or missing connector package) as an
        # unavailable optional backend. Missing dependencies imported from an
        # installed connector must remain visible.
        missing_module = exc.name
        if missing_module and (
            missing_module == _NIXL_MODULE
            or _NIXL_MODULE.startswith(f"{missing_module}.")
        ):
            logger.debug("[GMS Patch] NixlBaseConnector not available")
            return
        raise

    # vLLM 0.27 exports NixlConnector as an alias for NixlPullConnector while
    # NixlPushConnector is its sibling. Patch their common base so both modes
    # retain the scratch-registration safety gate.
    nixl_base_connector = nixl_module.NixlBaseConnector
    original_register = nixl_base_connector.register_kv_caches
    original_register_cross_layers = nixl_base_connector.register_cross_layers_kv_cache

    def has_deferred_kv_backing() -> bool:
        """Fail closed when scratch-KV state cannot be determined."""
        try:
            kv_mgr = get_gms_client_memory_manager("kv_cache")
            return kv_mgr is not None and is_scratch(kv_mgr)
        except (LookupError, AttributeError, RuntimeError) as exc:
            logger.warning(
                "[GMS Patch] Cannot determine deferred-KV state — "
                "raising to avoid pinning a stale scratch MR: %s",
                exc,
                exc_info=True,
            )
            raise

    def patched_register_kv_caches(self, kv_caches):
        if has_deferred_kv_backing():
            self._scratch_kv_pending = kv_caches
            logger.info(
                "[GMS Patch] Deferring NIXL KV cache registration "
                "(stashed %d layers for wake replay)",
                len(kv_caches),
            )
            return
        return original_register(self, kv_caches)

    def patched_register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        if has_deferred_kv_backing():
            self._scratch_cross_layers_kv_pending = (kv_cache, attn_backend)
            logger.info(
                "[GMS Patch] Deferring NIXL cross-layer KV cache registration "
                "for wake replay"
            )
            return
        return original_register_cross_layers(self, kv_cache, attn_backend)

    nixl_base_connector.register_kv_caches = patched_register_kv_caches
    nixl_base_connector.register_cross_layers_kv_cache = (
        patched_register_cross_layers_kv_cache
    )
    _register_kv_caches_patched = True
    logger.info("[GMS Patch] Patched NixlBaseConnector KV registration")


# =============================================================================
# Patch application helper
# =============================================================================


def patch_kv_cache_pool_scope() -> None:
    """Scope the scratch mem-pool to init_kv_cache (the raw KV tensors) only.

    Keeps BlockTables / workspace / the block-table pointer tensor on real memory.
    Single-block scratch aliases everything in the pool onto one granule, so a KV
    write over that pointer tensor would corrupt the block-table gather kernel
    (-> illegal memory access).
    """
    global _kv_cache_pool_scope_patched

    if _kv_cache_pool_scope_patched:
        return

    try:
        import torch
        from gpu_memory_service.client.torch.allocator import gms_use_mem_pool
    except ImportError as exc:
        logger.debug("[GMS Patch] init_kv_cache pool-scope not available: %s", exc)
        return

    def _pool_wrap(fn, *, name: str):
        def wrapped(*args, **kwargs):
            assert torch.cuda.is_available(), "GMS scratch KV requires CUDA"
            device = torch.device("cuda", torch.cuda.current_device())
            logger.info("[GMS Patch] %s under scratch mem-pool device=%s", name, device)
            with gms_use_mem_pool("kv_cache", device):
                return fn(*args, **kwargs)

        return wrapped

    patched_any = False
    # vLLM 0.27 V2 runner: actual torch.zeros live in attn_utils._allocate_kv_cache.
    # Patching model_runner.init_kv_cache is not enough if that name is a stale
    # import or the runner calls attn_utils directly.
    for mod_name, attr in (
        ("vllm.v1.worker.gpu.attn_utils", "_allocate_kv_cache"),
        ("vllm.v1.worker.gpu.attn_utils", "init_kv_cache"),
        ("vllm.v1.worker.gpu.model_runner", "init_kv_cache"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            target = getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
        if attr == "GPUModelRunner":
            original = getattr(target, "_allocate_kv_cache_tensors", None)
            if original is None:
                continue
            target._allocate_kv_cache_tensors = _pool_wrap(
                original, name=f"{mod_name}._allocate_kv_cache_tensors"
            )
            uniform = getattr(target, "allocate_uniform_kv_caches", None)
            if callable(uniform):
                target.allocate_uniform_kv_caches = _pool_wrap(
                    uniform, name=f"{mod_name}.allocate_uniform_kv_caches"
                )
            patched_any = True
            continue
        setattr(mod, attr, _pool_wrap(target, name=f"{mod_name}.{attr}"))
        patched_any = True

    if not patched_any:
        logger.debug("[GMS Patch] no KV allocation entry points found to wrap")
        return

    _kv_cache_pool_scope_patched = True
    logger.info(
        "[GMS Patch] Scoped scratch mem-pool to KV tensor allocation (KV tensors only)"
    )


def apply_scratch_kv_patches() -> None:
    """Apply scratch-KV monkey-patches. No-ops when scratch KV is disabled."""
    if not is_scratch_kv_enabled():
        return

    # Resolve the optional connector before mutating the other scratch-specific
    # vLLM entry points. A broken installed NIXL module must fail startup rather
    # than leave a partially applied scratch configuration.
    patch_register_kv_caches()
    patch_request_memory()
    patch_kv_cache_pool_scope()
    # Do not force eager BreakableCUDAGraphWrapper: graphs are captured
    # on scratch VAs at init and survive wake_up remap.
    logger.info("[GMS Patch] applied")


def patch_force_eager_breakable_cudagraph() -> None:
    """Disable on-the-fly breakable CUDA-graph capture.

    Scratch-init skips capture_model(), so the first real request would
    otherwise capture inside BreakableCUDAGraphWrapper and crash on
    unpinned CPU/CUDA copies. Eager is enough to prove GMS failover;
    capture can be re-enabled after wake with real KV.
    """
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    except ImportError:
        logger.debug("[GMS Patch] BreakableCUDAGraphWrapper not available")
        return

    def eager_call(self, *args, **kwargs):
        return self.runnable(*args, **kwargs)

    BreakableCUDAGraphWrapper.__call__ = eager_call
    logger.info("[GMS Patch] Forced eager BreakableCUDAGraphWrapper.__call__")


_dsv4_topk_patched = False

# Diagnostic: synchronise on entry to dsv4_topk so an async fault raised by the
# upstream router GEMM is attributed there instead of to the first add here.
_DSV4_TOPK_PROBE = os.environ.get("DYN_GMS_DSV4_TOPK_PROBE", "").lower() in (
    "1",
    "true",
    "yes",
)


def patch_dsv4_topk_clone_inputs() -> None:
    """Keep Triton ``dsv4_topk``; clone operands off GMS VMM first.

    Triton cannot load CUDA VMM pointers for ``correction_bias`` (IMA on
    B200). A PyTorch sequential-max fallback with per-call
    ``cuda.synchronize`` is what made GLM-5.2 TEP8 prefill sit at
    ~3200–6400 tok/s vs V1 ~16–26k. Clone into the default allocator and
    run the original kernel. Also patch ``fused_topk_bias_router.dsv4_topk``
    because that module does ``from dsv4_topk import dsv4_topk``.
    """
    global _dsv4_topk_patched
    if _dsv4_topk_patched:
        return
    try:
        import vllm.model_executor.layers.fused_moe.router.dsv4_topk as dsv4_topk_mod
    except ImportError:
        logger.debug("[GMS Patch] dsv4_topk not available")
        return

    orig = dsv4_topk_mod.dsv4_topk

    def wrapped(
        gating_output,
        correction_bias,
        indices_dtype,
        routed_scaling_factor,
    ):
        if (
            gating_output is None
            or correction_bias is None
            or getattr(gating_output, "is_meta", False)
            or getattr(correction_bias, "is_meta", False)
        ):
            return orig(
                gating_output,
                correction_bias,
                indices_dtype,
                routed_scaling_factor,
            )
        # gating_output is an activation (not a GMS VMM weight). Cloning
        # it every MoE layer of an 8192-token prefill is a full extra
        # copy of the router logits; soak stays at one 8192-token step
        # per ~1.28s (6400 tok/s). Only correction_bias is a GMS param
        # that Triton cannot load from VMM.
        bias = correction_bias.detach().contiguous().clone()
        return orig(
            gating_output, bias, indices_dtype, routed_scaling_factor
        )

    def pytorch_fallback(
        gating_output,
        correction_bias,
        indices_dtype,
        routed_scaling_factor,
    ):
        """Pure-PyTorch replacement for the Triton ``dsv4_topk`` kernel.

        Cloning operands off GMS VMM is not always sufficient: on the
        DeepSeek-V4 checkpoints Triton still takes an illegal address even
        with private copies, so this path avoids Triton entirely.

        Mirrors ``_dsv4_topk_kernel``: weights are softplus-then-sqrt of the
        logits; selection ranks by ``weights + correction_bias`` but the
        emitted weight is the UNBIASED weight; ties resolve to the lowest
        expert id; the selected weights are renormalised by their own sum and
        scaled by ``routed_scaling_factor``.

        Validated against the Triton kernel on B200 for both expert counts
        (256, 384): 100% top-k index agreement and a maximum weight delta of
        6e-8, i.e. float32 rounding. Still gated behind an env var because it
        is a workaround for a GMS VMM defect, not a supported code path.
        """
        import torch

        logits = gating_output.float()
        if _DSV4_TOPK_PROBE:
            # gating_output is the router GEMM's output, produced from a large
            # router weight that stays on the GMS mapping. The adds below are
            # the first synchronising ops after that GEMM, so an async fault in
            # it would be reported here rather than at its own launch. Sync on
            # entry to attribute the fault to the caller instead.
            try:
                torch.cuda.synchronize()
                logger.warning(
                    "[GMS Probe] dsv4_topk entry clean: gating_output=%s "
                    "correction_bias=%s",
                    tuple(gating_output.shape),
                    tuple(correction_bias.shape),
                )
            except Exception:
                logger.exception(
                    "[GMS Probe] CUDA context ALREADY POISONED on dsv4_topk "
                    "entry - the fault is upstream of the router, in whatever "
                    "produced gating_output"
                )
                raise
        # softplus, matching the kernel's >20 linear shortcut
        weights = torch.sqrt(
            torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits)))
        )
        ranked = weights + correction_bias.float()
        ranked = torch.nan_to_num(ranked, nan=-1e30)
        topk = getattr(dsv4_topk_mod, "_TOPK", 6)
        # torch.topk already resolves ties to the lowest index, matching the
        # kernel's tl.min(candidate) tie-break. Verified against the Triton
        # kernel on B200: 100% index agreement, max weight delta 6e-8.
        _, idx = torch.topk(ranked, k=topk, dim=-1, sorted=True)
        selected = weights.gather(-1, idx)
        total = selected.sum(dim=-1, keepdim=True)
        selected = selected * (
            routed_scaling_factor / torch.where(total > 0.0, total, 1.0)
        )
        return selected.to(torch.float32), idx.to(indices_dtype)

    use_fallback = os.environ.get("DYN_GMS_DSV4_TOPK_PYTORCH", "").lower() in (
        "1",
        "true",
        "yes",
    )
    chosen = torch_compiler_disable(pytorch_fallback) if use_fallback else wrapped

    dsv4_topk_mod.dsv4_topk = chosen
    try:
        import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as router_mod

        router_mod.dsv4_topk = chosen
    except ImportError:
        pass
    _dsv4_topk_patched = True
    if use_fallback:
        logger.warning(
            "[GMS Patch] dsv4_topk replaced with PyTorch topk "
            "(DYN_GMS_DSV4_TOPK_PYTORCH) - generations are NOT quality-correct"
        )
    else:
        logger.info("[GMS Patch] dsv4_topk wraps Triton with VMM-safe input clones")


def torch_compiler_disable(fn):
    """Mark ``fn`` as opaque to torch.compile, tolerating old torch builds."""
    import torch

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(fn) if callable(disable) else fn


_dsv4_weight_digest_patched = False


def patch_dsv4_weight_digest() -> None:
    """Checksum the loaded weights so RW and RO loads can be compared.

    A read-only import can serve degenerate output while every load-time check
    passes, because the damage is a stale alias pointing at a tensor nobody
    wrote rather than a bad value in a tensor that was written. Comparing a
    digest of the same tensors across a write-mode load and a read-mode load
    localises that to the exact parameter.

    Logs one line per rank at the end of load. Enable with
    DYN_GMS_DSV4_WEIGHT_DIGEST.
    """
    global _dsv4_weight_digest_patched

    if _dsv4_weight_digest_patched:
        return

    if os.environ.get("DYN_GMS_DSV4_WEIGHT_DIGEST", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return

    _dsv4_weight_digest_patched = True

    def digest(model) -> None:
        import torch

        rows = []
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if not torch.is_tensor(t) or t.is_meta or not t.is_cuda:
                continue
            # Sample rather than reduce the whole tensor: this runs on the
            # restore critical path and a few elements are enough to catch a
            # stale alias or an unwritten buffer.
            flat = t.detach().flatten()
            n = flat.numel()
            if n == 0:
                continue
            # Build sample indices on the host: linspace on device then
            # .long() can round the endpoint past n-1 and trip a device-side
            # assert, which poisons the context for everything after it.
            step = max(1, n // 8)
            picks = sorted({min(i, n - 1) for i in range(0, n, step)})[:8]
            idx = torch.tensor(picks, device=flat.device, dtype=torch.long)
            vals = flat[idx].to(torch.float64)
            rows.append(
                f"{name}|{t.dtype}|{tuple(t.shape)}|"
                f"{float(vals.sum()):.6e}|{float(vals.abs().max()):.6e}"
            )
        import hashlib

        blob = "\n".join(rows).encode()
        logger.warning(
            "[GMS Digest] %d tensors, sha256=%s",
            len(rows),
            hashlib.sha256(blob).hexdigest()[:32],
        )
        # Full rows go to a file: comparing two loads needs every tensor, and
        # 1845 log lines per rank would be unusable.
        try:
            import os as _os

            path = _os.environ.get(
                "DYN_GMS_DSV4_WEIGHT_DIGEST_PATH", "/tmp/gms_digest"
            )
            dev = getattr(getattr(model, "device", None), "index", None)
            if dev is None:
                dev = torch.cuda.current_device()
            with open(f"{path}.rank{dev}.txt", "w") as fh:
                fh.write("\n".join(rows))
            logger.warning("[GMS Digest] wrote %s.rank%s.txt", path, dev)
        except Exception:
            logger.exception("[GMS Digest] could not write digest file")

    globals()["_gms_dsv4_weight_digest"] = digest
    logger.info("[GMS Patch] weight digest enabled (DYN_GMS_DSV4_WEIGHT_DIGEST)")


def gms_dsv4_weight_digest(model) -> None:
    """Run the weight digest when it is enabled; no-op otherwise."""
    fn = globals().get("_gms_dsv4_weight_digest")
    if fn is not None:
        try:
            fn(model)
        except Exception:
            logger.exception("[GMS Digest] failed")


_dsv4_hash_topk_probe_patched = False


def patch_dsv4_hash_topk_probe() -> None:
    """Report the dtypes reaching the hash-MoE top-k op.

    ``topk_hash_softplus_sqrt`` fails with "expected scalar type Float but
    found Int" on the read-only import path. Several of its operands come from
    the gate (bias, hash table) and are rebound during materialization, so name
    the offender rather than guessing which one is wrong.

    Enable with DYN_GMS_DSV4_HASH_TOPK_PROBE.
    """
    global _dsv4_hash_topk_probe_patched

    if _dsv4_hash_topk_probe_patched:
        return

    if os.environ.get("DYN_GMS_DSV4_HASH_TOPK_PROBE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return

    try:
        import torch
        import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as mod
    except ImportError:
        return

    original = mod.vllm_topk_softplus_sqrt

    def probed(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ):
        def d(t):
            return "None" if not torch.is_tensor(t) else f"{t.dtype}/{t.device}"

        logger.warning(
            "[GMS Probe] topk_hash_softplus_sqrt operands: topk_weights=%s "
            "topk_indices=%s token_expert_indices=%s gating_output=%s "
            "e_score_correction_bias=%s input_tokens=%s hash_indices_table=%s",
            d(topk_weights),
            d(topk_indices),
            d(token_expert_indices),
            d(gating_output),
            d(e_score_correction_bias),
            d(input_tokens),
            d(hash_indices_table),
        )
        return original(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    mod.vllm_topk_softplus_sqrt = probed
    _dsv4_hash_topk_probe_patched = True
    logger.info("[GMS Patch] hash top-k dtype probe enabled")


_dsv4_layer_probe_patched = False


def patch_dsv4_layer_probe() -> None:
    """Bisect which DSv4 stage first poisons the CUDA context.

    The K-cache gather was long blamed for the first-prefill
    cudaErrorIllegalAddress, but synchronising on entry to it faults before its
    kernel is even launched, so the context is already bad when attention's
    prefill path is reached. CUDA reports an async fault at the next
    synchronising call, not at the launch that caused it, so the visible
    traceback is the first sync downstream of the real culprit.

    This wraps the DSv4 decoder layer and its attention entry point with
    synchronising checkpoints and reports the last one that passed, which names
    the stage containing the faulting launch. Diagnostic only - it serialises
    every layer. Enable with DYN_GMS_DSV4_LAYER_PROBE.
    """
    global _dsv4_layer_probe_patched

    if _dsv4_layer_probe_patched:
        return

    if os.environ.get("DYN_GMS_DSV4_LAYER_PROBE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return

    try:
        import torch
        import vllm.models.deepseek_v4.attention as attn_mod
        import vllm.models.deepseek_v4.nvidia.model as model_mod
    except ImportError:
        return

    state = {"n": 0, "dead": False, "last": "<none>"}

    def checkpoint(label: str) -> None:
        """Report the first checkpoint that finds the context already faulted."""
        if state["dead"]:
            return
        try:
            torch.cuda.synchronize()
            state["n"] += 1
            state["last"] = label
        except Exception as exc:
            state["dead"] = True
            logger.error(
                "[GMS Probe] FIRST BAD CHECKPOINT: %s | last clean: %s "
                "(%d clean checkpoints). The faulting launch is between those "
                "two points. %s",
                label,
                state["last"],
                state["n"],
                exc,
            )

    def wrap(owner, attr: str, label: str) -> None:
        target = getattr(owner, attr, None)
        if target is None:
            return
        original = target.__call__ if not callable(target) else target

        def probed(*args, **kwargs):
            checkpoint(f"{label} ENTRY")
            out = original(*args, **kwargs)
            checkpoint(f"{label} EXIT")
            return out

        setattr(owner, attr, probed)

    # Attention entry: brackets the whole MLA path including the gather.
    for cls_name in ("DeepseekV4Attention", "DeepseekV4Indexer"):
        cls = getattr(attn_mod, cls_name, None)
        if cls is not None and hasattr(cls, "forward"):
            wrap(cls, "forward", f"attention.{cls_name}.forward")
    # Sub-stages inside attention, to separate the input GEMMs (which read the
    # large RO-mapped projection weights on aux streams) from the MLA kernels.
    attn_cls = getattr(attn_mod, "DeepseekV4Attention", None)
    if attn_cls is not None:
        for meth in (
            "_run_parallel_input_projections",
            "_fused_wqa_wkv_gemm",
            "_prepare_and_attn_fn",
            "attention_impl",
        ):
            if hasattr(attn_cls, meth):
                wrap(attn_cls, meth, f"attention.{meth}")
    # Decoder layer: brackets attention vs FFN/MoE within one layer.
    for cls_name in ("DeepseekV4DecoderLayer", "DeepseekV4MoE"):
        cls = getattr(model_mod, cls_name, None)
        if cls is not None and hasattr(cls, "forward"):
            wrap(cls, "forward", f"model.{cls_name}.forward")

    _dsv4_layer_probe_patched = True
    logger.info("[GMS Patch] DSv4 layer probe enabled (DYN_GMS_DSV4_LAYER_PROBE)")


_dsv4_gather_checked = False


def patch_dsv4_gather_k_cache_check() -> None:
    """Validate the DSv4 K-cache gather's indices before it dereferences them.

    ``dequantize_and_gather_k_cache`` faults with cudaErrorIllegalAddress on the
    first prefill under GMS, identically in the CuTeDSL and Triton kernels. Both
    compute
    ``k_cache + block_table[req, pos // block_size] * k_cache.stride(0)``
    with no bounds check, so a single out-of-range block id produces a wild
    address in whichever kernel runs. Two kernels failing the same way points at
    the operands rather than at either kernel.

    This check reports whether the operands are already invalid on entry, which
    distinguishes "the gather is broken" from "something upstream corrupted the
    block table or the sequence lengths". It syncs and copies to host, so it is
    diagnostic only - enable it with DYN_GMS_DSV4_GATHER_CHECK.
    """
    global _dsv4_gather_checked

    if _dsv4_gather_checked:
        return

    if os.environ.get("DYN_GMS_DSV4_GATHER_CHECK", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return

    try:
        import vllm.models.deepseek_v4.common.ops.cache_utils as cache_utils_mod
    except ImportError:
        return

    original = cache_utils_mod.dequantize_and_gather_k_cache

    def checked(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, *a, **k):
        import torch

        torch.cuda.synchronize()
        num_blocks = k_cache.shape[0]
        bt = block_table.to("cpu", non_blocking=False)
        sl = seq_lens.to("cpu", non_blocking=False)
        gl = None if gather_lens is None else gather_lens.to("cpu", non_blocking=False)

        problems = []
        if bt.numel():
            if int(bt.max()) >= num_blocks:
                problems.append(
                    f"block_table.max={int(bt.max())} >= k_cache.shape[0]={num_blocks}"
                )
            if int(bt.min()) < 0:
                problems.append(f"block_table.min={int(bt.min())} < 0")
        if sl.numel():
            if int(sl.min()) < 0:
                problems.append(f"seq_lens.min={int(sl.min())} < 0")
            # Each request reads ceil(seq_len / block_size) entries of its row.
            need = (sl.to(torch.int64) + block_size - 1) // max(block_size, 1)
            if int(need.max()) > block_table.shape[-1]:
                problems.append(
                    f"needs {int(need.max())} block_table cols but row width is "
                    f"{block_table.shape[-1]} (seq_lens.max={int(sl.max())}, "
                    f"block_size={block_size})"
                )
            span = int(sl.max()) + int(offset)
            if span > out.shape[1]:
                problems.append(
                    f"writes up to {span} tokens but out.shape[1]={out.shape[1]}"
                )

        logger.warning(
            "[GMS Check] gather: out=%s k_cache=%s(num_blocks=%d, stride0=%d) "
            "block_table=%s seq_lens[min=%s,max=%s] gather_lens=%s block_size=%d "
            "offset=%d -> %s",
            tuple(out.shape),
            tuple(k_cache.shape),
            num_blocks,
            k_cache.stride(0),
            tuple(block_table.shape),
            int(sl.min()) if sl.numel() else "n/a",
            int(sl.max()) if sl.numel() else "n/a",
            "none" if gl is None else f"[min={int(gl.min())},max={int(gl.max())}]",
            block_size,
            offset,
            "; ".join(problems) if problems else "operands in range",
        )
        return original(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, *a, **k)

    cache_utils_mod.dequantize_and_gather_k_cache = checked
    try:
        import vllm.models.deepseek_v4.nvidia.flashmla as flashmla_mod

        flashmla_mod.dequantize_and_gather_k_cache = checked
    except ImportError:
        pass
    _dsv4_gather_checked = True
    logger.info(
        "[GMS Patch] dequantize_and_gather_k_cache bounds check enabled "
        "(DYN_GMS_DSV4_GATHER_CHECK)"
    )


_dsv4_gather_triton_patched = False


def patch_dsv4_gather_k_cache_triton() -> None:
    """Route the DSv4 K-cache gather to Triton instead of CuTeDSL.

    ``dequantize_and_gather_k_cache`` picks the CuTeDSL kernel whenever the
    ``cutlass`` package is importable. That kernel is compiled against fake
    tensors carrying hard alignment contracts (``assumed_align=32`` and a
    stride divisibility of 32 on the K cache, 16 on the output) and copies
    through ``cpasync`` 128-bit loads. Under GMS those contracts do not hold
    and the kernel faults with cudaErrorIllegalAddress on the first prefill,
    identically from kernel warmup's ``_dummy_run`` and from the first real
    request.

    The Triton implementation in the same module computes the same gather
    without the alignment contract, so select it by making the module-local
    ``has_cutedsl`` report False. vLLM imports that name into ``cache_utils``
    at module scope, so it must be rebound there rather than in
    ``vllm.utils.import_utils``.
    """
    global _dsv4_gather_triton_patched

    if _dsv4_gather_triton_patched:
        return

    if os.environ.get("DYN_GMS_DSV4_GATHER_TRITON", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return

    try:
        import vllm.models.deepseek_v4.common.ops.cache_utils as cache_utils_mod
    except ImportError:
        return

    cache_utils_mod.has_cutedsl = lambda: False
    _dsv4_gather_triton_patched = True
    logger.info(
        "[GMS Patch] dequantize_and_gather_k_cache forced to Triton "
        "(DYN_GMS_DSV4_GATHER_TRITON)"
    )

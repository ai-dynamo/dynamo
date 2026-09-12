"""Route the V1 model-runner's KV allocation through the GMS scratch mempool.

vLLM 0.25.1 split the model runner into V1 (vllm.v1.worker.gpu_model_runner,
DEFAULT: use_v2_model_runner=False) and V2 (vllm.v1.worker.gpu.model_runner).
Main's patch_kv_cache_pool_scope() wraps `vllm.v1.worker.gpu.model_runner.init_kv_cache`
— a symbol that does not exist in this vLLM — so its `except AttributeError` makes
it a silent no-op on the default V1 runner. Result: the shadow's KV torch.zeros
never lands in the scratch pool -> worker.py fail-closed guard fires
("Scratch-KV enabled but no KV allocation was routed through scratch").

Fix: wrap V1's `_allocate_kv_cache_tensors` (the fn that emits the raw KV buffer,
scoped to KV tensors only — not block tables) in gms_use_mem_pool("kv_cache").
Guarded to only engage when the kv_cache scratch pool is actually registered
(shadow mode), so it's a no-op otherwise. Gate: DYN_GMS_KVROUTE_V1=1.

This is a runtime stand-in for the proper upstream fix (correct
patch_kv_cache_pool_scope to target the V1 runner + V2's real method).
"""
import importlib.abc
import sys

_T = "vllm.v1.worker.gpu_model_runner"


def _patch():
    mod = sys.modules.get(_T)
    if mod is None:
        return
    R = getattr(mod, "GPUModelRunner", None)
    if R is None:
        return
    orig = getattr(R, "_allocate_kv_cache_tensors", None)
    if orig is None or getattr(orig, "_kvroute", False):
        return
    import torch
    from gpu_memory_service.client.torch.allocator import (
        get_gms_client_memory_manager,
        gms_use_mem_pool,
    )

    def patched(self, *a, **k):
        # Only route when the kv_cache scratch pool exists (shadow mode); else no-op.
        if get_gms_client_memory_manager("kv_cache") is not None:
            device = torch.device("cuda", torch.cuda.current_device())
            with gms_use_mem_pool("kv_cache", device):
                return orig(self, *a, **k)
        return orig(self, *a, **k)

    patched._kvroute = True
    R._allocate_kv_cache_tensors = patched
    print(
        "[KVROUTE] wrapped V1 GPUModelRunner._allocate_kv_cache_tensors in kv_cache pool",
        flush=True,
    )


class _F(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _T:
            return None
        for f in list(sys.meta_path):
            if f is self:
                continue
            spec = f.find_spec(name, path, target)
            if spec and spec.loader:
                oe = spec.loader.exec_module

                def ex(m, _oe=oe):
                    _oe(m)
                    _patch()

                spec.loader.exec_module = ex
                return spec
        return None


sys.meta_path.insert(0, _F())
_patch()
print("[KVROUTE] loaded (V1 KV-alloc scratch routing)", flush=True)

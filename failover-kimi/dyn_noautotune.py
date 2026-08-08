"""Minimal: no-op FlashInfer autotune when DYN_NO_AUTOTUNE=1.

FlashInfer autotune is the source of the GMS warmup memory burst (workspaces
not reused under the split-pool layout -> +25 GiB / peak 178 / 15 alloc-retries).
Skipping it drops the GMS bring-up peak to ~145 (below vanilla) — the headroom
lever for two-engine colocation / replenishment.

This module does NOTHING else (no _dummy_run wrapping, no cuda.synchronize), so
it is safe under cudagraph capture (non-eager). Loaded via sitecustomize, so it
reaches every TP worker process.
"""
import importlib.abc
import sys

_T = "vllm.model_executor.warmup.kernel_warmup"


def _patch():
    mod = sys.modules.get(_T)
    if mod is None:
        return
    fn = getattr(mod, "flashinfer_autotune", None)
    if fn is None or getattr(fn, "_noat", False):
        return

    def noop(*a, **k):
        print("[NOAT] flashinfer_autotune skipped (DYN_NO_AUTOTUNE=1)", flush=True)
        return None

    noop._noat = True
    mod.flashinfer_autotune = noop
    print("[NOAT] patched flashinfer_autotune -> noop", flush=True)


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
print("[NOAT] no-autotune module loaded", flush=True)

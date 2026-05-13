# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics for the MX+GMS module-graph walk.

When ``GMS_DIAG=1`` is set, this module wraps
``modelexpress.load_strategy.register_tensors`` so that every call
site (initial load in ``_load_*_mode`` and each wake via
``mx_bringup``) dumps two snapshots of the model tree just before the
walker runs:

- ``rank{R}_{phase}.txt``  — flat, sorted, diff-friendly tuple list
- ``rank{R}_{phase}.tree`` — human-readable indented tree

``phase`` is ``initial`` on the first call per rank, then ``wake1``,
``wake2``, ...  Output goes to ``$GMS_DIAG_DIR`` (default
``/tmp/gms-diag``).

Motivation: the MX tensor walker has tripped three distinct times on
objects attached to the model between initial load and wake. Diffing
the two snapshots pinpoints which module attributes and types are
new, which drives the entry catalog in
``modelexpress-design/troubleshooting.md``.

No overhead when ``GMS_DIAG`` is unset — ``maybe_install()`` returns
without touching anything.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

_DIAG_DIR = Path(os.environ.get("GMS_DIAG_DIR", "/tmp/gms-diag"))
_call_count: dict[int, int] = {}  # rank -> number of register_tensors calls seen


def _phase_label(rank: int) -> str:
    n = _call_count.get(rank, 0)
    _call_count[rank] = n + 1
    return "initial" if n == 0 else f"wake{n}"


def _describe_value(val: object) -> str:
    """Short, stable summary of a non-module attribute value."""
    t = type(val)
    qual = f"{t.__module__}.{t.__qualname__}"
    if isinstance(val, nn.Parameter):
        return f"{qual} shape={list(val.shape)} dtype={val.dtype}"
    if isinstance(val, torch.Tensor):
        return f"{qual} shape={list(val.shape)} dtype={val.dtype}"
    if isinstance(val, (list, tuple)):
        return f"{qual} len={len(val)}"
    if isinstance(val, dict):
        return f"{qual} len={len(val)}"
    return qual


# nn.Module.__init__ sets these on every module. They are stable
# across load / warmup / wake, so skipping them cuts snapshot noise
# without hiding any state that changes between snapshots. Parameters
# and buffers are still surfaced via the named_parameters /
# named_buffers passes in _dump_*.
_MODULE_INTERNAL_ATTRS = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    }
)


def _module_items(module: nn.Module):
    """Yield (attr_name, value) for user-visible, non-child-module attributes.

    Sorted for deterministic output. Skips:

    - dunders (matches the walker's own filter)
    - ``nn.Module`` internal bookkeeping attrs (hooks, training, etc.)
      which are stable and only add diff noise
    - child modules (already covered by ``named_modules``)

    Parameters and buffers of this module are *not* yielded here;
    they appear in the snapshot via separate named_parameters /
    named_buffers passes in ``_dump_flat`` / ``_dump_tree``.
    """
    try:
        d = vars(module)
    except TypeError:
        return
    for name in sorted(d):
        if name.startswith("__") or name in _MODULE_INTERNAL_ATTRS:
            continue
        val = d[name]
        if isinstance(val, nn.Module):
            continue
        yield name, val


def _tensor_desc(t: torch.Tensor) -> str:
    return f"shape={list(t.shape)} dtype={t.dtype} device={t.device}"


def _dump_flat(model: nn.Module, path: Path) -> None:
    """Flat tuple list, one line per (module, entry), sorted for ``diff -u``.

    Each module contributes up to four kinds of lines:

    - ``<module>\\t<self>\\t<type>``        — the module's own class
    - ``<module>\\t[param] <name>\\t<desc>`` — direct parameters
    - ``<module>\\t[buf]   <name>\\t<desc>`` — direct buffers
    - ``<module>\\t<attr>\\t<desc>``         — user-attached attributes
    """
    lines: list[str] = []
    for mod_path, module in model.named_modules():
        mod_name = mod_path or "<root>"
        mod_type = f"{type(module).__module__}.{type(module).__qualname__}"
        lines.append(f"{mod_name}\t<self>\t{mod_type}")
        for p_name, p in module.named_parameters(recurse=False):
            lines.append(f"{mod_name}\t[param] {p_name}\t{_tensor_desc(p)}")
        for b_name, b in module.named_buffers(recurse=False):
            lines.append(f"{mod_name}\t[buf] {b_name}\t{_tensor_desc(b)}")
        for attr_name, val in _module_items(module):
            lines.append(f"{mod_name}\t{attr_name}\t{_describe_value(val)}")
    lines.sort()
    path.write_text("\n".join(lines) + "\n")


def _dump_tree(model: nn.Module, path: Path) -> None:
    """Indented tree; parameters, buffers, and user attrs shown per module."""
    lines: list[str] = []
    for mod_path, module in model.named_modules():
        depth = mod_path.count(".") + 1 if mod_path else 0
        pad = "  " * depth
        name = mod_path.split(".")[-1] if mod_path else "<root>"
        lines.append(f"{pad}{name}: {type(module).__qualname__}")
        inner = "  " * (depth + 1)
        for p_name, p in module.named_parameters(recurse=False):
            lines.append(f"{inner}[param] {p_name}: {_tensor_desc(p)}")
        for b_name, b in module.named_buffers(recurse=False):
            lines.append(f"{inner}[buf] {b_name}: {_tensor_desc(b)}")
        for attr_name, val in _module_items(module):
            lines.append(f"{inner}[attr] {attr_name}: {_describe_value(val)}")
    path.write_text("\n".join(lines) + "\n")


def _snapshot(model: nn.Module, rank: int) -> None:
    phase = _phase_label(rank)
    try:
        _DIAG_DIR.mkdir(parents=True, exist_ok=True)
        flat = _DIAG_DIR / f"rank{rank}_{phase}.txt"
        tree = _DIAG_DIR / f"rank{rank}_{phase}.tree"
        _dump_flat(model, flat)
        _dump_tree(model, tree)
        logger.info(
            "[GMS-DIAG] rank=%d phase=%s -> %s + %s",
            rank,
            phase,
            flat,
            tree,
        )
    except Exception as e:
        logger.warning(
            "[GMS-DIAG] rank=%d phase=%s snapshot failed: %s",
            rank,
            phase,
            e,
        )


# Modules that import register_tensors by name. Each holds its own
# binding, so patching the source module alone isn't enough — we must
# also rebind every consumer.
_KNOWN_CONSUMERS = (
    "modelexpress.load_strategy.base",
    "modelexpress.load_strategy",
    "gpu_memory_service.integrations.vllm.model_loader",
    "gpu_memory_service.integrations.vllm.worker",
)


def install_register_tensors_snapshot() -> None:
    """Wrap ``register_tensors`` in every known namespace with a snapshot hook.

    Safe to call multiple times — no-op on the second call.
    """
    try:
        from modelexpress.load_strategy import base as _base
    except ImportError:
        logger.warning("[GMS-DIAG] modelexpress not installed; diagnostics disabled")
        return

    if getattr(_base.register_tensors, "_gms_diag_wrapped", False):
        return

    original = _base.register_tensors

    def wrapped(model, ctx, *args, **kwargs):
        # Pass through any future/existing kwargs (e.g. reuse_discovered=True
        # from mx_bringup) so we don't break the wrapped signature. We still
        # snapshot the model state at every call site regardless of flags —
        # the snapshot's value is comparing initial vs wake trees, whether
        # or not the walker actually runs inside register_tensors.
        try:
            _snapshot(model, ctx.global_rank)
        except Exception as e:
            logger.warning("[GMS-DIAG] snapshot wrapper error: %s", e)
        return original(model, ctx, *args, **kwargs)

    wrapped._gms_diag_wrapped = True  # type: ignore[attr-defined]

    patched = []
    for name in _KNOWN_CONSUMERS:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "register_tensors", None) is original:
            mod.register_tensors = wrapped
            patched.append(name)
    logger.info(
        "[GMS-DIAG] register_tensors snapshot hook installed (dir=%s, patched=%s)",
        _DIAG_DIR,
        patched,
    )


def maybe_install() -> None:
    """Install diagnostics iff ``GMS_DIAG=1``. Called from worker.py at import."""
    if os.environ.get("GMS_DIAG", "0") != "1":
        return
    install_register_tensors_snapshot()

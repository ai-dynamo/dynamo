# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup check for SGLang's ``--elastic-ep-backend mooncake``.

SGLang builds its elastic-EP process groups from the mooncake transfer
engine's torch ``ProcessGroup`` extension, deep inside engine startup and
after the model is loaded. The check has to run before model load to be worth
anything, and it must not import ``sglang`` or eagerly import ``torch``, so it
still answers in an image whose engine is the broken part.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Dict, List, Optional, Tuple

MOONCAKE_BACKEND = "mooncake"

# The torch backend names SGLang registers its elastic-EP process groups under
# (sglang/srt/distributed/parallel_state.py): the device group and the CPU group
# it uses for the metadata collectives. The extension registers both together, so
# one present without the other is a partial registration the engine falls over
# on later.
_REQUIRED_TORCH_BACKENDS = ("mooncake", "mooncake-cpu")

# mooncake renamed this extension ``mooncake.ep`` -> ``mooncake.pg``; SGLang
# v0.5.16 imports the old name and v0.5.18 the new one.
_PROCESS_GROUP_MODULES = ("mooncake.pg", "mooncake.ep")

# Where SGLang names that extension. Read as text: importing them to ask would
# need the working engine this check exists to doubt, and would pull in torch
# and CUDA before it can answer.
_SGLANG_ELASTIC_EP_SOURCES = (
    "srt/elastic_ep/elastic_ep.py",
    "srt/distributed/parallel_state.py",
)

# Both distributions ship the same extension; only one is normally installed.
_MOONCAKE_DISTRIBUTIONS = (
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine",
)

# Compiled ProcessGroup extensions are named for the exact torch release they
# were built against, e.g. ``pg_2_11_0`` / ``ep_2_9_1``.
_EXTENSION_PREFIXES = ("pg_", "ep_")


def _required_process_group_modules() -> Tuple[str, ...]:
    """The extension names the installed SGLang actually imports.

    An image can hold the wrong half of the ``mooncake.ep`` -> ``mooncake.pg``
    rename: the wheel imports cleanly and the engine still cannot start,
    because it asks for the other name. Reading the engine's own import line
    pins which name has to work. Falls back to accepting either when those
    sources cannot be read, since refusing a worker over an upstream file move
    would be worse than the late failure this check replaces.
    """
    try:
        spec = importlib.util.find_spec("sglang")
    except Exception:  # noqa: BLE001 - a broken engine must not mask the real error
        return _PROCESS_GROUP_MODULES
    if spec is None or not spec.submodule_search_locations:
        return _PROCESS_GROUP_MODULES

    required: List[str] = []
    for location in spec.submodule_search_locations:
        for relative in _SGLANG_ELASTIC_EP_SOURCES:
            try:
                with open(os.path.join(location, relative), encoding="utf-8") as handle:
                    source = handle.read()
            except OSError:
                continue
            for module_name in _PROCESS_GROUP_MODULES:
                if module_name in source and module_name not in required:
                    required.append(module_name)
    return tuple(required) or _PROCESS_GROUP_MODULES


def _import_process_group_extension() -> Optional[str]:
    """Return ``None`` when the extension imports, else the collected failures.

    Importing it is what SGLang itself does later; doing it here surfaces the
    torch-version skew (``mooncake.pg`` raises ``ImportError`` when no
    ``pg_<torch>`` module matches) before any GPU work has started.
    """
    failures: List[str] = []
    for module_name in _required_process_group_modules():
        try:
            importlib.import_module(module_name)
            return None
        except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return "; ".join(failures) if failures else "no attempt was made"


def _installed_mooncake_versions() -> Dict[str, str]:
    """Map each installed mooncake distribution to its version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _distribution_version

    found: Dict[str, str] = {}
    for distribution in _MOONCAKE_DISTRIBUTIONS:
        try:
            found[distribution] = _distribution_version(distribution)
        except PackageNotFoundError:
            continue
    return found


def _available_extension_modules() -> List[str]:
    """Names of the compiled ProcessGroup extensions the wheel actually ships.

    Resolved from the package directory rather than by import, so this stays
    cheap and side-effect free when the extension is the thing that is broken.
    """
    try:
        spec = importlib.util.find_spec("mooncake")
    except Exception:  # noqa: BLE001 - a broken package must not mask the real error
        return []
    if spec is None or not spec.submodule_search_locations:
        return []

    modules = set()
    for location in spec.submodule_search_locations:
        try:
            entries = os.listdir(location)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith(_EXTENSION_PREFIXES):
                continue
            modules.add(entry.split(".", 1)[0])
    return sorted(modules)


def _registered_torch_backends() -> Optional[Dict[str, Tuple[str, ...]]]:
    """torch's backend -> supported-device map, or ``None`` when unreadable.

    ``Backend.backend_capability`` is a torch internal. Reporting "could not
    read the registry" as "mooncake is not registered" would let a rename in
    somebody else's private attribute refuse every worker that asked for this
    transport, which is more fatal than the failure being guarded against.
    ``None`` keeps the two apart so the caller can skip only this assertion.
    """
    try:
        import torch.distributed as torch_distributed

        return dict(torch_distributed.Backend.backend_capability)
    except Exception:  # noqa: BLE001 - unreadable, not "nothing registered"
        return None


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # noqa: BLE001
        return "not installed"


def _format_versions(versions: Dict[str, str]) -> str:
    if not versions:
        return "none installed"
    return ", ".join(f"{name} {value}" for name, value in sorted(versions.items()))


def _build_diagnostic(
    import_failure: Optional[str],
    registered_backends: Dict[str, Tuple[str, ...]],
    missing_backends: Tuple[str, ...],
    enable_dp_attention: bool,
) -> str:
    mooncake_backends = {
        name: devices
        for name, devices in registered_backends.items()
        if MOONCAKE_BACKEND in name
    }
    lines = [
        "--elastic-ep-backend mooncake was requested, but the mooncake torch "
        "ProcessGroup backend is not usable in this image. SGLang builds its "
        "elastic-EP process groups from that backend after the model is "
        "loaded, so leaving this unchecked fails the worker late and from "
        "inside engine code.",
        f"  mooncake ProcessGroup extension: {import_failure or 'imported, but registered no usable backend'}",
        f"  installed mooncake distributions: {_format_versions(_installed_mooncake_versions())}",
        f"  torch: {_torch_version()}",
        f"  ProcessGroup extensions this mooncake ships: {', '.join(_available_extension_modules()) or 'none found'}",
        f"  mooncake backends registered with torch.distributed: {mooncake_backends or 'none'}",
    ]
    if missing_backends:
        lines.append(
            "  required by SGLang but not registered: " f"{', '.join(missing_backends)}"
        )
    if enable_dp_attention:
        lines.append(
            "  --enable-dp-attention is also set: DP attention synchronizes its "
            "MLP batch metadata with an all-gather over this group on the very "
            "first forward pass, so this combination reaches the backend "
            "immediately at startup."
        )
    lines.append(
        "Install a mooncake-transfer-engine build whose ProcessGroup extension "
        "matches the running torch, or drop --elastic-ep-backend."
    )
    return "\n".join(lines)


def check_elastic_ep_backend(
    elastic_ep_backend: Optional[str],
    enable_dp_attention: bool = False,
) -> None:
    """Reject ``--elastic-ep-backend mooncake`` when the image cannot serve it.

    Silent for every other configuration, including an unset backend: this must
    never turn into an unconditional startup failure for workers that asked for
    nothing of the sort.

    Raises:
        ValueError: The mooncake backend was requested but its torch
            ProcessGroup extension does not import, or imports without
            registering both of the backends SGLang asks torch for.
    """
    if not elastic_ep_backend:
        return
    if str(elastic_ep_backend).strip().lower() != MOONCAKE_BACKEND:
        return

    import_failure = _import_process_group_extension()
    registered_backends: Dict[str, Tuple[str, ...]] = {}
    missing_backends: Tuple[str, ...] = ()
    if import_failure is None:
        readable_backends = _registered_torch_backends()
        if readable_backends is None:
            return
        missing_backends = tuple(
            name for name in _REQUIRED_TORCH_BACKENDS if name not in readable_backends
        )
        if not missing_backends:
            return
        registered_backends = readable_backends

    raise ValueError(
        _build_diagnostic(
            import_failure,
            registered_backends,
            missing_backends,
            enable_dp_attention,
        )
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup check for SGLang's ``--elastic-ep-backend mooncake``.

SGLang builds its elastic-EP process groups from the mooncake transfer
engine's torch ``ProcessGroup`` extension: ``backend="mooncake"`` for the
device group and ``backend="mooncake-cpu"`` for the CPU group. That import
and that group construction happen deep inside engine startup, well after
the model is loaded, so a worker whose image cannot serve the requested
backend only finds out late and reports it from inside upstream engine code
that the deployer did not write. Checking the same preconditions while
Dynamo is still assembling ``ServerArgs`` turns that into an argument error
naming the flag that caused it.

This module deliberately imports no ``sglang`` symbol, and imports ``torch``
only inside the probe helpers. That keeps it usable -- and unit-testable --
in an image where the engine is not importable.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Dict, List, Optional, Tuple

MOONCAKE_BACKEND = "mooncake"

# The torch backend name SGLang registers the elastic-EP *device* group under
# (sglang/srt/distributed/parallel_state.py). The CPU group's "mooncake-cpu" is
# reported for context but not required: an engine/wheel pairing that only needs
# the device group must not be blocked here.
_MOONCAKE_DEVICE_BACKEND = "mooncake"

# mooncake renamed this extension ``mooncake.ep`` -> ``mooncake.pg``; SGLang
# v0.5.16 imports the old name and v0.5.18 the new one. Either resolving is
# enough, so try both rather than pinning the engine version this Dynamo build
# happens to sit next to.
_PROCESS_GROUP_MODULES = ("mooncake.pg", "mooncake.ep")

# Both distributions ship the same extension; only one is normally installed.
_MOONCAKE_DISTRIBUTIONS = (
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine",
)

# Compiled ProcessGroup extensions are named for the exact torch release they
# were built against, e.g. ``pg_2_11_0`` / ``ep_2_9_1``.
_EXTENSION_PREFIXES = ("pg_", "ep_")


def _import_process_group_extension() -> Optional[str]:
    """Return ``None`` when the extension imports, else the collected failures.

    Importing it is what SGLang itself does later; doing it here surfaces the
    torch-version skew (``mooncake.pg`` raises ``ImportError`` when no
    ``pg_<torch>`` module matches) before any GPU work has started.
    """
    failures: List[str] = []
    for module_name in _PROCESS_GROUP_MODULES:
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
            registering the device backend SGLang asks torch for.
    """
    if not elastic_ep_backend:
        return
    if str(elastic_ep_backend).strip().lower() != MOONCAKE_BACKEND:
        return

    import_failure = _import_process_group_extension()
    registered_backends: Dict[str, Tuple[str, ...]] = {}
    if import_failure is None:
        readable_backends = _registered_torch_backends()
        # An unreadable registry fails open: the import above is what catches
        # the reported failure, and refusing startup because torch moved a
        # private attribute would block images that are perfectly fine.
        if readable_backends is None:
            return
        if _MOONCAKE_DEVICE_BACKEND in readable_backends:
            return
        registered_backends = readable_backends

    raise ValueError(
        _build_diagnostic(import_failure, registered_backends, enable_dp_attention)
    )

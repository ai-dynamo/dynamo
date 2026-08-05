# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the **real** ``push_egress.py`` so the simulation exercises shipped code.

``components/src/dynamo/trtllm/request_handlers/push_egress.py`` imports only
``asyncio``, ``functools``, ``logging``, ``os``, ``typing`` and
``dynamo.common.utils.nvtx_utils`` -- and ``nvtx_utils`` in turn imports only
``functools``, ``inspect``, ``os``, degrading to no-op stubs when ``DYN_NVTX``
is unset or the ``nvtx`` wheel is absent. So the push-egress driver can be
loaded on a bare interpreter, without torch, tensorrt_llm or ``dynamo._core``.

Only ``dynamo/common/utils/__init__.py`` blocks a plain import, because it
eagerly pulls ``dynamo.llm`` (a compiled extension). The two files are
therefore loaded by path into a synthetic package tree, and *only* when a
normal import fails -- inside the container the installed package is used
as-is and nothing is stubbed.

This is what makes the sim a regression test rather than a lookalike: the
``__wrapped__`` deletion, the "return an async generator, not a coroutine"
invariant of 0fb02c2ea6, and the send/close/close_with_error call sequence are
all checked against the file that actually ships.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

#: egress_experiments/dynamo_sim/realcode.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPONENTS = REPO_ROOT / "components" / "src" / "dynamo"
_NVTX_UTILS = _COMPONENTS / "common" / "utils" / "nvtx_utils.py"
_PUSH_EGRESS = _COMPONENTS / "trtllm" / "request_handlers" / "push_egress.py"

_cached: Optional[types.ModuleType] = None
_failure: Optional[str] = None


def _load_by_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_standalone() -> types.ModuleType:
    """Path-load into a synthetic ``dynamo.common.utils`` tree."""
    for pkg in ("dynamo", "dynamo.common", "dynamo.common.utils"):
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[pkg] = module

    nvtx_utils = _load_by_path("dynamo.common.utils.nvtx_utils", _NVTX_UTILS)
    setattr(sys.modules["dynamo.common.utils"], "nvtx_utils", nvtx_utils)
    return _load_by_path(
        "egress_experiments._real_push_egress",
        _PUSH_EGRESS,
    )


def load_push_egress() -> Optional[types.ModuleType]:
    """Return the real push-egress module, or ``None`` with a reason recorded.

    Never raises: a missing checkout should skip the affected tests, not break
    the whole simulation.
    """
    global _cached, _failure
    if _cached is not None or _failure is not None:
        return _cached

    # Installed package first, so nothing is stubbed inside the container.
    try:
        _cached = importlib.import_module("dynamo.trtllm.request_handlers.push_egress")
        return _cached
    except Exception:
        pass

    try:
        _cached = _load_standalone()
        return _cached
    except Exception as exc:  # pragma: no cover - environment dependent
        _failure = f"{type(exc).__name__}: {exc}"
        return None


def load_failure() -> Optional[str]:
    return _failure

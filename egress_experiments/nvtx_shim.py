# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit NVTX through the repo's **own** ``nvtx_utils``, not a reimplementation.

The point of instrumenting the simulation is to run it under nsys and feed the
result back through :mod:`egress_experiments.capture_params` -- the same
extractor used on the real capture. That is only a meaningful check if the
range names, the domain, the colours and the enable-gate are the ones the real
worker uses, so this loads
``components/src/dynamo/common/utils/nvtx_utils.py`` itself.

Gate is unchanged: ``DYN_NVTX=1`` plus the ``nvtx`` wheel. Off, every call is a
no-op and the simulation runs exactly as before.

Loading it standalone is necessary because ``dynamo/common/utils/__init__.py``
eagerly imports ``dynamo.llm``, a compiled extension. Inside the container the
installed package is used instead and nothing is stubbed.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

#: egress_experiments/nvtx_shim.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
_NVTX_UTILS = (
    REPO_ROOT / "components" / "src" / "dynamo" / "common" / "utils" / "nvtx_utils.py"
)

_loaded: Optional[types.ModuleType] = None


class _NoOp:
    """Matches nvtx_utils' disabled branch."""

    @staticmethod
    def start_range(message: str, color: str = "white") -> Any:
        return None

    @staticmethod
    def end_range(rng: Any) -> None:
        return None

    @staticmethod
    def mark(message: str, color: str = "white") -> None:
        return None


def load() -> Any:
    """Return the real ``nvtx_utils`` module, or a no-op stand-in.

    Also registers it at ``dynamo.common.utils.nvtx_utils`` in ``sys.modules``,
    which is what lets :mod:`egress_experiments.dynamo_sim.realcode` path-load
    the real ``push_egress.py`` afterwards.
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    # Installed package first (the container), so nothing gets stubbed there.
    try:
        _loaded = importlib.import_module("dynamo.common.utils.nvtx_utils")
        return _loaded
    except Exception:
        pass

    try:
        for pkg in ("dynamo", "dynamo.common", "dynamo.common.utils"):
            if pkg not in sys.modules:
                module = types.ModuleType(pkg)
                module.__path__ = []  # type: ignore[attr-defined]
                sys.modules[pkg] = module

        spec = importlib.util.spec_from_file_location(
            "dynamo.common.utils.nvtx_utils", _NVTX_UTILS
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load nvtx_utils from {_NVTX_UTILS}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["dynamo.common.utils.nvtx_utils"] = module
        spec.loader.exec_module(module)
        setattr(sys.modules["dynamo.common.utils"], "nvtx_utils", module)
        _loaded = module
    except Exception:
        # A missing `nvtx` wheel is the common case and must not be fatal:
        # nvtx_utils itself degrades to no-ops, and so do we.
        _loaded = _NoOp()  # type: ignore[assignment]
    return _loaded


#: Module-level handle. Import this, not the loader.
nvtx = load()


def enabled() -> bool:
    """True when ranges will actually reach a profiler."""
    return os.environ.get("DYN_NVTX") == "1" and not isinstance(nvtx, _NoOp)


class range_:
    """Context manager over ``start_range``/``end_range``.

    Deliberately not ``nvtx_utils.annotate``: that uses the thread's nested
    push/pop stack, which interleaves incorrectly when another coroutine
    resumes on the same event loop. Every range in this simulation is on the
    loop, so start/end is the only safe form.
    """

    __slots__ = ("_message", "_color", "_handle")

    def __init__(self, message: str, color: str = "white") -> None:
        self._message = message
        self._color = color
        self._handle = None

    def __enter__(self) -> "range_":
        self._handle = nvtx.start_range(self._message, color=self._color)
        return self

    def __exit__(self, *exc) -> None:
        nvtx.end_range(self._handle)

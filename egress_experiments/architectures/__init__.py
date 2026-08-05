# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable response-path architectures.

The simulator core -- ``fake_trtllm``, ``costs``, ``probes``, ``harness`` --
is **frozen**. Architectures are added here instead, as small subclasses of
:class:`Architecture` that swap in their own LLM/proxy, handler or driver.
That keeps every experiment measurable against the same stick: if one
experiment could edit ``Costs`` or the dispatch thread, its number would not be
comparable with anyone else's.

Three hooks, matching the three places the response path can be restructured:

``build_llm``      the proxy side -- how responses cross from the engine and
                   how they are handed to the loop (``dispatch_result_task``,
                   ``notify_many``, the per-request ``AsyncQueue``).
``build_handler``  the worker side -- how a request consumes its responses and
                   builds output chunks.
``build_driver``   the Rust side -- how the loop is entered and left, and how
                   many GIL acquisitions that costs.

Rules for a new architecture
----------------------------
1. **Do not edit the core.** If you believe you have found a bug in it, report
   it -- it gets fixed centrally so every experiment moves together.
2. **Conserve work.** ``Costs`` is the modelled cost of real work; you may move
   it to another thread or process, or amortise it across a batch, but you may
   not delete it. ``costs.spin_ledger()`` shows where it went and the benchmark
   prints it.
3. **Justify against the real code.** An architecture that cannot be built in
   ``lib/bindings/python/rust`` + ``components/src/dynamo/trtllm`` is not a
   result. Cite what would change.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from egress_experiments.costs import Costs
from egress_experiments.dynamo_sim.probes import RequestRecord
from egress_experiments.dynamo_sim.rust_bridge import (
    Driver,
    PullDriver,
    PushDriver,
    TokioRuntime,
)
from egress_experiments.dynamo_sim.worker import TrtllmWorkerHandler
from egress_experiments.fake_trtllm.engine import EngineConfig
from egress_experiments.fake_trtllm.llm import FakeLLM


class Architecture:
    """One way of getting responses from the engine onto the wire."""

    #: Stable identifier used by ``--architecture`` and in reports.
    name: str = "baseline-push"
    #: One line, shown in the benchmark header.
    description: str = "shipped push egress (DYN_TRTLLM_PUSH_EGRESS=1)"
    #: Which Rust driver shape this uses, for reporting only.
    egress: str = "push"

    # -- hooks -------------------------------------------------------------

    def build_llm(self, engine_config: EngineConfig, costs: Costs) -> FakeLLM:
        return FakeLLM(engine_config, costs=costs)

    def build_handler(
        self,
        llm: FakeLLM,
        costs: Costs,
        records: Dict[str, RequestRecord],
    ) -> Any:
        return TrtllmWorkerHandler(llm, costs=costs, records=records)

    def build_driver(
        self,
        handler: Any,
        py_loop: Any,
        tokio: TokioRuntime,
        costs: Costs,
    ) -> Driver:
        return PushDriver(handler, py_loop, tokio, costs)

    # -- lifecycle ---------------------------------------------------------

    def on_started(self, llm: FakeLLM, driver: Driver) -> None:
        """Called once the loop, engine and dispatch thread are all up."""

    def on_finished(self, llm: FakeLLM, driver: Driver) -> None:
        """Called before teardown; use it to stop any threads you started."""

    def extra_report(self) -> Dict[str, Any]:
        """Architecture-specific counters to print alongside the result."""
        return {}


class BaselinePull(Architecture):
    name = "baseline-pull"
    description = "shipped pull egress (demand_driven_python_stream)"
    egress = "pull"

    def build_driver(self, handler, py_loop, tokio, costs) -> Driver:
        return PullDriver(handler, py_loop, tokio, costs)


class BaselinePush(Architecture):
    name = "baseline-push"
    description = "shipped push egress (DYN_TRTLLM_PUSH_EGRESS=1)"
    egress = "push"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], Architecture]] = {}


def register(factory: Callable[[], Architecture], name: Optional[str] = None) -> None:
    arch_name = name or factory().name
    if arch_name in _REGISTRY:
        raise ValueError(f"architecture {arch_name!r} is already registered")
    _REGISTRY[arch_name] = factory


def get(name: str) -> Architecture:
    if name not in _REGISTRY:
        raise KeyError(f"unknown architecture {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def names() -> List[str]:
    return sorted(_REGISTRY)


register(BaselinePull)
register(BaselinePush)

#: Import failures, so a broken experiment is visible rather than silent.
IMPORT_ERRORS: Dict[str, str] = {}


def _discover() -> None:
    """Import every sibling module so it can register itself.

    Auto-discovery rather than a hand-maintained list: experiments are written
    in parallel worktrees, and a shared list would be a merge conflict on every
    single one of them. Drop a file in this directory, call ``register`` at
    module scope, done.

    A module that fails to import is recorded and skipped -- it must not take
    the baselines down with it, because the baselines are what everything else
    is measured against.
    """
    import importlib
    import pkgutil

    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:  # pragma: no cover - experiment-dependent
            IMPORT_ERRORS[info.name] = f"{type(exc).__name__}: {exc}"


_discover()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load in-process user plugins from config specs.

Given a list of ``InProcessPluginSpec`` (see ``registry/config.py``),
import the named module, instantiate the named class with ``kwargs``,
and hand the instance to ``orchestrator.register_internal``.

``OrchestratorEngineAdapter.bootstrap_plugins`` invokes this helper after
registering builtins and before the Bootstrap fan-out, so operator-declared
in-process plugins coexist with builtins and receive the same lifecycle pass.
"""

from __future__ import annotations

import importlib
import logging
from typing import Sequence

from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.config import InProcessPluginSpec
from dynamo.planner.plugins.types import HoldPolicy

log = logging.getLogger(__name__)


def load_in_process_plugins(
    orchestrator: LocalPlannerOrchestrator,
    specs: Sequence[InProcessPluginSpec],
) -> None:
    """Iterate ``specs`` and register each via
    ``orchestrator.register_internal`` with ``is_builtin=False``.

    Any import / construction / registration failure is **re-raised** —
    startup should fail fast so operators notice a misconfigured
    ``in_process_plugins`` entry rather than silently running without
    the plugin. Tests catch common mistakes at
    ``test_in_process_loader.py``.
    """
    for spec in specs:
        try:
            module = importlib.import_module(spec.module)
        except ImportError as exc:
            raise ImportError(
                f"load_in_process_plugins: failed to import module "
                f"{spec.module!r} for plugin_id={spec.plugin_id!r}: {exc}"
            ) from exc
        try:
            cls = getattr(module, spec.class_)
        except AttributeError as exc:
            raise AttributeError(
                f"load_in_process_plugins: module {spec.module!r} has no "
                f"attribute {spec.class_!r} for plugin_id={spec.plugin_id!r}"
            ) from exc

        try:
            instance = cls(**spec.kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"load_in_process_plugins: failed to construct "
                f"{spec.class_!r} (plugin_id={spec.plugin_id!r}, "
                f"kwargs={spec.kwargs!r}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        hold_policy = HoldPolicy[spec.hold_policy]  # "ACCEPT_WHEN_IDLE" / "HOLD_LAST"
        orchestrator.register_internal(
            plugin_id=spec.plugin_id,
            plugin_type=spec.plugin_type,
            priority=spec.priority,
            instance=instance,
            execution_interval_seconds=spec.execution_interval_seconds,
            hold_policy=hold_policy,
            is_builtin=False,
            version="user-in-process",
            needs=list(spec.needs),
            requires_produced_fields=list(spec.requires_produced_fields),
            observation_window_seconds=spec.observation_window_seconds,
        )
        log.info(
            "load_in_process_plugins: registered plugin_id=%s module=%s class=%s",
            spec.plugin_id,
            spec.module,
            spec.class_,
        )


__all__ = ["load_in_process_plugins"]

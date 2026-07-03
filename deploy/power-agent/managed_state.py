# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the Power Agent's mutable managed-GPU state.

Why this lives in its own module instead of in ``power_agent.py``:

The daemon entrypoint runs ``power_agent.py`` as the top-level ``__main__``
module — every launch path does this:

  * the image ``ENTRYPOINT ["python", "/app/power_agent.py"]`` (Dockerfile),
  * the Helm DaemonSet ``command: [python, /app/power_agent.py]``
    (templates/daemonset.yaml), and
  * the dev-pod ``exec python3 /scripts/power_agent.py`` (templates/dev-pod.yaml).

Meanwhile ``actuator.py`` reaches back into the agent via ``import
power_agent`` (e.g. ``NvmlActuator.apply_cap`` delegates to
``power_agent._apply_cap`` and ``DcgmActuator._record_managed_state``
records the cap). Because the running module is ``__main__`` but the
actuator imports it under the name ``power_agent``, Python materialises
**two distinct module objects** in ``sys.modules`` — each with its own
module-level globals.

If the managed-GPU sets lived in ``power_agent.py`` they would therefore
exist as two independent copies: the actuator would record freshly-capped
GPUs into the ``power_agent`` copy while shutdown cleanup (running in
``__main__``) restored from the ``__main__`` copy — which would always be
empty. The failure is silent and total: every cap leaks past graceful
shutdown because the restore loop never sees a single managed GPU.

Hosting the state here — imported under its canonical name ``managed_state``
by ``power_agent.py`` — guarantees exactly one copy regardless of how the
agent was launched: the ``__main__`` instance and the canonical
``power_agent`` instance both ``import managed_state``, which resolves to the
*same* cached module object, so their ``_managed_gpu_indices`` /
``_previously_managed`` aliases converge on one set. (``actuator.py`` reaches
that set through ``power_agent`` — e.g. ``power_agent._managed_gpu_indices`` —
rather than importing ``managed_state`` itself.) This module deliberately
imports nothing from ``power_agent`` or ``actuator`` so it can never
participate in an import cycle.

NOTE FOR PACKAGING: any new launch surface must ship this file alongside
``power_agent.py`` / ``actuator.py`` (the image ``COPY`` and the dev-pod
script ConfigMap both include it). A missing ``managed_state.py`` fails
loudly at startup with ``ModuleNotFoundError: No module named
'managed_state'``.
"""

# Absolute path of the persisted managed-GPU state file (UUID-gated orphan
# recovery). Kept here so every module agrees on one location.
MANAGED_STATE_PATH = "/var/lib/dynamo-power-agent/managed_gpus.json"

# In-process set of physical GPU indices this running agent has capped.
# Populated by every successful cap write (NVML via ``power_agent._apply_cap``,
# DCGM via ``DcgmActuator._record_managed_state``) and drained by shutdown
# cleanup (``power_agent._shutdown_cleanup``, invoked from the reconcile loop
# after SIGTERM), which restores each one to default TGP before exit.
managed_gpu_indices: set[int] = set()

# Persisted across restarts (``MANAGED_STATE_PATH``): the UUIDs this agent
# currently OWNS a below-default cap on — added on a successful cap write and
# PRUNED again once that cap is released/restored to default (runtime release,
# SIGTERM sweep, or cold-start orphan recovery). It is NOT an append-only "ever
# capped" ledger (that is ``DcgmActuator._capped_uuids``); it is the live,
# cross-incarnation ownership set used for UUID-gated cold-start orphan recovery
# so a restart only touches GPUs it still owns. Always mutated in place — never
# rebind this name, or the alias that ``power_agent.py`` and ``actuator.py``
# hold would split and re-introduce the dual-copy bug described above.
previously_managed: set[str] = set()

# UUIDs whose cap acquisition completed (hardware cap is live and in-memory
# ownership was recorded) but whose durable ADD to MANAGED_STATE_PATH failed.
# This must be shared for the same reason as ``previously_managed``: cap writes
# run through the actuator's canonical ``import power_agent`` module, while the
# reconcile loop that flushes the retry queue runs in the entrypoint module.
pending_acquisition: set[str] = set()

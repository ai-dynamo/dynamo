# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The standard verb catalogue.

Declarations only — no implementation. A provider supplies the behaviour; this
file is the contract about what the names mean, what each is allowed to do, and
which of them gate a run.

The set is not invented. It covers the scenario suite's 26 event kinds (each maps
to exactly one verb, proved in ``tests/test_catalog.py``) plus the reach,
inference and lifecycle surface the deployment tests already need.

``aliases`` carry the existing document spellings, so scenario YAML written
against the current event names keeps working while the canonical name is the
one a reader sees in a plan.
"""

from __future__ import annotations

from .roles import Role
from .verbs import Grant, Phase, Receiver, verb

# --------------------------------------------------------------------- reach

url = verb(
    "url",
    phase=Phase.ARRANGE,
    grant=Grant.READ,
    default_role=Role.FRONTEND,
    summary="Address of a role's service port, from the test process.",
)
address = verb(
    "address",
    phase=Phase.ARRANGE,
    grant=Grant.READ,
    default_role=Role.FRONTEND,
    params={"vantage": "who is doing the reaching: test process or in-cluster"},
    summary="Address of a role, resolved for a given vantage point.",
)
models = verb(
    "models",
    grant=Grant.READ,
    default_role=Role.FRONTEND,
    summary="Model ids the endpoint advertises.",
)

# ----------------------------------------------------------------- inference

query = verb(
    "query",
    grant=Grant.INFER,
    default_role=Role.FRONTEND,
    params={"payload": "prompt or request body", "timeout": "seconds"},
    summary="One completion. The default is the frontend, so most tests say query(...).",
)
stream = verb(
    "stream",
    grant=Grant.INFER,
    default_role=Role.FRONTEND,
    params={"payload": "prompt or request body", "timeout": "seconds"},
    summary="One streamed completion, collected into chunks plus a finish reason.",
)
chat = verb(
    "chat",
    grant=Grant.INFER,
    default_role=Role.FRONTEND,
    params={"messages": "chat turns", "tools": "tool schemas", "timeout": "seconds"},
    summary="A chat-completions turn, including tool calls.",
)
probe = verb(
    "probe",
    grant=Grant.INFER,
    default_role=Role.FRONTEND,
    params={
        "prompt": "text",
        "n": "how many requests",
        "unique_prefix": "defeat KV reuse",
    },
    summary=(
        "Several requests with distinct prefixes, grouped by which replica served "
        "them. A fixed prompt lands on one replica through KV affinity and reads "
        "100% healthy or 100% broken, which hides a single bad replica."
    ),
)

# ---------------------------------------------------- direct-to-component reads

admin = verb(
    "admin",
    grant=Grant.READ,
    params={"route": "path", "method": "HTTP verb", "body": "request body"},
    summary="A request to a component's own system port.",
)
metrics = verb(
    "metrics",
    receiver=Receiver.JUDGE,
    phase=Phase.CHECK,
    grant=Grant.READ,
    summary="Metric families scraped from a role. A pure reader: returns data, gates nothing.",
)
logs = verb(
    "logs",
    receiver=Receiver.JUDGE,
    phase=Phase.CHECK,
    grant=Grant.READ,
    params={"since": "cursor", "previous": "the pre-restart container"},
    summary="A log slice for a role. A pure reader.",
)
exec_in = verb(
    "exec_in",
    grant=Grant.READ,
    params={"argv": "command to run", "timeout": "seconds"},
    summary="Run a command inside a replica.",
)

# ------------------------------------------------------------------- waiting

wait = verb(
    "wait",
    grant=Grant.READ,
    takes_selector=False,
    params={"seconds": "how long"},
    summary="Advance the timeline.",
    aliases=("Wait",),
)
wait_ready = verb(
    "wait_ready",
    phase=Phase.ARRANGE,
    grant=Grant.READ,
    params={"timeout": "seconds"},
    summary="Wait for replicas to report ready.",
    aliases=("WaitForStablePods",),
)
wait_serving = verb(
    "wait_serving",
    phase=Phase.ARRANGE,
    grant=Grant.READ,
    default_role=Role.FRONTEND,
    params={"timeout": "seconds", "model": "expected model id"},
    summary="Wait until the endpoint actually serves the model, not merely until pods are ready.",
    aliases=("WaitForModelReady",),
)
wait_log = verb(
    "wait_log",
    grant=Grant.READ,
    params={"pattern": "regex", "timeout": "seconds"},
    summary="Wait for a log line.",
    aliases=("WaitForLogPattern",),
)
wait_stable = verb(
    "wait_stable",
    grant=Grant.READ,
    params={"seconds": "quiet period", "timeout": "seconds"},
    summary="Wait until replica count and restart count stop changing.",
    aliases=("WaitForRecovery",),
)

# ----------------------------------------------------------------- lifecycle

start = verb("start", grant=Grant.LIFECYCLE, summary="Start a role's replicas.")
stop = verb(
    "stop",
    grant=Grant.LIFECYCLE,
    params={"graceful": "send SIGTERM first", "grace": "seconds before SIGKILL"},
    summary="Stop a role's replicas. Scoped to the role — never the whole deployment.",
)
restart = verb(
    "restart",
    grant=Grant.LIFECYCLE,
    params={"settings": "flags to change on the way back up"},
    summary="Restart a role, optionally with different flags.",
)
scale = verb(
    "scale",
    grant=Grant.LIFECYCLE,
    params={"n": "replica count"},
    summary="Change a role's replica count.",
)
reconfigure = verb(
    "reconfigure",
    grant=Grant.LIFECYCLE,
    params={"settings": "live-updatable settings"},
    summary="Change settings in place, with no restart.",
    aliases=("SetBusyThreshold",),
)
rolling_replace = verb(
    "rolling_replace",
    grant=Grant.LIFECYCLE,
    params={"image": "new image", "max_unavailable": "replicas down at once"},
    summary="Replace replicas one at a time.",
    aliases=("RollingReplace", "RollingUpgrade"),
)

# -------------------------------------------------------------------- faults
#
# Every fault declares what its effect proves. A fault whose effect cannot be
# observed is indistinguishable from one that silently did nothing, and a test
# built on that passes for the wrong reason.

kill_process = verb(
    "kill_process",
    grant=Grant.FAULT,
    params={"signal": "signal number"},
    proves=("process_exited", "restart_count_increased"),
    summary="Signal a process inside a replica.",
    aliases=("TerminateProcess",),
)
stall_process = verb(
    "stall_process",
    grant=Grant.FAULT,
    params={"seconds": "how long, or None to hold until released"},
    proves=("process_stopped", "process_resumed"),
    summary="SIGSTOP a process, then resume it.",
    aliases=("StallProcess",),
)
delete_replica = verb(
    "delete_replica",
    grant=Grant.FAULT,
    params={"force": "skip graceful termination"},
    proves=("replica_deleted", "replica_replaced"),
    summary="Delete a replica outright.",
    aliases=("DeletePod",),
)
partition = verb(
    "partition",
    grant=Grant.FAULT,
    params={
        "peer": "the other side",
        "seconds": "duration",
        "flush_conntrack": "drop existing flows",
    },
    proves=("traffic_blocked", "traffic_restored"),
    summary="Block traffic between two roles.",
    aliases=("NetworkPartition",),
)
reset_connections = verb(
    "reset_connections",
    grant=Grant.FAULT,
    params={"count": "how many RSTs", "from_inside": "originate inside the pod"},
    proves=("connections_reset",),
    summary="Force TCP resets against a role.",
    aliases=("RstInjection", "RstFromInsidePod"),
)
restart_infra = verb(
    "restart_infra",
    grant=Grant.INFRA,
    takes_selector=False,
    params={"what": "nats or etcd"},
    summary=(
        "Restart shared infrastructure. Separate from lifecycle because in a "
        "shared namespace this breaks tests that are not yours."
    ),
)
inject = verb(
    "inject",
    grant=Grant.FAULT,
    params={"backend": "fault backend", "kind": "fault kind"},
    proves=("fault_armed", "fault_observed"),
    summary="A fault delegated to an external injector, such as a GPU Xid.",
    aliases=("CudaFaultInjection", "UpstreamGpuXidInjection"),
)
arm = verb(
    "arm",
    phase=Phase.ARRANGE,
    grant=Grant.FAULT,
    params={"fault": "what to arm"},
    proves=("fault_armed",),
    summary="Prepare a fault before readiness, for faults that must exist at start-up.",
    aliases=("PrepareCudaFaultInjection",),
)

# ------------------------------------------------------------------ producers

capture = verb(
    "capture",
    phase=Phase.COLLECT,
    grant=Grant.READ,
    takes_selector=False,
    params={"what": "evidence kind", "tag": "label for the artifact"},
    summary="Collect evidence into the bundle.",
    aliases=("CaptureMetrics", "PrintProcessTree"),
)
monitor = verb(
    "monitor",
    grant=Grant.READ,
    params={"interval": "seconds between samples", "include": "which resources"},
    summary="Sample resource usage in the background for the rest of the timeline.",
    aliases=("ResourceMonitor", "PeriodicSnapshot"),
)
run_command = verb(
    "run_command",
    grant=Grant.READ,
    takes_selector=False,
    params={"argv": "command", "timeout": "seconds"},
    summary="Run a command on the test host, recorded in the timeline.",
    aliases=("RunCommand",),
)

# ----------------------------------------------------------------------- load

start_load = verb(
    "start_load",
    grant=Grant.INFER,
    default_role=Role.FRONTEND,
    params={"workload": "what to send", "name": "handle for later reference"},
    summary="Begin a background load generator.",
    aliases=("StartLoad",),
)
stop_load = verb(
    "stop_load",
    grant=Grant.INFER,
    takes_selector=False,
    params={"name": "which load"},
    summary="Stop a background load generator.",
    aliases=("StopLoad",),
)
await_load = verb(
    "await_load",
    grant=Grant.READ,
    takes_selector=False,
    params={"name": "which load", "timeout": "seconds"},
    summary="Wait for a load generator to finish.",
    aliases=("WaitForLoadCompletion",),
)

# ------------------------------------------------- in-timeline ACT assertions
#
# require_* raises immediately and records. Distinct from expect_*, which is
# evaluated after the fact against collected evidence.

require_restarted = verb(
    "require_restarted",
    grant=Grant.READ,
    params={"since": "handle to compare against", "within": "seconds"},
    summary="Assert now that a role restarted since a point in the timeline.",
    aliases=("AssertPodsRestarted",),
)
require_replaced = verb(
    "require_replaced",
    grant=Grant.READ,
    params={"since": "handle to compare against", "within": "seconds"},
    summary="Assert now that replicas were replaced, not merely restarted.",
)

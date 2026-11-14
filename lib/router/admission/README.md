<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-admission

Admit, hold, or reject now. The router's admission stage: the per-class policy queue (`PolicyQueue`, `QueueRejection`, `QueueLimitKind`, `QueueSnapshot`, stats), the placement a held request is waiting for (`WorkerPlacement`), the resolved profiles (`PolicyProfile`, one flat class table built through `PolicyProfile::from_classes`; `PolicyClassConfig` with its queue policy, TTFT budget and controlled delay; `RouterQueuePolicy`; `UnknownPolicyClassError` for a class name the profile does not know), and the classifier side: the plugin API (`RequestClassifier`, `ClassifyRequest`, `ClassifyEvent`, `RequestClassifierContext`, `RequestProgress`), the `WorkerView` a classifier reads workers through, the `RequestClassifierRuntime` that runs the plugin under panic capture and shutdown, and the `RequestLifecycle` state machine. `lib/router/kv-router` re-exports the crate at `plugins::request_classifier` and `scheduling::{policy_queue, policy_config, request_classifier}`, and keeps the kernel side: the YAML policy file and its resolver, the plugin registry, the gate over the slot projection, the `WorkerView` implementation over worker selection's rows, and the conversion of `ClassificationError` into its own error.

Dependencies: `dynamo-router-protocol`, `dynamo-slot-tracker` (for the fullness rule, `RankLoad` and `BusyThresholds`, which the slot tracker owns), and `dynamo-session-affinity` (for `WorkerAffinityTarget`, the target a session binds to) and nothing router-side beyond them; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. Tokio stays for the queue's instants and the classifier's delivery task; the stage rule forbids sockets and HTTP, not async.

See `lib/router/README.md` for how the crates fit together.

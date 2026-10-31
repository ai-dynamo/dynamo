<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-admission

Admit, hold, or reject now. The router's admission stage: the per-class policy queue (`PolicyQueue`, `QueueRejection`, `QueueLimitKind`, `QueueSnapshot`, stats), the placement a held request is waiting for (`WorkerPlacement`), the resolved profiles (`PolicyProfile`, `PolicyClassConfig`, `RouterQueuePolicy`, built through a validated `FamilyBucketTable`), and the classifier side: the plugin API (`RequestClassifier`, `ClassifyRequest`, `ClassifyEvent`, `RequestClassifierContext`, `RequestProgress`), the `WorkerView` a classifier reads workers through, the `RequestClassifierRuntime` that runs the plugin under panic capture and shutdown, and the `RequestLifecycle` state machine. `lib/kv-router` re-exports the crate at `plugins::request_classifier` and `scheduling::{policy_queue, policy_config, request_classifier}`, and keeps the kernel side: the YAML policy file and its resolver, the plugin registry, the gate over the slot projection, the `WorkerView` implementation over worker selection's rows, and the conversion of `ClassificationError` into its own error.

Dependencies: `dynamo-router-protocol` and nothing router-side beyond it; no axum, reqwest, zmq, Prometheus, or `dynamo-runtime` under any feature. Tokio stays for the queue's instants and the classifier's delivery task; the stage rule forbids sockets and HTTP, not async.

Part of the router split (see `lib/router/README.md`).

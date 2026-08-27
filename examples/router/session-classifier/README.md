<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Session Classifier

This standalone crate implements Dynamo's `RequestClassifier` interface. Within one router process, it permits one active request per session while requests from different sessions proceed independently. Deployments with multiple router replicas need an external coordination mechanism to enforce the same limit across replicas.

`classify()` remains pending when the same session already has an active request. A `Completed` or `Aborted` lifecycle event releases the session and wakes its next request. Requests without a request ID or session metadata pass through immediately.

The router invokes the classifier for schedulable requests. Advisory query-only worker-selection probes bypass it because they do not own a request lifecycle.

The classifier only controls when a request may continue. It does not choose a worker or implement affinity; a worker selector can independently implement session stickiness.

## Configure the classifier

Create the classifier in the frontend's router construction path:

```rust
use dynamo_session_classifier_example::SessionClassifier;

let classifier = SessionClassifier::default();
```

Pass the classifier to [`KvRouter::new_with_request_classifier`](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_router.rs), then drive the router through [`RoutingHost`](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_router/routing_host.rs) so response lifecycle events return to the classifier. The ordinary `KvRouter::new` path has no classifier and retains its existing behavior.

## Test the example

From the Dynamo repository root, run:

```bash
cargo test -p dynamo-session-classifier-example
```

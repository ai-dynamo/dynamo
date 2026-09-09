<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overload-Aware Soft Pinning

The `soft-pin-repin` policy keeps a session on its current worker until that worker exceeds an active-request threshold. It then selects the least-loaded alternative and lets Dynamo store the selected worker as the new soft binding.

The policy implements only `WorkerPicker`. It requests `WorkerInputs::LOAD`; Dynamo continues to own worker discovery, eligibility, reservations, accounting, dispatch, and session binding state.

## Policy Behavior

| Affinity state | Decision |
|---|---|
| No target | Select the worker with the fewest active requests |
| Eligible target at or below `max_active_requests` | Retain the target |
| Eligible target above `max_active_requests` | Select the least-loaded non-target worker |
| Overloaded target with no alternative | Retain the target |
| Target absent from the eligible set | Select the worker with the fewest active requests |

Load ties use `WorkerWithDpRank`, so selection does not depend on Dynamo's unspecified candidate-row order. A target without a data-parallel rank matches every rank on that worker; the policy retains its least-loaded matching rank.

## Configuration

[`worker-selection.yaml`](worker-selection.yaml) configures a threshold of zero for the Mocker demonstration:

```yaml
worker_selection:
  aggregated: soft-pin-repin
  instances:
    - name: soft-pin-repin
      type: soft-pin-repin
      parameters:
        max_active_requests: 0
```

The threshold is inclusive. With `max_active_requests: 0`, an idle target remains pinned and any in-flight request makes that target eligible for repinning.

Build the Python extension against the example catalog before starting the frontend. Follow [Run With the Python Frontend](../README.md#run-with-the-python-frontend) for the catalog-link command.

## Run With Two Mockers

Run each command from the Dynamo repository root in its own terminal. These source-development commands use Dynamo's private Mocker launcher because Mocker is not currently exposed as a public CLI.

### 1. Start the Frontend

```bash
DYN_ROUTER_WORKER_SELECTION_POLICY=soft-pin-repin \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config examples/router/custom-policy-example/soft-pin-repin/worker-selection.yaml \
  --router-session-affinity-ttl-secs 60 \
  --router-session-affinity-mode soft \
  --discovery-backend file \
  --http-port 8000
```

### 2. Start Two Mocker Workers

```bash
python3 -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --decode-speedup-ratio 0.1 \
  --num-workers 2
```

The decode slowdown leaves enough time to send a second request while the first request remains active.

### 3. Establish and Hold the Initial Pin

Start a 128-token streaming request with session ID `soft-pin-repin-example`:

```bash
curl --fail --silent --show-error --no-buffer \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: soft-pin-repin-example' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hold the initial soft pin"}],
    "max_tokens": 128,
    "stream": true,
    "nvext": {"extra_fields": ["worker_id"]}
  }'
```

Wait for the first `data:` line, then read `nvext.worker_id.decode_worker_id` as worker A. Leave the request running while you send the next request.

### 4. Cross the Threshold

In another terminal, send a second request with the same session ID:

```bash
curl --fail --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: soft-pin-repin-example' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "repin the overloaded session"}],
    "max_tokens": 4,
    "stream": false,
    "nvext": {"extra_fields": ["worker_id"]}
  }' | jq --exit-status --raw-output '.nvext.worker_id.decode_worker_id'
```

The first request gives A one active request, which exceeds the configured threshold of zero. The policy selects worker B, and Dynamo stores B as the new soft binding.

### 5. Verify the New Pin

After the first request finishes, send a third request with the same session ID:

```bash
curl --fail --silent --show-error \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Session-ID: soft-pin-repin-example' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "retain the new soft pin"}],
    "max_tokens": 4,
    "stream": false,
    "nvext": {"extra_fields": ["worker_id"]}
  }' | jq --exit-status --raw-output '.nvext.worker_id.decode_worker_id'
```

The worker sequence must be:

```text
A -> B -> B
```

The second request proves that the picker can move an overloaded soft target. The third request proves that Dynamo updated the binding to B and that the policy retained B after its load drained.

## Test the Policy

```bash
cargo test -p dynamo-custom-policy-example-soft-pin-repin
```

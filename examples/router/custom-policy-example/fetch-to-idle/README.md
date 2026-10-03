<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fetch to an Idle Worker

The `fetch-to-idle` policy keeps a request on the worker that already caches its prompt prefix. When that worker is busy, the policy sends the request to the least-loaded worker instead, and that worker fetches the prefix from the busy one rather than computing it again.

The policy has two parts:

- A `WorkerPicker` chooses the worker. It requests `WorkerInputs::CACHE` and `WorkerInputs::LOAD`.
- An experimental `KvTransferPolicy` decides whether the chosen worker fetches KV, and from which source. Dynamo builds the `kv.fetch` hint from that choice.

> [!WARNING]
> The fetch policy uses `plugins::worker_selection::experimental`. That API can change in any release.

## Policy Behavior

The picker counts a worker's cached prefix as its device plus host-pinned overlap. The holder is the worker with the most cached blocks.

| Situation | Picker decision |
|---|---|
| The holder has fewer than `busy_active_requests` active requests | Select the holder |
| The holder has at least `busy_active_requests` active requests | Select the worker with the fewest active requests |
| No worker caches any of the prefix | Select the worker with the fewest active requests |

Ties go to the lower worker ID, so the choice does not depend on candidate-row order.

After the picker runs, Dynamo calls the fetch policy only if another worker or KV pool holds a longer prefix than the chosen worker. The fetch policy receives every eligible source, longest prefix first.

| Situation | Fetch decision |
|---|---|
| The longest source adds fewer than `min_fetch_blocks` beyond the chosen worker's prefix | `Skip`: no hint, the worker computes the prefix |
| Several sources hold the longest prefix and one is a KV pool | `FetchFrom` the KV pool, so a busy worker's GPU does not serve the transfer |
| Otherwise | `FetchFrom(0)`: the longest source |

## Configuration

[`worker-selection.yaml`](worker-selection.yaml):

```yaml
worker_selection:
  aggregated: fetch-to-idle
  instances:
    - name: fetch-to-idle
      type: fetch-to-idle
      parameters:
        busy_active_requests: 1
        min_fetch_blocks: 64
```

| Parameter | Meaning |
|---|---|
| `busy_active_requests` | A holder with at least this many active requests is busy. Must be at least 1 |
| `min_fetch_blocks` | The smallest number of KV blocks worth fetching. Defaults to 0, which fetches whenever a source has more |

Build the Python extension against the example catalog before starting the frontend. Follow [Run With the Python Frontend](../README.md#run-with-the-python-frontend) for the catalog-link command, then start the frontend with `--router-policy-config` pointing at this file.

## Requirements

A fetch only happens when the workers can consume `kv.fetch` hints and the router keeps the KV block chain. The [KVCR deployment examples](../../../backends/vllm/deploy/kvcr/README.md) meet both requirements. With other workers, the picker still works and the fetch policy is never called.

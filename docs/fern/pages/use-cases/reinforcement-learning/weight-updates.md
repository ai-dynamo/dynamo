---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Distribute and Update Rollout Weights
subtitle: Move policy weights into the fleet, then coordinate refresh and recovery
---

A live policy refresh is more than tensor transfer. The RL framework must select the target workers, gate generation, apply one policy, clear stale cache state, verify readiness, and decide when new rollouts can begin. Dynamo exposes backend controls but does not provide a fleet-wide atomic update.

## Choose a ModelExpress Source

[ModelExpress](https://github.com/ai-dynamo/modelexpress/tree/main/modelexpress_client/python/modelexpress_rl) can move a policy from a trainer, object storage, or another inference worker. Its RL refit client separates transfer from installation so an inference worker can stage a version, apply it at an orchestrator-selected safe point, and then publish that applied version as a compatible peer source.

ModelExpress can stage a version from three sources:

- **Trainer to inference:** trainer ranks publish the GPU shards they already own and each inference rank pulls the ranges required by its own layout over NIXL.
- **Artifact to inference:** an inference worker prepares a canonical checkpoint from S3, including the current exact-base XOR delta format, before engine installation.
- **Inference to inference:** after an inference worker applies a version, a rank-compatible worker can pull that version from it instead of returning to the trainer.

All three refit paths are Experimental. The [ModelExpress RL refit package](https://github.com/ai-dynamo/modelexpress/tree/main/modelexpress_client/python/modelexpress_rl) contains the current source strategies. The [Dynamo vLLM refit example](https://github.com/ai-dynamo/modelexpress/tree/main/examples/rl/dynamo_vllm_refit) validates the full-weight trainer-to-inference path only; it does not qualify the S3 delta path or every backend and topology. ModelExpress startup loading remains a separate boot and scale-out workflow.

> [!NOTE]
> ModelExpress startup loading and live RL refit have different interfaces. vLLM 0.23 and later use the native `--load-format modelexpress`; `mx` remains a backward-compatible alias. Earlier plugin-based images generally use `mx`, while legacy split-loader images can expose `mx-source` and `mx-target`. Match the loader to the installed vLLM, ModelExpress package, and runtime image instead of inferring it from the refit API. See [Dynamo deprecations](../../reference/general/releases/deprecations.mdx) for the Dynamo-owned loader migration.

## Choose the Update Path

| Path | Transfer and control | Current boundary |
|---|---|---|
| ModelExpress trainer source | Trainer publication plus receiver-driven NIXL staging and engine apply | Experimental. Prove source ownership, layout conversion, installation, and safe-point orchestration for the exact framework/backend pair. |
| ModelExpress inference peer | An already-updated, rank-compatible inference worker republishes the version for P2P staging | Experimental. Source selection is a transfer optimization, not proof that the source or target may serve. |
| ModelExpress S3 artifact | Canonical full checkpoint or exact-base XOR delta is prepared locally and applied by the engine adapter | Experimental and S3-specific. The required base and artifact identity must match. |
| verl | Recipe-owned Ray/ZMQ control and colocated CUDA IPC | Follow the recipe's trainer/rollout rank mapping. |
| NeMo RL | Framework-owned NCCL sender and fixed worker URLs | Managed Slurm/Ray vLLM path only. |
| Dynamo vLLM from disk | `update_weights_from_disk` through each worker's `system_url` | Per-worker pause, apply, cache reset, and version; no fleet transaction. |
| Dynamo vLLM distributed | Group lifecycle plus `update_weights_from_distributed` | Request body, rank map, and transport remain integration-specific. |
| Dynamo SGLang | Fixed `/engine/control/update_weights_from_*` routes or allowlisted methods | The integration must obtain each SGLang system URL separately. |

The transport name does not determine compatibility. Record checkpoint format, source and destination parallel layouts, rank mapping, dtype, group membership, resharding, network transport, and failure behavior.

## Know the Current Contract Boundaries

| Concern | Current contract |
|---|---|
| Fleet-wide atomic update | Dynamo and ModelExpress expose per-worker and per-version building blocks, not one cross-backend fleet transaction. The framework gates generation and decides the success set. |
| Status and cancellation | ModelExpress weight versions have `STAGING`, `READY`, and `RELEASING` lifecycle state. Changing or deleting a version does not cancel an engine installation already in progress or prove fleet readiness. |
| Delta reconstruction | The current ModelExpress generator path reconstructs its canonical XOR delta from an exact S3 base. Delta refit is not a generic Dynamo capability or an implicit property of NIXL and P2P transfer. |
| Quantization and derived state | The inference adapter and engine own conversion, scales, fused parameters, compiled-graph storage, and post-load processing. Validate the exact model and engine path. |
| Replacement-worker admission | ModelExpress can help a new worker obtain a version; the framework or deployment decides which version it must load and when it may join the rollout pool. |
| Per-token policy identity | The current shared serving response does not attach a policy-version identity to every generated token. Preserve request, attempt, target-version, and worker evidence in the orchestrator. |

## Use the Shared Refresh Lifecycle

Follow the canonical [policy-refresh lifecycle](integration-reference.md#coordinate-policy-refresh) for every transfer path. Persist the target policy identity and every worker result; do not infer fleet success from one worker, one HTTP 200 response, or one trainer send. The sections below describe only the backend- and framework-specific differences.

## Update a vLLM Worker from Disk

Start the RL listener and worker as described in [RL Integration Reference](integration-reference.md#discover-workers-safely). Discover workers with `GET /v1/rl/workers`, require protocol version `1`, and select only workers that advertise `pause_generation`, `update_weights_from_disk`, `get_weight_version`, and `resume_generation`.

Use each returned `system_url`. The following shape updates one worker and keeps it paused if validation fails:

```bash
set -euo pipefail

WORKER_URL=http://10.0.0.12:8081
TARGET_VERSION=step-42
TARGET_PATH=/models/checkpoint-42

curl --fail-with-body "$WORKER_URL/engine/pause_generation" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"wait","clear_cache":false}' | jq -e '.status == "ok"'

curl --fail-with-body "$WORKER_URL/engine/update_weights_from_disk" \
  -H 'Content-Type: application/json' \
  -d "{\"model_path\":\"$TARGET_PATH\",\"weight_version\":\"$TARGET_VERSION\"}" \
  | jq -e --arg version "$TARGET_VERSION" '.status == "ok" and .version == $version'

curl --fail-with-body "$WORKER_URL/engine/get_weight_version" \
  -H 'Content-Type: application/json' \
  -d '{}' | jq -e --arg version "$TARGET_VERSION" '.status == "ok" and .version == $version'

curl --fail-with-body "$WORKER_URL/engine/resume_generation" \
  -H 'Content-Type: application/json' \
  -d '{}' | jq -e '.status == "ok"'
```

Repeat the operation under one framework-owned barrier for the complete target set. The version is caller-supplied metadata, not a tensor digest; pair it with update success, cache handling, and post-update generation.

For distributed vLLM updates, use the advertised group lifecycle and distributed-update routes only with the exact request schema and rank mapping validated by the integration. Treat group-initialization timeout as worker failure because the backend process can remain blocked.

## Update an SGLang Worker from Disk

**Experimental.** SGLang workers do not currently register with `GET /v1/rl/workers`. Set `DYN_SYSTEM_PORT` on each worker and obtain its trusted system-server URL from the framework or deployment. Do not derive that URL from the shared frontend address.

SGLang exposes fixed weight-update routes under `/engine/control/*`. Generation pause and continue methods must be explicitly allowlisted when the integration needs a framework-owned update barrier:

```bash
export DYN_SGLANG_ENGINE_ROUTES="pause_generation:tm continue_generation:tm"
DYN_SYSTEM_PORT=8081 python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B
```

After the framework gates new rollout work, update one worker with the request schema supported by the installed SGLang version:

```bash
set -euo pipefail

WORKER_URL=http://10.0.0.12:8081
TARGET_VERSION=step-42
TARGET_PATH=/models/checkpoint-42

curl --fail-with-body "$WORKER_URL/engine/pause_generation" \
  -H 'Content-Type: application/json' \
  -d '{}' | jq -e '.status == "ok"'

curl --fail-with-body "$WORKER_URL/engine/control/update_weights_from_disk" \
  -H 'Content-Type: application/json' \
  -d "{\"model_path\":\"$TARGET_PATH\",\"weight_version\":\"$TARGET_VERSION\",\"flush_cache\":true,\"keep_pause\":true}" \
  | jq -e '.success == true'

curl --fail-with-body "$WORKER_URL/engine/continue_generation" \
  -H 'Content-Type: application/json' \
  -d '{"torch_empty_cache":false}' | jq -e '.status == "ok"'
```

Repeat this sequence across the framework-selected worker set under one barrier. The fixed update response reports `success`, `message`, and paused-request count, but SGLang does not currently expose the vLLM discovery and `get_weight_version` contract. Keep the target version in the orchestrator, check every response body, and require post-update generation before admitting the worker.

The built-in disk update can flush SGLang's local cache. Validate any additional host, disk, or shared cache tier separately. Distributed, tensor, and IPC update routes use the request schemas of the installed SGLang version; the [SGLang engine-route reference](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md#engine-routes) documents the fixed routes and explicit method allowlist.

## verl Colocated Update

The public verl recipe sends generation through Dynamo but keeps sleep, wake, and weight transfer in recipe-owned Ray actors and a ZMQ/CUDA IPC bridge. Do not replace this path with public worker discovery unless the integration itself changes.

Verify that every data-parallel shard receives the same trainer step, old cache state is handled, and every worker resumes before post-update rollout generation. See [verl Integration](verl.md#verify-the-run).

## NeMo RL Managed Update

NeMo RL records a fixed vLLM fleet, creates one trainer-plus-inference NCCL world, drains generation, applies the checkpoint to every engine, clears cache state in a separate pause phase, and resumes only after the framework collects all results.

This path prevents a dead or replaced worker from silently joining with initial weights, but it is not elastic and has no fleet-wide rollback. Keep the rollout phase gated after any worker, refit, cache, or resume failure. See [NeMo RL Integration](nemo-rl.md#policy-refit).

## Keep Cache and Version State Correct

KV entries created under one policy are invalid under another policy even when token IDs are unchanged. Identify every device, host, disk, or shared cache tier and verify how each tier is cleared before generation resumes. Separate required cold-cache warm-up from steady-state measurements.

Use an immutable checkpoint ID or digest as the target policy identity. Record the framework step, intended worker set, previous and target versions, transfer and cache-reset timing, readiness, and post-update request. A readable version string alone does not prove which tensors are resident, and the current router does not select workers by that value.

## Handle Partial Failure

| Failure | Safe default |
|---|---|
| Membership changes before transfer | Refresh the target set and rebuild any distributed group. |
| Pause fails | Do not transfer to or admit the worker. |
| One transfer fails after peers succeed | Keep the fleet gated; retry, replace, or roll back under one explicit policy. |
| Cache reset fails | Exclude the worker even if transfer succeeded. |
| Version check differs | Do not reopen a synchronous fleet. |
| Resume or post-update generation fails | Keep the worker out of the target pool and preserve diagnostics. |

Always check both HTTP status and the backend-specific response body. A practical rollback can require reapplying the previous checkpoint or replacing the worker; test that path before enabling asynchronous updates.

## Validate the Complete Update

Record the target policy, selected worker set, request gate, transfer parameters, per-worker apply and cache results, version and liveness checks, post-update generation, and update duration. Inject at least one missing-worker, transfer, and post-update failure.

Do not call a path supported when only transfer bandwidth or one-worker success was measured. Readiness, cache correctness, failure recovery, and useful post-update rollout generation are part of the contract.

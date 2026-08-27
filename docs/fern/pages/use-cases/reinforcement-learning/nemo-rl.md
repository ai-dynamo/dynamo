---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Integrate with NeMo RL
subtitle: Run NeMo RL's managed Dynamo vLLM backend on Slurm
---

**Experimental.** NeMo RL includes a managed Dynamo generation backend with a pinned runtime and dedicated GPU functional test. NeMo RL launches and owns a fixed Dynamo vLLM fleet inside its Slurm/Ray allocation; it does not connect to an existing Dynamo deployment or require Kubernetes.

## Integration Shape

| Dimension | Current path |
|---|---|
| Runtime | NeMo RL-managed Slurm allocation and Ray virtual cluster |
| Backend | `dynamo.vllm` with BF16 generation |
| Placement | Training and generation are not colocated |
| Routing | Managed Dynamo frontend; the example uses KV-aware routing |
| Policy update | NeMo RL NCCL sender to a fixed vLLM fleet |
| Supported engine layout | Each tensor-parallel × pipeline-parallel engine group fits on one node |

The reviewed NeMo RL integration pins `ai-dynamo[vllm]==1.3.0.post1` and its compatible vLLM environment. Do not replace that runtime with current Dynamo `main` or a newer wheel without rerunning the functional and training checks.

## Prerequisites

- A Slurm site supported by NeMo RL's Ray launcher
- A full-node allocation with at least two GPUs available to the recipe
- A container registry and image-conversion path readable by the Slurm site
- Model, data, results, and container paths shared where required by the allocation

## Build the Runtime

Clone the reviewed NeMo RL source and build its opt-in Dynamo layer:

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git
git -C RL checkout 6ae035784fe40fd9c9e31d27fffa4a403243a0bd
cd RL

export IMAGE=registry.example.com/nemo-rl:dynamo-6ae03578
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg BUILD_DYNAMO=1 \
  --target release \
  --file docker/Dockerfile \
  --tag "$IMAGE" \
  --push \
  .
```

Replace the registry with one available to your site and record the resolved image digest. Convert the image to the format expected by the Slurm environment using the site's normal NeMo RL workflow.

## Configure the Backend

Start from the pinned `examples/configs/grpo_math_1B_dynamo.yaml`. Its essential generation settings are:

```yaml
policy:
  generation:
    backend: dynamo
    dynamo_cfg:
      engine: vllm
      frontend_args:
        router_mode: kv
    vllm_cfg:
      tensor_parallel_size: 1
      pipeline_parallel_size: 1
      expert_parallel_size: 1
      precision: bfloat16
      kv_cache_dtype: auto
    colocated:
      enabled: false
      resources:
        gpus_per_node: 1
        num_nodes: 1
```

NeMo RL validates which vLLM options are translated, managed, unsupported, or ignored by the Dynamo backend. Treat configuration warnings and errors as contract checks rather than assuming every normal vLLM field reaches `dynamo.vllm`.

## Run the Training Smoke

Set the site-specific allocation values and submit the pinned two-step recipe from the NeMo RL repository root:

```bash
export CONTAINER=/shared/images/nemo-rl-dynamo-6ae03578.sqsh
export MOUNTS="$PWD:$PWD"
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=your-partition
export GPUS_PER_NODE=8
export BASE_LOG_DIR="$PWD/results/dynamo-smoke/logs"
printf -v COMMAND '%q ' \
  /opt/nemo_rl_venv/bin/python -u "$PWD/examples/run_grpo.py" \
  --config "$PWD/examples/configs/grpo_math_1B_dynamo.yaml"
export COMMAND

sbatch \
  --nodes=1 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  ray.sub
```

Set `GPUS_PER_NODE` to the physical GPU count expected by the partition. The launcher requests an exclusive full node even though the small recipe uses one training GPU and one generation GPU.

The upstream functional entry point is:

```bash
uv run --no-sync bash tests/functional/grpo_dynamo.sh
```

Run it only in the purpose-built Dynamo image on a compatible allocation. A passing result covers the pinned configuration, not other models, topologies, Dynamo versions, or Slurm environments.

## Verify the Run

### Token Correctness

Direct GRPO sends token-ID prompts to `/v1/completions` and consumes returned completion token IDs and log probabilities. NeMo Gym uses a local chat wrapper with `nvext.token_data`. Validate both paths separately when your workload uses both; missing token IDs, missing log probabilities, or mismatched lengths must fail the sample.

### Policy Refit

NeMo RL fixes worker membership, creates a trainer-plus-inference NCCL world, drains generation, applies the target checkpoint to each worker, clears stale cache state, and resumes the fleet. Verify that every worker completes the refit and cache barrier before post-update generation begins. A per-worker success is not a fleet transaction, and this path does not replace failed workers in place.

### Routing and Telemetry

Compare the example's `kv` router with `round-robin` while holding prompts, concurrency, engine count, update cadence, and cache-reset behavior fixed. NeMo RL can poll per-worker Dynamo and vLLM metrics, but the current integration does not provide a lossless rollout-to-Dynamo request identity. Keep trainer step, rollout, attempt, target policy, and accepted sample identity in NeMo RL records.

## Troubleshoot

| Symptom | Check first |
|---|---|
| Frontend never becomes ready | etcd/NATS health, worker exits, expected registration count, and `/v1/models` |
| Training cannot score a completion | Returned token IDs, logprob lengths, tokenizer identity, and response adapter |
| Refit hangs or fails | Fixed worker list, NCCL world geometry, vLLM patch, and first failed worker result |
| Post-update output is inconsistent | Refit count, cache invalidation, pause/resume failures, and post-update sample |
| Worker exits | Ray actor, GPU reservation, process group, and frontend registration |
| Shutdown leaves processes | Managed teardown, ports, temporary directories, and next-job startup |

Keep new rollout admission gated after any refit or cache-control failure. See [Update Rollout Weights](weight-updates.md#nemo-rl-managed-update) and [Observe and Simulate RL Rollouts](operations-and-simulation.md) for the shared lifecycle and telemetry boundaries.

## Current Limitations

- Managed Slurm/Ray and vLLM only; no external Dynamo fleet, Kubernetes deployment, SGLang, or TensorRT-LLM path
- Fixed, non-colocated fleet; no elastic worker replacement during the update lifecycle
- No general policy-version transaction or automatic rollback
- No current lossless framework rollout-to-Dynamo request join
- Supported status requires an independent reproduction with token correctness, refit, post-update generation, and failure recovery

## Upstream Resources

- [Managed Dynamo generation guide](https://github.com/NVIDIA-NeMo/RL/blob/6ae035784fe40fd9c9e31d27fffa4a403243a0bd/docs/guides/dynamo-generation.md)
- [Two-GPU configuration](https://github.com/NVIDIA-NeMo/RL/blob/6ae035784fe40fd9c9e31d27fffa4a403243a0bd/examples/configs/grpo_math_1B_dynamo.yaml)
- [Dedicated functional test](https://github.com/NVIDIA-NeMo/RL/blob/6ae035784fe40fd9c9e31d27fffa4a403243a0bd/tests/functional/grpo_dynamo.sh)

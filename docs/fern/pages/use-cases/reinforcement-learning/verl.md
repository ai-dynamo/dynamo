---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Integrate with verl
subtitle: Run the public verl-recipe Dynamo rollout backend
---

**Experimental.** verl-recipe provides an asynchronous Dynamo rollout backend with a shared frontend, vLLM workers, routing, and colocated policy updates. Use the upstream recipe as the implementation source of truth; this page covers the shortest Dynamo workflow and its support boundary.

## Integration Shape

| Concern | Current path |
|---|---|
| Training and rollout orchestration | verl and verl-recipe |
| Generation | Dynamo vLLM workers behind one shared frontend |
| Routing | Native Dynamo router when ThunderAgent is disabled; ThunderAgent when explicitly enabled |
| Policy update | Recipe-owned Ray/ZMQ control with colocated CUDA IPC |
| Service discovery | Recipe-managed etcd and NATS |
| Deployment | Local or multi-node Ray processes; Kubernetes is not required |

The native router and ThunderAgent are different scheduling paths. Choose one before building the environment and do not compare their results as if only a router flag changed.

## Prerequisites

- A Linux GPU environment that satisfies the selected verl, Dynamo, vLLM, CUDA, and PyTorch versions
- `git`, Python, `etcd`, and `nats-server`
- Model and dataset paths visible on every participating node
- Enough GPUs for the trainer and rollout layout; the validation smoke is not a full training run

## Prepare the Source

The reviewed recipe snapshot is `461b830c`, and its `REQUIRED_VERL.txt` selects the matching core verl commit. The core commit records an older recipe submodule, so use the installer and then set the nested recipe checkout to the same reviewed snapshot:

```bash
git clone https://github.com/verl-project/verl-recipe.git
git -C verl-recipe checkout 461b830cfee4f5a67c21edc300c24373230babc7

cd verl-recipe
./install_verl.sh --recipe dynamo --method git --dest ../verl

git -C ../verl/recipe fetch origin 461b830cfee4f5a67c21edc300c24373230babc7
git -C ../verl/recipe checkout 461b830cfee4f5a67c21edc300c24373230babc7
test -f ../verl/recipe/dynamo/main_dynamo.py
```

The recipe does not pin a complete Dynamo/vLLM image for the native-router path. Build one clean environment, record its immutable image and package versions, and keep the Dynamo and nested recipe checkouts clean during validation.

## Run the Validation Smoke

Run the upstream validation-only smoke from the resulting verl checkout:

```bash
cd ../verl

MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
TRAIN_FILE=/data/dapo-math-17k.parquet \
TEST_FILE=/data/aime-2024.parquet \
RAY_DATA_HOME=/data/verl \
bash recipe/dynamo/smoke_dynamo_v1.sh
```

The smoke starts recipe-managed etcd, NATS, a Dynamo vLLM worker, and the shared frontend. `PASS: Dynamo validation smoke completed` confirms the validation command completed; it does not prove an optimizer step or policy refresh.

## Run a Training Iteration

After the smoke passes, run at least one optimizer step with the same environment. This example selects the native Dynamo router explicitly:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

python3 -m recipe.dynamo.main_dynamo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/gsm8k/train.parquet \
  data.val_files=/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=dynamo \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
  ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=2
```

Adjust model, data, and resource values for your environment. A passing run must include rollout generation, reward or advantage computation, an actor update, policy synchronization, and generation after the update.

## Verify the Run

Check three boundaries before scaling:

1. **Generation correctness:** The completion token IDs and selected log probabilities consumed by verl match in length and order. Record terminal and canceled attempts separately.
2. **Policy update:** Every intended rollout shard receives the same trainer step through the recipe's CUDA IPC path, stale KV state is handled, and post-update generation succeeds.
3. **Routing:** With ThunderAgent disabled, compare `round-robin` and `kv` using the same prompts, concurrency, cache state, and update cadence. Report useful framework output, not only request throughput.

Set `request_completion_token_ids=true` when the framework must score the exact engine tokens. Use [RL Integration Reference](integration-reference.md#preserve-token-authority) for the shared response checks and [Route RL Rollouts](routing.md) for the routing experiment.

## Observe and Recover

When `enable_worker_system_metrics=true`, the recipe writes worker endpoint files to `VERL_DYNAMO_WORKER_METRICS_DIR`. Run the provided metrics sidecar if you need per-worker JSONL:

```bash
python3 recipe/dynamo/metrics_sidecar.py \
  --endpoints-glob "$VERL_DYNAMO_WORKER_METRICS_DIR/*.endpoints" \
  --output /tmp/verl-dynamo/kv-metrics.jsonl \
  --label dynamo-kv \
  --interval 30
```

| Symptom | Check first |
|---|---|
| Frontend never becomes ready | etcd/NATS health, worker logs, port reachability, and expected registration count |
| Trainer token scores differ | Completion token IDs, logprob alignment, tokenizer/model version, and response adaptation |
| One shard remains stale | Ray actor, ZMQ bridge, CUDA rank mapping, and per-shard update acknowledgements |
| Cache reuse drops after update | Cache reset, worker wake-up, and the warm-up boundary |
| Shutdown leaves processes | Recipe watchdog and frontend, worker, NATS, then etcd teardown |

See [Observe and Simulate RL Rollouts](operations-and-simulation.md) for request tracing and [Update Rollout Weights](weight-updates.md#verl-colocated-update) for the policy-update boundary.

## Current Limitations

- The recipe does not provide one complete native-path Dynamo/vLLM image pin.
- The documented policy update is the recipe's colocated CUDA IPC path, not public Dynamo worker discovery or ModelExpress.
- Multi-node and large-model layouts require separate topology and failure validation.
- Supported status requires an independent run with token correctness, policy refresh, post-update generation, and request, worker, and update recovery.

## Upstream Resources

- [Dynamo rollout backend](https://github.com/verl-project/verl-recipe/tree/461b830cfee4f5a67c21edc300c24373230babc7/dynamo)
- [Required verl version](https://github.com/verl-project/verl-recipe/blob/461b830cfee4f5a67c21edc300c24373230babc7/dynamo/REQUIRED_VERL.txt)
- [Validation smoke](https://github.com/verl-project/verl-recipe/blob/461b830cfee4f5a67c21edc300c24373230babc7/dynamo/smoke_dynamo_v1.sh)

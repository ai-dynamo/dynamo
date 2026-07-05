<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Terminal-Bench 2.1

Pinned Harbor/Terminus-2 harness for the GLM-5.2 NScale comparison.

## Immutable inputs

| Input | Pin |
|---|---|
| Harbor | `v0.17.1`, commit `1afcacf22df82e6b3f36152fcd11bfef4ec96c4f` |
| Dataset | Harbor Hub `terminal-bench/terminal-bench-2-1@6` |
| Dataset content digest | `7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a` |
| Agent | `terminus-2` from the pinned Harbor commit |
| Official run | 89 tasks, 5 attempts each, 445 trials |
| Smoke | first 3 tasks from the pinned dataset, 1 attempt each |

`bootstrap.sh` checks out Harbor at the full commit and runs `uv sync --frozen
--no-dev` against Harbor's checked-in lockfile. It refuses a different remote,
commit, package version, or modified checkout.
Every run repeats a read-only `uv sync --frozen --no-dev --check`, rejects tracked or
untracked checkout changes, and records the canonical installed package inventory and
digest. Resumes must reproduce the same environment identity.
Bootstrap and every real run also resolve revision 6 from Harbor Hub and reject
any content digest, version ID, or task count other than the pinned 89-task set.

## Bootstrap

Requirements: Git, uv, Docker with enough capacity for the selected concurrency,
and Python 3 for the metadata/summarization scripts. Harbor's Python 3.12 runtime
is created by uv.

```bash
cd benchmarks/glm52-nscale/eval/terminalbench
./bootstrap.sh
```

The default installation lives in ignored `.cache/`. Set `HARBOR_SOURCE_DIR` to
put the immutable checkout and virtual environment on a persistent volume.

## Endpoint contract

The endpoint must expose OpenAI Chat Completions and `GET /v1/models`. Pass a base
URL that includes `/v1`. `--model` is the LiteLLM model string; for a custom
OpenAI-compatible endpoint it should use the `openai/` provider prefix.
The endpoint preflight accepts `context_window` or `max_model_len` only when every
advertised non-null alias equals 409,600, then persists canonical `context_window`.

```bash
export OPENAI_API_KEY=EMPTY  # replace only when the endpoint enforces auth
API_BASE=http://glm52-dynamo-vllm-frontend:8000/v1
```

The wrapper verifies that `GET ${API_BASE}/models` advertises
`zai-org/GLM-5.2` before starting. Validation diagnostics may override the API and
LiteLLM names separately with `--served-model` and `--model`; full runs require the
pinned names. The API key is read from the environment and is never written to command
metadata.

The wrapper supplies both the LiteLLM model registration and explicit
`llm_call_kwargs.max_tokens` and `top_p` values. Harbor 0.17.1 uses
`max_output_tokens` for context accounting but does not forward it to Chat
Completions, so the call kwargs are required to reproduce the model-card recipe.

The campaign matches the published GLM-5.2 Terminal-Bench 2.1 / Terminus-2
settings: JSON parser, temperature 1.0, top-p 1.0, 48,000 output tokens, at most
500 episodes, a four-hour task timeout, and a 262,144-token context. Greedy calibration at temperature 0
was rejected after three simultaneous turns decoded continuously for seven
minutes; it was not representative of the published evaluation setup.

## Three-task smoke

Run this through the campaign continuity guard after a `validation` deployment. The
real endpoint must be the exact cluster-DNS service recorded in the active runtime
binding; localhost/port-forward endpoints are diagnostics-only and cannot be imported.

```bash
variant=dynamo-vllm
phase=validation
summary=/artifacts/glm52-nscale/terminalbench/summaries/${variant}/${phase}
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${summary}/runtime-continuity.json" -- \
  bash -lc "/workspace/eval/terminalbench/run-smoke.sh \
    --api-base http://glm52-dynamo-vllm-frontend:8000/v1 \
    --label ${variant} --phase ${phase} \
    --job-name ${variant}-${phase}-terminalbench21-smoke \
    --summary-dir ${summary}"
```

Use labels `vllm-serve`, `dynamo-sglang`, and `sglang-serve` for the remaining
variants. A smoke run is accepted only if the Harbor job records exactly three
unique tasks and three completed trial results.

Inspect the fully resolved Harbor configuration without contacting the endpoint
or launching Docker tasks:

```bash
./run-smoke.sh \
  --api-base http://127.0.0.1:18080/v1 \
  --label dynamo-vllm \
  --phase validation \
  --dry-run
```

## Official 89 x 5 run

Raw Harbor jobs can be large. Put `--jobs-dir` on the NScale artifact PVC and
copy or point `--summary-dir` at the campaign's compact result directory.

```bash
variant=dynamo-vllm
phase=ab
jobs=/artifacts/glm52-nscale/terminalbench/jobs
summary=/artifacts/glm52-nscale/terminalbench/summaries/${variant}/${phase}
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${summary}/runtime-continuity.json" -- \
  bash -lc "/workspace/eval/terminalbench/run-full.sh \
    --api-base http://glm52-dynamo-vllm-frontend:8000/v1 \
    --label ${variant} --phase ${phase} \
    --job-name ${variant}-${phase}-terminalbench21-r6-89x5 \
    --jobs-dir ${jobs} --summary-dir ${summary} --n-concurrent 4"
```

The full wrapper deliberately exposes no task filter or attempt override. It
always executes revision 6 in full with five attempts, zero Harbor retries,
Terminus-2, Docker task environments, task cleanup enabled, and exactly four concurrent
trials. Full-run concurrency is immutable because changing it can alter serving-engine
batching and sampling behavior.
The model/served-model, temperature, top-p, turn limit, context/output limits, and timeout
multiplier are equally immutable in full mode; those CLI overrides are validation-only
diagnostics.

Harbor persists each completed trial. Resume an interrupted run with identical
arguments and the same explicit job name:

```bash
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${summary}/runtime-continuity.json" -- \
  bash -lc "/workspace/eval/terminalbench/run-full.sh \
    --api-base http://glm52-dynamo-vllm-frontend:8000/v1 \
    --label ${variant} --phase ${phase} \
    --job-name ${variant}-${phase}-terminalbench21-r6-89x5 \
    --jobs-dir ${jobs} --summary-dir ${summary} --n-concurrent 4 --resume"
```

Harbor itself rejects a changed job configuration, and the wrapper separately
rejects resume metadata that differs from the original run specification.
A successful Harbor invocation is accepted only after every scoped result is bound to
its pinned package ref and common task checksum, the content-addressed `task.toml` is
parsed, and the requested Docker image resolves locally to a `sha256:` image ID and at
least one immutable RepoDigest. The deterministic evidence is identical across
variants/phases when the actual task images match. Resumes recompute it and refuse to
overwrite different existing evidence.
A nonzero guarded invocation leaves a validated continuity record. The retry archives
that record by content digest before running and writes the successful canonical
`runtime-continuity.json`; an existing successful attestation is never overwritten.

## Artifacts

Each job contains:

- `config.json` and `lock.json`: Harbor's requested and resolved configuration.
- `dataset-metadata.json`: resolved dataset content digest and all 89 immutable
  task package digests.
- `run-metadata.json`: schema-v2 source revision, immutable pins, endpoint model identity,
  exact secret-free argv, verified campaign source tree, lock-checked Harbor package
  inventory, system identity, timings, and input hashes.
- `task-images.json`: schema-v1 per-task package ref/checksum, cached `task.toml`
  hash, requested Docker ref, immutable image ID and RepoDigests, and exact scoped
  task/trial counts attested after a successful Harbor run.
- the runtime-binding envelope inside metadata: canonical deployment/evaluator
  identity with separate deployment and full-content digests.
- one subdirectory per trial with result, verifier, agent, and terminal artifacts.
- `summary/summary.json`: validation, aggregate score, pass@1 through pass@5,
  errors, timing, token counts, hashes, embedded task-image evidence, and per-task
  results. Terminal summary schema v2 requires task-image schema v1.
- `summary/task-images.json`: physical copy of the validated task-image evidence;
  its SHA-256 is `summary.json.input_hashes.task_images_sha256`.
- `summary/tasks.csv`: one row per task with attempts, passes, errors, reward, and
  pass@k.
- `summary/trials.csv`: one row per attempt with reward, failure/error details,
  model identity, timing, token counts, and result hash.
- `summary/runtime-continuity.json`: privacy-safe pre/post controller, pod, image,
  container, and restart identity written by the guarded local launcher.

`summarize.py --strict` runs automatically. It writes the summary even when the
population is incomplete, then exits nonzero unless the observed task count,
trial count, per-task attempts, Harbor total, and Harbor completed count all
match the selected smoke or full mode. A verifier reward of zero is a valid model
failure. A Harbor trial exception or missing verifier reward invalidates the run:
those outcomes usually indicate runner, image, or harness failure and must be
replayed before the result can be promoted as complete.

To regenerate a compact summary from retained raw results:

```bash
python3 summarize.py \
  --job-dir /path/to/jobs/dynamo-vllm-terminalbench21-r6-89x5 \
  --output-dir ../../results/terminalbench/dynamo-vllm \
  --metadata /path/to/jobs/dynamo-vllm-terminalbench21-r6-89x5/run-metadata.json \
  --task-images /path/to/jobs/dynamo-vllm-terminalbench21-r6-89x5/task-images.json \
  --expected-tasks 89 \
  --expected-attempts 5 \
  --strict
```

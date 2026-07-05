<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SWE-bench agent evaluations

This harness runs the same mini-SWE-agent v2.4.4 scaffold against each GLM-5.2
endpoint, then scores its patches with the official pinned evaluators.

## Immutable inputs

| Input | Pin |
| --- | --- |
| mini-SWE-agent | v2.4.4, `4fe36a38941abde8a332bda950ca6c0de653a19f` |
| SWE-bench evaluator | v4.1.0, `726c5461e2ef52d83cf1ea2107870a8bb3328d57` |
| SWE-bench Pro evaluator | `ca10a60a5fcae51e6948ffe1485d4153d421e6c5` |
| Verified dataset | `SWE-bench/SWE-bench_Verified@91aa3ed51b709be6457e12d00300a6a596d4c6a3` |
| Multilingual dataset | `SWE-bench/SWE-bench_Multilingual@2b7aced941b4873e9cad3e76abbae93f481d1beb` |
| Pro dataset | `ScaleAI/SWE-bench_Pro@7ab5114912baf22bb098818e604c02fe7ad2c11f` |

`common.sh` verifies these against `../pins.env` before doing any work. Bootstrap
installs under the 101-package `constraints.lock`, permits exactly the two pinned
editable source checkouts, and fails unless the normalized post-install freeze is an
exact match. Raw lock, raw freeze, and normalized-freeze hashes are bound into each run.

## Setup

The runner needs Python 3.11, `uv`, `git`, a Linux Docker daemon, disk space for task
images, and network access to Hugging Face and Docker Hub.

```bash
export SWEBENCH_WORK_ROOT=/mnt/artifacts/glm52/swebench
./bootstrap.sh
```

Bootstrap materializes all three datasets at their exact Hugging Face revisions and
asserts row counts of 500 Verified, 300 Multilingual, and 731 public Pro tasks.
Every run freezes the live venv again, verifies it against `constraints.lock`, and
stores immutable raw and normalized freeze files in the run directory. A stale
bootstrap-time freeze cannot mask a later package change.

The Pro adapter follows the public scaffold exactly:

```text
problem_statement

Requirements:
requirements

New interfaces introduced:
interface
```

It maps each row's authoritative `dockerhub_tag` to
`docker.io/jefzda/sweap-images:<dockerhub_tag>` for both agent execution and patch
evaluation. It does not reconstruct tags from repository names. Pro generation also
runs from `/app` and clears the images' `/bin/bash` entrypoint so mini-SWE-agent can
start its long-lived task container correctly.

## Run

Use a durable artifact volume for full runs. `run-name` identifies the serving stack
and must be unique within the artifact root.

```bash
export SWEBENCH_WORK_ROOT=/mnt/artifacts/glm52/swebench
export SWEBENCH_RESULTS_ROOT=/mnt/artifacts/glm52/results/swebench
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://glm52-dynamo-vllm-frontend:8000/v1

./run.sh verified dynamo-vllm-ab all --phase ab
./run.sh multilingual dynamo-vllm-ab all --phase ab
./run.sh pro dynamo-vllm-ab all --phase ab
```

Repeat with phase-qualified run names such as `vllm-serve-ab`, `dynamo-sglang-ab`,
and `sglang-serve-ab` after changing only `OPENAI_BASE_URL`. Generation and evaluation
can be resumed independently:

```bash
./run.sh verified dynamo-vllm-ab generate --phase ab
./run.sh verified dynamo-vllm-ab evaluate --phase ab
```

For a smoke test, use a distinct run name and set `INSTANCE_SLICE=0:1`:

```bash
INSTANCE_SLICE=0:1 ./run.sh verified dynamo-vllm-validation all --phase validation
```

The wrapper records the exact filtered/sliced IDs in `run-scope.json`. Resumed phases
reuse that scope when the selector variables are omitted, and reject any attempt to
change a run name's scope. Smoke summaries report `scope: "smoke"`, their target and
excluded dataset counts, and `score_on_scope`; `benchmark_score` is `null` because a
slice is not a full benchmark result. Full runs require the exact dataset ID set and
fail if generation has missing or unexpected IDs, or evaluation has missing,
unexpected, incomplete, or infrastructure-error IDs.

Full runs require phase `ab` or `ba`; filtered/sliced smoke runs require `validation`.
The delimited phase token must appear in the run name. Before any endpoint request, the
harness validates `/artifacts/glm52-nscale/runtime-bindings/<variant>/active.json`
against the requested variant, phase, endpoint, context, model, and runtime image.

Schema-v3 `run-metadata.json` is created read-only before generation and validated before any
existing prediction can be skipped. It binds the run name and suite to the normalized
endpoint, model, exact target scope, merged config digests, evaluator JSONL and dataset
provenance, SWE pins, source lock, and the clean checked-out commits of all three
upstream repositories. It embeds `runtime-binding.json` with separate canonical
`deployment_sha256` and full-envelope `content_sha256` digests. Its
`content.deployment` is the agreed hashed Kubernetes binding, while
`content.evaluator` holds the runtime source/TP contract, exact endpoint evidence, and
fully resolved evaluator config. It also binds the effective generation worker/batch
counts and evaluator worker, timeout, backend, and Docker platform settings.
The top-level `campaign_source` equals the identity inside the runtime envelope's
evaluator content and proves the current `/workspace/campaign.env` plus `eval/` content
hashes and permission modes against the deployment recipe source commit.
The exact `/v1/models` response is retained. The selected served model may advertise
either `context_window` or native-vLLM `max_model_len`, but every present non-null
alias must agree exactly on `409600`; the immutable selected-model identity is
canonicalized to `context_window: 409600`. Missing, conflicting, and 262144-token
values are rejected. Any drift fails closed and requires a new run name. The endpoint
is probed before the metadata is created, so a typo does not reserve a run name.
Predictions found without pre-existing metadata are never adopted as a resumable run.

Every prediction must have a pinned mini-SWE-agent 1.1 trajectory whose instance,
model, submission, clean terminal status, positive API-call count, config evidence,
and final exit message agree with `preds.json`. Missing/corrupt trajectories and
recorded API, container, or runner exceptions are infrastructure errors and cannot be
scored as model failures. An empty patch remains a valid scored model failure only
when that trajectory evidence is complete. Each trajectory's complete effective agent,
environment, model, endpoint, temperature, top-p, 32768-token output limit, and
14400-second wall limit must equal the immutable binding and every other trajectory.

Before each task image is pruned, `task-images.json` records its requested reference,
immutable Docker image ID, and all RepoDigests. Generation completeness requires the
exact scoped instance map and checks each trajectory's environment image against it.
`generation-summary.json` embeds the map and canonical digest, allowing all four stack
variants to be rejected unless they used the same task-image content map.

Evaluation cache keys include a canonical SHA-256 digest of `preds.json`, recorded in
`evaluation-predictions-sha256.txt`. Regenerating any patch therefore cannot reuse a
prior patch's test report. `REDO_EVALUATION=1` additionally forces a fresh evaluator
run for unchanged predictions.

Full runs pin `AGENT_WORKERS=16`, `GENERATION_BATCH_SIZE=8`, `EVALUATOR_WORKERS=8`,
and `EVALUATOR_TIMEOUT=3600`; overrides are rejected. Smoke runs may override them but
are not importable as benchmark results. Optional controls are `INSTANCE_FILTER`,
`REDO_EXISTING=1`, and `REDO_EVALUATION=1`. Full Pro runs additionally require local Docker on
`linux/amd64`; smoke runs may opt into the pinned evaluator's Modal path with
`PRO_EVAL_BACKEND=modal` or select a different `DOCKER_PLATFORM`. Pro's Docker wait
uses `EVALUATOR_TIMEOUT` as a hard per-instance limit; a timeout kills and removes the
container and records a replayable evaluator-error status.

Generation runs the immutable scope in batches and, between batches, removes only
unused `sweb.eval.*` and `jefzda/sweap-images` task images without `--force`. Docker
therefore protects images referenced by running containers, including other benchmarks
sharing the DinD daemon. Standard evaluation enables the pinned evaluator's per-task
image cleanup; Pro removes each unused task image after evaluation and retries deferred
cleanup at the suite boundary. Both evaluators guard every task-image lookup and pull,
rejecting mutable tags whose image ID or RepoDigests differ from generation evidence.
`docker-cleanup.log` records `docker system df` before and after cleanup, preventing the
runner's 400 GiB image store from accumulating a full suite's hundreds of distinct
images.

The upstream prompt's working directory is rendered from the pinned Docker environment
config. It remains `/testbed` for Verified and Multilingual and becomes `/app` for Pro,
matching the Pro images and the digest-bound `config/pro.yaml` override.

## Artifacts

Each `${SWEBENCH_RESULTS_ROOT}/<run-name>/<suite>` directory contains:

- `agent/preds.json`: evaluator-ready patches.
- `agent/<instance>/<instance>.traj.json`: complete model/tool trajectory.
- `run-scope.json`: immutable full-run or smoke target IDs and selection inputs.
- `run-metadata.json`: immutable endpoint/model/config/dataset/pin/source identity.
- `runtime-binding.json`: embedded schema-v1 runtime and evaluator contract.
- `effective-config.json`: exact Pydantic-normalized mini-SWE config.
- `environment.freeze.txt` and `environment.normalized.freeze.txt`: live per-run venv identity.
- `task-images.json`: exact per-instance task-image IDs and RepoDigests.
- `evaluation-predictions-sha256.txt`: canonical prediction-set cache identity.
- `generation-summary.json`: prediction completeness, exit statuses, patch sizes,
  and API-call counts.
- `evaluation/`: per-instance test logs, reports, patches, and raw evaluator score.
- `score.json`: normalized pass/fail/error IDs, completeness-gate status, scope score,
  and a full benchmark score only for complete unsliced runs.
- exact generation/evaluation commands, endpoint model response, timestamps, logs,
  dataset provenance, and Docker cleanup/storage logs.

The wrappers never record `OPENAI_API_KEY`.

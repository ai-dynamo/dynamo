<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 campaign runbook

Run every command from the repository root. Raw artifacts live on the evaluation PVC;
only sanitized compact/task-level results and the self-contained report are committed.

## One-time setup

```bash
cp benchmarks/glm52-nscale/campaign.local.env.example \
  benchmarks/glm52-nscale/campaign.local.env
# Fill in the ignored local file, then provision nvcr-secret and hf-token-secret.

benchmarks/glm52-nscale/eval/deploy-runner.sh
# Choose exactly one first-time cache command:
benchmarks/glm52-nscale/cache/migrate.sh                    # legacy VAST cache
# benchmarks/glm52-nscale/cache/migrate.sh --initialize-empty  # fresh namespace
# Later reconciliations of the initialized cache use:
# benchmarks/glm52-nscale/cache/deploy.sh
benchmarks/glm52-nscale/cache/configure-runner.sh
benchmarks/glm52-nscale/cache/assert-ready.sh
benchmarks/glm52-nscale/cache/verify.sh
benchmarks/glm52-nscale/eval/exec-runner.sh \
  /workspace/eval/bfcl/scripts/bootstrap.sh
benchmarks/glm52-nscale/eval/exec-runner.sh \
  /workspace/eval/swebench/bootstrap.sh
benchmarks/glm52-nscale/eval/exec-runner.sh \
  /workspace/eval/terminalbench/bootstrap.sh
```

`cache/deploy.sh` is fail-closed and only starts an already initialized dedicated
cache PVC. It never creates an empty cache implicitly. Upgrade a legacy
artifact-PVC-backed cache with `cache/migrate.sh`, or explicitly initialize a
fresh cache with `cache/migrate.sh --initialize-empty`. Migration preserves and
hashes the former store, binds the marker hash to the new PVC, and retains both
storage backends. Emergency VAST rollback uses the committed
`cache/registry-mirror-vast-rollback.yaml` manifest.

The campaign output subtree uses a second, nested Cinder mount so VAST remains
available for persistent harness environments while raw benchmark output avoids the
shared ancestor quota. For the existing campaign, complete the image prefill, export
its runner-local checkpoint, then perform the one-time read-only migration and runner
cutover:

```bash
snapshot=/Users/rmccormick/codex/glm52-campaign-artifacts/cache-prefill/verified-20260707T030800Z
benchmarks/glm52-nscale/artifact-storage/migrate.sh
benchmarks/glm52-nscale/eval/teardown-runner.sh
benchmarks/glm52-nscale/artifact-storage/deploy-runner.sh
benchmarks/glm52-nscale/artifact-storage/restore-prefill.sh "${snapshot}"
benchmarks/glm52-nscale/artifact-storage/assert-ready.sh
```

This overlay is deliberately outside `eval/`: the pinned evaluator source stays at
the immutable campaign commit. Migration hashes an exact VAST-to-Cinder copy, retains
both PVs, verifies the authoritative task-image manifest, and requires the clean
`vllm-serve-ab-r2` replay target to be absent. The old failed attempt remains rejected
evidence; the clean replay uses the new run name.

`campaign.source_commit` must name the immutable scaffold commit. Runner sync and
deployment both refuse source drift from that commit. `deploy-runner.sh` never deletes a
live runner implicitly; use `teardown-runner.sh` only after `assert-runner-idle.sh` passes.

## Endpoint map

| Variant | API base |
|---|---|
| `dynamo-vllm` | `http://glm52-dynamo-vllm-frontend:8000/v1` |
| `vllm-serve` | `http://glm52-vllm-serve:8000/v1` |
| `dynamo-sglang` | `http://glm52-dynamo-sglang-frontend:8000/v1` |
| `sglang-serve` | `http://glm52-sglang-serve:8000/v1` |

Never use a localhost or port-forward URL for an importable run. Every harness verifies
the exact endpoint, served model, 409,600-token serving window, phase, and active runtime
binding before issuing requests.

## Validation deployment

For each variant, create a `validation` deployment, run the API smoke and one-instance
suite smokes, then tear it down:

```bash
variant=dynamo-vllm
benchmarks/glm52-nscale/deploy/deploy.sh "${variant}" validation
benchmarks/glm52-nscale/deploy/smoke.sh "${variant}"
# Run the suite-specific validation commands below with phase=validation.
benchmarks/glm52-nscale/deploy/teardown.sh
```

`deploy.sh` captures and publishes a privacy-safe active binding. `teardown.sh` first
refuses active evaluator work, deletes every serving/Grove resource, proves zero remaining
GPU requests, and removes the active binding.

## Full order-balanced schedule

Each row below is one independent suite experiment. Deploy a fresh server, run only that
suite, and tear it down before the next stack. Phase `ab` runs Dynamo then native; phase
`ba` reverses the order. This produces 40 full result cells:

| Framework | Phase | First | Second |
|---|---|---|---|
| vLLM | `ab` | `dynamo-vllm` | `vllm-serve` |
| vLLM | `ba` | `vllm-serve` | `dynamo-vllm` |
| SGLang | `ab` | `dynamo-sglang` | `sglang-serve` |
| SGLang | `ba` | `sglang-serve` | `dynamo-sglang` |

Repeat that four-row order for each of `bfcl-v4`, `swebench-verified`,
`swebench-pro`, `swebench-multilingual`, and `terminal-bench-2.1`. The full BFCL
rows additionally require `SERPAPI_API_KEY` through the optional
`glm52-benchmark-secrets` Secret.

For every cell:

```bash
benchmarks/glm52-nscale/artifact-storage/assert-ready.sh
benchmarks/glm52-nscale/deploy/deploy.sh "${variant}" "${phase}"
benchmarks/glm52-nscale/deploy/smoke.sh "${variant}"
# Execute exactly one guarded suite command below.
benchmarks/glm52-nscale/deploy/teardown.sh
```

## Guarded BFCL command

```bash
run_dir=/artifacts/glm52-nscale/bfcl/${variant}/${phase}/${variant}-${phase}-full
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${run_dir}/runtime-continuity.json" -- \
  bash -lc "export GLM52_OPENAI_BASE_URL=${api_base}; \
    /workspace/eval/bfcl/scripts/run-full.sh \
    ${variant} ${phase} ${run_dir}"
```

For validation, replace the final command with
`run-smoke.sh ${variant}
/artifacts/glm52-nscale/bfcl/${variant}/validation/${variant}-validation-smoke`.
Full BFCL is fixed at 5,106 scored IDs, 5,217 generated IDs, 16 request threads,
temperature 0, 64,000 output tokens, and zero inference errors.

## Guarded SWE-bench command

Use suite `verified`, `pro`, or `multilingual`:

```bash
benchmarks/glm52-nscale/cache/assert-ready.sh
suite=verified
run_name=${variant}-${phase}
run_dir=/artifacts/glm52-nscale/swebench/results/${run_name}/${suite}
if [[ "${suite}" == verified ]]; then
  benchmarks/glm52-nscale/cache/require-complete.sh verified \
    /artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified/task-images.json
fi
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${run_dir}/runtime-continuity.json" -- \
  bash -lc "export OPENAI_BASE_URL=${api_base}; export OPENAI_API_KEY=EMPTY; \
    /workspace/eval/swebench/run.sh \
    ${suite} ${run_name} all --phase ${phase}"
```

Full SWE runs are fixed at 16 generation workers, batches of 8, temperature/top-p 1,
32,768 output tokens, and a 14,400-second agent wall limit. For validation, use a
`-${phase}-` run name with `phase=validation` and prefix the remote command with
`INSTANCE_SLICE=0:1`.

For the current Verified recovery, prefill its exact image population from the
authoritative completed `task-images.json` before another official cell.
`cache/prefill.sh start verified` is idempotent and its named tmux session
intentionally makes the runner-idle gate fail while work is active. Do not deploy
the next fresh server until `cache/prefill.sh status verified` reports
`state=complete`, `completed=total`, and `tmux_active=false`; see
`cache_binding_matches=true`; see `cache/README.md` for the command and evidence
paths. The implementation is Verified-only; Pro's shared multi-tag repository
needs a separate tag-aware path.

The native-vLLM A/B recovery cell uses `run_name=vllm-serve-ab-r2`; do not reuse
the immutable failed `vllm-serve-ab` run metadata.

## Guarded Terminal-Bench command

```bash
jobs=/artifacts/glm52-nscale/terminalbench/jobs
summary=/artifacts/glm52-nscale/terminalbench/summaries/${variant}/${phase}
benchmarks/glm52-nscale/eval/run-guarded.sh "${variant}" \
  --phase "${phase}" \
  --attestation "${summary}/runtime-continuity.json" -- \
  bash -lc "/workspace/eval/terminalbench/run-full.sh \
    --api-base ${api_base} --label ${variant} --phase ${phase} \
    --job-name ${variant}-${phase}-terminalbench21-r6-89x5 \
    --jobs-dir ${jobs} --summary-dir ${summary} --n-concurrent 4"
```

For validation, use `run-smoke.sh`, phase `validation`, and a distinct job/summary path.
Official runs are fixed at 89 tasks, five attempts, four concurrent trials, Terminus-2,
temperature/top-p 1, 48,000 output tokens, 500 turns, a 262,144-token advertised agent
context, and a four-hour task timeout.

## Import and report

Import only strict full artifacts after their guarded command succeeds. The importer
requires phase, runtime binding, continuity, source/environment/task-image identities,
exact populations, and zero infrastructure errors; it materializes sanitized task-level
artifacts and paired disagreements under `results/`. See `results/README.md` for exact
commands. `/artifacts` exists only inside the runner; use the allowlisted fetcher to stage
compact evidence locally before import:

```bash
remote_dir=/artifacts/glm52-nscale/swebench/results/${variant}-${phase}/${suite}
local_dir=benchmarks/glm52-nscale/report/runs/swebench-${suite}/${phase}/${variant}
benchmarks/glm52-nscale/eval/fetch-result.sh \
  swebench "${remote_dir}" "${local_dir}"
python3 benchmarks/glm52-nscale/report/import_result.py \
  --variant "${variant}" --suite "swebench-${suite}" --phase "${phase}" \
  --artifact "${local_dir}/score.json"

python3 benchmarks/glm52-nscale/report/generate.py
python3 benchmarks/glm52-nscale/report/generate.py --check
```

Do not mark the campaign complete until all 40 variant/suite/phase rows validate and the
AB/BA aggregate report is current.

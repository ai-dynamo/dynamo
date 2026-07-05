<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Result summary contract

`summary.json` is the only input to the HTML report. It contains at most one
full-run result for each `(variant, suite, phase)` key; smoke-run results remain in the
raw artifact tree.

Do not hand-edit aggregate result rows. The artifact PVC is mounted only in the
evaluation runner, so first copy the allowlisted compact evidence into the ignored
`report/runs/` staging tree, then invoke the fail-closed importer:

```bash
remote=/artifacts/glm52-nscale/bfcl/dynamo-vllm/ab/full
local=benchmarks/glm52-nscale/report/runs/bfcl-v4/ab/dynamo-vllm
benchmarks/glm52-nscale/eval/fetch-result.sh bfcl "${remote}" "${local}"
python3 benchmarks/glm52-nscale/report/import_result.py \
  --variant dynamo-vllm \
  --suite bfcl-v4 \
  --phase ab \
  --artifact "${local}/summary.json"

remote=/artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified
local=benchmarks/glm52-nscale/report/runs/swebench-verified/ab/dynamo-vllm
benchmarks/glm52-nscale/eval/fetch-result.sh swebench "${remote}" "${local}"
python3 benchmarks/glm52-nscale/report/import_result.py \
  --variant dynamo-vllm \
  --suite swebench-verified \
  --phase ab \
  --artifact "${local}/score.json"

remote=/artifacts/glm52-nscale/terminalbench/summaries/dynamo-vllm/ab
local=benchmarks/glm52-nscale/report/runs/terminal-bench-2.1/ab/dynamo-vllm
benchmarks/glm52-nscale/eval/fetch-result.sh terminalbench "${remote}" "${local}"
python3 benchmarks/glm52-nscale/report/import_result.py \
  --variant dynamo-vllm \
  --suite terminal-bench-2.1 \
  --phase ab \
  --artifact "${local}/summary.json"
```

`fetch-result.sh` refuses an existing local destination and transfers only the fixed
top-level files required by that suite's importer. It never copies trajectories,
task-container files, prompts, or raw logs.

The BFCL import requires sibling `summary.json` and
`complete-validation.json`, `expected-ids.json`, `failures.jsonl`, environment-lock,
and runtime-continuity artifacts. A SWE-bench import requires sibling
`score.json`, `generation-summary.json`, `run-metadata.json`, `run-scope.json`, and
both live environment-freeze artifacts. The
SWE run name must begin with the exact serving variant ID so evidence cannot be
assigned to another stack. Terminal-Bench uses its strict compact summary,
embedded run metadata, task rows, `trials.csv`, and a physical sibling
`task-images.json`. The importer requires the embedded task-image evidence, its
summary input hash, and the fetched file to agree; it validates all 89 ordered
pinned task refs, task and `task.toml` checksums, requested image references,
SHA-256 image IDs, nonempty canonical RepoDigests, and the 445-trial count. Every
suite also requires a sibling `runtime-continuity.json`. Imports reject smoke or
partial runs, missing IDs/trials, evaluation or infrastructure errors, aggregate
values that do not reconcile with outcome rows, and cross-variant metadata.

Every imported row records a phase-qualified logical `artifact://` URI and SHA-256
for each compact evidence artifact; host paths are never published. The importer
materializes sanitized task outcomes under `task-level/<suite>/<phase>/<variant>.jsonl`
and paired disagreements under `paired-disagreements/<suite>/<phase>/<pair>.jsonl`.
Existing rows are immutable by default; `--replace` is an
explicit, fully revalidated operation. The summary update is locked and atomically
renamed only after the complete report schema validates. Sidecar writes are rolled
back if the summary replacement fails.

Generate the HTML after importing and verify freshness before publication:

```bash
python3 benchmarks/glm52-nscale/report/generate.py
python3 benchmarks/glm52-nscale/report/generate.py --check
```

The HTML embeds the canonical SHA-256 of the parsed summary. `--check` renders
in memory and fails if the checked-in HTML differs from the current summary. Every
task-level and paired-disagreement JSONL is also read, schema-checked, reconciled,
counted, and hashed; missing, tampered, or orphaned sidecars fail validation.

Once `campaign.source_commit` is pinned, both import and report generation require
the current `campaign.env`, complete `eval/` tree, `report/import_result.py`, and
`report/generate.py` to equal that commit. Result files, report output, tests, and
documentation remain mutable without weakening the executable-source guard.

Campaign status and time fields are validated. `started_at` and any non-null
`completed_at` must be timezone-aware ISO-8601 timestamps. Pending and in-progress
campaigns must not set `completed_at`; a complete campaign must set it no earlier
than `started_at`.

Every variant must appear in exactly one Dynamo/native framework pair. A campaign
may omit result rows while it is in progress, but `status: complete` is a strict
closure assertion: the schema must define exactly 40 variant/suite/phase combinations,
all 40 result rows must be present and complete, every suite/variant/phase cell must
use a globally fresh deployment/controller/pod identity, and every variant status
must be complete. Capture timestamps prove, per suite and framework, the full order
`ab` Dynamo, `ab` native, `ba` native, `ba` Dynamo.

Each result row has this shape:

```json
{
  "variant": "dynamo-vllm",
  "suite": "bfcl-v4",
  "phase": "ab",
  "run_type": "full",
  "status": "complete",
  "completeness": {
    "generated_units": 5217,
    "evaluated_units": 5106,
    "completed_trials": 5106
  },
  "metrics": {
    "overall_accuracy": 0.72,
    "correct_cases": 3676,
    "failed_cases": 1430,
    "inference_errors": 0
  },
  "evidence": {
    "importer": "glm52-result-import/v1",
    "sources": [
      {
        "role": "bfcl-summary",
        "path": "artifact://bfcl-v4/ab/dynamo-vllm/bfcl-summary/summary.json",
        "sha256": "<64 lowercase hexadecimal characters>"
      }
    ]
  },
  "task_level": {
    "path": "results/task-level/bfcl-v4/ab/dynamo-vllm.jsonl",
    "sha256": "<64 lowercase hexadecimal characters>",
    "records": 5106
  },
  "runtime_identity": {
    "deployment_sha256": "<64 lowercase hexadecimal characters>",
    "content_sha256": "<64 lowercase hexadecimal characters>",
    "captured_at": "2026-07-05T03:00:00Z"
  },
  "wall_time_seconds": 12345.6
}
```

- Percent metrics are fractions in `[0, 1]`; the report formats them as
  percentages.
- `units` is the official scored/evaluated population. A suite may set a larger
  `generation_units`; BFCL generates 5,217 IDs because its 5,106 scored IDs
  depend on 111 unscored memory-prerequisite IDs.
- A `complete` row must have every expected unit generated and evaluated, all
  expected trials completed, and every suite-specific required metric
  populated. It must also carry validated source-evidence lineage.
- Complete BFCL rows require zero inference errors. Complete Terminal-Bench rows
  require zero errored and zero no-reward attempts; ordinary verifier failures
  remain valid benchmark outcomes. Every Terminal-Bench variant and AB/BA phase
  must carry the same compact task-image map count and digest, as well as the same
  exact Harbor environment identity.
- Derivable aggregate metrics are reconciled against their outcome counts:
  SWE-bench `benchmark_score` is resolved instances divided by the full dataset,
  `score_on_submitted` is resolved divided by submitted, and Terminal-Bench
  `pass_at_1` is passed attempts divided by all completed trials.
- Non-complete rows may carry partial metrics for machine-readable progress,
  but the report withholds them until the official full evaluation completes.
- `generate.py --validate-only` rejects unknown references, malformed metrics,
  false completeness, and duplicate `(variant, suite, phase)` rows.

Metric names intentionally match or directly map to the normalized harness
summaries:

- BFCL: official `Overall Acc`, plus correct, failed, and inference-error case
  counts.
- SWE-bench variants: `benchmark_score`, `score_on_submitted`, passed, failed,
  and missing instances.
- Terminal-Bench 2.1: task-mean `pass_at_1` through `pass_at_5`, all-trial mean
  reward, and attempt outcome counts.

Pair deltas are computed by the report as `Dynamo - native`; they are never
stored separately, so they cannot become stale.

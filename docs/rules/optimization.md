# Optimization Rules

Use evidence before spending GPU time on candidate changes.

## Evidence Before GPU Spend

- Treat AIPerf results, endpoint behavior, pod logs, Kubernetes events, and recipe diffs as evidence, not proof.
- Do not deploy or benchmark a candidate change unless it is linked to the target workload, the metric or constraint it should improve, and the exact knob being changed.

## One Purpose Per Candidate

- Change one knob family at a time unless multiple changes are required for correctness.
- Keep topology changes, engine-runtime changes, and benchmark-load changes separate.

## No Promotion Without Comparison

- Do not recommend a candidate unless it has been compared against the current baseline or best known result on the same
  workload.
- Report regressions, failed runs, and limitations just as is done with wins.

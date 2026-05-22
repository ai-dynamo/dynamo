<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Known Issues — `dynamo-optimize`

Reference content for `dynamo-optimize/SKILL.md`. Entries follow the
6-element shape from `dynamo-skill-author/references/body-shape.md`:
Symptom, Root cause, Affected, Fix, Verify, (optional) Reference.

Walk this list before falling through to `dynamo-troubleshoot/SKILL.md`.

---

### Recipe envelope mismatch — pod stuck Pending with SchedulingFailed

**Symptom:** `kubectl get pods -n <ns>` shows the worker pod in `Pending`;
`kubectl describe pod <pod>` reports a SchedulingFailed event referencing
`Insufficient nvidia.com/gpu` or "no nodes match selector".

**Root cause:** The recipe pinned a specific GPU SKU or count
(e.g. `h200_sxm` × 16) and the cluster has different hardware. Recipes
encode a tested envelope (G1, G3); the scheduler cannot place workers
outside that envelope.

**Affected:** Any recipe applied to a cluster whose hardware differs
from the recipe's tested envelope; especially common when copying a
qwen3-32b recipe (16x H200) onto a smaller dev cluster.

**Fix:** Either pick a recipe whose tested envelope matches the cluster
(re-run Phase 2 of `dynamo-optimize` with the observed `kubectl get
nodes -L nvidia.com/gpu.product` output as the GPU constraint), or
explicitly accept that you are deploying outside the tested envelope
and patch GPU count in `deploy.yaml`. Skill default is to refuse the
mismatch unless the user says "proceed at my own risk".

**Verify:** `kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=<name> -n <ns> --timeout=1200s` succeeds.

**Reference:** `dynamo-optimize/SKILL.md` Phase 2 Decision points.

---

### AIPerf 0.8.0 vs recipe base-image transformers conflict (DYN-2878 pattern)

**Symptom:** `pip install aiperf==0.8.0` in the benchmark Pod completes,
but `aiperf profile ...` fails on import with a transformers traceback
(`AttributeError: '...' object has no attribute 'tokenizer'` or a
`transformers.utils` import error).

**Root cause:** The recipe's base image ships a `transformers` version
that pre-dates a feature `aiperf==0.8.0` depends on. DYN-2878 is the
canonical instance: `recipes/deepseek-v32-fp4/trtllm/{agg-round-robin,
disagg-kv-router}/perf.yaml` co-pins `transformers==4.57.6` on top of
the base image's older transformers to work around exactly this issue.

**Affected:** Any recipe whose `perf.yaml` ALSO pins a non-default
transformers version, or whose base image is stale. Detect with
`kubectl exec <perf-pod> -- pip show transformers`.

**Fix:** Two options.
1. Honour the recipe's transformers co-pin: add the recipe's
   `pip install transformers==<recipe-pin>` to the AIPerf install step.
2. Roll the base image forward to one that ships a compatible
   transformers, then drop the co-pin.

The skill always pins `aiperf==0.8.0` regardless of the recipe's pin
(locked decision 2026-05-22). When this conflict surfaces, the skill
records it in the output contract `dry_run_result: failed: <reason>` —
it does NOT silently fall back to the recipe's older AIPerf pin.

**Verify:** `kubectl logs <perf-pod>` shows the AIPerf "starting profile"
banner instead of the transformers traceback.

**Reference:** DYN-2878 (internal); `corpus/aiperf/version.yaml`
`recipe_pin_observed` notes.

---

### `--goodput` rejects user-declared tag with ValueError

**Symptom:** `kubectl logs <perf-pod>` shows:

```
ValueError: Unknown metric tag(s) in --goodput: <tag>.
```

**Root cause:** The user declared an SLO using a tag that isn't in
AIPerf's metric registry. Common offenders: `tokens_per_second` (appears
in AIPerf's own `slos.py` docstring as an example but is NOT a real
tag); `ttft` and `itl` (those are short headers, not tags); a tag from
a pre-0.8.0 AIPerf version that was renamed.

**Affected:** Any first-time SLO declaration that wasn't copy-pasted
from a verified source.

**Fix:** Replace with a registered tag. See
[slo-shape.md](slo-shape.md) "Supported `--goodput` metric tags" table.
Verified tags include `time_to_first_token`, `inter_token_latency`,
`request_latency`, `output_token_throughput_per_user`.

**Verify:** Re-run `python3 scripts/measure_slo.py --slo "<corrected>" ...`;
the script proceeds past startup and emits PASS/FAIL lines.

**Reference:** [slo-shape.md](slo-shape.md) Pitfalls table.

---

### AIConfigurator silent naive-fallback — DGDR reports Ready, config is naive

**Symptom:** A DGDR completes successfully (`status.phase: Ready` or
`Deployed`), but the deployed configuration looks like a memory-fit TP
calculation rather than an AIC-driven choice (e.g. straight `TP=GPU_count`
with no prefill/decode split tuning, no parallelism beyond TP).

**Root cause:** AIC silently falls back to a naive memory-fit TP
calculation when the model/hardware/backend combination is unsupported
(G10). The DGDR controller does NOT emit an event distinguishing
"AIC-produced config" from "naive-fallback config", so the only
signal is the configuration itself.

**Affected:** Any DGDR for a model/SKU/backend combination AIC doesn't
support. Less common on flagship combos (Qwen3, Llama-3 on H100/H200);
more common on newer FP4/FP8 variants or non-standard SKUs.

**Fix:** Inspect `status.profilingResults.selectedConfig` and compare
against the recipe's known-good parallelism for the same model. If they
match (or if the selected config is "TP=N, no prefill/decode split"),
treat as naive fallback. Either (a) use the recipe directly (skip the
AIC chain), (b) hand off to `dynamo-plan` to run AIC standalone with
explicit `searchStrategy: thorough`, or (c) accept the naive config
explicitly.

**Verify:** Phase 2 of `dynamo-optimize` records `ai_configurator_chain.chained_to: detected_naive_fallback` in the output contract.

**Reference:** `corpus/aiconfigurator/feasibility.md` Recommended chain;
`docs/components/profiler/profiler-guide.md:87`.

---

### DGDR is immutable — can't edit an in-flight DGDR

**Symptom:** `kubectl edit dgdr <name>` succeeds locally, but the
controller doesn't reprofile and the deployed DGD doesn't change. Or
`kubectl edit` fails with an admission-webhook reject citing
immutability.

**Root cause:** DGDRs are immutable by design (G8); to change SLAs or
configuration, you delete and recreate (`profiler-guide.md:262-263`).

**Affected:** Any user iterating on SLA targets via DGDR.

**Fix:** `kubectl delete dgdr <name>` then `kubectl apply -f <new-dgdr.yaml>`. NOTE: the DGDR does NOT own the DGD it created (no
owner reference per `dgdr.md` lines 208-212), so deleting the DGDR does
NOT delete the running deployment. The new DGDR will create a new DGD
with a different name; you'll need to delete the old DGD explicitly to
free its GPUs.

**Verify:** `kubectl get dgd -n <ns>` shows the new DGD; `kubectl get
dgdr -n <ns>` shows only the new DGDR.

**Reference:** [`docs/components/profiler/profiler-guide.md`](https://github.com/ai-dynamo/dynamo/blob/release/1.2.0/docs/components/profiler/profiler-guide.md) lines 262-263.

---

### `/v1/models` empty after DGD reports Ready

**Symptom:** `kubectl get dgd <name>` shows `state: successful`, all
pods are Ready, but `curl http://localhost:8000/v1/models` returns
`{"data": []}` for 30-120 seconds.

**Root cause:** DGD `state: successful` indicates pods are Ready, but
workers register endpoints with the Frontend via NATS after Ready, not
before. There's a registration lag.

**Affected:** All DGDs, all recipes. Worse with cold image pulls.

**Fix:** Poll `/v1/models` until non-empty. The skill's Phase 4.3 smoke
test does this automatically with a 10-second poll loop. If the poll
exceeds 5 minutes, hand off to `dynamo-troubleshoot/SKILL.md` — the
NATS connection may be broken.

**Verify:** `curl http://localhost:8000/v1/models` returns at least one
model entry under `data`.

**Reference:** Same pattern is documented in
`dynamo-deploy/references/known-issues.md` ("DGD successful but /v1/models empty").

---

### Model-download Job exceeds 600s timeout on large models

**Symptom:** `kubectl wait --for=condition=Complete job/model-download
... --timeout=600s` times out; `kubectl get job model-download -o yaml`
shows the download still in progress.

**Root cause:** The qwen3-32b recipe (and most recipes) pin
`--timeout=600s` (G25). For very large models (qwen3-235b, deepseek-v4,
llama-3-70b) over slow networks, 600s is insufficient. Anish's substrate
used `--timeout=6000s` for exactly this reason; the in-tree README is
the canonical 600s default.

**Affected:** Models ≥ 50 GB, networks slower than 100 MB/s sustained.

**Fix:** Re-run the wait with a longer timeout: `kubectl wait --for=condition=Complete job/model-download -n <ns> --timeout=3600s`.
Do NOT re-apply the model-download Job — that creates a duplicate Job
and may corrupt the partial download.

**Verify:** `kubectl get job model-download -o jsonpath='{.status.succeeded}'` returns `1`.

**Reference:** `corpus/dynamo/recipes-qwen3-32b-README.yaml` extract
`qwen3-32b-model-download-timeout`.

---

### Workstation pre-validation on disagg-* recipe produces misleading numbers

**Symptom:** User ran `dynamo-serve` locally with a disagg recipe's
config, declared SLO PASSed at workstation, but cluster deploy FAILed
the same SLO.

**Root cause:** Disagg modes depend on the cross-node KV transfer path
(NIXL/UCX). The workstation cannot exercise this path — it runs prefill
and decode in the same process. Workstation perf is not representative
of cluster perf for `disagg-*` modes.

**Affected:** All `disagg-*` recipes. The skill refuses workstation
pre-val on these specifically.

**Fix:** Don't run workstation pre-val on `disagg-*`. Skip directly to
the cluster apply + measure-slo loop. If you do want a fast local
sanity check, apply the recipe's `agg-*` sibling (if any) for the
shakedown and confirm the model serves; then move to disagg on the
cluster.

**Verify:** Phase 3 output contract records `workstation_preval: disagg: skipped (correlation)`.

**Reference:** `dynamo-optimize/SKILL.md` Phase 3.2.

---

### Comparing AIPerf 0.6.0 baseline against 0.8.0 post-deploy

**Symptom:** `measure_slo.py --baseline <baseline.json>` emits
`INVALID|<metric>|reason=different_aiperf_versions` and exits non-zero.

**Root cause:** The skill always pins `aiperf==0.8.0` post-deploy
(G27, locked decision). If the baseline was captured under an older
AIPerf version, the metric definitions may differ — comparing avg TTFT
across versions can be misleading if the version difference touches
how the metric is computed.

**Affected:** Users with pre-existing baselines captured under an older
AIPerf pin.

**Fix:** Re-capture the baseline using the skill's `measure_slo.py
--mode baseline`, which uses the same `aiperf==0.8.0` pin. The
re-captured baseline is comparable to the post-deploy measurement.

**Verify:** `baseline.json` records `aiperf_version: "0.8.0"`; the
delta block emits valid `DELTA|<metric>|...` lines.

**Reference:** [slo-shape.md](slo-shape.md) "Delta vs baseline (optional)".

---

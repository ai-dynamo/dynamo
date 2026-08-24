---
name: author-baseline-dgd
description: >-
  Drafts a candidate baseline DynamoGraphDeployment from interview requirements when no catalog recipe matches the
  user's model, hardware, and backend, presenting per-decision evidence for the user's confirmation. Use only from
  user-interviewer at interview time, at rung 3 of the baseline-source ladder, and never to deploy or to replace a
  baseline the user already provided.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - workload
    - interview
    - optimization
---

# Author Baseline DGD

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Draft ONE candidate baseline DGD for a greenfield engagement and present it for the user's explicit confirmation.
Do not deploy, benchmark, apply, or record anything as the baseline: an unconfirmed draft is a proposal, and only
the user's confirmation makes it a user-provided baseline.

## Inputs

Require:

- the interview fact table from `synthesize-user-workload` (model source and revision, hardware type and count,
  backend and precision preferences, workload shape, SLOs, Kubernetes context and namespace);
- the recipe catalog scan that established rung 3 (no exact or close recipe), including the nearest recipes
  considered and why each was rejected as a base; and
- any user-stated constraints (`resources.pinned` candidates, budgets) already collected.

If model identity or hardware type and count is missing, return the question to `user-interviewer` instead of
guessing. Backend is different: when the user explicitly has no preference, CHOOSE it here with evidence - prefer
the backend whose nearest catalog recipe scaffolds this model family and hardware, per the knob guides' coverage -
and record the choice and its evidence in the decision table the user confirms. A confirmed draft's backend is a
confirmed decision, not an invented default; the contract's `preferences.framework` still records only what the
user themselves stated.

## Read The Applicable Knowledge

Always read:

- all three files under `agent-docs/guides/model-sizing/` (memory fit, `min_tp`, classification);
- `agent-docs/guides/knob-tuning/tuning-hierarchy.md`;
- the chosen backend's guide (`agent-docs/guides/knob-tuning/vllm.md`, `sglang.md`, or `tensorrt-llm.md`) -
  when choosing the backend here, read the candidates' guides as needed to make the choice;
- `agent-docs/guides/knob-tuning/dynamo.md`; and
- the nearest catalog recipes' manifests, as structural scaffolding only.

Read `agent-docs/guides/rate-matching/matching.md` only when the draft is disaggregated (rare for a baseline;
prefer aggregated unless the user's SLOs demand otherwise).

## Author The Draft

1. **Size the model**: compute weight bytes, `min_tp`, and `headroom_ratio` per `memory.md`, showing the
   arithmetic. Choose the serving TP per `parallelism.md` (prefer lower TP and more replicas for throughput
   workloads; raise TP above `min_tp` only when headroom demands it, recording the replica cost).
2. **Choose topology conservatively**: an aggregated single-node layout unless the user's hardware or SLOs force
   otherwise. The baseline's job is to run and measure, not to win; the optimization loop owns improvement.
3. **Scaffold from the nearest recipe**: copy its structure (components, probes, service wiring, image versions
   for the chosen backend) and replace model, parallelism, resources, and any hardware-bound fields, naming every
   replacement. Never carry a hardware-bound topology, transport, or checkpoint choice across without evidence it
   fits the target.
4. **Set knobs to the backend guide's defaults**, deviating only where the sizing arithmetic requires it
   (e.g. `gpu_memory_utilization`, `max_model_len` capped to the workload). Leave optimization headroom alone.
5. **Validate the draft**: parse as YAML, exactly one `DynamoGraphDeployment` document, no secret values, and
   confirm it would pass `kubectl apply --dry-run=server` semantics (correct API version, resource names, required
   fields) to the extent checkable offline.

## Present For Confirmation

Return to `user-interviewer`, for relay to the user:

- the complete draft manifest;
- a per-decision evidence table: each major choice (TP, replicas, memory settings, backend, image, topology) with
  the guide citation or arithmetic that produced it;
- the nearest recipes considered and why each was rejected as a base; and
- the explicit statement that this draft is unvalidated on hardware and iteration 0 will characterize it.

Do not proceed on silence, enthusiasm, or a partial answer: confirmation is the user's explicit acceptance of THIS
manifest (or of it as amended by the user). The confirmed manifest goes to `synthesize-user-workload` for canonical
capture with `deployment.origin: agent-authored` and `deployment.origin_source: inputs/baseline-evidence.md`
(the interviewer writes the evidence table and confirmation there at capture time, per `run-artifacts.md`).

## Do Not

- Deploy, benchmark, or apply anything.
- Record an unconfirmed draft anywhere a downstream role could mistake it for the baseline.
- Author when a user DGD exists (that engagement has a baseline) or when rung 1 or 2 produced a viable base.
- Invent model, hardware, or SLO facts; missing facts return to the interview.
- Embed secret values or Kubernetes `Secret` resources.

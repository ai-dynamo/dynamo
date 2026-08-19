---
name: create-optimization-hypothesis
description: >-
  Materializes one evidence-backed proposal from a flexible knowledge-consult.md into a challenger-ready
  deploy-draft.yaml by applying only its selected DGD change to the current successful manifest. Use after
  consult-perf-knowledge writes a proposed consultation in the current deployment iteration's next-candidate directory.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - optimization
    - hypothesis
    - kubernetes
    - yaml
---

# Create Optimization Hypothesis

Materialize an already-reasoned proposal. Treat `knowledge-consult.md` as a flexible reasoning record, not a rigid
schema. Do not select a different lever, broaden the proposal, deploy, benchmark, or approve it.

## Inputs

Require:

- the current `DEPLOY_ROOT`;
- `DEPLOY_ROOT/applied_manifests/deploy.yaml`; and
- `DEPLOY_ROOT/next-candidate/knowledge-consult.md` written by `consult-perf-knowledge`.

Use:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/next-candidate/
```

as `HYPOTHESIS_ROOT`. The enclosing `deploy-iter-<NNN>` remains the analyzed source iteration. Do not create the next
deployment-iteration directory; `recipe-deployer` owns it after challenger approval.

## Read The Consultation

Read the entire consultation. Keep its freedom of form: do not require a particular bullet order, table shape, or
subsection beyond the core `Decision`, `Evidence`, `Proposed Change`, and `Materialization Handoff` sections.

Inspect `Decision` first:

- For `Status: no-proposal` or `Status: blocked`, stop without creating `deploy-draft.yaml`.
- For `Status: proposed`, continue only when the remaining content makes one candidate actionable.
- For a missing, conflicting, or unknown status, return the consultation without creating a draft.

For a proposed candidate, require the consultation to communicate, anywhere in its relevant sections:

- the successful source manifest path and SHA256;
- at least three distinct qualifying evidence categories, including AIPerf profiler data;
- the selected primary knob, its owner, and an exact target setting or state;
- whether the candidate is `single-knob` or `coupled-bundle`;
- one intended mechanism connecting the change to the expected measurable effect;
- the important risks or metrics that may regress; and
- for a coupled bundle, every required setting, the qualifying coupling reason, and any required follow-up ablation.

Accept evidence and reasoning as concise prose or tables. Do not require the consultation to precompute YAML paths,
current values, a source-to-draft diff, a separate validation plan, or a materialization-status field. Do not repeat
the performance analysis or re-rank the selected lever. Count the evidence categories represented by the entries; do
not rely only on the declared category count.

If the proposed target value, affected component, bundle membership, or intended mechanism is ambiguous, return the
consultation to `consult-perf-knowledge`. Do not choose a value, add a related optimization, or infer a broader
candidate.

## Resolve The Manifest Change

Require the recorded source path to identify the current successful manifest. Recompute its SHA256 and require it to
match `Source manifest SHA256` in `Materialization Handoff`. Then translate the selected proposal into the smallest
mechanical manifest edit:

1. Locate the selected knob in `DEPLOY_ROOT/applied_manifests/deploy.yaml`.
2. Resolve its exact YAML path and current value from that manifest.
3. If it is embedded in a command or argument string, identify the owning YAML path and change only the selected
   fragment.
4. If the consultation explicitly requires adding an absent setting, add only that setting in the location used by
   the owning component.
5. For a coupled bundle, resolve every explicitly named setting and no others.

Use cited source or official documentation only to confirm the selected knob's syntax and placement. Do not use it to
choose a different knob or target value. If the selected knob maps to multiple components or the requested state cannot
be mapped unambiguously, stop and return the consultation.

## Materialize The Draft

Create `${HYPOTHESIS_ROOT}/deploy-draft.yaml` from the exact successful source manifest:

1. Preserve the source formatting and key order where practical.
2. Apply only the resolved change selected in `knowledge-consult.md`.
3. For a flag embedded in a command or argument string, replace only the selected fragment.
4. Preserve API version, kind, DGD name, model identity, images, secret references, and benchmark wiring.
5. Do not edit the tracked recipe, current applied manifest, benchmark files, or prior iteration artifacts.
6. Do not add optional tuning to a functionality-required bundle.
7. Remove incidental formatting or ordering changes from the source-to-draft diff.

Render and validate in a temporary file before replacing the final draft. If an existing `deploy-draft.yaml` already
has the same source hash and resolved semantic diff, validate and reuse it. If it differs, do not overwrite it; report
the conflict for review.

## Validate The Draft

Before finalizing:

- parse the draft as YAML;
- require the source and draft to remain `DynamoGraphDeployment` resources;
- preserve the source API version, kind, and `metadata.name`;
- compute a source-to-draft semantic diff;
- require the semantic diff to contain exactly the resolved selected change;
- require all fields in an allowed coupled bundle and no others;
- confirm one independently testable mechanism under
  `agent-docs/rules/optimization/one-variable.md`;
- preserve target-fixed model, framework, precision, hardware, workload, and SLO constraints;
- verify the draft contains no secret values; and
- compute SHA256 hashes for the source and draft.

Perform local validation only. Do not mutate Kubernetes, run `kubectl apply`, use server-side dry run, launch a smoke
test, or run AIPerf.

## Complete The Handoff Record

After the final draft passes validation, update only the `Materialization Handoff` section of
`knowledge-consult.md`:

- replace `Draft manifest SHA256: pending` with the final draft SHA256;
- add `Materialization result: created`;
- add the local YAML and semantic-diff validation result; and
- add a compact table containing every exact YAML path and its verified before and after values.

Preserve the consultation's free-form reasoning and all evidence. Do not rewrite its `Decision`, `Reasoning`,
`Evidence`, or `Proposed Change` sections. Do not write separate `hypothesis-ledger.md` or `hypothesis-ledger.json`
files; `knowledge-consult.md` is the single reasoning and handoff record.

## Return

Return these two files to `hypothesis-challenger`:

- `${HYPOTHESIS_ROOT}/knowledge-consult.md`;
- `${HYPOTHESIS_ROOT}/deploy-draft.yaml`.

The draft is a proposal, not authorization to deploy.

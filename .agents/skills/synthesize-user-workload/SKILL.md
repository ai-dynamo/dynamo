---
name: synthesize-user-workload
description: >-
  Synthesizes a canonical user_workload.yaml and captures the user's immutable DynamoGraphDeployment from an
  optimization user's initial request, attachments, and minimal follow-up interview. Use as the first skill in a new
  Dynamo recipe optimization run, or to validate supplied workload and DGD inputs before deployment or benchmarking.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - workload
    - interview
    - optimization
---

# Synthesize User Workload

Create the durable workload contract and user-provided baseline DGD that every later optimization role receives. Do
not search for or select a recipe, deploy, benchmark, or propose tuning changes.

## Inputs

Require:

- the user's initial optimization request exactly as received;
- the user-provided DGD as an attachment, local file path, or pasted YAML;
- any attached workload descriptions, traces, or other local file paths;
- an optional caller-supplied `EXP_ID` or `EXP_ROOT`; and
- an existing `user_workload.yaml` only when the user is refining the interview before downstream work begins.

Read `agent-docs/rules/execution/user-workload.md`, `agent-docs/rules/execution/run-artifacts.md`, and
`agent-docs/references/definitions.md` before interviewing or writing the file.

## Extract Before Asking

Build a fact table from the initial request and attachments. Record only facts the user supplied or that a referenced
artifact proves. Keep the source of each fact while interviewing, but do not copy private conversation text into the
final YAML.

Resolve these blocking fields:

- one concrete user-provided YAML document containing a `DynamoGraphDeployment`;
- workload profile name, type, and a concrete description of the serving traffic;
- exact model source and revision when the user fixes one, plus the fallback policy when the requested revision
  may be unsupported by the available engine or weights (fall back to a named alternative and mark the target
  blocked, or halt);
- the user's current production deployment configuration when one exists (path or paste): it anchors every
  later comparison and seeds topology priors; when unavailable, record that explicitly as a known comparison
  limitation;
- allowed hardware type and count, including heterogeneous allocations;
- Kubernetes context and existing namespace; and
- either an exact trace or enough traffic-shape information to configure a defensible benchmark. A parametric
  description is a first-class option between presets and traces: AIPerf synthesizes static shapes (ISL/OSL mean
  and stddev, prefix reuse via the prefix-prompt pool and length controls) and multiturn sessions (session count,
  turns per session mean/stddev, per-turn delay) in a single command — offer "describe your workload" whenever the
  user has neither a matching preset nor a trace, and record the elicited parameters in the contract.

Objectives, SLOs, framework, precision, topology, storage class, and exact token or load values may remain unspecified
when the user explicitly has no constraint or preference. Represent those values with the schema's empty value; do not
invent defaults.

## Interview Minimally

Ask only for blocking facts that remain unknown or contradictory after reading all supplied context.

- Group related questions into one concise turn.
- Prefer a bounded choice only when the available choices are supported by the repository or user context.
- Explain why a requested fact blocks DGD handoff or reproducible measurement.
- Accept natural-language answers; do not make the user author YAML.
- Do not ask for secret values, kubeconfig contents, registry credentials, or Kubernetes Secret data.
- Do not add a ceremonial confirmation round when the user's message already provides an unambiguous value.
- When the user supplies resource limits — a ceiling on total concurrent GPU use, or knobs they forbid changing —
  record them in the contract's `resources` block. Ask about limits only when the run's scope makes them blocking
  (for example, when architectural changes such as replica scaling or disaggregation are in scope and the ceiling
  determines what may be proposed). Candidates run serially under the current iteration model; parallel candidate
  execution is not supported.
- If the user has not stated budgets, ask for them in the same interview batch — GPU-hours, wall clock, and
  failed-deploy limit — proposing sensible defaults for this engagement's scope; record the answers verbatim in the
  contract's `budgets` block, and `null` for any limit the user declines to state (a null budget leaves that
  limit ungated).
- When the user's production deployment shape differs from the unit under test (for example, fleet-level traffic
  numbers but a single-replica test deployment), derive the per-unit load explicitly and confirm it with the user
  before recording it.

Before writing the contract, enumerate every fact the downstream roles (deployer, benchmarker, hypothesis loop,
analyzer) will require, and ask all still-missing ones in the single upfront batch: a question that first surfaces
after the optimization loop has started is an interview defect. Record any fact the user defers as an explicit
assumption in the contract so the loop can proceed non-blockingly.

If blocking facts remain, return the questions to the parent and stop without handing work to a downstream role.

## Establish The Experiment Root

Use the exact caller-supplied `EXP_ROOT` when present. Otherwise create one unused directory under `runs/` using a
stable, filesystem-safe `EXP_ID` derived from the UTC creation timestamp and workload profile slug. Never reuse or
overwrite an existing experiment directory.

The user interviewer owns creation of `EXP_ROOT`; downstream roles must receive its exact path rather than infer it
from directory order or modification time. Create `<EXP_ROOT>/manifest.yaml` alongside it with the session
metadata `run-artifacts.md` defines (timestamps, repo commit, cluster context name, agent versions when known).

## Capture The User-Provided DGD

Create the canonical baseline input:

```text
<EXP_ROOT>/inputs/user_provided_dgd.yaml
```

- When the user supplies a file, copy its bytes without editing the source or canonical copy.
- When the user pastes YAML, materialize that YAML without changing its configuration.
- Parse the canonical copy as YAML and require at least one mapping document whose `kind` is
  `DynamoGraphDeployment`.
- Reject embedded secret values or Kubernetes `Secret` resources; references to pre-existing Secret names are allowed.
- Reject a recipe directory, catalog choice, generated substitute, or inferred default in place of the user's DGD.
  A specific manifest the user explicitly presents as their baseline is a user-provided DGD, whatever its origin;
  the rejection targets substitutes the USER did not supply.
- Do not patch cluster compatibility or performance settings during capture.
- Compute the canonical copy's SHA256 before writing the workload contract.
- If the DGD contradicts an explicit workload constraint, return the contradiction as a blocking question; do not
  silently choose which input wins.

Never overwrite the canonical DGD after it is captured. A changed user DGD starts a new experiment.

## Write And Validate The Workload

Write exactly:

```text
<EXP_ROOT>/user_workload.yaml
```

Follow the schema and rules in `agent-docs/rules/execution/user-workload.md`.

Before finalizing:

1. Preserve all explicit user constraints without rounding or reinterpretation.
2. Record the exact canonical DGD path and SHA256 under `deployment`.
2a. Record the user's stated budgets (GPU-hours, wall clock, failed-deploy limit) under `budgets`, verbatim;
    leave each `null` when the user declined to state one.
3. Represent permitted unknowns as `null`, `""`, or `[]` according to the schema.
4. Resolve supporting artifact paths beneath the expected filesystem and verify each existing local path.
5. Parse the result as YAML and require exactly one mapping document.
6. Require every local supporting path to exist, unless the user explicitly identified it as a future input.
7. Reject secret values and sensitive kubeconfig content.
8. Ensure no DGD body, deployment result, or optimization hypothesis appears in the file.
9. Compute the final file's SHA256.

Do not overwrite the contract after handoff to `recipe-deployer`. A different performance question may use a new
benchmark series within this contract; a material change to the user workload requires a new experiment root.

## Contract Corrections

A correction of fact is the user fixing a misreported description of the same workload (a wrong traffic number, a
fleet-level figure that should have been per-replica, a mistaken SLO value).

**Before handoff to `recipe-deployer`**: the contract is still this skill's working file; amend it, note the
correction inline, and recompute its SHA256.

**After handoff**: the contract and its hash are frozen and historical artifacts reference them; do not mutate either.
Instead:

1. Start a new experiment root via this skill, carrying the corrected values forward and re-capturing the canonical
   DGD copy there.
2. Write one append-only supersedence marker in the old root, `<OLD_EXP_ROOT>/SUPERSEDED`, recording the corrected
   field(s), the reason, the timestamp, and the new `EXP_ROOT`. Do not edit or delete any other artifact in the old
   root: its runs remain valid evidence about the conditions they actually measured.
3. Re-establish the baseline in the new root before proposing any candidate. Prior conclusions do not carry over;
   they may be cited as evidence about the superseded conditions.

This makes a post-handoff correction operationally identical to a material workload change (both open a new root);
the distinction is recorded by the SUPERSEDED marker, not by editing frozen files.

## Return

Return:

- exact `EXP_ID` and `EXP_ROOT`;
- exact `user_workload.yaml` path and SHA256;
- exact `user_provided_dgd.yaml` path and SHA256;
- a concise summary of fixed constraints and intentionally unspecified preferences; and
- any non-blocking limitations downstream roles must preserve.

The user interviewer hands both path-and-hash pairs directly to `recipe-deployer`. The workload path and hash remain
supporting context for `perf-analyzer`, `hypothesis-generator`, and `hypothesis-challenger`.

End with an operator handoff line before any deployment starts: state that the interview is complete, give the
contract path, and tell the operator that the engagement now runs a long unattended loop — without goal mode the
session pauses for input at every turn end — and that this is the moment to arm it (AGENTS.md, Long-Running Runs,
has the condition template; fill in the budget).

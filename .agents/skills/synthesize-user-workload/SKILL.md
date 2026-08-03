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
- exact model source and revision when the user fixes one;
- allowed hardware type and count, including heterogeneous allocations;
- Kubernetes context and existing namespace; and
- either an exact trace or enough traffic-shape information to configure a defensible benchmark.

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

If blocking facts remain, return the questions to the parent and stop without handing work to a downstream role.

## Establish The Experiment Root

Use the exact caller-supplied `EXP_ROOT` when present. Otherwise create one unused directory under `runs/` using a
stable, filesystem-safe `EXP_ID` derived from the UTC creation timestamp and workload profile slug. Never reuse or
overwrite an existing experiment directory.

The user interviewer owns creation of `EXP_ROOT`; downstream roles must receive its exact path rather than infer it
from directory order or modification time.

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
3. Represent permitted unknowns as `null`, `""`, or `[]` according to the schema.
4. Resolve supporting artifact paths beneath the expected filesystem and verify each existing local path.
5. Parse the result as YAML and require exactly one mapping document.
6. Require every local supporting path to exist, unless the user explicitly identified it as a future input.
7. Reject secret values and sensitive kubeconfig content.
8. Ensure no DGD body, deployment result, or optimization hypothesis appears in the file.
9. Compute the final file's SHA256.

Do not overwrite the contract after handoff to `recipe-deployer`. A different performance question may use a new
benchmark series within this contract; a material change to the user workload requires a new experiment root.

## Return

Return:

- exact `EXP_ID` and `EXP_ROOT`;
- exact `user_workload.yaml` path and SHA256;
- exact `user_provided_dgd.yaml` path and SHA256;
- a concise summary of fixed constraints and intentionally unspecified preferences; and
- any non-blocking limitations downstream roles must preserve.

The user interviewer hands both path-and-hash pairs directly to `recipe-deployer`. The workload path and hash remain
supporting context for `perf-analyzer`, `hypothesis-generator`, and `hypothesis-challenger`.

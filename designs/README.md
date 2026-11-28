<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design invariants

This directory records the design rules contributors and agents must preserve.
DEPs hold proposals, alternatives, and decision history. These documents derive
scoped invariants and review checks from those decisions; they do not transcribe
DEPs or require readers to reconstruct issue discussions.

## Find the applicable rules

Before implementing or reviewing a change, read every document whose scope
covers the change and its effects on callers, consumers, and generated resources.
Select by behavior as well as changed paths: a component can violate an operator
boundary without changing operator code.

| Document | Scope |
| --- | --- |
| [Component boundaries](COMPONENT_BOUNDARIES.md) | Deployment interpretation, component configuration, cross-component interfaces, DGD access, and related RBAC |
| [DGDR v1beta2 API design](operator/DGDR_V1BETA2_API_DESIGN.md) | DGDR v1beta2 typed inputs, defaults, validation, AISimulate lowering, input overrides, and Run snapshots |
| [Structural defaulting](operator/STRUCTURAL_DEFAULTING.md) | Operator mutating admission and its tests, across resources and API versions |
| [Structural validation](operator/STRUCTURAL_VALIDATION.md) | Operator validating admission and its tests, across resources and API versions |

For each applicable invariant, follow the affected producer and consumer paths
and check the document's review criteria. Report a violation with the invariant,
code location, and concrete evidence. Identify checks that could not be verified.
A rule in an existing base-branch document is the review baseline; changes to the
rule itself need explicit review and decision context.

## Maintain the rules

- Put globally applicable rules directly in `designs/`. Put area-specific rules
  in a subdirectory matching the responsible approver area, such as `operator/`.
  Keep document filenames in `UPPER_SNAKE_CASE.md`.
- Normally derive rules from a decided DEP. Cite the decision, state the scope,
  write actionable invariants and review checks, and record explicit exceptions.
  Existing engineering conventions may cite their implementation history instead.
- Keep motivation only where it explains how to apply a rule. Leave alternatives
  and discussion history in the DEP. Several DEPs may inform one document, and a
  DEP may affect more than one document.
- Update the affected documents when a later decision changes a rule. Do not
  silently broaden an exception or present a proposed DEP as already approved.
- Add new documents to the table above and link them from the relevant
  `AGENTS.md` scopes. Keep one canonical copy of each rule set.
- Declare ownership in `.github/codeowners/areas.yaml` and regenerate
  `CODEOWNERS`. Global design rules and the index belong to Architecture;
  `operator/` belongs to Operator. Multiple CODEOWNERS provide review visibility;
  GitHub accepts approval from any listed owner, rather than requiring each team.

These documents describe required design direction, including explicitly listed
migration exceptions. Adding them does not by itself migrate existing workloads.

---
name: investigate-protocol-compatibility
description: Investigates client-visible protocol compatibility between Dynamo and a framework server using schema candidates, source traces, and controlled behavioral probes. Use when triaging request or response differences, testing a compatibility claim, or developing an authorized focused fix; not for exhaustive schema equality or general deployment debugging.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - frontend
    - protocol
    - compatibility
    - testing
---

# Investigate Protocol Compatibility

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Use OpenAPI to narrow candidates, controlled requests to establish behavior, and
focused regressions to verify authorized fixes. The objective is reproducible
client-visible findings, not complete schema equality.

## Scope and Authority

Start from the user's client workflow, reported failure, or candidate list.
Establish the endpoints, unary/streaming transports, features, reference version,
model/tokenizer revisions, parser settings, Dynamo revision/configuration, and
deployment mode needed for the claim. State missing information and narrow the
claim rather than assuming a universal compatibility target.

Prioritize ordinary successful requests and client-breaking failures: rejection,
ignored controls, malformed responses, and broken streaming or error handling.
Keep priority separate from correctness; a deferred difference is not compatibility.

Investigation does not authorize implementation, policy alignment, issue/PR
publication, or CI triggers. Follow the user's scope and applicable compute,
artifact, ownership, and publication rules. Ask for direction before changing
intentional policies or extending the agreed workload/resource budget.

## Resources and Ownership

Read resources for the decisions they own; do not copy their manuals into the
investigation record.

- Read [Protocol Field Handling](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/protocol-field-handling.md)
  for source contracts, semantic ownership, lifecycle tracing, value semantics,
  and the distinction between admission and support.
- Consult the [frontend configuration reference](../../../docs/fern/pages/reference/components/frontend-configuration.mdx)
  and [NVIDIA request extensions](../../../docs/fern/pages/developer-guide/additional-resources/nvidia-request-extensions-nvext.md)
  for documented settings and Dynamo-specific fields. These do not replace the
  pinned framework's source contract or execution evidence.
- Before adding regression tests, read the [pytest guidelines](../../../.ai/pytest-guidelines.md)
  and [test model-size guardrails](../../../.ai/test-model-size-guardrails.md);
  reuse fixtures and tests in the affected subsystem.
- When a persistent bug worklog is needed, use [debug-session](../debug-session/SKILL.md).
  For an authorized fix, follow [issue-first](../issue-first/SKILL.md).

Optional OpenAPI tooling is being introduced by
[the comparator PR](https://github.com/ai-dynamo/dynamo/pull/15914) and
[the vLLM assessment PR](https://github.com/ai-dynamo/dynamo/pull/15921).
Check availability in the selected checkout; these are not prerequisites for this
skill. When present, read `scripts/protocol_compatibility/README.md` for commands
and links to maintained backend assessment recipes, and the frontend
`openapi-request-contract-comparison.md` guide for interpretation. Read the
selected assessment's completion record for its actual scope, pins, and evidence;
a recipe alone does not establish that an assessment ran. Treat historical
results as revision-scoped observations.
After those resources merge, prefer checkout-relative links to their maintained
owners rather than copying commands or pinning the skill to their original PRs.

## Narrow the Candidates

OpenAPI differences, client reports, code inspection, and interaction failures
are all valid entry points. Skip discovery when a useful reproducer already
exists. Reuse evidence only when its source, configuration, and claim match.

For schema inputs, record provenance and check reference resolution, composition,
normalization, and comparator coverage before selecting candidates. Distinguish
the raw comparison, augmented or rendered report, reviewed assessment, and newer
capture recipe. Preserve their source/configuration pins and artifact identities.
When using another representation, record an explicit crosswalk and verify the
selected declarations and deltas; added metadata can change bytes without changing
comparison evidence. Do not assume byte identity or silently regenerate the report.

An unsupported construct or missing schema is a coverage gap, not proof of a
runtime defect or compatibility. If coarse root findings or gaps hide individual
candidates, retain them and, within the authorized investigation scope, define a
bounded source-first candidate set with pointers to each parent finding or gap
and pinned source. Keep these candidates distinct from generated findings; an
empty generated category does not establish that no migration risk exists.

| Candidate | First question |
| --- | --- |
| Missing field declaration | Does runtime already accept and implement it? |
| Default, alias, or composition difference | Do equivalent requests actually behave differently? |
| Different limit or admission rule | Is this an intentional policy, and what client workflow is affected? |
| Field declared by both | Is it preserved and enforced, including relevant combinations? |
| Response difference | Does it affect client parsing, interpretation, or control flow? |

Group findings by likely implementation cause while preserving their original
identities and endpoint-specific evidence. One shared field may need one
investigation, but a chat result does not establish completion behavior.

Trace parsing, admission/validation, normalization, frontend semantics, worker
transport, backend effect, and response serialization as applicable. Inspect the
reference implementation too: an exposed option can be ignored or overridden by
its pinned execution path.

## Match Evidence to the Claim

Label what each result establishes rather than treating all checks as equivalent:

- **Source expectation:** the pinned implementation suggests a path or effect;
  it does not show that a request exercised it.
- **Isolated helper execution:** demonstrates the tested helper's behavior, not
  that production callers supply those inputs or preserve its result.
- **Full-path CPU test:** exercises the included admission, transformation, and
  serialization path. State mocked or omitted layers; it does not automatically
  establish engine/model effects.
- **Paired HTTP/inference probe:** compares direct-framework and Dynamo behavior
  for the pinned request and configuration. State whether inference actually ran;
  HTTP-only or mock-backed checks do not establish generation-effect parity.

Choose evidence for the decision, not a mandatory progression through every level.
Use the field-handling contract above to identify which semantic owners a check
covers. Before gathering more evidence, state which possible outcome would change
the scoped disposition, priority, or next action.

## Establish a Controlled Counterexample

For a suspected runtime difference, send the same request to pinned direct
framework and Dynamo servers with comparable model and relevant configuration.
Record unavoidable differences. Start with a successful control and one changed
input that distinguishes the hypothesis. Assert the requested effect, not only
HTTP 200, field acceptance, or backend delivery.

Choose observable behavior rather than exact stochastic text equality. For
example, ask for one value while constraining output to another to test constraint
enforcement. Use tokenizer-verified forced tokens plus a stop condition to test
stopping. Combine options when that interaction is the suspected failure; a
single-option pass does not cover combinations.

Retain the request, status, relevant headers, raw JSON or server-sent events
(SSE), assertions, logs, and source/configuration pins. For streaming, distinguish
pre-header HTTP rejection from an error after HTTP 200; inspect event order,
payloads, termination, and cancellation when relevant. Capture raw events before
client libraries aggregate them. Redact secrets before sharing evidence.

Expand a reproduced witness to relevant boundaries and interactions, not an
unbounded parameter matrix. Preserve failed attempts; separate setup failures
and inconclusive probes from product failures. A mock can isolate a layer, but
cannot establish direct-engine effect parity. Reproduce the reference's requested
effect before treating its output as the compatibility target.

## Decide and, When Authorized, Fix

Give each tested behavior an evidence-backed disposition:

- **Runtime defect:** demonstrated client behavior violates the intended contract;
  identify the responsible layer and propose a focused fix.
- **Schema-only defect:** runtime meets the tested contract but the declaration
  does not; correct the declaration when authorized.
- **Comparison false positive:** the reported mismatch does not establish the
  claimed incompatibility; retain the witness and likely comparator cause.
- **Intentional divergence:** record the contract difference and impact; request a
  policy decision before alignment.
- **Reference limitation or inconclusive:** name what was not established and why;
  do not report compatibility or silently emulate a reference bug.

A disposition applies to the tested behavior, not automatically the whole field.
For example, equivalent omission behavior can dismiss a missing-default finding
without proving equal ranges, null handling, or multi-choice responses.

For an authorized runtime fix, preserve the parent-failing regression and run it
unchanged against the fix, then test adjacent behavior likely to regress. Explain
any unavoidable source/environment differences. A declaration-only change needs
schema/serialization evidence, not automatic GPU execution. Keep focused
regressions separate from broad stack qualification.

Target `main` when the fix is independent. Separate schema or test packaging when
it would otherwise create an unnecessary dependency. Retain real dependencies
when the implementation or regression needs an unmerged API or behavior; do not
duplicate that prerequisite just to claim independence.

## Handoff and Completion

For each finding, report the client-visible claim and priority, affected scope,
control/reproducer and evidence links, cause/confidence, disposition, and next
action. Include regression and fix results only when actually performed. Keep
tool false positives and reference limitations as durable Markdown records with
reproducers and likely causes for later tool improvements, in the task's approved
evidence location rather than an invented public support matrix.

Finish when the agreed scope has dispositions and explicit unresolved limits, or
report the specific authority/resource blocker. List untested candidates as such.
Triage may finish with ranked, source-supported unknowns and concrete next checks;
this is not a runtime compatibility claim or permission to implement a fix.
Gather more evidence only when it could change the scoped decision. Stronger
runtime claims and authorized fixes still need witnesses covering the claimed
behavior, including the parent-failing regression for a fix.
Do not turn a successful focused test into a whole-field, combined-stack, or
all-backend compatibility claim, and do not keep searching for exhaustive parity.

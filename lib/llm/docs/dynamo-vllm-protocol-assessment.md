<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Assess Dynamo against vLLM serve

Use the direct assessment to answer: **what differs, what changed, what remains
unresolved, and what needs action?** Start with `report.md`, not an inventory diff.
The upstream-to-upstream tools in [the B1 guide](vllm-protocol-tooling.md) remain
available as supporting source-change analysis; they cannot discover a
longstanding Dynamo gap when upstream has not changed.

## Scope and invariants

The initial adapter compares vLLM serve request declarations against Dynamo's
effective Rust request declarations and selected handling paths:

- `/v1/chat/completions` is the primary endpoint.
- `/v1/completions` is included for compatibility.

SGLang and TensorRT-LLM adapters are not implemented. Response schemas, streaming
wire contracts, error behavior, tokenizer/model behavior, and exhaustive runtime
conformance are deferred. Known response-projection implications may appear as
handling evidence, but are not response-parity claims.

Keep these invariants when changing the tooling or a framework version:

1. Compare Dynamo directly with the native server even on the first run and when
   neither side changed. Missing history never suppresses existing gaps.
2. Record exact source revisions and extraction provenance. Never import upstream
   engine modules to discover their declarations.
3. Preserve unknown facts and coverage failures. Missing extraction is not a
   permissive schema, support, a resolved gap, or a successful compatibility check.
4. Derive handling from implementation sources. A typed field, accepted input,
   passthrough vocabulary, or forwarding path alone does not establish support.
5. Separate observations from reviewed decisions and runtime evidence. Never
   manufacture an intentional divergence or carry runtime evidence across versions.
6. Keep generated inventories separate from reviewed decisions and support policy.
   Ordinary Rust builds continue to use the committed compact `vllm_fields.rs`.

## Prerequisites

Use Python 3.12 and `PyYAML==6.0.2`, a Dynamo Git repository containing the selected
commits, and a vLLM Git repository containing the selected upstream commits. Local
uncommitted runtime changes are not assessed: create an identifiable commit first.
The extractor's own dirty source state is separately identified by tool hashes.

Provide a `--crate-cache` containing the `dynamo-protocols` and `async-openai`
`.crate` archives named by the selected Dynamo `Cargo.lock`. Their checksums are
verified against that lockfile before parsing; neither crate is compiled. Missing
sources become explicit coverage diagnostics. The CI workflow downloads these
archives from crates.io and verifies the same checksums.

Native external SDK sources are not imported or guessed. Unavailable externally
defined types are unresolved, with their reference names retained. This can make
current real-source assessments fail extraction completeness. Restore coverage
or explicitly change the agreed assessment scope through review; do not label a
partially extracted report green.

## Workflows

Always use a new output directory. Commands below do not check out source branches,
modify runtime pins, or write decisions. `FULL_*_SHA` means a full lowercase
40-character commit SHA, not a tag or branch name.

### Initial baseline

Select a configured platform, because CPU/XPU and CUDA may pin different versions:

```sh
python scripts/run_protocol_assessment.py \
  --dynamo-commit FULL_DYNAMO_SHA --platform cpu \
  --upstream-repo /path/to/vllm --crate-cache /path/to/crates \
  --output-dir /path/to/new-baseline
```

The driver verifies that the selected revision's `container/context.yaml` and
`vllm_pins.json` agree. Open `new-baseline/report.md`, then the linked
`current/report.md`. The generated `workflow.json` records platform, configured
pin, input hashes, candidate selection, and the command result.

### Framework-version bump or Dynamo-only change

Commit the proposed runtime change. For a framework bump, update the container
version, immutable source pin, and generated compact vocabulary together. Compare
the pre-change and proposed Dynamo revisions:

```sh
python scripts/run_protocol_assessment.py \
  --dynamo-commit FULL_PROPOSED_DYNAMO_SHA \
  --baseline-dynamo FULL_PRE_CHANGE_DYNAMO_SHA --platform cpu \
  --upstream-repo /path/to/vllm --crate-cache /path/to/crates \
  --output-dir /path/to/new-change-assessment
```

Both assessments are generated from their respective committed pins. A Dynamo-only
change follows the same procedure: the upstream revision stays the same. Baseline
action-required results do not prevent generating the proposed assessment. Tool
failures do stop the run and must be fixed before reviewing its output.

For a historical Dynamo revision that predates `vllm_pins.json`, use the direct
CLI and explicitly select the upstream commit. Generate a baseline, then assess
the proposed Dynamo revision against the **same** upstream commit:

```sh
python scripts/assess_protocol_compatibility.py \
  --dynamo-repo /path/to/dynamo \
  --dynamo-commit FULL_PRE_CHANGE_DYNAMO_SHA \
  --upstream-repo /path/to/vllm --upstream-commit FULL_UPSTREAM_SHA \
  --crate-cache /path/to/crates --output-dir /path/to/historical-baseline
# Exit 1 means the report exists but action remains; inspect it before continuing.
python scripts/assess_protocol_compatibility.py \
  --dynamo-repo /path/to/dynamo \
  --dynamo-commit FULL_PROPOSED_DYNAMO_SHA \
  --upstream-repo /path/to/vllm --upstream-commit FULL_UPSTREAM_SHA \
  --crate-cache /path/to/crates \
  --previous /path/to/historical-baseline/report.json \
  --output-dir /path/to/historical-comparison
```

This is an explicit-revision comparison, not verification of configured runtime
pins at the historical revision. Do not invent missing historical pin metadata.

To preserve longer-running decision and lost-coverage history, supply
`--previous /path/to/prior/current/report.json` instead of `--baseline-dynamo`.
Previous scope must match exactly, including pipeline. A scope change starts a
new baseline, rather than silently resolving old findings.

### Explicit candidate or periodic check

```sh
python scripts/run_protocol_assessment.py \
  --dynamo-commit FULL_DYNAMO_SHA --platform cpu \
  --upstream-candidate FULL_CANDIDATE_VLLM_SHA \
  --upstream-repo /path/to/vllm --crate-cache /path/to/crates \
  --output-dir /path/to/new-candidate-assessment
```

Unless `--previous` is supplied, this compares the configured native version and
the candidate at the **same** Dynamo revision. It does not adopt the candidate.
Rechecking the same candidate still reports existing Dynamo differences.

For arbitrary exact revision pairs without configured-platform selection, use
`scripts/assess_protocol_compatibility.py --dynamo-repo ... --dynamo-commit ...
--upstream-repo ... --upstream-commit ... --output-dir ...`.

### Reproducible real-source version-bump rehearsal

These are real source revisions, not synthetic models:

- Dynamo integrated compatibility revision:
  `adf9ca44d90ca14137cf178f80475c4b5953c0a6`.
- Configured CPU vLLM 0.29.0:
  `98dff2a81d747d1dba01a47f939f48c3526d4206`.
- Candidate vLLM 0.30.0:
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

Use the candidate command above with these values and `--platform cpu`. The
baseline and current reports contain both revision pairs, differences, selected
behavioral changes, and gate outcomes. This is a pre-adoption bump rehearsal;
it is not a claim that those Dynamo runtime pins were changed. Counts depend on
the extractor source and are recorded in the generated artifacts. The current
bounded extraction has known unresolved facts, so exit 1 is expected until the
required coverage and review work are addressed.

## Read and review findings

The report separates structural differences, handling observations, Dynamo-only
inputs, selected upstream behavior changes, and coverage diagnostics. Short
summaries lead each finding; expandable facts retain exact values and fingerprints.
Source and evidence links connect observations to their pinned implementations.

Lifecycle has a precise meaning:

| Lifecycle | Meaning |
| --- | --- |
| New | First observed in this assessment history, not necessarily newly introduced upstream |
| Changed | Same logical finding, different relevant facts |
| Unchanged | Same identity and fact fingerprint |
| Resolved | Difference disappeared while the relevant comparison retained coverage |
| No longer assessable | Difference disappeared, but coverage was lost; restore coverage |

Handling can include interpretation, forwarding, both, and conditional rejection.
Incomplete path analysis remains unresolved even when some effects are identified.
Predicates and implementation hashes are source evidence, not evaluated execution
paths. Pipeline selection records the assessment scope; it does not turn unproven
reachability into a support claim.

## Decisions and evidence

Store direct-assessment decisions in
`lib/llm/src/protocols/openai/compatibility/assessment/decisions.json` or pass another
reviewed registry with `--decisions`. The committed registry starts empty: no
finding has been silently approved. Re-run into a new directory after review.

An illustrative entry, **not an approval**, is:

```json
{
  "identity": "/v1/chat/completions#some_field:default",
  "fingerprint": "COPY_64_CHARACTER_FINGERPRINT_FROM_REPORT",
  "scope": {"COPY": "exact scope object from report.json"},
  "reviewed_revisions": {"dynamo": "FULL_SHA", "vllm": "FULL_SHA"},
  "disposition": "tracked_gap",
  "owner": "responsible-team",
  "rationale": "Explain the observed difference and intended outcome.",
  "tracking": "https://github.com/example/project/issues/1",
  "next_action": "Implement or test the specified behavior.",
  "carry_static_disposition": false,
  "evidence": [{"kind": "source", "url": "https://github.com/example/project/blob/FULL_SHA/file"}]
}
```

Allowed dispositions are `tracked_gap`, `intentional_divergence`, `unsupported`,
`non_material`, and `needs_runtime_evidence`. Every entry requires identity,
fingerprint, exact scope and revision applicability, rationale, owner, next action
(including an explicit no-action reason when appropriate), and evidence.
Unresolved work requires tracking. Evidence kinds are `source`, `implementation`,
and `runtime`; runtime entries additionally require exact `revisions` and `scope`.

Matching is deterministic: identity, fingerprint, scope, then revisions. A reviewer
may set `carry_static_disposition: true` to retain an unchanged static decision
across revisions; it does not carry runtime evidence. Changed contract or handling
facts invalidate the decision. Source-location moves alone do not. Evidence URLs
are references, not fetched or independently verified by this tool.

Legacy B1 files under `compatibility/decisions/PREVIOUS_SHA-CANDIDATE_SHA.json`
remain valid only for upstream-pair drift reports. They are not silently migrated
or accepted by the direct-assessment registry. Review the corresponding direct
finding and create a new entry with its own identity, facts, scope, and evidence.

## Gate and release policy

| Gate | What it establishes |
| --- | --- |
| Extraction | Required facts were assessed within the declared static scope |
| Review | Required findings have applicable reviewed dispositions |
| Runtime conformance | Not established by this static command; requires scoped test evidence |
| Release policy | Only the supplied static policy was checked; runtime release approval remains separate |

Exit 0 means the enforced static gates passed. Exit 1 means a report was generated
and coverage, review, lost-history, or support-policy action remains. Exit 2 means
invalid inputs or a tool/Git failure. Malformed decisions cannot approve findings.
Unchanged findings with applicable decisions need no repeated triage; historical
unreviewed findings still need a decision. Incomplete extraction cannot pass even
if every coverage item has a disposition.

Pass `--support-policy .../assessment/support-policy.json` to evaluate the initial
static policy. It requires complete extraction/review and starts with no additional
field-specific release promises. Maintainers can add finding identities to
`required_findings_absent` when a promised invariant forbids those differences.
A tracked gap then blocks that policy even after review. This registry is a release
gate, not a duplicate supported-field catalog. Policy satisfaction never replaces
behavioral testing or the maintainer's release decision.

## CI, periodic ownership, and escalation

`.github/workflows/protocol-assessment.yml` connects this workflow to relevant PRs,
manual dispatch, and a weekly check. PRs assess the head commit against the base
commit's configured pins when available; initial rollout still assesses the head
when the base has no pins. The platform matrix covers the committed CPU, CUDA,
and XPU configurations. Update the matrix when configured platforms change.

Weekly runs resolve the explicit ref in `assessment/candidate.json` to a full
commit before assessment. The default candidate is vLLM's `refs/heads/main`, not
a runtime-version adoption policy. All source revisions and tool hashes are
retained in artifacts. A failed job raises action through the workflow status;
reports are uploaded even when the assessment returns 1. No issue or PR is
automatically posted, and the workflow has read-only repository permissions.

Frontend CODEOWNERS own finding triage, compatibility decisions, and candidate
policy. Ops CODEOWNERS own workflow operation and artifact availability. After an
action-required run, the frontend reviewer records a disposition and opens or
links follow-up work for unresolved findings. Before a version bump merges, they
must inspect `current/report.md`, restore required coverage, review new/changed
findings, and run targeted behavior tests for the promised support contract.

The workflow file is an implementation, not evidence that scheduling or branch
protection is already enabled upstream. It becomes scheduled on the default branch
after merge with Actions enabled. Repository administrators must add its platform
checks to required checks if merge enforcement is desired. Until then, the version
bump reviewer must run and attach the assessment manually. CI artifacts expire after
30 days; retain release/review evidence in the project's durable validation archive.

## Extraction boundaries and development checks

Python inheritance, aliases, reachable local nested types, field defaults,
bounded integer constant expressions, and supported Pydantic alias configuration
are parsed statically. Unsupported configuration, dynamic factories, external
types, and metadata remain diagnostics. Declaration defaults and constraints do
not prove the effects of validators or request-time overrides.

Rust extraction follows declarations, import aliases/reexports, flattening,
selected serde attributes, and the lockfile-verified declaration crates. Custom
deserializers, conditional definitions, recursive schemas, unsupported enum forms,
and unproven handling paths remain explicit coverage gaps. B5 admission vocabulary
is combined with its actual validation and passthrough code, not treated as a
support allowlist.

Known nested object locations are retained separately even if a sibling type is
unresolved. Same-named nested fields, such as a native root input versus a field
under `nvext`, are placement candidates for review, not inferred aliases or
semantic equivalents. They do not replace the missing root input's handling
finding. Custom deserializers do not grant declaration-only nested locations;
arrays, maps and enum alternatives remain in structural schemas. Segment arrays
distinguish literal dotted JSON keys from nested object paths.

Selected request accessors (`get_*`), validation methods, and `response_generator`
reads identify frontend interpretation candidates. A response-generator read also
links same-named payload accesses in that endpoint's delta and aggregation source.
These links include implementation fingerprints and source predicates, so changes
invalidate prior handling facts. They are review evidence, not proven data-flow
edges: invocation, effective response placement, streaming serialization, and
runtime behavior remain unverified. Another endpoint's same-named payload is not
substituted when the selected endpoint has no identified projection candidate.

Behavior-change candidates are grouped request-class method bodies and selected
validator/normalizer helpers in endpoint source modules. Affected field lists are
candidate scope, not whole-program dataflow proof. Unrelated module bindings are
not automatically triage requirements. Broader behavior stays outside this scope.

Run the source-only regression suite with:

```sh
python -m unittest discover -s scripts -p 'test_*protocol*.py' -v
```

The Git/CLI tests use synthetic source repositories and are labeled accordingly;
they validate tooling mechanics, not native-server runtime behavior. Real-source
assessments and targeted runtime probes must remain distinguishable in evidence.

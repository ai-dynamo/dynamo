<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Assess Dynamo against vLLM serve

## Design decision: layered compatibility assessment

Status: B1 (`codex/vllm-protocol-tooling`) implements the request-contract base
in assessment format v2. Optional source investigation and its tests are a
separate stacked change (`codex/vllm-source-investigation`).
Behavioral conformance remains a separate, unimplemented test workflow.
Native-schema export migration is still in progress on its own branch.

There are two acceptance layers (contract and behavior), plus optional supporting
investigation—not three sequential acceptance gates. The design below describes
their relationship; the current implementation section describes B1 alone.

### Context and decision

Developers must be able to identify and address contract differences, then
separately establish behavioral conformance. A source-analysis limitation must
not be presented as a protocol mismatch or as incomplete schema extraction.
Keep one discoverable workflow, but separate mechanisms, evidence, counts,
review decisions, and acceptance gates:

```text
Pinned Dynamo + pinned native server + explicit endpoint/platform scope
                              |
          1. Compare declared request contracts
             -> fix differences or review exceptions -> reassess
                              |
          2. Establish scoped behavioral conformance
             -> run targeted tests -> fix or review exceptions -> retest

Optional source investigation assists either stage; it is not a prerequisite.
```

This is a developer sequence, not a requirement to finish every field before
testing any behavior. Stage 2 can start for a selected subset while other contract
work continues. Each stage must be independently runnable and report its own
result. A contract-only run must not start serving workloads or require a prior
report, an upstream version change, or completion of source-path analysis.

### Stage 1: contract differences

Compare one selected Dynamo revision directly against the configured native pin,
or an explicitly selected native revision. The first run reports longstanding
differences without requiring history. B1's initial target is vLLM serve, primarily
chat completions, with completions included for compatibility; other server
adapters and response-schema comparison are not implied by this design change.

Use implementation-owned schema exports, supported metadata, and a structural
diff to compare field placement/names/aliases, types and nesting, requiredness,
nullability, declared defaults, and expressible constraints. These describe the
declared request contract, not every input accepted by arbitrary runtime code.
The exporter migration is the planned implementation, not a current capability.

Report separately:

- Contract differences, with exact values, source/schema references and direction
  of the difference. A wider Dynamo input domain is not automatically a defect.
- Contract coverage gaps, such as missing referenced schemas, unsupported keywords,
  or custom deserialization that makes an in-scope declared shape uncertain.
- Dynamo-only inputs, informational unless an explicit contract requirement is
  violated; do not count them as native-field mismatches.

The developer restores required schema coverage, fixes unintended differences,
records scoped exceptions with rationale/owner/evidence, and reassesses. A reviewed
tracked gap is not a fix: contract review can be complete while acceptance remains
blocked by the selected support policy. A coverage-gap disposition alone cannot
make extraction complete. Scope exclusions require an explicit reviewed scope
change, not merely a decision on a finding.

Stage 1 is sufficient for the selected contract scope when required contract facts
are available, required differences are fixed or covered by approved exceptions,
and the contract review/policy gates pass. Its success says nothing about execution,
forwarding, response projection, or behavioral conformance.

### Stage 2: behavioral conformance

Use a separate test mechanism, reusing existing differential/runtime infrastructure
where suitable. For an agreed finite support scope, compare Dynamo and the native
server with equivalent model/tokenizer revisions, backend settings, requests, and
serving paths. Record exact server/Dynamo commits, test versions, configuration,
and raw results. Explicit expectations cover reviewed intentional divergences;
do not silently normalize away the behavior under test.

Tests may cover runtime validation and defaults, interpretation/forwarding,
conditional rejection, response projection, errors and streaming behavior as
promised by that scope. Define assertions and tolerances appropriate to each test;
do not require byte-identical stochastic model output to establish protocol behavior.
Behavior tests are needed even when Stage 1 finds no schema difference.

Report pass, fail, blocked, and not tested separately, with coverage denominators
and scoped exceptions. Missing or stale test evidence is not a pass. Stage 2 is
sufficient only when the required suite has applicable passing evidence or approved
exceptions under the behavioral support policy. Not every native-server feature
is automatically required, and review alone cannot substitute for execution.

This stage is not implemented by B1's current static command. Contract-only success
must display behavioral conformance as not assessed. Overall acceptance, when
requested, requires both stage-specific acceptance results for the agreed scope;
neither stage overwrites or implicitly approves the other.

### Optional source investigation

Retain useful references and fingerprints for validators, forwarding, rejection,
transformation and projection paths as investigation notes. Upstream source changes
can help select tests even when schemas do not change. These are hypotheses and
review aids, not proof of runtime reachability or conformance.

Notes have independent provenance, freshness and unresolved status. Missing or
incomplete tracing must not fail the contract gate, inflate contract-difference
counts, or be counted as failed behavioral tests. A source-only change can stale
an investigation note or require behavioral retesting without invalidating an
unchanged schema-only decision. If investigation discovers a schema-export defect,
record a separate evidence-backed contract coverage gap rather than silently
changing the stage's result.

### Reporting, applicability and acceptance criteria

One top-level report should link independent contract results, behavioral results
and optional investigation notes. Do not publish a combined "compatibility issue"
count or one undifferentiated coverage gate. Show contract coverage/review/policy,
behavioral test coverage/outcomes/policy, and investigation availability separately.
No new commands or artifact filenames in this section are an implemented CLI promise.

Decision identities, fingerprints and applicability must be layer-scoped. Keep
exact revision/configuration applicability for runtime evidence; unchanged schema
facts do not carry runtime evidence forward. Preserve findings and decisions during
migration with an explicit old-to-new mapping and review any changed meaning.

Implementation acceptance requires demonstrations that:

1. An initial contract assessment needs neither comparison history nor behavior tests.
2. A contract defect or missing required schema fails Stage 1, but missing source
   tracing does not. Source investigation can be disabled without changing its result.
3. Complete contract facts and approved differences allow Stage 1 to pass while
   Stage 2 remains not assessed; no overall parity claim is emitted.
4. A behavioral regression with unchanged schemas fails Stage 2 independently.
5. Reviewed gaps, absent/stale tests and loss of coverage cannot silently become passes.
6. Historical handling observations remain available as investigation evidence,
   while contract decisions no longer depend on unrelated handling fingerprints.

### Options and consequences

Keeping the former combined static gate required fewer changes but conflated
unknown execution paths with missing contract facts. Completely disconnected
tools make scope/provenance and decisions harder to find. The selected approach
uses separate mechanisms behind one entry point/report index: more explicit
result interfaces, but independently actionable stages and unambiguous claims.

The base implements declaration fingerprints, completeness, review gates and
contract counts without source investigation. JSON `findings`, `retired_findings`,
and `gates` describe contracts only; `investigation.status` is `not_implemented`
and `behavioral_conformance.status` is `not_assessed`.
Previously recorded advisory notes remain under `investigation.unobserved_notes`;
legacy v1 migration records remain available. Neither is regenerated or evaluated.
The optional stacked investigation layer must preserve identical contract
findings/gates for the same inputs.
The legacy `gates.runtime_conformance` reminder is retained for consumers, but is
not a behavioral result or a contract gate. CLI exit 0 is not overall acceptance.

Behavioral suite selection, execution and evidence evaluation remain separate work.
Existing endpoint tests can be reused, but merely linking a test or recording a
runtime evidence URL does not make this CLI evaluate or approve it.

## Current implementation

Use the direct assessment to answer: **what differs, what changed, what remains
unresolved, and what needs action?** Start with `report.md`, not an inventory diff.
The base runs contract comparison independently of source investigation.
Unlike an upstream-only diff, it discovers longstanding Dynamo gaps
even when upstream has not changed. See the [tooling package](../../../scripts/protocol_compatibility/README.md)
for its module layout and command reference.

## Scope and invariants

The initial adapter compares vLLM serve request declarations against Dynamo's
effective Rust request declarations:

- `/v1/chat/completions` is the primary endpoint.
- `/v1/completions` is included for compatibility.

SGLang and TensorRT-LLM adapters are not implemented. Response schemas, streaming
wire contracts, error behavior, tokenizer/model behavior, and exhaustive runtime
conformance are deferred. Source-projection investigation belongs to the optional
stacked change and would not establish response parity.

Keep these invariants when changing the tooling or a framework version:

1. Compare Dynamo directly with the native server even on the first run and when
   neither side changed. Missing history never suppresses existing gaps.
2. Record exact source revisions and extraction provenance. Never import upstream
   engine modules to discover their declarations.
3. Preserve unknown facts and coverage failures. Missing extraction is not a
   permissive schema, support, a resolved gap, or a successful compatibility check.
4. Do not infer behavior from declarations. A typed field, accepted input,
   passthrough vocabulary, or forwarding path alone does not establish support.
5. Separate observations from reviewed decisions and runtime evidence. Never
   manufacture an intentional divergence or carry runtime evidence across versions.
6. Keep generated inventories separate from reviewed decisions and support policy.
   Ordinary Rust builds continue to use the committed compact `vllm_fields.rs`.

## Prerequisites

Use Python 3.12 with the [pinned tooling dependencies](../../../scripts/protocol_compatibility/requirements.txt),
a Dynamo Git repository containing the selected
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

Run all commands from the repository root. The single entry point is
`python -m scripts.protocol_compatibility`. Internal modules are not standalone
scripts. Choose `assess --platform` for configured pins or `assess
--upstream-commit` for an explicit revision; the two modes are mutually exclusive.

### Verify pins and regenerate artifacts

```sh
python -m scripts.protocol_compatibility check-pins --upstream-repo /path/to/vllm
python -m scripts.protocol_compatibility generate-inventory --upstream-repo /path/to/vllm
python -m scripts.protocol_compatibility generate-inventory --upstream-repo /path/to/vllm --check
```

Only the compact `vllm_fields.rs` vocabulary is committed. Add
`--inventory-output /path/to/inventory.json` when detailed declarations are useful
for debugging; routine assessment review does not require this file. Generated
vocabulary membership means a native field is known, not that Dynamo supports it.

Historical N-2 adapter fixtures remain consumed by separate mixed-version tests.
Regenerate them with `python -m scripts.protocol_compatibility
generate-release-fixtures` and check freshness with `--check`. Their generation
does not execute historical source or establish mixed-version interoperability.

### Initial baseline

Select a configured platform, because CPU/XPU and CUDA may pin different versions:

```sh
python -m scripts.protocol_compatibility assess \
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
python -m scripts.protocol_compatibility assess \
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
python -m scripts.protocol_compatibility assess \
  --dynamo-repo /path/to/dynamo \
  --dynamo-commit FULL_PRE_CHANGE_DYNAMO_SHA \
  --upstream-repo /path/to/vllm --upstream-commit FULL_UPSTREAM_SHA \
  --crate-cache /path/to/crates --output-dir /path/to/historical-baseline
# Exit 1 means the report exists but action remains; inspect it before continuing.
python -m scripts.protocol_compatibility assess \
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
python -m scripts.protocol_compatibility assess \
  --dynamo-commit FULL_DYNAMO_SHA --platform cpu \
  --upstream-candidate FULL_CANDIDATE_VLLM_SHA \
  --upstream-repo /path/to/vllm --crate-cache /path/to/crates \
  --output-dir /path/to/new-candidate-assessment
```

Unless `--previous` is supplied, this compares the configured native version and
the candidate at the **same** Dynamo revision. It does not adopt the candidate.
Rechecking the same candidate still reports existing Dynamo differences.

For arbitrary exact revision pairs without configured-platform selection, use
`python -m scripts.protocol_compatibility assess --dynamo-repo ... --dynamo-commit ...
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
baseline and current reports contain both revision pairs, contract differences,
and gate outcomes. This is a pre-adoption bump rehearsal;
it is not a claim that those Dynamo runtime pins were changed. Counts depend on
the extractor source and are recorded in the generated artifacts. The current
bounded extraction has known unresolved facts, so exit 1 is expected until the
required coverage and review work are addressed.

## Read and review findings

The report separates declared contract differences, Dynamo-only
inputs, and contract coverage diagnostics. Short
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
across revisions; it does not carry runtime evidence. Changed contract facts
invalidate the decision; handling-only changes do not. Source-location moves alone do not. Evidence URLs
are references, not fetched or independently verified by this tool.

The contract registry uses `dynamo-native-decisions/v2` with `"layer": "contract"`.
Wrap the illustrative entry above in its `decisions` array. Nonempty v1 registries
are rejected: re-review and bind decisions to the new contract-only fingerprints;
an empty v1 registry is harmless and remains accepted. Handling/source notes cannot
be placed in the contract registry or `required_findings_absent` policy list.

Previous direct v1 reports are accepted as history, not as approvals. The
`history_migration` object records an explicit mapping and retains legacy handling
observations as investigation evidence. Contract observations receive declaration-only
fingerprints; missing native input slots are now explicit `input_slot` observations.
Their old handling observations are not mislabeled as resolved contract defects.
V2 reports keep the migration record when history is carried forward.

The consolidated tooling has one contract decision registry. Experimental upstream-pair
decisions from earlier B1 revisions are not accepted or silently reinterpreted.
Retain those historical reports as evidence, then review the corresponding direct
finding and create an entry with its own identity, facts, scope, and evidence.
There is no separate `--require-triage` mode: assessment always evaluates the
extraction and review gates.

## Gate and release policy

| Gate | What it establishes |
| --- | --- |
| Extraction | Required declaration facts were assessed within the contract scope |
| Review | Required contract findings have applicable reviewed dispositions |
| Runtime conformance | Not established by this static command; requires scoped test evidence |
| Release policy | Only the supplied static policy was checked; runtime release approval remains separate |

Exit 0 means the enforced contract gates passed. Exit 1 means a report was generated
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
deserializers, conditional definitions, recursive schemas and unsupported enum forms
remain explicit contract coverage gaps. Handling paths and B5 admission/forwarding
analysis are outside the base. A passthrough vocabulary is not a typed schema
or support allowlist.

Known nested object locations are retained separately even if a sibling type is
unresolved. Same-named nested fields, such as a native root input versus a field
under `nvext`, are placement candidates for review, not inferred aliases or
semantic equivalents. They do not replace the missing root input's declaration
finding. Custom deserializers do not grant declaration-only nested locations;
arrays, maps and enum alternatives remain in structural schemas. Segment arrays
distinguish literal dotted JSON keys from nested object paths.

The optional investigation branch owns request-accessor, validation, forwarding,
response-projection and selected upstream behavior-change analysis, including
their tests. Shared source snapshots may retain method fingerprints for provenance
and inventory generation; B1 does not interpret them as handling or conformance.

Run the source-only regression suite with:

```sh
python -m unittest discover -s scripts/protocol_compatibility/tests -t . -v
```

The Git/CLI tests use synthetic source repositories and are labeled accordingly;
they validate tooling mechanics, not native-server runtime behavior. Real-source
assessments and targeted runtime probes must remain distinguishable in evidence.

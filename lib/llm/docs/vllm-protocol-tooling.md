<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM protocol source tooling

These command-line tools inventory native-server declarations and report source
changes for `/v1/chat/completions` and `/v1/completions`. They do not establish
runtime parity, change request admission, or enable a CI/scheduled workflow.

## Inputs and outputs

- `container/context.yaml` remains the authority for configured engine versions.
- `lib/llm/src/protocols/openai/compatibility/vllm_pins.json` maps each configured
  platform/version to an immutable upstream commit.
- `vllm_fields.rs` is the committed, generated field vocabulary. Normal builds
  do not need an upstream checkout or detailed JSON inventory.
- Optional inventory JSON records request declarations, inherited fields,
  defaults, input aliases, and source provenance. Generate it with
  `--inventory-output /path/to/inventory.json`; it is not required in Git.
  Neither the vocabulary nor the inventory is a support allowlist.
- `summary.md` in each drift run is the developer-readable review entry point:
  comparison/coverage gates, all changes, before/after values, pinned source links,
  reachable consumers, recorded dispositions, ownership, evidence, and actions.
  Detailed JSON reports and source snapshots remain alongside it for automation.
- `components/src/dynamo/vllm/tests/fixtures/protocol_releases/` contains pinned
  historical adapter excerpts and source hashes, not executable release runtimes.

The current pins are vLLM 0.29.0
(`98dff2a81d747d1dba01a47f939f48c3526d4206`, CPU/XPU) and 0.30.0
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, CUDA). Release excerpts reference
Dynamo 1.4.0 (`03014943323e78feb5bd672ef08b72caea0918ac`) and 1.5.0
(`b83b1d9304ebfc624709ac46db32b1b6f1ff1615`).

## Regenerate and verify

Use Python with `PyYAML==6.0.2`, an upstream Git checkout containing the pinned
commits and release tags, and a Dynamo checkout containing the historical
commits. Upstream Python modules are parsed as source, never imported/executed.

```sh
python -m unittest discover -s scripts -p 'test_*protocol*.py' -v
python scripts/check_protocol_pins.py --upstream-repo /path/to/vllm
python scripts/generate_protocol_inventory.py --upstream-repo /path/to/vllm
python scripts/generate_protocol_inventory.py --upstream-repo /path/to/vllm --check
python scripts/generate_protocol_inventory.py --upstream-repo /path/to/vllm \
  --inventory-output /path/to/artifacts/vllm_inventory.json
python scripts/generate_protocol_release_fixtures.py --check
```

The inventory generator's `--check` verifies `vllm_fields.rs` without reading or
creating inventory JSON. If `--inventory-output` is also supplied, it checks that
explicit JSON output too. Both must match when explicitly requested.
Change source pins or the generator and regenerate; do not hand-edit outputs.
Unresolvable request inheritance and dynamic input-name aliases fail vocabulary
generation rather than silently dropping accepted input names.

## Source coverage

The extractor retains B1's selected server/renderer modules and expands coverage
from chat/completion request and response classes plus `SamplingParams`. It
follows repository-local imports (including relative imports and re-exports),
type aliases, forward references, inherited declarations, and nested models.
It never imports or executes upstream Python. Changes in reached files appear
even if the referencing request declaration did not change. Entire reached
module contracts are compared, so unrelated edits in such a module can produce
review candidates; reachability does not prove a public API impact.

The snapshots and report list unresolved references and their consumers.
External package sources, star imports, dynamic definitions, and unsupported
name resolution are not silently treated as complete coverage. Recursive models
terminate normally; unresolved alias/re-export cycles are diagnosed. Typing,
builtin, and selected Pydantic primitives are explicit leaves; their library
implementations are not resolved. Python syntax must be supported by the
interpreter running the extractor, whose version is recorded in provenance.

Use `--require-complete-coverage` on either drift entry point to fail if either
snapshot has unresolved references, even when there are no changes or all
dispositions are recorded. This is separate from `--require-triage`. External
SDK source/version ingestion and arbitrary runtime schema evaluation are not
implemented; a strict check on current vLLM sources can therefore fail coverage.
The report is still written so the uncovered references can be investigated.

## Review a framework-version bump

Update the intended container version and its source pin together. Compare the
new state against the full Dynamo commit before the bump, using a fresh output
directory for each run:

```sh
python scripts/run_protocol_drift_check.py \
  --upstream-repo /path/to/vllm \
  --baseline-dynamo FULL_PRE_BUMP_DYNAMO_SHA \
  --output-dir /path/to/new-report --require-triage
```

For an explicit upstream candidate instead, replace `--baseline-dynamo` with
`--upstream-candidate FULL_UPSTREAM_SHA`. This does not adopt the candidate or
change the pins. Reports record source revisions, input hashes, and tool
provenance. A changed validator/helper is an investigation candidate, not proof
of incompatible runtime behavior.

Open `summary.md`, not a raw inventory diff, to review the run. The comparison
table links to each revision pair; each candidate has a stable ID, before/after
payloads, source links, and its recorded disposition. Large source payloads and
dependency lists are expandable. Missing fields are distinguished from explicit
JSON null. Method/function changes expose AST hashes with source links rather
than pretending to infer their changed runtime semantics.

The standalone `protocol_drift.py --previous SHA --candidate SHA` entry point also
writes `summary.md`, alongside `previous.json`, `candidate.json`, and `report.json`.
It does not apply decision files; `--fail-on-drift` fails on any detected change.

### Example end-to-end candidate check

With the configured 0.29.0 CPU/XPU and 0.30.0 CUDA pins, compare against 0.30.0:

```sh
python scripts/run_protocol_drift_check.py \
  --upstream-repo /path/to/vllm \
  --upstream-candidate ced6857afa0ea7b2e3f0846a62e1394e90f15607 \
  --output-dir /path/to/new-report --require-triage
```

The output directory contains `summary.md`, one JSON report per distinct revision
pair, and `inventory-<SHA>.json` source snapshots. CPU/XPU share one comparison;
CUDA compares 0.30.0 to itself. Without decisions, changed pairs fail the triage
gate; unchanged pairs need no decision file. Exact counts depend on extractor
revision and are recorded in the generated output, not a maintained golden count.

For example, a report entry for `ChatCompletionRequest.watermarking` shows an
added declaration with annotation `bool` and default expression `True`, assessment
`Unverified`, and missing ownership/evidence until a decision is recorded. The
tool does not invent an owner, tracking issue, resolution, or compatibility claim.

Record decisions in the path printed for that comparison, then rerun into a
different empty directory. A missing/invalid decision file produces the report
and exits 1 with `--require-triage`; valid complete dispositions exit 0 unless
another required gate fails. Tracked unresolved gaps can pass triage. Without
`--require-triage`, an incomplete review can exit 0, explicitly labeled as an
unenforced triage gate. Evidence links are recorded references, not fetched or
verified by the renderer; reviewers must inspect them.

The decision file is
`lib/llm/src/protocols/openai/compatibility/decisions/PREVIOUS_SHA-CANDIDATE_SHA.json`.
Its top-level keys are `previous_upstream_commit`, `candidate_upstream_commit`,
and `changes`. The last maps each exact report change ID to a reviewed decision.
The gate requires exact candidate coverage and validates all entries before
annotating the report; stale or partial decisions cannot approve a new diff.

Material decisions include status, owner, rationale, next step, and either a
tracking issue or explicit decision. `Compatible` requires runtime evidence.
Unresolved gaps require tracking; unsupported/intentional differences require
evidence. A non-material source change instead records `material: false`, the
exact `scope_endpoints`, ownership, rationale, next step, evidence, and a
no-action decision; it cannot claim runtime compatibility.

## Limits

Static extraction cannot prove runtime normalization, model-specific validators,
streaming behavior, or frontend/backend transport. Historical function excerpts
do not prove N-2 interoperability. Passing decision-schema checks does not replace
human review. Initial full drift classification, approved real-world decision
examples, runtime integration, and CI ownership/execution remain separate work.

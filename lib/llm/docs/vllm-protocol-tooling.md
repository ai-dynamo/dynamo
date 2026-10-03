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
- `vllm_inventory.json` records request declarations, inherited fields, defaults,
  and statically resolved input aliases. `vllm_fields.rs` contains the derived
  field vocabulary. Both are generated; neither is a support allowlist.
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
python scripts/generate_protocol_release_fixtures.py --check
```

`--check` fails when the corresponding committed generated output is stale.
Change source pins or the generator and regenerate; do not hand-edit outputs.
Unresolvable inheritance and dynamic aliases fail extraction rather than
silently becoming support claims.

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

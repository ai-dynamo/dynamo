<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# API surface tracker

## Purpose

The `api-surface` check on every pull request extracts the public API surface
at the merge base and at the PR head, diffs them, and fails when a `stable`
symbol that was not deprecated at the base is removed, renamed, relocated, or
changed incompatibly. Changes to `preview` and `experimental` symbols are
informational only. The policy it enforces lives at
[docs/fern/pages/community/contributing/api-stability-policy.md](../../docs/fern/pages/community/contributing/api-stability-policy.md).

## Layout

- `.github/api_surface/` — the tracker itself (this directory); extractors,
  diff, ledger, stability inference, validator, PR impact analysis, CLI.
- `.github/api-surface/` — the data it reads and writes: `annotations.yaml`,
  `suppressions.yaml`, `snapshots/`, `ledger.json`, `crate-provenance.yaml`.
  Owned by the process codeowners; edits need their approval.

## Commands

All commands run from the repo root with no build:

```bash
PYTHONPATH=.github python -m api_surface extract --repo . --release 1.4.0 --json /tmp/snap.json
PYTHONPATH=.github python -m api_surface analyze-pr --base-repo ../base --head-repo . --release 1.4.0
PYTHONPATH=.github python -m api_surface render --from-json impact.json --format md
PYTHONPATH=.github python -m api_surface validate --repo .
PYTHONPATH=.github python -m api_surface list-deprecations --repo .
```

## Marking a Deprecation

| Surface | How to Mark |
|---|---|
| **Rust** | `#[deprecated(note = "...")]` |
| **Python** | `@deprecated(...)` or `warnings.warn(..., DeprecationWarning, stacklevel=...)` |
| **HTTP API** | `deprecated: true` in the OpenAPI spec, plus the runtime headers below |
| **CRD** | `deprecated: true` and `deprecationWarning` on the old `spec.versions` entry while it remains served |
| **Config, env, Helm, metrics** | Keep the old surface working, mark it in user-facing help or reference docs, and record its symbol ID in `annotations.yaml` |

## Promoting a components Symbol

A `components/` module publishes nothing stable unless it declares a literal
`__all__`. To make a handler or helper a stable contract, add it to `__all__`
or add its symbol ID to `.github/api-surface/annotations.yaml` with
`stability: stable`.

## Waivers

A genuine exception goes in `.github/api-surface/suppressions.yaml` with
`id` or `pattern`, an optional `surface` guard, a `reason`, and an `until`
release after which the waiver expires. The file is owned by the process
codeowners, so a PR that edits it needs their approval. The waiver ships in
the same pull request as the change it covers.

## Running the Tests

```bash
PYTHONPATH=.github python -m pytest .github/api_surface/tests -p no:cacheprovider --override-ini="addopts=" --override-ini="filterwarnings="
```

## Required Check

The `api-surface` job in `.github/workflows/api-surface.yml` is a required
status check. Renaming the job breaks branch protection.

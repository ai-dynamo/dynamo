---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: API Stability and Deprecation Policy
subtitle: What counts as public, how stability tiers work, and the deprecation rule CI enforces
---

Dynamo commits to a stability promise for its public surface. This page defines that surface,
assigns every public symbol a stability tier, and sets the deprecation rule that must run before a
`stable` symbol is removed. The `api-surface` check on every pull request enforces the rule.

> [!NOTE]
> The `api-surface` check fails a pull request that removes, renames, relocates, or incompatibly
> changes a `stable` public symbol unless the merge base already carries a deprecation marker for
> it, or the pull request includes a reviewed waiver. Additions and changes to `preview` or
> `experimental` symbols never fail the check.

## Public Surface

Public means documented for external use. Runtime reachability is supporting evidence for HTTP
routes and exported metrics, but reachability alone does not make an undocumented configuration
switch or importable helper public. Everything else is internal and carries no stability promise.

| Surface | Public | Not Public |
|---|---|---|
| **Python bindings** | Symbols in the `dynamo._core`, `dynamo.prometheus_metrics`, and `kvbm._core` stubs | Underscore-prefixed names |
| **Python components** | Symbols a module lists in `__all__`, plus symbols promoted in `annotations.yaml` | Everything else under `components/src/dynamo/` |
| **HTTP API** | Routes in the generated OpenAPI spec; the OpenAI-compatible endpoints | Internal control-plane routes not in the spec |
| **Configuration** | `DYN_*` environment variables and CLI flags visible in generated `--help` | Hidden internal configuration and wire types |
| **Helm values** | Keys in published chart `values.yaml` files | Templates, internal subchart plumbing |
| **CRDs** | Fields in served CRD versions | Unserved or internal versions |
| **Metrics** | Prometheus metric names and labels exported by the runtime | Internal counters not registered for export |
| **Rust** | The C ABI and the crate-root facade of each `lib/` crate | Items below the crate root |

A `components/` module that does not define a literal `__all__` publishes nothing stable. To make a
handler or helper a stable contract, add it to `__all__` or add its symbol ID to
`.github/api-surface/annotations.yaml` with `stability: stable`.

## Stability Tiers

Every public symbol carries one of three tiers. The tier sets the promise; the versioning rule
below turns it into a release rule.

- **`stable`**: load-bearing. Will not be removed or changed incompatibly without running the
  deprecation lifecycle below. This is the default for anything public unless a signal says
  otherwise.
- **`preview`**: usable and supported, but the shape can still change. Maps to `beta` maturity.
  May change incompatibly in a minor release with a note.
- **`experimental`**: explicitly provisional. Maps to `alpha`. May change or disappear in any
  release.

**Tier Assignment**

A symbol's tier follows from code-grounded signals, and an explicit annotation always beats an
inferred one:

- **CRD version names** encode maturity: `v1alpha1` is experimental, `v1beta2` is preview, `v1` is
  stable.
- **Name and path keywords**: `experimental` or `alpha` in a symbol's name makes it experimental;
  `preview` or `beta` makes it preview.
- **Declared visibility**: a Rust item below the crate-root facade, or a Python components symbol
  not listed in `__all__`, is experimental.
- **Explicit annotations**: `.github/api-surface/annotations.yaml` sets `stability` for any
  extracted symbol ID and overrides inference.
- **Weakest signal wins**: a symbol is only as stable as its least-stable signal. A `preview` field
  under an `alpha` CRD version is experimental unless an annotation records a reviewed exception.

## Versioning Rule

Dynamo ships a minor release about once a month. The rule is written against that cadence and
does not require a major version for every stable removal.

- A `stable` symbol is deprecated for at least one release before the release that removes it.
  Two releases is the target; one is the floor.
- HTTP routes and CRD fields are deprecated for at least three releases, because external systems
  pin them hardest.
- `preview` and `experimental` symbols may change or disappear in any minor release with a
  release note.
- Patch releases carry no incompatible change to any public surface.
- Major versions are reserved for coordinated breaks across many surfaces. They are not required
  to remove one deprecated symbol.

Enforcement starts with the release that ships the `api-surface` check. Removals that landed
before it are not treated as broken promises.

## Marking a Deprecation

Mark deprecations in code, at the symbol, with a one-line note that says what to use instead. A
deprecation with no migration path is a bug.

| Surface | How to Mark |
|---|---|
| **Rust** | `#[deprecated(note = "...")]` |
| **Python** | `@deprecated(...)` or `warnings.warn(..., DeprecationWarning, stacklevel=...)` |
| **HTTP API** | `deprecated: true` in the OpenAPI spec, plus the runtime headers below |
| **CRD** | `deprecated: true` and `deprecationWarning` on the old `spec.versions` entry while it remains served |
| **Config, env, Helm, metrics** | Keep the old surface working, mark it in user-facing help or reference docs, and record its symbol ID in `annotations.yaml` |

The annotation manifest has one entry per symbol ID. Every deprecation records its removal target
there. For surfaces without a native marker, it also supplies the tier and the migration note:

```yaml
symbols:
  "metric:frontend_service::OLD_REQUESTS_TOTAL":
    stability: stable
    deprecated: true
    note: "Use REQUESTS_TOTAL"
    removal_target: "v1.6.0"
```

An annotation for an unknown symbol or an invalid tier fails extraction, so a typo cannot weaken
the contract.

**HTTP Runtime Signals**

Code markers are invisible to a running client. Deprecated HTTP routes also emit:

- `Deprecation` header (RFC 9745) on every response, `Sunset` (RFC 8594) with the removal date,
  and a `Link` to the successor or migration guide.
- `410 Gone` with a migration pointer after removal, not a bare `404`.

## Deprecation Lifecycle

```text
Announce  ->  Deprecate (>= minimum window)  ->  Remove (at or after the removal target)
```

1. **Announce.** Add the marker in a pull request, name the removal-target release, and add the
   `deprecation` label.
2. **Deprecate.** The symbol keeps working with the marker active for at least the minimum window.
   It appears in the release notes Breaking Changes section.
3. **Remove.** At or after the removal target, remove the symbol. Removing it before the promised
   target breaks the promise even after waiting the window.

**Communications**

- **Release notes**: every deprecation appears in Breaking Changes in the release it is announced
  and again in the release it is removed.
- **`deprecation` label**: on the pull request, so deprecations are filterable.
- **Slack**: material `stable` deprecations get a heads-up at announce time.

## Enforcement

The `api-surface` check runs on every pull request. It extracts the public surface from the merge
base and from the pull request head, diffs them, and posts an impact comment listing breaking,
waived, informational, and added symbols. It fails when a `stable` symbol that was not deprecated
at the merge base is removed, renamed, relocated, or changed incompatibly. Extraction failure on
either tree fails the check. Removing or changing a `preview` or `experimental` symbol is
informational.

**Waivers**

A genuine exception, such as a security-driven removal, goes in
`.github/api-surface/suppressions.yaml` with a reason and an `until` release after which the
waiver expires. The file is owned by the process codeowners, so any pull request that edits it
needs their approval. The waiver ships in the same pull request as the change it covers.

```yaml
suppressions:
  - id: "python:dynamo._core.Foo.bar"
    reason: "Leaked into the stub by mistake; never documented."
    until: "1.5.0"
```

**Tooling**

The tracker lives in `.github/api_surface/` and runs without a build:

```bash
PYTHONPATH=.github python -m api_surface extract --repo . --release 1.4.0 --json /tmp/snap.json
PYTHONPATH=.github python -m api_surface analyze-pr --base-repo ../base --head-repo . --release 1.4.0
```

See `.github/api_surface/README.md` for the full command set.

## Options Considered

**Stability model**

| Option | Why Not |
|---|---|
| **No written policy** | Nothing to enforce, and users cannot tell stable from churning surface. |
| **Two tiers (stable / unstable)** | Loses the `alpha` versus `beta` distinction that CRD versions and the codebase already encode. |
| **Hand-maintained tier list** | Rots immediately. Tiers should fall out of code-grounded signals. |
| **Three tiers, code-inferred (chosen)** | Matches Kubernetes conventions, maps onto signals already in the code, and scopes the promise cleanly. |

**Versioning**

| Option | Why Not |
|---|---|
| **Strict SemVer: major for every stable break** | A monthly-cadence platform would pin itself to 1.x forever or ship a major every quarter. |
| **No fixed window** | Unenforceable. |
| **12-month runway** | Right for a slow REST API, too slow for an inference platform. |
| **One release enforced, two targeted, three for HTTP and CRDs (chosen)** | Enforceable, predictable, and tier-scoped. |

## Risks

**"This slows us down."** New surface ships as `experimental` or `preview` with no removal
obligation, and `components/` modules publish nothing stable until they opt in through `__all__`.
Only `stable` carries the deprecation cost.

**"Inference will mislabel a tier."** Inference only relaxes the default, and an annotation
overrides it. The failure mode is "marked less stable than it is," which is safe.

**"Emergency removals cannot wait."** A security-driven removal takes a reviewed waiver and a
release-note callout. A rule that bends with a paper trail beats no rule.

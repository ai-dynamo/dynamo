<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Page anatomy

What the page takes from the README, and what belongs to Fern.

## From the README

Everything substantive. The page mirrors its recipe's READMEs in both
directions: every fact they carry appears, and the page asserts nothing they do
not support.

**Mirror their heading structure and order**, with their wording and
capitalisation. Recipes differ, and that is fine — a reader moving between
GitHub and the docs should recognise the same page. Three shapes are common:

```
Configurations · Supported features · Prerequisites · Quick Start
  · Optimization targets · Performance results · Known issues

Available Configurations · Prerequisites · Quick Start
  · Test the Deployment · Model Details · Notes

Results · Experiment Overview · Dataset · Prerequisites · Quick Start
  · Expected Results · Cleanup · References
```

Sub-READMEs matter as much as the top-level one. `perf/README.md` usually owns
the benchmark procedure — trace staging, the environment variables to edit, the
concurrency sweep. Skip it and the benchmark section will not work.

## Fern scaffolding

Not in any README, and legitimately so:

- **Frontmatter** — SPDX header, `title`, one-sentence `subtitle`. Fern renders
  the title, so the body has no `#` H1.
- **Intro paragraph** — what it deploys and the strongest proof point.
- **The target picker** and its summary panels. This re-presents the README's
  configuration table one target at a time; it is the UX, not extra
  information, as long as every value in it comes from a README or a manifest.
- **`data-*` wrappers** and the duplication they force — the same command shown
  once per target.
- **`## Source`** — links back to the README and the manifests. Always last.
- **Cross-links** to other Fern pages.

## Tables

Two kinds, treated oppositely. Getting this backwards is easy and the symptom
is silent.

**Comparison tables** — Performance results, Optimization targets, Benchmark
Results, Expected Results. The point is seeing every target side by side, and
the README prints them in full. **No `data-*` on any row**, or the global CSS
hides the rows that do not match and the table quietly loses most of itself.

**Per-target configuration** — the shipped settings for one target. These live
in the picker summary panels rather than a table, so a reader sees their
target's full configuration in one place.

## Where a README is wrong

READMEs drift from their own manifests: a `port-forward` naming a service no
`deploy.yaml` defines, an image tag nothing pins, a context length the engine
config contradicts.

**The page follows the manifest**, because a command that fails helps nobody,
and the README bug goes in the PR description for its owner. This is the only
case where the page diverges from its README, and it should be called out
rather than left silent.

## Reader impact ordering

When deciding whether something belongs, ask what happens if it is missing:

1. **Breaks the reader** — a command fails, or they deploy the wrong thing.
   Service names, image tags, GPU counts, trace staging, required env vars.
2. **Misleads** — they form a false belief. Limitations, known issues,
   unsupported combinations.
3. **Informational** — useful, not load-bearing.

The first two are why the page exists. Resist adding beyond what the READMEs
carry, especially maturity framing — "Day-0", "experimental", "not yet
promoted" — unless a README says it. That status lives in the catalog for
internal use and should not leak into reader-facing prose on its own authority.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overlap review: new front door vs. the existing benchmarking guide

Does the performance analysis front door on `gluo/perf-analysis-front-door` duplicate
`docs/fern/pages/recipes/feature-benchmarks/benchmarking-guide.md` (452 lines)?

**Verdict: mostly no — roughly 90% is disjoint. Two genuine overlaps, and one defect in the
new work.** Verified against `origin/main` @ `c574e4d7c1a`.

---

## What each owns

| | Benchmarking guide | New front door |
| --- | --- | --- |
| Subject | One tool: AIPerf | Choosing among all performance tools |
| Contains | Prerequisites, port-forwarding, 5-step workflow, concurrency sweeps, plots, in-cluster Job, flag quick-reference, troubleshooting | Question classes, routing tables, preconditions, run protocol, reporting contract |
| Flags and commands | Many, by design | None, by design |
| Page type | How-to | Explanation plus routing |

Its `## Choosing Your Benchmarking Approach` (line 31) reads like competition but is not: it
chooses between **client-side and server-side placement of AIPerf**, a deployment-mechanics
decision *inside* one tool. The new pages choose *between* tools. Different axis, no conflict.

---

## Overlap 1 — DynoSim and Mocker routing. Real, and already inconsistent

`benchmarking-guide.md:421`, `## Testing with DynoSim / Mocker`, performs routing: when to use
simulation, and DynoSim Runs versus Sweeps. That is the same job as the new pages' sizing and
no-GPU routes.

More importantly, the two now say different things about when simulation is appropriate:

- **That section** frames it as "for development and testing purposes", "testing deployments
  without expensive GPU infrastructure", "benchmarking framework validation". Grepping it for
  `uncalibrat|not model|fidelity|does not replace` returns **nothing** — there is no fidelity
  caveat on the page.
- **The new pages** state that simulation ranks candidates but does not measure them, that the
  default timing model is uncalibrated, that several paths including multimodal encoder
  compute are not modelled, and that the winner must be validated on hardware.

Both are defensible in isolation. Together they let a reader conclude that simulation output
is a benchmark result. This is precisely the drift the one-fact-one-home rule exists to
prevent, appearing immediately.

**Options.**

- **(a)** Reduce `## Testing with DynoSim / Mocker` to a pointer at the simulation pages, so
  the "when and how much to trust it" question is answered in one place. Cleanest.
- **(b)** Leave the section and add the fidelity caveat to it. Cheaper, but keeps two homes
  for the same fact and will drift again.
- **(c)** Leave as is. Not recommended: the inconsistency is live now.

Option (a) edits a page outside the front door's original scope, so it is a deliberate choice
rather than an oversight to fix silently.

---

## Overlap 2 — recipes-first advice. Minor, acceptable

`benchmarking-guide.md:14`, `## Start from Dynamo Recipes When Possible`, says what the new
Kubernetes page's characterization route says. The new page's version is two sentences that
link onward rather than explaining, so this is duplication of a *pointer*, not of content.

No action needed.

---

## Defect in the new work — the guide is not linked from the front door

None of the three new pages link to `benchmarking-guide.md`. They link to the Operations
`benchmarking-with-aiperf.mdx` pages and to `browse-all-benchmarks.mdx`. `benchmarks/README.md`
points at "the AIPerf benchmarking guides in the same Operations sections", also not the guide.

The guide is reachable in two hops, because both Operations AIPerf pages link to it under
their next-steps:

- `docs/fern/pages/kubernetes/operations/benchmarking-with-aiperf.mdx:71`
- `docs/fern/pages/cli/operations/benchmarking-with-aiperf.mdx:68`

For pages whose entire purpose is routing a reader to the tool that answers their question,
failing to link the fullest treatment of the most-used tool is a real defect. The plan's own
"two hops from question to runnable command" check should have caught it.

**Fix.** Link `benchmarking-guide.md` directly from the characterization route on both router
pages:

- `docs/fern/pages/kubernetes/operations/performance-analysis.md`
- `docs/fern/pages/cli/operations/performance-analysis.md`

---

## Note: the guide has a pre-existing structural problem

Not caused by, and not in scope for, this branch — recorded because it will complicate any
edit to that file.

`benchmarking-guide.md` is two guides concatenated into one page, and the seam is visible:

- **Two body `# H1` headings** — `# Client-Side Benchmarking (Local)` at line 87 and
  `# Server-Side Benchmarking (In-Cluster)` at line 311. This violates a must-fix style rule:
  Fern renders the page H1 from the nav `page:` label, so a body `# H1` produces a duplicate
  title. The page therefore renders three H1s.
- **Duplicate `## Prerequisites`** at lines 91 and 315, plus two parallel quick-start flows
  (`## User Workflow` at 103, `## Quick Start` at 321).

The internal jump links from `## Choosing Your Benchmarking Approach`
(`#client-side-benchmarking-local`, `#server-side-benchmarking-in-cluster`) **do** resolve —
they target those two body H1s. So fixing the H1 violation by demoting them to `##` preserves
the anchors, since the slug is unchanged.

The natural fix is to split the file into two pages, one per approach, which removes the
duplicate H1s and the duplicate Prerequisites at the same time. That is a larger change than
either overlap fix above and belongs in its own pass.

---

## Recommended action

1. Link the guide from both router pages. Small, clearly correct, closes the defect above.
2. Decide between options (a) and (b) for the DynoSim section.
3. File the duplicate-`## Prerequisites` and possibly-broken-anchor issue separately.

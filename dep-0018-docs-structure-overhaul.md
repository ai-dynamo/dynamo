# Documentation Structure Overhaul

**Status**: Draft

**Authors**: [Jonathan Tong](https://github.com/Jont828)

**Category**: Guidelines

**Replaces**: N/A

**Replaced By**: N/A

**Sponsor**: TBD

**Required Reviewers**: TBD

**Review Date**: TBD

**Pull Request**: TBD

**Implementation PR / Tracking Issue**: [ai-dynamo/dynamo#8658](https://github.com/ai-dynamo/dynamo/issues/8658) (K8s subset)

# Summary

Restructure the Dynamo documentation site to fix a long-standing
problem flagged by NVIDIA QA: the docs mix high-level guides with
component-specific implementation details and offer no defined path
from Concepts → Quick Start → Reference. This DEP proposes a new
top-level information architecture that cleanly separates **tutorial**
content (how to do things) from **knowledge bank** content
(architecture, concepts, components, features), with two parallel
tutorial tracks for CLI and Kubernetes usage. The K8s-specific
walkthrough rewrite is covered by [ai-dynamo/dynamo#8658](https://github.com/ai-dynamo/dynamo/issues/8658)
and slots into the new "Kubernetes Usage" section defined here.

# Motivation

QA feedback (paraphrased): the `docs/` directory mixes high-level
guides with component-specific implementation details, with no defined
path from Concepts → Quick Start → Reference. Critical configuration
options are explained in prose across multiple pages rather than
consolidated in reference tables. Users cannot tell whether a given
page is meant to teach them how to use Dynamo, explain how it works
internally, or serve as a reference they look up later.

Concretely, the current sidebar has several structural problems:

1. **Local and K8s deployment paths are not clearly parallel.** It's
   not obvious that Dynamo can be run either locally (CLI) or on a
   Kubernetes cluster, or which pages are relevant to each path. K8s
   deployment has its own top-level section, but the local deployment
   content is scattered across Getting Started, User Guides, and
   Backends rather than elevated to a peer section.

2. **No overview of how to navigate a section.** Sections present a
   flat list of pages with no landing page explaining which pages to
   read or in what order. For example, the current K8s Deployment
   Guide is a flat list:

   ```
   Quickstart
   Installation Guide
   Model Deployment Guide
   DGDR Reference
   Dynamo Operator
   Service Discovery
   Webhooks
   Minikube Setup
   Managing Models with DynamoModel
   Autoscaling
   Rolling Update
   Developing with Tilt
   Inference Gateway (GAIE)
   Snapshot
   Shadow Engine Failover
   Disagg Communication
   ```

   A first-time reader has no way to tell which of these is required
   for a basic deployment vs. an advanced operator-level reference.

3. **No separation between "do" and "understand".** Tutorials and
   reference/explanation content are interleaved. A user looking for
   "how do I deploy on K8s?" lands next to "how does the request
   plane work internally?".

4. **For tutorials, basic vs. advanced is not surfaced.** The K8s
   section, for example, mixes a quickstart with optional components
   (Grove, model caching) and prerequisites (GPU Operator, RDMA) that
   are never surfaced at the point in the flow where the user needs
   them.

5. **No top-down conceptual path.** While we have docs for components
   and backends, there's no top-level tl;dr overview that gives a
   high-level picture of how the pieces fit together without reading
   through all the details. We should, for example, make it clear that
   the backends are modular and that the profiler/planner are part of
   the operator.

6. **"User Guides" is a dumping ground.** It mixes backend tutorials
   (SGLang/TRT-LLM/vLLM examples), feature explanations (fault
   tolerance, tool calling, reasoning, LoRA), use-case content
   (multimodal, diffusion, agents), developer content (writing Python
   workers, mocker), and observability — none of which belong
   together.

7. **Cross-cutting features are scattered.** Observability, fault
   tolerance, benchmarking, tool calling, etc. each have their own
   ad-hoc home, with no single section a user can scan to see what
   Dynamo can do.

8. **K8s docs read like an encyclopedia, not a walkthrough.** This is
   the specific instance covered by
   [ai-dynamo/dynamo#8658](https://github.com/ai-dynamo/dynamo/issues/8658):
   prerequisites (GPU Operator, RDMA, Grove, model caching) are
   documented somewhere but never surfaced at the point in the
   deployment flow where the user needs them. The structural changes
   in this DEP are a precondition for the #8658 rewrite to have a
   coherent home.

## Goals

* Establish a clear separation between basic tutorials, advanced
  tutorials/use cases, concepts/explanations, and reference content,
  so contributors know where new pages belong and readers can find
  the right kind of content.
* Make CLI and Kubernetes usage parallel, first-class tutorial tracks.
* Provide a top-down conceptual path (Architecture Overview → Concepts
  → Components → Backends) so a new reader can build a mental model
  before diving into reference.
* Give every section a landing page that orients the reader.
* Give the K8s walkthrough rewrite from
  [#8658](https://github.com/ai-dynamo/dynamo/issues/8658) a coherent
  home it can link out from.

### Non Goals

* Rewriting the **content** of every page. This DEP is about
  structure; per-page content rewrites are out of scope except where
  required to fill new landing pages or split tutorial vs. reference
  content that currently lives in a single page.
* The K8s content rewrite itself — that is tracked by
  [#8658](https://github.com/ai-dynamo/dynamo/issues/8658). This DEP
  defines the section it lands in.
* Migrating the docs to a different docs platform (e.g., off Fern).
* Changing the source repository layout of `docs/` beyond what is
  required to match the new sidebar structure.

## Requirements

### REQ 1 New Top-Level Sidebar Structure

The site **MUST** adopt the new top-level structure described in the
Proposal. Every existing page **MUST** be mapped to a new location.
There **MUST NOT** be orphans or broken links after the migration.

### REQ 2 Tutorial vs. Knowledge Bank Separation

Tutorial pages (CLI Usage, Kubernetes Usage, Getting Started)
**MUST** be separated from knowledge-bank content (Architecture
Overview, Concepts, Components, Backends). Cross-cutting reference
content (Features) **MUST** live in the knowledge bank and **MUST
NOT** be styled as a walkthrough.

### REQ 3 Section Landing Pages

Every newly introduced collapsible section (Usage, Architecture
Overview, Advanced, Developer's Guide, and their children where
non-trivial) **MUST** have a landing page that explains what the
section is for, what is required vs. optional, and what order to read
the pages in.

### REQ 4 Parallel CLI and Kubernetes Tracks

CLI Usage and Kubernetes Usage **MUST** be peer top-level sections.
Each **MUST** contain installation, a basic deployment walkthrough,
and observability setup at minimum. Each **SHOULD** be slim — get the
user to a working deployment, then link out to knowledge content for
depth.

### REQ 5 Tutorial Pages Are Copy-Pasteable

Basic usage/tutorial pages **SHOULD** be straightforward and to the
point. Commands **SHOULD** be copy-pasteable directly into a terminal.
User-configurable options (e.g., name, namespace) **SHOULD** be
parameterized via environment variables. Tutorials **SHOULD** explain
only what the user needs to follow the steps, and **SHOULD** link to
component/feature reference pages for everything else.

### REQ 6 Redirects for Moved Pages

URL changes from page moves **MUST** be handled via redirects so
external bookmarks and search results continue to resolve.

### REQ 7 #8658 K8s Rewrite Lands in Kubernetes Usage

The K8s content rewrite from
[#8658](https://github.com/ai-dynamo/dynamo/issues/8658) **MUST**
land inside the new Kubernetes Usage section defined by this DEP. The
two efforts **SHOULD** land together — restructuring without
rewriting K8s leaves the walkthrough problem; rewriting K8s without
restructuring leaves the new walkthrough in a section that still
neighbors "Design Docs" and "User Guides".

# Proposal

Adopt a new top-level information architecture organized around two
tutorial tracks (CLI Usage, Kubernetes Usage) plus a knowledge bank
that flows from a high-level overview down through concepts,
components, and backends, with advanced content (features,
applications, integrations) grouped separately.

## Design Principles

1. **Clear separation by content type.** Basic tutorials, advanced
   tutorials/use cases, concepts/explanations, and reference each
   live in their own area. This makes the site easier to navigate and
   gives contributors clear guidance on where a new page should go.
   It also lets us surface the most important content at the top.

2. **Basic tutorials are copy-pasteable and minimal.** Commands paste
   directly into a terminal; user-configurable options like
   name/namespace are env vars. Tutorials explain only what you need
   to follow the steps and leave the rest to component/feature
   reference pages.

3. **Two parallel tutorial tracks.** Make it clear that Dynamo can be
   run on the CLI or on Kubernetes, and that both are first-class
   options. "CLI Usage" (run processes locally) and "Kubernetes
   Usage" (apply manifests on a cluster) are the "do stuff" sections.
   Each is slim — get the user to a working deployment, then link out
   to knowledge content for depth.

4. **Knowledge bank follows a "giving a talk" flow.**

   Architecture Overview (what Dynamo is and what it's made of) →
   Concepts (what you need to understand first) →
   Components (the actual pieces of the system) →
   Backends (engine-specific reference).

   This mirrors the way the architecture is presented in talks: a
   single high-level slide introduces each concept/component, then
   each topic is covered in detail with more granularity.

5. **Advanced content is organized, not dumped.** The current
   "User Guides" section is a catch-all for anything that isn't a
   basic tutorial or reference. Split it into:
   - **Features** — cross-cutting capabilities (tool calling, fault
     tolerance, observability reference, benchmarking, LoRA, perf
     tuning)
   - **Applications** — use-case domains (multimodal, diffusion,
     agentic workloads)
   - **Integrations** — third-party integrations (LMCache, FlexKV,
     custom KV events)

6. **Developer content is segregated.** Contributing, building from
   source, writing Python workers, mocker, and local K8s dev tooling
   (Minikube, Tilt) live in a "Developer's Guide" section, not mixed
   into user-facing tutorials.

## New Top-Level Sidebar

```
Getting Started               ← intro + CLI quickstart + K8s quickstart
Usage
  ├── CLI Usage               ← run locally (slim tutorial)
  └── Kubernetes Usage        ← run on a cluster (slim tutorial; #8658 lands here)
Architecture Overview         ← big-picture tl;dr (standalone)
  ├── Concepts                ← deep dives into theory
  ├── Components              ← operator, frontend, router, planner, profiler, KVBM
  └── Backends                ← SGLang, TRT-LLM, vLLM reference
Advanced
  ├── Features                ← tool calling, fault tolerance, observability ref, benchmarking, LoRA, perf tuning
  ├── Applications            ← multimodal, diffusion, agentic workloads
  └── Integrations            ← LMCache, FlexKV, custom KV events
Developer's Guide             ← contributing, building, writing workers, mocker, local K8s dev
Resources                     ← support matrix, feature matrix, release artifacts, examples, glossary
Blog                          ← unchanged
Documentation                 ← unchanged (docs-about-docs)
```

The page-by-page realization of this tree (every existing page
mapped to its new location) is captured in
[`sidebar-restructure-proposal-v3.md`](https://github.com/ai-dynamo/dynamo/blob/dynamo-full-docs-refactor/sidebar-restructure-proposal-v3.md)
on the `dynamo-full-docs-refactor` branch of `ai-dynamo/dynamo`. A
first-pass `index.yml` implementing this tree exists on the same
branch.

## Where Current Content Moves

| Current Section | What happens |
|---|---|
| **Getting Started** | Slimmed to intro + CLI quickstart + K8s quickstart. Local Installation → CLI Usage. Building from Source → Developer's Guide. Contribution Guide → Developer's Guide. |
| **Kubernetes Deployment** | Renamed to "Kubernetes Usage". Dev tools (Minikube/Tilt) → Developer's Guide. Content rewrite covered by [#8658](https://github.com/ai-dynamo/dynamo/issues/8658). |
| **User Guides** | **Eliminated.** Backend examples → CLI Usage. Feature content → Features. Multimodal/Diffusion/Agents → Applications. Writing Python Workers → Developer's Guide. Benchmarking → Features. Mocker → Developer's Guide. Observability (Local) split across CLI Usage + Features. |
| **Backends** | Examples pages move to CLI Usage. Reference guides, deep-dives (disaggregation, HiCache, KV offloading, chat processor), and observability stay as knowledge-bank Backends (vLLM gains parity with SGLang/TRT-LLM). |
| **Components** | Unchanged structurally. Component design docs move here from Design Docs (e.g., `router-design.md` joins Components > Router). |
| **Integrations** | Unchanged in content, moved under "Advanced". |
| **Design Docs** | **Eliminated.** `architecture.md` → standalone Architecture Overview. Conceptual docs (`dynamo-flow`, `disagg-serving`, `distributed-runtime`, communication planes) → Concepts. Component design docs → their respective Components sections. |
| **Resources** | Unchanged. |
| **Blog / Documentation** | Unchanged. |

## Relationship to #8658

Issue [#8658](https://github.com/ai-dynamo/dynamo/issues/8658) covers
the **content rewrite** of the K8s docs into a single linear
walkthrough (modeled on Cluster API's quickstart), with prerequisites
and optional components surfaced at the point in the flow where they
become relevant, and DGDR as the primary deployment entrypoint.

This DEP is the **structural container** for that work: it defines
the "Kubernetes Usage" section the rewrite lives in, and the parallel
sections (CLI Usage, Concepts, Components, Features, etc.) that the
K8s walkthrough links out to instead of duplicating.

# Alternate Solutions

## Alt 1 Incremental Cleanup of "User Guides" Only

**Pros:**
- Smaller scope, lower risk
- No URL changes outside one section

**Cons:**
- Leaves the tutorial vs. knowledge-bank confusion intact
- Doesn't address the missing CLI Usage track
- Doesn't give the K8s rewrite a coherent peer section

**Reason Rejected:** Doesn't address the root QA feedback about
overall structure.

## Alt 2 Rewrite K8s Docs Only (Just #8658)

**Pros:**
- Single focused effort, clear deliverable
- Addresses the most acute pain point

**Cons:**
- The walkthrough has to link out for concepts, components, and
  features, and those targets are currently scattered or duplicated
- Reader experience regresses as soon as they follow any link out of
  the new walkthrough

**Reason Rejected:** The K8s rewrite cannot stand on its own without
a coherent surrounding structure to link into.

## Alt 3 Adopt Full Diátaxis (Tutorials / How-To / Reference / Explanation)

**Pros:**
- Industry-standard taxonomy with clear definitions for each content
  type
- Forces a clean separation by content type

**Cons:**
- Forces four parallel sections per topic, which fragments content
- Heavier lift to migrate every page into a strict four-way split
- Tends to produce shallow per-quadrant pages instead of cohesive
  topic pages

**Reason Rejected:** The proposed structure is Diátaxis-flavored
(tutorial tracks vs. knowledge bank) without the full taxonomy
overhead.

**Notes:** Worth revisiting if the proposed structure proves
insufficient over time.

## Alt 4 Defer the Restructure and Ship #8658 into the Current Sidebar

**Pros:**
- No structural risk
- Faster delivery of the K8s walkthrough

**Cons:**
- QA feedback explicitly cites structure as the problem, not just K8s
  content
- Shipping #8658 alone leaves the broader critique unaddressed
- The new K8s walkthrough would still neighbor "Design Docs" and
  "User Guides"

**Reason Rejected:** Doesn't address the documented QA feedback.

# References

- [ai-dynamo/dynamo#8658](https://github.com/ai-dynamo/dynamo/issues/8658) — docs: refactor Kubernetes and update to include v1beta1 DGDR (the K8s content rewrite this DEP creates a home for)
- Branch [`dynamo-full-docs-refactor`](https://github.com/ai-dynamo/dynamo/tree/dynamo-full-docs-refactor) on `ai-dynamo/dynamo` — first-pass `index.yml` implementing the proposed tree
- [`sidebar-restructure-proposal-v3.md`](https://github.com/ai-dynamo/dynamo/blob/dynamo-full-docs-refactor/sidebar-restructure-proposal-v3.md) — full page-by-page navigation tree
- [`k8s-docs-refactor-draft.md`](https://github.com/ai-dynamo/dynamo/blob/dynamo-full-docs-refactor/k8s-docs-refactor-draft.md) — draft of the K8s walkthrough content (#8658)
- [Cluster API quickstart](https://cluster-api.sigs.k8s.io/user/quick-start) — model for walkthrough-style K8s docs (used in #8658)
- [Diátaxis framework](https://diataxis.fr/) — influence on the tutorial vs. knowledge-bank split

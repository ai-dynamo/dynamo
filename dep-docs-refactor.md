# DEP: Documentation Structure Overhaul

**Status:** Draft
**Area:** docs
**Related:** [#8658 — docs: refactor Kubernetes and update to include v1beta1 DGDR](https://github.com/ai-dynamo/dynamo/issues/8658)

---

## Summary

Restructure the Dynamo documentation site to fix a long-standing
problem flagged by NVIDIA QA: the docs mix high-level guides with
component-specific implementation details and offer no defined path
from Concepts to Quick Start to Reference. This DEP proposes a new
top-level information architecture that cleanly separates **tutorial**
content (how to do things) from **knowledge bank** content
(architecture, concepts, components, features), with two parallel
tutorial tracks for CLI and Kubernetes usage. The K8s-specific
walkthrough rewrite is covered by issue #8658 and slots into the new
"Kubernetes Usage" section defined here.

## Motivation

QA feedback (paraphrased): the `docs/` directory mixes high-level
guides with component-specific implementation details, with no defined
path from Concepts → Quick Start → Reference. Critical configuration
options are explained in prose across multiple pages rather than
consolidated in reference tables. Users cannot tell whether a given
page is meant to teach them how to use Dynamo, explain how it works
internally, or serve as a reference they look up later.

Concretely, the current sidebar has several structural problems:


1. **Local and K8s deployment paths are not clear**. It's not clear that you can either run Dynamo locally or on a K8s cluster, and it's not clear which pages are relevant to each path. K8s deployment has its own section, but the local deployment content is not elevated to the same level.
2. No overview on how to navigate each page or section. The K8s deployment > Deployment guide section has the following pages in a list. There's no landing page explaining which pages to read or what order to read them in. 
   
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

3. **No separation between "do" and "understand".** Tutorials and
   reference/explanation content are interleaved. A user looking for
   "how do I deploy on K8s?" lands next to "how does the request
   plane work internally?".

4. **For tutorials, it's not clear what's required/basic use case vs what's optional/advanced.** The K8s section, for example, includes
   content on optional components (Grove, model caching) and
   prerequisites (GPU Operator, RDMA) that are never surfaced at the
   point in the flow where the user needs them.

5. **No top-down conceptual path.** While we have docs for the components and backends, there's no top level "tl;dr" overview that gives a high-level picture of how the pieces fit together without needing to read through all the details. We should for example, make it clear that the backends are modular and profiler/planner are part of the operator.

6. **"User Guides" is a dumping ground.** It mixes backend tutorials
   (SGLang/TRT-LLM/vLLM examples), feature explanations
   (fault tolerance, tool calling, reasoning, LoRA), use-case content
   (multimodal, diffusion, agents), developer content (writing Python
   workers, mocker), and observability — none of which belong
   together.

7. **Cross-cutting features are scattered.** Observability, fault
   tolerance, benchmarking, tool calling, etc. each have their own
   ad-hoc home, with no single section a user can scan to see what
   Dynamo can do.

8. **K8s docs read like an encyclopedia, not a walkthrough.** This is
   the specific instance covered by #8658: prerequisites
   (GPU Operator, RDMA, Grove, model caching) are documented somewhere
   but never surfaced at the point in the deployment flow where the
   user needs them. The structural changes in this DEP are a
   precondition for the #8658 rewrite to have a coherent home.

## Proposal

Adopt a new top-level information architecture organized around two
tutorial tracks (CLI Usage, Kubernetes Usage) plus a knowledge bank
that flows from a high-level overview down through concepts,
components, backends, and features.

### Design principles

1. Have a clear separation between pages for basic tutorials, advanced tutorials/use cases, concepts/explanations, and reference. This makes it easier for users to navigate the site and gives structure to where a new page should go. It also allows us to surface the most important content at the top.
2. Basic usage/tutorial pages should be straightforward and to the point. They should have cmds that can be copy/pasted directly into the terminal, and user configurable options like name/namespace should be in an env variable. These tutorials should explain only what you need to know and leave the rest to component/feature reference pages.
3. 
4. **Two tutorial tracks.** Make it clear that dynamo can either be run on CLI or K8s and both are first class options. "CLI Usage" (run processes locally) and
   "Kubernetes Usage" (apply manifests on a cluster). These are the
   "do stuff" sections. Each is slim — get the user to a working
   deployment, then link out to knowledge content for depth.

5. **Knowledge bank follows a "giving a talk" flow.** 
   Architecture Overview/Landing page (what is dynamo and what are the things its made of) →
   Concepts (what you need to understand first) →
   Components (the actual pieces of the system) →
   Backends (engine-specific reference).

   For example, this image was used in a talk to introduce each of the concepts/components at a high level. Then, each topic was covered in detail with more granularity.
![alt text](image.png)


6. Advanced content/guides are organized. Currently, the "User Guides" section is a dumping ground for anything that isn't a basic tutorial or reference. One potential solution is to split this into "Features" (tool calling, fault tolerance, observability reference, benchmarking, LoRA, perf tuning), "Applications" (multimodal, diffusion, agents), and "Integrations" (LMCache, FlexKV, custom KV events).

7. **Developer content is segregated.** Contributing, building from
   source, writing Python workers, mocker, and local K8s dev tooling
   (Minikube, Tilt) live in a "Developer's Guide" section, not mixed
   into user-facing tutorials.

### New top-level sections

- Getting Started
- Usage
   - CLI Usage
   - Kubernetes Usage
 - Architecture Overview
   - Concepts
   - Components
   - Backends
 - Advanced
   - Features
   - Applications
   - Integrations
 - Developer's Guide
 - Resources (reference matrices, release artifacts, examples, glossary)


### Where current content moves

| Current Section | What happens |
|---|---|
| **Getting Started** | Slimmed to intro + CLI quickstart + K8s quickstart. Local Installation → CLI Usage. Building from Source → Developer's Guide. Contribution Guide → Developer's Guide. |
| **Kubernetes Deployment** | Renamed to "Kubernetes Usage". Dev tools (Minikube/Tilt) → Developer's Guide. Content rewrite covered by #8658. |
| **User Guides** | **Eliminated.** Backend examples → CLI Usage. Feature content → Features. Multimodal/Diffusion/Agents → Applications. Writing Python Workers → Developer's Guide. Benchmarking → Features. Mocker → Developer's Guide. Observability (Local) split across CLI Usage + Features. |
| **Backends** | Examples pages move to CLI Usage. Reference guides, deep-dives (disaggregation, HiCache, KV offloading, chat processor), and observability stay as knowledge-bank Backends (vLLM gains parity with SGLang/TRT-LLM). |
| **Components** | Unchanged structurally. Component design docs move here from Design Docs (e.g., `router-design.md` joins Components > Router). |
| **Integrations** | Unchanged. |
| **Design Docs** | **Eliminated.** `architecture.md` → standalone Architecture Overview. Conceptual docs (`dynamo-flow`, `disagg-serving`, `distributed-runtime`, communication planes) → Concepts. Component design docs → their respective Components sections. |
| **Resources** | Unchanged. |
| **Blog / Documentation** | Unchanged. |

### Proposed navigation tree

See [`sidebar-restructure-proposal-v3.md`](./sidebar-restructure-proposal-v3.md)
for the full proposed `index.yml` tree with every page mapped to its
new location. The summary above captures the structural intent; v3 is
the page-by-page realization. The branch `dynamo-full-docs-refactor`
contains a first-pass implementation of this tree.

### Relationship to #8658

Issue [#8658](https://github.com/ai-dynamo/dynamo/issues/8658) covers
the **content rewrite** of the K8s docs into a single linear
walkthrough (modeled on Cluster API's quickstart), with prerequisites
and optional components surfaced at the point in the flow where they
become relevant, and DGDR as the primary deployment entrypoint. The
draft of that content lives in
[`k8s-docs-refactor-draft.md`](./k8s-docs-refactor-draft.md).

This DEP is the **structural container** for that work: it defines
the "Kubernetes Usage" section the rewrite lives in, and the parallel
sections (CLI Usage, Concepts, Components, Features, etc.) that the
K8s walkthrough links out to instead of duplicating. The two efforts
should land together — restructuring without rewriting K8s leaves the
walkthrough problem; rewriting K8s without restructuring leaves the
new walkthrough in a section that still neighbors "Design Docs" and
"User Guides".

## Alternate Solutions

1. **Incremental cleanup of "User Guides" only.** Split User Guides
   into per-topic top-level sections without touching the rest of the
   sidebar. Rejected: leaves the tutorial/knowledge-bank confusion
   intact, doesn't address the missing CLI Usage track, and doesn't
   give the K8s rewrite a coherent peer section.

2. **Rewrite K8s docs only (just #8658).** Land the K8s walkthrough in
   the current sidebar. Rejected: the walkthrough has to link out for
   concepts, components, and features, and those targets are
   currently scattered or duplicated. The reader experience would
   regress as soon as they followed any link.

3. **Adopt Diátaxis (tutorials / how-to / reference / explanation) as
   the top-level structure.** Rejected (for now): pure Diátaxis
   would force four parallel sections per topic, which is a heavier
   lift and tends to fragment content. The proposed structure is
   Diátaxis-flavored (tutorial tracks vs. knowledge bank) without the
   full taxonomy.

4. **Defer the restructure and ship K8s rewrite into current
   sidebar.** Rejected: the QA feedback explicitly cites structure as
   the problem, not just K8s content. Shipping the K8s rewrite alone
   would leave that critique unaddressed.

## Requirements

- New `index.yml` tree matching the structure in
  `sidebar-restructure-proposal-v3.md`, with all existing pages
  mapped to a new location (no orphans, no broken links).
- Redirects for any URL changes so external bookmarks and search
  results continue to resolve.
- Landing pages for newly introduced collapsible sections
  (CLI Usage, Concepts, Components, Features, Applications,
  Developer's Guide). A first pass at these already exists on the
  `dynamo-full-docs-refactor` branch.
- An expanded Architecture Overview that serves as a true big-picture
  tl;dr (the current `design-docs/architecture.md` may need a rewrite
  rather than a straight move).
- Deduplication of pages that currently appear in multiple sections
  (see "Pages that appear in multiple sections" in v3).
- The K8s content rewrite from #8658 landing inside the new
  Kubernetes Usage section.

## References

- [#8658 — docs: refactor Kubernetes and update to include v1beta1 DGDR](https://github.com/ai-dynamo/dynamo/issues/8658)
- [`sidebar-restructure-proposal-v3.md`](./sidebar-restructure-proposal-v3.md) — full proposed navigation tree
- [`k8s-docs-refactor-draft.md`](./k8s-docs-refactor-draft.md) — draft of the K8s walkthrough content (#8658)
- [`sidebar-restructure-proposal.md`](./sidebar-restructure-proposal.md), [`sidebar-restructure-proposal-v2.md`](./sidebar-restructure-proposal-v2.md) — earlier iterations, retained for design history
- Branch `dynamo-full-docs-refactor` — first-pass implementation of the new sidebar (commits 32ad556, 0acb9e7, 2913620, 4a4251d)
- [Cluster API quickstart](https://cluster-api.sigs.k8s.io/user/quick-start) — model for walkthrough-style K8s docs (used in #8658)
- [Diátaxis framework](https://diataxis.fr/) — influence on the tutorial vs. knowledge-bank split

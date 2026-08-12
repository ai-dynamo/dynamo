---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Documentation Content Guidelines
subtitle: Decide where new Dynamo documentation belongs and how to structure it.
---

Use these guidelines to choose a content type, owning tab, and page structure before drafting new
documentation. For writing style, frontmatter, links, terminology, and validation requirements, see
the [Documentation Style Guide](../pages/community/contributing/documentation/documentation-style-guide.md).

## Choose a Content Type

Split documentation into four primary types. Give each page one primary purpose.

| Content type | Reader goal | Typical location |
|---|---|---|
| Installation | Prepare a cluster, host, dependency, or integration | Kubernetes Guide or CLI Guide installation section |
| Tutorial | Complete a task with Dynamo | Kubernetes Guide, CLI Guide, or Use Cases |
| Knowledge base | Understand architecture, design, or implementation | Developer Guide |
| Reference | Look up an exact field, flag, API, environment variable, or contract | Reference |

Before adding a page, ask:

- Can this information fit cleanly into an existing page?
- Does it describe a substantial workflow or a distinct body of technical knowledge?
- Is it instruction, explanation, or lookup material?
- Will creating a new page make the topic easier to find, or fragment related information?

Prefer extending an existing page when the new material serves the same reader goal.

## Choose the Owning Tab

Use the topic and execution environment to choose the tab:

| Tab | Content |
|---|---|
| Kubernetes Guide | Core Dynamo workflows performed with Kubernetes resources, DGD or DGDR specifications, Helm, and `kubectl` |
| CLI Guide | Core Dynamo workflows performed with local Python processes, containers, shell commands, or repository scripts |
| Use Cases | Features and applications of Dynamo that build on the core deployment workflows |
| Developer Guide | Architecture, implementation, design decisions, communication flows, and contributor knowledge |
| Reference | Complete configuration surfaces and exact technical contracts |
| Recipes | Validated, copy-ready model deployment configurations and feature benchmarks |

Put core functionality and model-deployment workflows in the Kubernetes or CLI Guide. Use the Use
Cases tab for applications and cross-cutting features that build on those core workflows.

## Organize a Topic

Use an overview page when a topic has multiple workflows or optional branches:

```text
Topic
├── Overview
├── Subtopic 1
└── Subtopic 2
```

The overview should introduce the topic, provide a basic path or high-level summary, and link to the
next pages. Deeper pages should cover optional or advanced workflows that readers can choose after
understanding the overview.

Match the source tree to the site structure:

- Use top-level directories that match the tab directories under `docs/fern/pages/`.
- Use nested directories for sidebar sections and topic groups.
- Keep filenames aligned with page titles and URL slugs.
- Do not use `README.md` as a published page name.

## Quickstarts

The site has one primary quickstart for Kubernetes and one for local CLI usage. Keep them as the
shortest reliable path to a working result.

- Use a minimal, copy-ready workflow.
- Choose one representative default instead of presenting configuration decisions.
- Show the success signal directly.
- Exclude architecture diagrams, implementation detail, tuning, and optional branches.
- Link to installation, tutorials, the Developer Guide, and Reference for additional detail.

## Installation Pages

Use installation pages for prerequisites that readers must prepare before following a tutorial.

- Treat the main Kubernetes and CLI installation pages as the canonical baseline.
- Do not repeat baseline steps such as installing the GPU Operator or Dynamo platform in a
  branch-specific installation page.
- Document only what readers must add, replace, or configure differently from the baseline.
- End with a direct verification that proves the dependency or environment is ready.
- Keep feature usage and deployment workflows in tutorials.

Start with the matching template:

- [Kubernetes installation template](kubernetes/installation.mdx)
- [CLI installation template](cli/installation.mdx)

## Tutorials

Tutorials are concise, action-oriented walkthroughs for completing a task.

- Put prerequisites before the procedure and link to canonical installation pages.
- Use the Fern `<Steps>` component for the main sequence.
- Keep one primary action in each step.
- Explain the decisions users must make and the configuration values that materially affect the
  result.
- Keep architecture, implementation detail, exhaustive options, and field definitions in the
  Developer Guide or Reference.
- End with a request, status check, or other direct verification.

A tutorial should describe a reusable workflow rather than one narrowly tailored deployment. Its
examples must still be concrete and copy-ready. Use tabs when readers follow the same sequence with
different backends, deployment modes, providers, or environments.

Match examples to the owning tab:

- Kubernetes tutorials use DGD, DGDR, Helm, and `kubectl` where applicable.
- CLI tutorials use local Python, container, shell, or repository-script commands.
- Do not place operator-specific examples in a CLI tutorial or local-process examples in a
  Kubernetes tutorial.

Start with the matching template:

- [Kubernetes tutorial template](kubernetes/tutorial.mdx)
- [CLI tutorial template](cli/tutorial.mdx)

## Knowledge Base Pages

Use the Developer Guide knowledge base to explain how or why Dynamo works.

- Cover architecture, component responsibilities, data and control flow, lifecycle, invariants,
  design decisions, tradeoffs, and failure behavior.
- Put architecture and sequence diagrams here rather than in tutorials or quickstarts.
- Include source locations when they help contributors navigate the implementation.
- Keep proposals and major architecture changes in Dynamo Enhancement Proposals (DEPs).
- Link to Kubernetes or CLI tutorials for operational procedures.
- Link to Reference for exact fields, flags, defaults, and allowed values.

## Reference Pages

Reference pages provide complete, structured lookup material.

- Group references by interface or category, such as Python, Rust, Kubernetes, components, or
  backends.
- Cover the full supported configuration surface in scope.
- Include types, defaults, allowed values, validation rules, precedence, and interactions.
- Use Fern parameter components such as `<ParamField>` for fields, arguments, and environment
  variables when appropriate.
- Keep examples short and illustrative. Link to a tutorial for end-to-end usage.
- Prefer generated reference content when the source contract can produce it reliably.

Reference coverage should resemble the completeness of a command's `--help` output or a system
manual page, while remaining organized for the interface being documented.

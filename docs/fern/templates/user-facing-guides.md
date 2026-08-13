---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing User-Facing Guides
subtitle: Create quickstarts, installation pages, and tutorials for Kubernetes, CLI, and use-case workflows.
---

User-facing guides help readers get Dynamo running or use it to complete a task. They live in the
Kubernetes Guide, CLI Guide, and Use Cases, and follow one of three patterns: quickstart,
installation, or tutorial.

For guidance on choosing between user-facing, knowledge-base, and reference content, see the
[Documentation Content Guidelines](docs-guidelines.md). For prose, formatting, links, terminology,
and validation requirements, see the
[Documentation Style Guide](../pages/community/contributing/documentation/documentation-style-guide.md).

## Choose the Owning Tab

Choose the tab based on the reader's goal and execution environment.

| Tab | Use it for |
|---|---|
| Kubernetes Guide | Core Dynamo workflows performed with Kubernetes resources, DGD or DGDR specifications, Helm, and `kubectl` |
| CLI Guide | Core Dynamo workflows performed with local Python processes, containers, shell commands, or repository scripts |
| Use Cases | Features and applications that build on the core Kubernetes or CLI deployment workflows |

Put foundational installation and model-deployment workflows in the Kubernetes or CLI Guide. Put a
workflow in Use Cases when its primary purpose is applying Dynamo to a feature or application, such
as tool calling, multimodal inference, or observability.

## Choose a Guide Type

| Guide type | Reader goal | Typical placement |
|---|---|---|
| Quickstart | Reach a first working result | The primary Kubernetes or CLI entry point |
| Installation | Prepare a cluster, host, dependency, or integration | Kubernetes or CLI installation sections; a use-case section only for unique prerequisites |
| Tutorial | Complete a reusable task | Kubernetes Guide, CLI Guide, or Use Cases |

A page should follow one primary pattern. If a tutorial needs substantial environment preparation,
link to an installation page instead of embedding a second installation workflow. If a page becomes
an exhaustive list of fields, flags, or defaults, move that material to Reference and link to it.

## Write Quickstarts

The site has one primary quickstart for Kubernetes and one for local CLI usage. Keep them as the
shortest reliable path to a working result.

- Use a minimal, copy-ready workflow.
- Choose one representative default instead of presenting configuration decisions.
- Show the success signal directly.
- Exclude architecture diagrams, implementation detail, tuning, and optional branches.
- Link to installation, tutorials, the Developer Guide, and Reference for additional detail.

Quickstarts are few and deliberately curated, so they do not need a generic starter template.

## Write Installation Pages

Use installation pages for prerequisites that readers must prepare before following a tutorial.

- Treat the main Kubernetes and CLI installation pages as the canonical baseline.
- Do not repeat baseline steps such as installing the GPU Operator or Dynamo platform in a
  branch-specific installation page.
- Document only what readers must add, replace, or configure differently from the baseline.
- End with a direct verification that proves the dependency or environment is ready.
- Keep feature usage and deployment workflows in tutorials.

## Write Tutorials

Tutorials are concise, action-oriented walkthroughs for completing a task.

- Put prerequisites before the procedure and link to canonical installation pages.
- Use the Fern `<Steps>` component for the main sequence.
- Keep one primary action in each step.
- Explain the decisions users must make and the configuration values that materially affect the
  result.
- Keep architecture, implementation detail, exhaustive options, and field definitions in the
  Developer Guide or Reference.
- End with a request, status check, or other direct verification.

Describe a reusable workflow rather than one narrowly tailored deployment. Keep examples concrete
and copy-ready. Use tabs when readers follow the same sequence with different backends, deployment
modes, providers, or environments.

## Keep Environment-Specific Instructions Separate

Match commands and configuration to the owning tab:

- Kubernetes guides use DGD, DGDR, Helm, and `kubectl` where applicable.
- CLI guides use local Python, container, shell, or repository-script commands.
- Use-case guides center the feature or application. Link to foundational Kubernetes or CLI setup
  instead of repeating it.
- At the start of a use-case guide, state whether the workflow supports Kubernetes, CLI, or both.
  Make every copy-ready example match the stated path. Use tabs when both paths follow the same
  sequence; use separate sections or pages when the procedures differ materially.
- When a guide relies on a launch script, show the command that invokes it and the arguments readers
  must understand instead of presenting the script as a black box.
- Do not place operator-specific examples in a CLI guide or local-process examples in a Kubernetes
  guide.

## Start from a Template

Keep shared authoring guidance in this page rather than copying it into each environment directory.
The environment directories contain only starter files whose structure or examples differ by
execution environment.

| Page type | Kubernetes | CLI |
|---|---|---|
| Installation | [Kubernetes installation template](kubernetes/installation.mdx) | [CLI installation template](cli/installation.mdx) |
| Tutorial | [Kubernetes tutorial template](kubernetes/tutorial.mdx) | [CLI tutorial template](cli/tutorial.mdx) |

For a Use Cases page, start with the Kubernetes or CLI tutorial template that matches the primary
execution path, then adapt the framing around the feature or application. Add a dedicated use-case
template only after multiple pages share a stable structure that the existing templates do not
capture.

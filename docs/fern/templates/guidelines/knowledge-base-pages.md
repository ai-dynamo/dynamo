---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Knowledge Base Pages
subtitle: Explain Dynamo architecture, behavior, and implementation in the Developer Guide.
---

Use the Developer Guide knowledge base to explain how or why Dynamo works. These pages build a
mental model for readers rather than walking them through an operational task or listing a complete
configuration surface.

For help choosing another content type, see the
[Documentation Content Guidelines](docs-guidelines.md).

## Cover System Behavior

Include the technical context readers need to understand the subject:

- Architecture and component responsibilities
- Data and control flow
- Lifecycle and state transitions
- Invariants and failure behavior
- Design decisions and tradeoffs
- Interactions with other Dynamo components

Put architecture and sequence diagrams here rather than in tutorials or quickstarts.

## Organize the Explanation

Start with the purpose and the reader's mental model before moving into implementation detail. Use
an overview page when a component or subsystem has several concepts, flows, or optional branches.

Include source locations when they help contributors navigate the implementation, but explain the
behavior in the page instead of requiring readers to reconstruct it from code.

## Keep Content Boundaries Clear

- Link to [user-facing guides](user-facing-guides.md) for installation and operational procedures.
- Link to [Reference](reference-pages.md) for exact fields, flags, defaults, and allowed values.
- Keep proposals and major architecture changes in Dynamo Enhancement Proposals (DEPs).
- Keep examples illustrative. Do not turn an explanation page into a second tutorial or exhaustive
  configuration reference.

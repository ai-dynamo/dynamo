<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions — docs, examples, recipes

When creating or editing files under `docs/`, `examples/`, or `recipes/`, follow the
[documentation style guide](documentation-style-guide.md). Non-negotiables:

- SPDX header on every file: frontmatter `#` form for Fern docs, `<!-- -->` for plain READMEs,
  full Apache block for code/config; copyright range `2025-2026`.
- Fern docs: `---` frontmatter with SPDX + at least one metadata key (`title`/`subtitle`/
  `sidebar-title`). Fern renders the page H1 from the nav `page:`, so do **not** add a body `# H1`
  (it duplicates the title); start the body at `##`.
- Admonitions GitHub-style (`> [!NOTE]`); the Fern build converts them — don't hand-write `<Note>`.
- Links: relative + extension within `docs/`; absolute `github.com/ai-dynamo/dynamo` URLs for
  targets outside `docs/` (no `../` escapes).
- Code fences language-tagged (`bash`, not `sh`); backend casing vLLM / SGLang / TensorRT-LLM.
- No internal/sensitive refs (NVBug/JIRA IDs, internal hosts, secrets, TODO/FIXME) in shipped docs.
- Write for humans: no marketing/bombast, no filler, be concrete.

The Dynamo Docs Bot enforces the deterministic subset pre-merge.

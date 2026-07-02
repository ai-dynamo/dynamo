---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Documentation Style Guide
subtitle: How to write and structure Dynamo docs, examples, and recipes
---

This is the documentation standard for NVIDIA Dynamo. Follow it for every page under `docs/`, and for
the READMEs and configuration under `examples/` and `recipes/`. Consistent structure, accurate
cross-references, and plain technical prose keep the docs usable across both the Fern-published site
and GitHub.

These are defaults, not rigid rules. Deviate when it makes the content clearer, and say why. This
guide takes precedence; for anything it doesn't cover, defer to the
[Google developer documentation style guide](https://developers.google.com/style), then the
[Microsoft Writing Style Guide](https://learn.microsoft.com/style-guide).

**Must-fix vs guidance.** The bot enforces five things as must-fix: an SPDX header, frontmatter with
at least one metadata key (and **no** stray body `# H1`), a nav entry for every new page, the link
rules, and no internal references. Treat everything else here as guidance, and deviate with a reason.

## Frontmatter and title

Fern generates the page H1 from the nav `page:` value (`docs/index.yml` and the versioned navs), so
**do not add a body `# H1`**: it renders a second, duplicate title. Start the body at `##`.
Frontmatter holds the SPDX header plus metadata, and **must contain at least one real YAML key**, or
the SPDX `#` comments are read as body content and render as H1s.

```yaml
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configuration and Tuning
subtitle: Router flags, event transport, and tuning guidance
---

Intro paragraph (no body `# H1`), then `##` sections.
```

- `title`: sets the page H1, the browser-tab `<title>`, and the sidebar label. Optional; if omitted,
  Fern uses the nav `page:` value as the H1. Set it when the heading should differ from a short nav
  label (e.g. nav "Introduction", H1 "Introduction to Dynamo"), and pair with `sidebar-title`.
- `subtitle` (**recommended**): renders as a visible subtitle under the H1 (and as the meta description).
- `sidebar-title`: short label for the sidebar nav; use when the title is long.
- Optional: `description`, `keywords` (search), `last-updated`, `max-toc-depth`.

## SPDX headers

Every file carries an SPDX header with the copyright **range** `Copyright (c) 2025-2026 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.` (not a single year). The form depends on the file:

- **Fern docs** (`.md`/`.mdx` with frontmatter): two `#` lines **inside** the `---` block (above).
- **Plain markdown** (READMEs without frontmatter): an HTML comment block:
  ```markdown
  <!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
  -->
  ```
- **Code and config** (`.py`, `.sh`, `.yaml`, `Dockerfile`): the full Apache-2.0 header block as
  `#` (or `//`) comments at the top.

> [!WARNING]
> Keep SPDX lines inside the frontmatter. A bare `#` SPDX line in the markdown body renders as an H1.

## Page types

Each page serves one of four needs ([Diátaxis](https://diataxis.fr/)); keep them distinct rather than
blending:

- **Tutorial**: a learning-oriented first lesson (`getting-started/`).
- **How-to**: goal-oriented steps for one task (`backends/<engine>/`, `kubernetes/`).
- **Reference**: dry technical description of flags, APIs, config (`reference/support-matrix.md`).
- **Explanation**: background and rationale (`design-docs/`).

A `backends/` how-to that drifts into a flag reference, or a `getting-started/` tutorial that explains
the KV router architecture, is the most common docs smell. Split it and cross-link.

## Headings and structure

- Start the body at `##`: Fern generates the page H1 from the nav, so a body `# H1` duplicates it (see Frontmatter and title). Open with a short intro paragraph, then `##` sections.
- Use a logical hierarchy (`##` → `###`); do not skip levels.
- Headings use **Title Case for short label / noun-phrase headings** ("Routing Behavior", "KV Event
  Transport and Persistence"); a heading that reads as a **full phrase or clause** stays sentence case
  ("Choosing a checkpoint flow"). Above all, be **consistent within a page**: don't mix the two
  arbitrarily. The Title-Case-for-labels convention is a deliberate Dynamo deviation from the
  Google/Microsoft sentence-case default.
- No end punctuation on headings, or on short (≤3-word) list items; save periods for body copy.
- Renaming a heading changes its anchor and breaks inbound links, so rename deliberately.

## Procedures

- Put the **condition before the instruction**: "To enable KV-aware routing, set `--router-mode kv`",
  not "Set `--router-mode kv` to enable KV-aware routing". A reader who doesn't need it can then skip
  the step.
- One action per numbered step.

## Writing for humans

Much of this prose is now drafted by agents. Edit it so it does not read that way:

- **Cut marketing and bombast.** No "seamless, robust, powerful, blazing-fast, cutting-edge,
  effortless, unlock, leverage, delve, comprehensive, rich ecosystem, world-class, game-changing."
  Say what it does.
- **Cut filler and hedging.** Drop "it's important to note that", "it's worth mentioning",
  "generally speaking", "simply", "just". Write "to", not "in order to".
- **Start sentences with a verb** where you can; cut weak openers like "you can" and "there is/are".
- **Prefer simple verbs.** Avoid be/have/make/do as the main verb and multi-word phrases: "use" not
  "utilize", "connect" not "establish connectivity". Omit weak adverbs (quite, very).
- **Drop difficulty words**: "easy", "easily", "simple", "simply". What's easy varies by reader.
- **No empty framing.** Don't open with "In this section we will…" or close with "In conclusion…".
  Lead with the content. **Don't restate the heading** in the first sentence.
- **Be concrete.** Name the flag, the default, the command, not "configure the appropriate
  settings". Specifics over adjectives. No rule-of-three padding unless each term carries weight.
- Short declarative sentences; active voice; present tense; second person imperative
  ("Set `--flag` to…"). Contractions ("it's", "you'll") are fine; they read as human.
- Avoid the AI tic of frequent em-dash asides. If a sentence can be cut, cut it.

## Terminology

- Backend names, exact casing: **vLLM**, **SGLang**, **TensorRT-LLM** (or **TRT-LLM**). Never "vllm",
  "Sglang", or "TensorRT LLM".
- Product and component casing: **NVIDIA Dynamo** on first mention, then **Dynamo**; **KV router**,
  **NIXL**, **GPU**; **Kubernetes**, not "k8s", in prose.
- Use the same word for the same concept; don't reuse one word for two concepts.
- Expand acronyms on first use ("Time To First Token (TTFT)", "Expert Parallelism (EP)"), then use
  the acronym.
- **Inclusive terms**: "denylist"/"allowlist", not "blacklist"/"whitelist"; "primary"/"replica", not
  "master"/"slave".
- **No needless jargon**: don't use a term when a more familiar one exists; cut marketing-speak.
- Mark feature lifecycle inline: **Experimental.** for preview features; **Deprecated.** (with a
  `> [!WARNING]`) for removed or legacy ones. Note availability for new features ("Available since
  v0.X").

## Formatting

- **Lists by purpose:** numbered (`1.`) for sequences and steps, bulleted (`-`) otherwise; consistent
  within a page. Tables stay scannable; keep long prose out of cells.
- **File names** are kebab-case (`router-configuration.md`); Chinese translations live under
  `fern/translations/zh-CN/pages-dev/` at the same relative path and file name as the English page
  (Fern's native localization pairs them and adds the language picker automatically).
- **Code fences** always tag the language (`bash`, `yaml`, `python`, `json`, `rust`, `text`,
  `mermaid`; `bash`, not `sh`). Keep commands copy-pasteable: no `$`/`#` prompt prefixes, real flags
  not placeholders. Put output in its own `text` block. In prose, wrap flags (`--router-mode`),
  paths, and env vars (`DYN_*`) in backticks.

```bash
python3 -m dynamo.frontend --router-mode kv
```

- **Diagrams** use ` ```mermaid ` blocks. **Images** live under `docs/assets/img/` with descriptive
  alt text: `![KV-aware routing data flow](assets/img/kv-routing.svg)`.
- Whitespace, trailing newlines, and line endings are normalized by pre-commit.

## Links

Link targets are handled differently depending on whether they live inside `docs/`:

- **Within `docs/`** (doc → doc): a **relative path with the file extension**, such as
  `[Routing Concepts](router-concepts.md)` or `[Deployment](../kubernetes/README.md)`. Fern resolves
  these to published-site URLs. Don't hardcode `https://docs.nvidia.com/...` links to pages in this
  repo.
- **Outside `docs/`** (examples, recipes, source, `container/`, sibling repos): an **absolute GitHub
  URL** like `https://github.com/ai-dynamo/dynamo/blob/main/<path>` (use `/tree/main/` for a
  directory). A relative `../` path that escapes `docs/` (e.g. `../../../../examples/...`) breaks on
  the published site and after version path-rewrites, so don't use it.
- **Link text describes the destination**: never "click here" or "this page". Lead with the concept
  so the page stays skimmable.

Every internal link and `#anchor` must resolve to a real file or heading.

## Internal and sensitive references

Published docs must not contain internal-only references: NVBug numbers, Linear or JIRA IDs, internal
`*.nvidia.com` hostnames, credentials or tokens, customer names, or `TODO`/`FIXME` markers. Keep
tracker references in commits and pull requests, not in shipped docs.

## Admonitions

Write admonitions GitHub-style in `docs/` source. They render on GitHub, and the Fern build converts
them to Fern callouts when publishing:

```markdown
> [!NOTE]
> Additional context users should know.

> [!TIP]
> A helpful suggestion.

> [!IMPORTANT]
> Key information.

> [!WARNING]
> Something to watch out for.

> [!CAUTION]
> A risk or negative outcome.
```

The build maps `[!NOTE]→<Note>`, `[!TIP]→<Tip>`, `[!IMPORTANT]→<Info>`, `[!WARNING]→<Warning>`,
`[!CAUTION]→<Error>`. Don't use bold-text pseudo-admonitions (`> **Note:**`); they aren't converted.

## Fern build transforms

Author against `docs/` on `main`; the publish step applies transforms, so you don't hand-write
Fern-specific output:

- **Callouts:** `fern/convert_callouts.py` converts GitHub admonitions to Fern components.
- **Versioned paths:** content is copied `pages-dev/ → pages-vX.Y.Z/`, and nav paths are rewritten
  when a version is cut.

Write portable GitHub-flavored Markdown; don't pre-bake Fern components or version-specific paths.

## Navigation and placement

- Add every new `docs/` page to `docs/index.yml` under the right `section`, as a `- page:` + `path:`
  entry. A page that isn't in the nav is unreachable.
- Match the topic directory: `getting-started`, `reference`, `kubernetes`, `backends/<engine>`,
  `features`, `components/<component>`, `observability`, `design-docs`, `tool-calling`, `benchmarks`,
  `agents`, `integrations`, `performance`.
- Don't duplicate content across pages; link the canonical page. Prefer extending an existing page
  over adding a new file.

## Examples and recipes

- **Examples** (`examples/`): code-first, each in a topic directory with a `README.md`, surfaced
  from the relevant `docs/<area>/*-examples.md` page.
- **Recipes** (`recipes/`): one `<model>/` directory each, with a `README.md`, `Dockerfile`, and
  configs. Add every new recipe to the **Available Recipes** table in `recipes/README.md`.
- Their READMEs use the HTML-comment SPDX form (no frontmatter).

## Pre-merge checklist

These are checked automatically on every docs/examples/recipes pull request; resolve each before
merge:

- [ ] SPDX header present, correct form for the file type, `2025-2026` range
- [ ] No body `# H1` (Fern renders the title from the nav `page:`); frontmatter has SPDX + at least one key (`title`/`subtitle`/`sidebar-title`)
- [ ] New, moved, or deleted page reflected in the right index (`docs/index.yml`, `*-examples.md`, or
      `recipes/README.md`)
- [ ] Links: relative + extension within docs, absolute GitHub URL outside docs (no `../` escapes);
      link text describes the destination; every internal link and `#anchor` resolves
- [ ] Code fences language-tagged, no shell prompts, output in `text`; admonitions GitHub-style
- [ ] Lists typed by purpose; images have alt text and live under `assets/img/`
- [ ] Heading case is consistent within the page (Title Case for short labels, sentence case for full phrases); no end punctuation; one page type (tutorial/how-to/reference/explanation)
- [ ] No internal or sensitive references (NVBug/JIRA/Linear IDs, internal hosts, secrets, TODO/FIXME)
- [ ] Terminology: correct casing (vLLM / SGLang / TensorRT-LLM / Dynamo / Kubernetes), inclusive
      terms, acronyms expanded on first use, no needless jargon
- [ ] Prose reads like a person wrote it: verb-first, concrete, no marketing, filler, or empty framing
- [ ] Content is consistent with existing docs and accurate against the current code

## References

This guide layers on, and defers to, established authorities: the
[Google developer documentation style guide](https://developers.google.com/style), the
[Microsoft Writing Style Guide](https://learn.microsoft.com/style-guide),
[Write the Docs](https://www.writethedocs.org/guide/), and [Diátaxis](https://diataxis.fr/).

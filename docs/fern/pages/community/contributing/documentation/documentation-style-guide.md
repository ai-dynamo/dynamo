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

The refactored site uses four primary page types: **installation**, **tutorial**, **design document**,
and **reference**. A **quickstart** is a deliberately minimal tutorial subtype. Give each page one
primary purpose; split and cross-link content instead of blending page types.

Less is more across the site. Keep only what the reader needs for the page's purpose. Reference is the
exception: it should exhaustively document the supported configuration surface.

| Page type       | Reader goal                                         | Default shape                                                                                | Keep out                                                                      |
| --------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Installation    | Prepare a machine, cluster, dependency, or platform | Prerequisites, ordered install steps, verification, common blockers                          | Feature tutorials, architecture, duplicated deployment workflows              |
| Tutorial        | Complete a specific user or operator task           | Outcome, prerequisites, `<Steps>`, verification, focused troubleshooting                     | Architecture diagrams, implementation detail, exhaustive configuration tables |
| Design document | Understand how or why Dynamo works                  | Architecture, lifecycle, data or control flow, invariants, rationale, diagrams               | Production setup instructions and exhaustive field catalogs                   |
| Reference       | Look up an exact contract or option                 | Complete fields, types, defaults, allowed values, APIs, metrics, status codes, compatibility | Narrative walkthroughs and repeated architecture background                   |

### Installation pages

Use an installation page when readers must install or prepare something before they can use Dynamo or
a major integration. Keep installation canonical: link to it from tutorials instead of repeating it.

- State supported environments and prerequisites first.
- Use an ordered flow, preferably `<Steps>` in an `.mdx` page, for actions that must happen in
  sequence.
- Provide copyable commands with safe defaults and no shell prompt characters.
- End with one direct verification that proves the installation works.
- Include only common installation blockers. Move operational troubleshooting to the relevant guide.
- Use tabs for parallel platform or environment variants when the steps remain substantially the same.
- Do not explain a feature, create a DynamoGraphDeployment (DGD), or repeat the platform quickstart
  unless that action is itself part of the installation. Link the canonical procedure.

Most features do not need their own installation section. Add one only when the user must perform a
distinct prerequisite or install step.

### Quickstarts

A quickstart is the shortest reliable path from a prepared environment to a working result. It is a
tutorial subtype, but its admission bar is stricter. A reader should be able to copy, paste, and run it
without understanding Dynamo internals or making unnecessary choices.

- Use the fewest steps and commands that produce a useful working result.
- Make every prerequisite explicit, but link to installation instead of repeating it.
- Choose one safe, representative default. Do not make readers tune or compare options.
- Provide complete copyable commands or manifests. Do not rely on unstated edits, placeholders that
  are not defined, or knowledge from another page.
- Show the expected success signal: a response, status, pod state, or one verification command.
- Make failure-prone details explicit so the path is hard to misuse.
- Do not include architecture diagrams, design rationale, long explanations, exhaustive alternatives,
  performance tuning, or broad troubleshooting. Link to tutorials, design documents, and Reference.
- Remove any paragraph or option that is not required to reach the working result.

### Tutorials

Use a tutorial for a concrete task that requires several user actions, such as enabling a deployment
mode, configuring a backend, or validating an operational behavior. Tutorials in the refactored site
are task-oriented how-to guides, not conceptual lessons.

- Open with the outcome and only the prerequisites specific to the task.
- Use `<Steps>` and `<Step>` in `.mdx` for a meaningful sequence. Keep one primary action per step.
- Make step titles imperative and specific, such as “Enable KV routing” or “Verify the workers.”
- Put the condition immediately before the action that depends on it.
- Include enough explanation to make the action safe, then stop.
- End with verification. Add troubleshooting only for likely failures of this exact workflow.
- Do not include architecture diagrams unless the task cannot be completed safely without one. In
  general, link to the Developer Guide design document instead.
- Do not reproduce exhaustive flag, environment-variable, API, or metric definitions. Link to
  Reference.
- If a capability works out of the box, it probably does not need a tutorial. If enabling it requires
  one simple flag or field, prefer a short section on an existing page plus a Reference entry.

Consolidate parallel variants with `<Tabs>` when they follow the same step flow and the combined page
stays easy to scan. Common tab sets include **vLLM / SGLang / TensorRT-LLM** and
**Aggregated / Disaggregated**. Keep shared actions outside the tabs and put only the differing
commands or configuration inside them. Split variants into separate pages when their prerequisites,
sequence, or troubleshooting diverge substantially, or when tabs make the page too long.

### Design documents

Use design documents in the Developer Guide knowledge base for architecture and implementation
knowledge. Length is acceptable when the detail helps a developer understand the system, but organize
it so readers can find the relevant subsystem or flow.

- Explain component responsibilities, lifecycle, data flow, control flow, invariants, tradeoffs, and
  failure behavior.
- Use architecture and sequence diagrams here rather than in quickstarts or tutorials.
- Distinguish current behavior from proposals or historical context.
- Name implementation concepts and source locations when they help contributors navigate the code.
- Do not turn a design document into a deployment guide. Link to the User Guide for actions.
- Do not catalog every supported option. Link to Reference for exact configuration.

### Reference pages

Use Reference for lookup, not instruction. Reference is the one page type where completeness is more
important than brevity. It should describe the entire supported surface accurately and consistently.

- Document every supported flag, environment variable, configuration field, API endpoint, status
  code, metric, allowed value, default, validation constraint, and compatibility condition in scope.
- Prefer structured fields, concise tables, and generated sources over narrative prose.
- State interactions and precedence when one option changes another.
- Mark deprecated, experimental, backend-specific, and version-specific behavior explicitly.
- Keep examples small and illustrative. Link to a tutorial for an end-to-end workflow.
- Verify runtime-sensitive defaults and behavior against current code or generated schemas.

### Choosing whether to add a page

Do not create a page because an old page existed or because a feature has a name. Create one only when
it serves a distinct reader goal that an existing page cannot serve cleanly. Prefer, in order:

1. No new content when the behavior is automatic and needs no user decision.
2. A Reference entry for an exact option or contract.
3. A short section on an existing tutorial or installation page.
4. A new page only for a substantial, discoverable workflow or a distinct body of design knowledge.

A tutorial that drifts into architecture, a design document that becomes a setup guide, or a
quickstart that asks readers to make several configuration decisions is a page-type failure. Split and
cross-link it.

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

## Procedures and variants

- Put the **condition before the instruction**: “To enable KV-aware routing, set
  `--router-mode kv`,” not “Set `--router-mode kv` to enable KV-aware routing.” Readers who do not
  need the condition can skip the action.
- Use one primary action per step. Keep setup, action, and verification in that order.
- Use `<Steps>` and `<Step>` for meaningful tutorial and installation sequences. Use a plain ordered
  list for two trivial actions that do not benefit from named steps.
- Use `<Tabs>` and `<Tab>` to consolidate parallel backends, deployment modes, operating systems, or
  providers when the surrounding workflow is shared. Do not duplicate shared prose inside every tab.
- Keep tab labels short and consistent: `vLLM`, `SGLang`, `TensorRT-LLM`, `Aggregated`, and
  `Disaggregated`.
- Do not hide materially different workflows in tabs. Split them when the sequence or prerequisites
  differ enough that a shared flow becomes confusing.
- Author pages that use Steps or Tabs as `.mdx`. Prefer plain Markdown for pages that do not need
  interactive structure.

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
- Mark feature lifecycle inline: **Experimental.** for preview features; **Deprecated.** (with
  `<Warning>` in `.mdx` or `> [!WARNING]` in `.md`) for removed or legacy ones. Note availability
  for new features ("Available since v0.X").

## Formatting

- **Lists by purpose:** numbered (`1.`) for sequences and steps, bulleted (`-`) otherwise; consistent
  within a page. Tables stay scannable; keep long prose out of cells.
- **File names** are kebab-case (`router-configuration.md`); Chinese translations live under
  `docs/fern/translations/zh-CN/pages/` at the same relative path and file name as the English page
  (Fern's native localization pairs them and adds the language picker automatically).
- **Code fences** always tag the language (`bash`, `yaml`, `python`, `json`, `rust`, `text`,
  `mermaid`; `bash`, not `sh`). Keep commands copy-pasteable: no `$`/`#` prompt prefixes, real flags
  not placeholders. Put output in its own `text` block. In prose, wrap flags (`--router-mode`),
  paths, and env vars (`DYN_*`) in backticks.

```bash
python3 -m dynamo.frontend --router-mode kv
```

- **Diagrams** use ` ```mermaid ` blocks. **Images** live under `docs/fern/assets/img/` with
  descriptive alt text, referenced by a relative path from the page:
  `![KV-aware routing data flow](../../../assets/img/kv-routing.svg)`. Blog posts use their own
  `pages/blog/_assets/` tree instead.
- Whitespace, trailing newlines, and line endings are normalized by pre-commit.

## Links

Link targets are handled differently depending on whether they live inside `docs/`:

- **Within `docs/`** (doc → doc): a **relative path with the file extension**, such as
  `[Routing Concepts](router-concepts.md)` or `[Deployment](../kubernetes/quickstart.mdx)`. Fern resolves
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

Match the admonition syntax to the source file extension. In `.mdx` pages, use Fern callout
components:

```jsx
<Note>
Additional context users should know.
</Note>

<Tip>
A helpful suggestion.
</Tip>

<Info>
Key information.
</Info>

<Warning>
Something to watch out for.
</Warning>

<Error>
A risk or negative outcome.
</Error>
```

In `.md` pages, use GitHub-style blockquote admonitions. They render on GitHub, and the Fern build
converts them to Fern callouts when publishing:

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

For `.md` pages, the build maps `[!NOTE]→<Note>`, `[!TIP]→<Tip>`, `[!IMPORTANT]→<Info>`,
`[!WARNING]→<Warning>`, `[!CAUTION]→<Error>`. Don't use bold-text pseudo-admonitions
(`> **Note:**`); they aren't converted.

## Fern components and build behavior

Author source pages under `docs/fern/pages/`. Use the lightest format that supports the page:

- Use `.mdx` when a tutorial or installation page needs `<Steps>`, `<Tabs>`, cards, or another Fern
  component.
- Use `.md` for straightforward prose that works as portable GitHub-flavored Markdown.
- In `.mdx`, write admonitions with Fern callout components.
- In `.md`, write admonitions with GitHub syntax; `docs/fern/scripts/convert_callouts.py` converts them during the
  build.
- Do not add components as decoration. Components must improve sequencing, branching, disclosure, or
  scanning.
- The release workflow builds versioned page trees from the tagged documentation and rewrites the
  navigation paths for each snapshot. Do not hardcode version-specific paths.

## Navigation and placement

- Add every new page to `docs/fern/index.yml` under the right tab and `section`, as a `- page:` +
  `path:` entry. `path:` is relative to `docs/fern/`, so it starts with `pages/`. A page that isn't in
  the nav is unreachable.
- The site is tab-based, and each tab is rooted at one directory under `docs/fern/pages/`:
  `kubernetes/` and `cli/` (parallel deployment guides), `use-cases/`, `recipes/`,
  `developer-guide/`, `reference/`, `blog/`, `community/`, and `home/`.
- Choose the tab before the directory. `kubernetes/` and `cli/` cover the same features for two
  different readers, so place a page by the surface its instructions target — manifests, Helm, CRDs,
  or `kubectl` in `kubernetes/`; `dynamo` commands, local processes, and env vars in `cli/`. Content
  that genuinely serves both gets two pages, each complete for its reader; a contract that's
  independent of how Dynamo is launched belongs in `reference/`.
- Within the tab, place the file beside the nearest existing page on the same topic and reuse that
  sibling's subdirectory. Read `index.yml` for the live structure.
- Don't duplicate content across pages; link the canonical page. Prefer extending an existing page
  over adding a new file.

## Examples and recipes

- **Examples** (`examples/`): code-first, each in a topic directory with a `README.md`, surfaced
  from the relevant `*-examples.md` page or from the topic page that needs it.
- **Recipes** (`recipes/`): one `<model>/` directory each, with a `README.md`, `Dockerfile`, and
  configs. Add every new recipe to the **Available Recipes** table in `recipes/README.md`.
- Their READMEs use the HTML-comment SPDX form (no frontmatter).

## Pre-merge checklist

These are checked automatically on every docs/examples/recipes pull request; resolve each before
merge:

- [ ] SPDX header present, correct form for the file type, `2025-2026` range
- [ ] No body `# H1` (Fern renders the title from the nav `page:`); frontmatter has SPDX + at least one key (`title`/`subtitle`/`sidebar-title`)
- [ ] New, moved, or deleted page reflected in the right index (`docs/fern/index.yml`, `*-examples.md`, or
      `recipes/README.md`)
- [ ] Links: relative + extension within docs, absolute GitHub URL outside docs (no `../` escapes);
      link text describes the destination; every internal link and `#anchor` resolves
- [ ] Code fences language-tagged, no shell prompts, output in `text`; admonitions match the source
      file extension
- [ ] Lists typed by purpose; images have alt text and live under `assets/img/`
- [ ] Heading case is consistent within the page (Title Case for short labels, sentence case for full phrases); no end punctuation; one page type (installation/tutorial/design/reference; quickstart is a tutorial subtype)
- [ ] Quickstarts are copy-paste-ready and minimal: no architecture diagrams, tuning, or optional branches
- [ ] Tutorials use Steps for meaningful sequences, Tabs only for genuinely parallel variants, and link rather than duplicate Reference or design detail
- [ ] Reference pages cover the complete supported configuration surface in scope
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

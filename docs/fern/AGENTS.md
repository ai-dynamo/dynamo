<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions — docs, examples, recipes

When creating or editing files under `docs/`, `examples/`, or `recipes/`, follow the
[documentation style guide](pages/community/contributing/documentation/documentation-style-guide.md).
For **where** a new page belongs in the tab structure, read
[`pages/AGENTS.md`](pages/AGENTS.md) first — placement is decided before anything below matters.

## Authoring non-negotiables

- SPDX header on every file: frontmatter `#` form for Fern docs, `<!-- -->` for plain READMEs,
  full Apache block for code/config; copyright range `2025-2026`.
- Fern docs: `---` frontmatter with SPDX + at least one metadata key (`title`/`subtitle`/
  `sidebar-title`). Fern renders the page H1 from the nav `page:`, so do **not** add a body `# H1`
  (it duplicates the title); start the body at `##`.
- Every new page needs a `- page:` entry in `index.yml`. A page not in the nav is unreachable.
- Admonitions follow the source extension: use Fern callout components (`<Note>`, `<Tip>`,
  `<Info>`, `<Warning>`, `<Error>`) in `.mdx`, and GitHub-style blockquotes (`> [!NOTE]`) in `.md`.
- Links: relative + extension within `docs/`; absolute `github.com/ai-dynamo/dynamo` URLs for
  targets outside `docs/` (no `../` escapes).
- Code fences language-tagged (`bash`, not `sh`); backend casing vLLM / SGLang / TensorRT-LLM.
- No internal/sensitive refs (NVBug/JIRA IDs, internal hosts, secrets, TODO/FIXME) in shipped docs.
- Write for humans: no marketing/bombast, no filler, be concrete.

`docs/fern/scripts/docs_lint.py` enforces the deterministic subset as the `Docs Lint` job on every pull
request, and `fern check` plus `fern docs broken-links` run alongside it. Reproduce all three
locally before pushing (see [Validate](#validate)).

The linter separates the rules that fail the job from the ones it only reports:

| | Rules |
|---|---|
| **Blocking** (job fails) | Missing or misplaced SPDX header; frontmatter with no YAML key; a relative link that breaks or escapes `docs/`; a dangling `path:` in `index.yml`; a tracker ID or NVBug reference |
| **Advisory** (annotated, does not fail) | Body `# H1`; `TODO`/`FIXME`; internal-looking host; hardcoded `docs.nvidia.com` self-link; a page file missing from the nav |

Two rules are advisory for a reason: the generated Kubernetes API reference carries a body `# H1`
that comes from the `crd-ref-docs` template rather than the page, and a `TODO` sometimes marks a
page whose fate is an open decision. Advisory does not mean optional — fix them in the pages you
touch. The linter scans `docs/` by default, matching the CI job. Pass `--scan docs,examples,recipes`
to cover example and recipe READMEs too, but expect pre-existing findings there: no job gates those
trees, so they carry violations this PR does not clear.

## This directory

`docs/fern/` holds both the content tree and the site configuration. Content lives in `pages/`;
everything else here is machinery:

| Path | Role |
|---|---|
| `pages/` | Every docs page — see [`pages/AGENTS.md`](pages/AGENTS.md) |
| `index.yml` | Navigation: tab map + per-tab layout. The only place a page becomes reachable |
| `docs.yml` | Site config, locales, landing page, versions, and `redirects:` |
| `main.css` | Site styles, including the pure-CSS recipe target-picker vocabulary |
| `components/` | React `.tsx` components used by `.mdx` pages |
| `scripts/` | Build and sync tooling (callout conversion, translation links, snapshot rewrites) |
| `translations/` | Locale mirrors of `pages/` (`zh-CN/pages/<same relative path>`) |
| `assets/` | Images, diagrams, fonts |

Two gates on the machinery:

- Editing `main.css` requires running `python3 docs/fern/scripts/sync_site_css.py` so the footer's
  CSS mirror stays in sync. Pre-commit enforces this.
- The `docs-website` branch is CI-managed and must **never** be edited by hand. All authoring
  happens on `main` or a feature branch based on it.

## Generated files: commit, publish-time, or post-merge

Commit the artifact when any input is external or time-varying. Generate at publish when the artifact is a pure function of sources committed in the same commit and the toolchain to compute it is already present in the publish runner. Where the toolchain is not in the publish runner, or committed history matters, regenerate post-merge: a workflow on push to main runs the generator and bot-commits changed outputs, so PRs carry source edits only.

`releases.json`, `releases-atom.xml`, and the Python/Rust API reference pages are generated at
publish time. The marker-spliced release pages and the Kubernetes `full-api-reference.mdx` stay
committed for review (the Kubernetes page is the one committed, freshness-gated API reference).

## API references

The Python and Rust pages under `pages/reference/api/python/` and `pages/reference/api/rust/` are
publish-time artifacts: gitignored, never committed, generated by `fern-docs.yml` against the exact
ref being published (and regenerated in pre-merge to prove a source PR cannot break generation).
The Kubernetes `pages/reference/kubernetes-api/full-api-reference.mdx` page regenerates from a
committed docs-side Markdown file, so it stays committed and freshness-gated. Every generated page
carries a `GENERATED by … do not edit` marker. To change one, edit the source and rerun its
generator, never the page.

The curated DGD, DGDR, and DCD pages under `pages/reference/kubernetes-api/` are manually maintained
user-facing references, not generator outputs. Keep them aligned with the Go API types and generated
full reference by following
[`templates/guidelines/kubernetes-api-reference.md`](templates/guidelines/kubernetes-api-reference.md).

Python and Rust are one hop: `gen_python_api.py` and `gen_rust_api.py` read docstrings through
griffe.

```bash
python3 docs/fern/scripts/gen_python_api.py        # writes into gitignored paths; nothing to commit
```

The full Kubernetes reference has two stages, and the first lives outside `docs/`.
`make generate-api-docs` in `deploy/operator/` runs `crd-ref-docs` over the CRD Go types to produce
`pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md`.
`gen_kubernetes_api.py` parses that committed Markdown and renders only `full-api-reference.mdx`. It
never reads the Go types, so changing a type and rerunning it alone is a no-op — refresh the
intermediate first.

```bash
make -C deploy/operator generate-api-docs         # Go types -> api-reference-k8s.md
python3 docs/fern/scripts/gen_kubernetes_api.py   # that Markdown -> full-api-reference.mdx
```

The `Generated API References` pre-merge job runs the Python/Rust generators in write mode (a
source PR that breaks generation fails fast; there is no committed output to diff) and applies
`--check` only to the Kubernetes reference. On a pull request the Kubernetes check runs with
`--since <base>`, which scopes it to what the branch itself changed; on `main` the flag is omitted
and the strict gate applies. Because these references track source outside `docs/`, a branch's
output goes stale whenever `main` lands a commit touching a symbol it documents — so regenerate
immediately before merge rather than earlier in review. Reviewers must separately verify
that any affected curated Kubernetes page remains aligned with the generated schema.

## Validate

```bash
python3 docs/fern/scripts/docs_lint.py                          # the `Docs Lint` pull request job
python3 docs/fern/scripts/docs_lint.py --scan docs,examples,recipes  # ungated trees; expect existing findings
fern check                                            # nav + frontmatter structure
fern docs broken-links                                # link resolution
python3 docs/fern/pages/recipes/_catalog/validate.py  # recipe or benchmark changes only
```

For how the site builds and publishes, see
[Building and Publishing](pages/community/contributing/documentation/building-and-publishing.md).

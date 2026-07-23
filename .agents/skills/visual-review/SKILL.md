---
name: visual-review
description: Create self-contained interactive HTML code-review dashboards from GitHub or GitLab pull requests, checked-out branch diffs, or supplied unified diffs, with correctness and safe-to-merge scores, smell indicators, interactive component and finding diagrams, structured finding packages, per-finding diff tabs with important references, expandable full context, API manifests, annotated red/green diffs, minimaps, validation cases, and fix prompts. Use when Codex is asked for a visual PR review or any of those visual review artifacts.
license: Apache-2.0
metadata:
  author: Stefan Schimanski
  tags:
    - dynamo
    - code-review
    - visualization
    - pull-request
---

# Visual Review

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Stefan Schimanski
SPDX-License-Identifier: Apache-2.0
-->

Produce a deep code review and render it as one portable HTML file with no external dependencies.

## Workflow

1. Read repository instructions from the active worktree before inspecting code.
2. Acquire the exact unified diff, immutable revision metadata, and full context for every changed file.
   - Remote PR: use the repository CLI/API without checking out unless requested; fetch base/head file contents when the normal patch omits expandable context.
   - Current branch: diff against the actual merge base with full-file context, for example `git diff --unified=100000 <base> <head> -- <changed-files>`.
   - Supplied patch: preserve it verbatim and request or derive base/head file contents when the patch alone cannot support context expansion.
3. Read changed code plus enough callers, types, tests, and invariants to support every finding. Do not run tests when the user requests a read-only review.
4. Write a review spec following [references/review-schema.md](references/review-schema.md). Keep prose concise and evidence-based. Always include an information-flow component diagram. Add finding-specific sequence or graph diagrams when they materially clarify causality. Mark findings that involve API objects and attach concrete manifest examples.
5. Render the dashboard:

```bash
python3 <skill-dir>/scripts/render_review.py \
  --spec /absolute/path/review.json \
  --diff /absolute/path/review.diff \
  --output <workspace-artifact-dir>/<review-name>.html
```

   Resolve `<workspace-artifact-dir>` inside the active Codex workspace, not inside a temporary review worktree. Prefer a writable, VCS-ignored `.codex/reviews/`; otherwise use a writable, VCS-ignored workspace directory such as `tmp/codex-reviews/`. Confirm the chosen directory with `git check-ignore` when the workspace uses Git. The renderer creates the output directory when needed.
6. Re-run with `--validate-only` after edits. If a local headless browser exists, inspect the rendered page; otherwise rely on the renderer's structural, anchor, and annotation validation.
7. Return a clickable Markdown file link to the workspace-local HTML artifact, followed by a fenced `markdown` code block containing a copy-ready text version of every finding. Link the bare absolute `.html` path with no `#fragment`; Codex workspace file links with URL fragments may fail to open. Internal anchors remain available after the HTML file is open. Never hand off a final review from `/tmp`, `$TMPDIR`, `~/Library/Caches`, or any path outside the active workspace; those locations are temporary working storage only.
   - Keep the findings in severity order and include severity, title, exact `file:line`, summary, labeled evidence lists, and suggested fix. Preserve the dashboard's structure instead of flattening each finding into one prose paragraph.
   - Emit only findings in this block, not scope boundaries, validation cases, logical flows, or implementation notes.
   - Emit the fenced block even when there are no findings; use `No findings.` as its content.
8. Treat user feedback about a generated dashboard as feedback on the reusable skill. Fix the current artifact and update the relevant skill instruction, renderer, or template in the same turn so future visual reviews inherit the improvement. Revalidate both the skill and the regenerated artifact.

## Review contract

- Include the complete diff with old/new line numbers and GitHub-style red/green rows.
- Put a tab bar above the code diff with `All changes` plus one tab per finding. The all-changes tab shows the complete PR; each finding tab shows only `code_blocks` attached to that finding and a compact copy of its important reference snippets. Keep filenames, annotations, heat, smell, and exact line links functional in every view.
- Attach every `code_blocks` entry to exactly one finding through `finding_ids`; use those associations as the source of truth for finding-specific diff tabs.
- Embed full changed-file context but collapse it to the normal review radius initially. Insert GitHub-style `[+]` expansion rows for each hidden context run so reviewers can reveal more context without leaving the dashboard. Finding tabs must not reveal unrelated changed rows merely because context is expanded.
- Anchor each actionable finding to an exact changed line. Do not label the requested implementation itself as an additional finding; distinguish known scope from newly found defects.
- Render a findings overview immediately after the summary. Make each overview entry the canonical finding package with severity, title, concise summary, labeled bullet groups, and a direct link to the annotated code.
- Never write a finding as one long prose paragraph. Require a one-sentence `summary`, two or more labeled `details` groups with short bullet items, a structured `suggested_fix`, and a per-finding `agent_prompt`.
- Put every finding-specific artifact inside its canonical finding package and keep it collapsed by default: Suggested fix, diagrams, API manifests, validation cases, logical flows, reference snippets, and agent fix prompt. Attach each supporting artifact to exactly one finding through a single-entry `finding_ids` array. Do not render mixed PR-wide sections for these artifacts. Only inherently PR-wide material such as the primary information-flow component map, scope, scores, file map, and complete diff stays global.
- Always render an information-flow component diagram covering every runtime component involved in the PR, including unchanged external components needed to understand the flow. Distinguish changed, existing, external, and test-only nodes. Show the direction and payload or action on each connection.
- Render component diagrams with the vendored Cytoscape.js and Dagre libraries. Keep the output self-contained with no CDN or runtime network dependency. Component maps must use automatic directed layout, display labels on edges, allow node dragging, zooming, and panning, provide visible Relayout and Fit controls, and expose keyboard-accessible links for every linked node and edge.
- Assess every finding for a visual explanation. Use a sequence diagram for timing, ordering, retries, races, or lifecycle behavior; use a component or graph illustration for cross-component dependencies and data paths. For ordering defects, show both the failing sequence and the corrected sequence. Attach focused diagrams to their finding through `finding_ids` and render them in its collapsed diagram section. Do not add decorative diagrams for localized findings that are clearer in one code annotation.
- For each finding involving Kubernetes or other API objects, include a linked manifest or focused manifest snippet with the exact fields that trigger or demonstrate the issue. Label server-observed snapshots so they are not mistaken for apply-ready manifests.
- Always duplicate the final findings in the chat response as a fenced `markdown` code block so the reviewer can paste them into another review surface without opening the dashboard.
- Keep dashboard concepts explicit and separate: findings are defects or risks; scope states review boundaries; validation cases describe how to prove behavior; logical flows explain multi-location causality. Never present these as peer lists without labels.
- Link multi-step behavior through flow cards and reference snippets when the bug crosses three or more locations.
- Keep logical flows in the main content only. Do not duplicate them as sidebar navigation.
- Make every internal anchor destination unmistakable: flash the actual destination on navigation and leave it visibly highlighted. Diff-line anchors must highlight the whole diff row, not only the line number, and clicking the current anchor again must retrigger the flash. Preserve a strong static highlight when reduced motion is requested.
- Make reference anchors focus the evidence rather than the outer reference container: center the inline reference annotation in the viewport and apply the anchor palette to both the decisive code lines and the annotation card. Do not rely on an outline around the full reference block.
- When `source_url` is a GitHub pull request, normalize direct review links to the current `/files` view and derive exact diff anchors for changed files and old/new lines. Inspect the PR page for files GitHub leaves behind a `Load diff` placeholder and list them in `github_collapsed_files`: their line anchors do not exist until the user expands the diff, so link to the always-present file anchor without an `L`/`R` suffix and say `Load diff`, then identify the old/new target line in the visible label and tooltip. Use exact `L`/`R` anchors only for files GitHub renders initially. Place a GitHub-style blue `+` button on the boundary between the number gutters and code, sized like GitHub's control. Reveal it only when the code cell is hovered or the button receives keyboard focus, not when the line numbers or another part of the row is hovered. Open GitHub in a new tab without replacing the local review; use both `target="_blank"` with `rel="noopener noreferrer"` and an explicit user-click `window.open(..., "_blank", "noopener,noreferrer")` handler because embedded app browsers may ignore the target attribute. Do not claim that this opens GitHub's comment editor: the user must use GitHub's own controls; state the necessary action in the button tooltip. Keep the line number itself as the dashboard's internal anchor, and also expose direct PR links from finding summaries, inline annotations, validation cases, and file headers.
- When `source_url` is a GitLab merge request, normalize direct review links to its `/diffs` view. Pin file links with GitLab's SHA-1 filename anchor and derive exact line links with GitLab's `<sha1(path)>_<old-position>_<new-position>` line code. Expose these MR links in the same file headers, finding summaries, inline annotations, validation cases, and hover `+` buttons as GitHub links, and identify GitLab accurately in every label and tooltip.
- Use the minimap for changed files, important callers/types/tests, and explicit out-of-scope areas.
- Define larger logical diff ranges once in `code_blocks`. Use `high`/red for code that must be inspected, `medium`/yellow for code worth a look, and `low`/green for routine changes. Render the heat as a 5 px bar at the far left of the number gutters; do not recolor the number or code backgrounds, because those already encode diff semantics.
- Classify each `code_blocks` entry independently with `smell: high`, `medium`, or `none`. Smell is exclusively about code quality: structural debt, mixed responsibilities, duplication, avoidable complexity, poor encapsulation, readability, or maintainability. Never derive smell from correctness, finding severity, risk, or missing behavior; those belong to findings and review heat. Use high for concentrated structural debt, medium for a refactoring concern, and none for clean or routine code. Every high or medium smell must include a concise `smell_reason` naming the quality problem and its maintenance cost. Render that reason in the code-edge and aggregate hover explanations. Render high as a red ☁ and medium as an orange ☁ overlaid at the far-right edge of the block's first code row; render no row icon for none. Beside the PR title, aggregate the maximum smell as red, orange, or green.
- Use exactly one custom tooltip for smell and score badges; never also set a native HTML `title`. Structure metric tooltips with a heading, a short summary or scale explanation, and a bulleted list for multiple smell reasons or score factors. Do not concatenate long explanations into one paragraph with inline bullet characters.
- Always score correctness improvement and safe-to-merge confidence from 1 through 10 and render both badges immediately after the aggregate smell icon. Label the second badge `Safe to merge`, never merely `Risk`: 10 is green and means low residual risk / safe to merge, while 1 is red and means unsafe to merge. Ten is also good for correctness and means a clear net correctness improvement. Color both continuously through red at 0, orange at 3, yellow at 6, and green at 10. Include a concise summary and an auditable base-plus-factors breakdown in each hover.
- Score correctness as a delta against the base branch, not as an absolute bug count. Use base 10 for a complete P1/P2-class fix, base 7 when the target fix is materially useful but incomplete, and base 5 for no material correctness change. Deduct only regressions introduced by the PR: P1 −9, P2 −5, and P3 −2. Do not call a pre-existing unresolved ordering a newly introduced regression or mix smell into correctness.
- Score safe-to-merge confidence from base 10, deducting for unresolved severity, likelihood, blast radius, and recovery difficulty while avoiding double-counting. Apply the aggregate smell once as change-risk debt: high/red −3 or medium/orange −1. Add small positive factors only for concrete mitigations such as focused tests. Clamp both scores to 1–10. Keep the internal spec key `risk_score` for compatibility, but never expose that ambiguous name as the dashboard label.
- Render a right-side 10,000-foot rail beside each file diff from the same `code_blocks` data. Align one concise sentence with the start of each larger block; explain what the block does and avoid line-by-line narration or repeating findings.
- Keep file diffs in the page's single vertical scroll flow. Allow horizontal overflow for long code, but explicitly hide vertical overflow on that horizontal scroller so browser overflow-axis normalization cannot create a tiny nested vertical scroll area that captures wheel input.
- State what is in scope and out of scope when ownership spans multiple PRs.
- Make each finding's agent fix prompt short and independently actionable. Include only that finding's non-obvious behavior, constraints, and tests; do not combine unrelated findings or repeat repository instructions that the agent will load from `AGENTS.md` or equivalent files.
- Report tests as unrun unless they were actually executed.
- Keep the final HTML review inside a writable, preferably VCS-ignored directory in the active workspace so the Codex app can open the file link reliably.
- Never post review comments, mutate the PR, or change source code unless the user separately requests it.

## Spec guidance

- Put findings in severity order (`P0` through `P3`, then notes).
- Use the structured finding schema from [references/review-schema.md](references/review-schema.md). Keep each bullet focused on one trigger, causal step, impact, or proof point.
- Set `involves_api_objects: true` on findings whose trigger or consequence depends on API object fields. Every such finding must be covered by at least one `manifests` entry.
- Use stable IDs for findings, flows, references, and scope sections so links remain meaningful.
- Prefer plain text with backticks for identifiers; the renderer escapes all content.
- Add exact validation cases to the matrix when coverage is part of the review. Every row must identify an exact changed code line with `file`, `side`, and `line`; link the row to that diff anchor.
- Keep reference snippets short and focused on the causal path, but include a few real source lines before and after the decisive code. Mark the decisive one-based snippet lines in `highlight_lines`; render them in light blue so evidence is distinguishable from context at a glance. Render the reference `summary` directly after the last highlighted line as a neutral gray annotation card with the same structure as P1/P2 inline annotations, but never present it as an additional finding.

### Diagrams

- Include exactly one primary `component` diagram that covers the full information flow. Additional focused component diagrams are allowed.
- Let Dagre place component nodes and edge labels. Do not hand-place the graph or depend on `x`, `y`, `label_x`, or `label_y` coordinates; preserve concise labels so the automatic planar-style layout stays readable.
- Keep component-map typography legible at the initial Fit view: use at least 15 px node labels and 13 px edge labels, and increase dense, larger graphs enough that Fit does not shrink their rendered text below normal UI caption size.
- Component maps are interactive: reviewers can drag nodes, zoom or pan the canvas, click linked nodes or edges, and restore the automatic layout or viewport with Relayout and Fit.
- Use `sequence` diagrams for finding-specific event ordering. Keep participant names stable across failing and corrected views so they compare directly.
- Link diagram nodes, edges, and events to findings, diff annotations, flows, or references when evidence exists.
- Keep labels short; put nuance in the diagram description or outcome.

### API object examples

- Add one or more `manifests` entries for every finding marked `involves_api_objects`.
- Show only fields relevant to the finding. Use multi-document YAML when before/after or old/new incarnations must be compared.
- Prefix server-populated fields such as `uid` and `resourceVersion` with a comment that the document is an observed snapshot, not apply-ready input.

The renderer and HTML template live in `scripts/render_review.py` and `assets/review-template.html`.

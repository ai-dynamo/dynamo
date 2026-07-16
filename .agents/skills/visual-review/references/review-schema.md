# Review specification

The renderer accepts one UTF-8 JSON object. `title`, `findings`, `correctness_score`, `risk_score`, and a primary information-flow `component` diagram are required; all other fields are optional.

## Top-level fields

```json
{
  "version": 1,
  "title": "PR #123 · fix(component): summary",
  "subtitle": "main ← feature · commit abc123",
  "source_url": "https://github.com/org/repo/pull/123/files",
  "revision": "abc123",
  "github_collapsed_files": ["path/to/large-diff.go"],
  "status": "2 findings",
  "summary": "One-sentence review outcome.",
  "correctness_score": {},
  "risk_score": {},
  "scope": {
    "in": ["Behavior owned by this change"],
    "out": ["Behavior owned elsewhere"]
  },
  "findings": [],
  "diagrams": [],
  "manifests": [],
  "flows": [],
  "file_map": [],
  "code_blocks": [],
  "test_matrix": [],
  "references": [],
  "validation_note": "Static review only; tests not run."
}
```

For a GitHub pull request, set `source_url` to the PR, `/files`, or `/changes`
page. The renderer normalizes direct review links to GitHub's `/files` view.
The renderer derives each changed file's GitHub `diff-<sha256(path)>` anchor and
the corresponding `L<line>` or `R<line>` target. The dashboard uses these for
hover `+` buttons and direct links from findings, cases, annotations, and file
headers. Other source providers retain the generic source link.

List a changed path in `github_collapsed_files` when GitHub initially renders
that file behind `Load diff` (including large or generated diffs). GitHub has
not created its `L`/`R` line elements at that point, so the dashboard links to
the existing file anchor and tells the reviewer to load the diff and then use
the named old/new line. Do not emit a knowingly dead exact-line anchor.

## Findings

Every finding must point to a line present in the unified diff.
Findings are the review's defects and risks. Each finding is the canonical
container for its explanation and all supporting artifacts.

```json
{
  "id": "finding-stale-restore",
  "severity": "P1",
  "title": "Preserved data overrides a live deletion",
  "summary": "A live deletion is overwritten by a stale preserved value.",
  "details": [
    {"label": "Trigger", "items": ["The object was deleted after the initial read."]},
    {"label": "Failure path", "items": ["The stale value wins during reconstruction."]},
    {"label": "Impact", "items": ["The deleted API field reappears."]}
  ],
  "suggested_fix": {
    "summary": "Make the live source authoritative.",
    "steps": ["Check the live object before restoring preserved data."],
    "tests": ["Cover deletion between read and write."]
  },
  "agent_prompt": "Fix only the stale-restore ordering and add its focused regression test.",
  "file": "path/to/file.go",
  "side": "new",
  "line": 551,
  "involves_api_objects": true,
  "links": [
    {"label": "logical flow", "target": "flow-live-source"},
    {"label": "restore helper", "target": "ref-restore-helper"}
  ]
}
```

`severity` accepts `P0`, `P1`, `P2`, `P3`, or `note`. Use `side: "old"` only for deleted lines.
Set `involves_api_objects: true` when API object fields participate in the trigger or consequence. Every such finding must be covered by a linked manifest example.
Do not use a single prose `body`. Keep `summary` to one sentence; use labeled
`details[].items` for trigger, failure path, impact, and evidence. The dashboard
renders Suggested fix and all related artifacts as collapsed sections inside
the finding. `agent_prompt` must be independently usable for that finding.

## Correctness and safe-to-merge scores

Both scores use 1 through 10 with higher always better. `correctness_score`
measures the PR's net correctness improvement against the base branch.
`risk_score` is the compatibility key for the badge labeled `Safe to merge`.
It measures merge confidence after residual risk: 10 means low residual risk
and safe to merge, while 1 means unsafe to merge. The title renders both
immediately after the smell icon on a continuous red-0, orange-3, yellow-6,
green-10 gradient. Never label the second badge merely `Risk`, because that
would make the good direction of the scale ambiguous.

```json
{
  "correctness_score": {
    "base": 10,
    "value": 5,
    "summary": "Fixes the target P1 but introduces one P2 compatibility regression.",
    "factors": [
      {"label": "Introduced P2 regression", "delta": -5}
    ]
  },
  "risk_score": {
    "base": 10,
    "value": 6,
    "summary": "The main path is covered, but a cross-component race remains.",
    "factors": [
      {"label": "Residual asynchronous ordering risk", "delta": -3},
      {"label": "Focused regression coverage", "delta": 1},
      {"label": "Aggregate medium smell", "delta": -1}
    ]
  }
}
```

The renderer verifies `value == clamp(base + sum(delta), 1, 10)`. Use base 10
for a complete P1/P2-class fix, base 7 for a materially useful but incomplete
target fix, and base 5 when the PR has no material correctness effect. Deduct
only newly introduced regressions from correctness: P1 −9, P2 −5, and P3 −2.
Represent a pre-existing unresolved case through the lower base instead of
mislabeling it as introduced. Keep smell out of correctness.

Start safe-to-merge confidence at 10. Deduct for unresolved severity, likelihood,
blast radius, and recovery difficulty without double-counting the same fact.
Apply only the maximum aggregate smell once: high/red −3 or medium/orange −1.
Positive factors require concrete mitigation such as focused regression tests.
Keep every factor label evidence-based because the full formula appears on
hover.

## Diagrams

Every spec must contain one primary `component` diagram covering all runtime components involved in the PR. Add `sequence` diagrams for races, retries, lifecycle changes, or other ordered interactions. For an ordering defect, include separate failing and corrected sequences with the same participants.

Component diagrams are automatically laid out as directed graphs with the vendored Cytoscape.js 3.33.4, Dagre 0.8.5, and cytoscape-dagre 2.5.0 libraries. The rendered graph has labeled arrows, draggable nodes, zoom and pan gestures, clickable evidence links with keyboard-accessible text equivalents, and Relayout and Fit controls. The libraries are inlined into the output, so the review remains self-contained and offline-capable.

Component diagram:

```json
{
  "id": "diagram-information-flow",
  "type": "component",
  "primary": true,
  "finding_ids": [],
  "title": "Information flow",
  "description": "Changed components are blue; external dependencies are dashed.",
  "nodes": [
    {"id": "api", "label": "API server", "detail": "Persists objects", "kind": "external"},
    {"id": "controller", "label": "Controller", "detail": "Reconciles desired state", "kind": "changed", "target": "finding-stale-restore"}
  ],
  "edges": [
    {"from": "controller", "to": "api", "label": "Update object", "tone": "info", "target": "finding-stale-restore"}
  ]
}
```

`kind` accepts `changed`, `existing`, `external`, or `test`. Edge and event `tone` accepts `danger`, `warning`, `success`, `info`, or `neutral`. Component placement is automatic; legacy `x`, `y`, `label_x`, and `label_y` fields are tolerated but are not required or used for layout.

Sequence diagram:

```json
{
  "id": "diagram-race-current",
  "type": "sequence",
  "primary": false,
  "finding_ids": ["finding-stale-restore"],
  "title": "Current failing order",
  "description": "The watch delete arrives before the local mutation is recorded.",
  "participants": ["Controller", "API server", "Informer", "Cache"],
  "events": [
    {"from": "Controller", "to": "API server", "label": "Update", "tone": "info"},
    {"from": "API server", "to": "Informer", "label": "Delete event", "tone": "danger", "target": "finding-stale-restore"}
  ],
  "outcome": "The stale response is cached.",
  "outcome_tone": "danger"
}
```

Node, edge, and event `target` values are optional links to registered review IDs.
The one PR-wide component map has `primary: true` and no `finding_ids`. Every
other diagram must name one or more findings and is rendered inside them.

## API object manifests

Use focused YAML or JSON examples for each finding marked `involves_api_objects`.
Attach each example to exactly one finding so it appears inside that finding's
collapsed API-object section.

```json
{
  "id": "manifest-replacement",
  "title": "Observed replacement snapshots",
  "description": "Server-populated metadata is shown for comparison; this is not apply-ready YAML.",
  "language": "yaml",
  "finding_ids": ["finding-stale-restore"],
  "code": "# observed snapshot\\napiVersion: example.io/v1\\nkind: Widget\\nmetadata:\\n  name: same-key\\n  uid: old-uid\\n  resourceVersion: \\\"2\\\""
}
```

## Logical flows

Use flows for causal paths with multiple locations.
Flows explain findings; they are not additional findings. Give each flow a
single-entry `finding_ids` array so it is rendered inside that finding rather
than in a mixed PR-wide section.

```json
{
  "id": "flow-live-source",
  "finding_ids": ["finding-stale-restore"],
  "title": "Deleted live mount returns",
  "description": "The preserved payload is consulted before origin reconstruction.",
  "steps": [
    {"label": "Step 1", "title": "Restore runs", "code": "restorePreserved(...) ", "target": "finding-stale-restore"},
    {"label": "Step 2", "title": "Gate passes", "code": "matches(firstCache)", "target": "ref-restore-helper"}
  ]
}
```

## File minimap

`total_lines` and marker `line` place clickable marks proportionally. Files need not be present in the diff.

```json
{
  "path": "path/to/file.go",
  "label": "core conversion",
  "total_lines": 2500,
  "markers": [
    {"line": 551, "label": "restore ordering", "target": "finding-stale-restore", "kind": "risk"},
    {"line": 2200, "label": "origin tests", "target": "ref-tests", "kind": "test"}
  ]
}
```

Marker `kind` accepts `relevant`, `risk`, `test`, or `out`.

## Code blocks, heatmap, smell, and 10,000-foot rail

Use one `code_blocks` entry per larger logical diff block. The same entry drives
a 5 px heat bar at the far left of its line-number gutters and one sentence in
the right-side 10,000-foot rail. It also places a smell cloud on the right code
edge when `smell` is `medium` or `high`.

```json
{
  "id": "block-restore-ordering",
  "finding_ids": ["finding-stale-restore"],
  "file": "path/to/file.go",
  "side": "new",
  "start_line": 530,
  "end_line": 570,
  "heat": "high",
  "smell": "high",
  "smell_reason": "One function combines parsing, policy, state mutation, and output formatting, so each policy change must touch the whole control path.",
  "summary": "Merges preserved and live mounts, with the live-source precedence decided here."
}
```

`heat` accepts `high` (red: must inspect), `medium` (yellow: worth a look), or
`low` (green: routine). `smell` is independent and accepts `high`, `medium`, or
`none`. It measures code quality only: structural debt, mixed responsibilities,
duplication, avoidable complexity, poor encapsulation, readability, or
maintainability. Do not raise smell because code is incorrect, risky, or missing
a behavior; represent that with findings and `heat`. High and medium smells
require a concise `smell_reason` that identifies the quality issue and its
maintenance cost. The renderer uses it for the code-edge and aggregate hover
explanations. High smells render a red ☁, medium smells render an orange ☁, and
`none` renders no code-edge icon. The title aggregates the highest block smell:
red for any `high`, orange for any `medium` without `high`, and green when all
blocks are `none`. `start_line` and `end_line`
must both be rendered lines on the selected diff side. Keep `summary` to one
sentence and describe what the block does, not a line-by-line restatement.
Every code block must have a single-entry `finding_ids` array. The dashboard
uses this relation to build one focused code-diff tab per finding. Supply the
renderer with full changed-file context; it initially collapses distant context
and exposes `[+]` controls that reveal each hidden context run. Finding tabs
also render the related reference snippets above their filtered diff.

## Test matrix

Test-matrix rows are validation cases, not findings. Every row must link to the
exact changed line whose behavior the case validates.

```json
{
  "id": "case-same-key",
  "finding_ids": ["finding-stale-restore"],
  "case": "same key from both sources",
  "expected": "one merged entry",
  "status": "missing",
  "tone": "required",
  "file": "path/to/file.go",
  "side": "new",
  "line": 551
}
```

`tone` accepts `required`, `existing`, or `neutral`. `side` accepts `old` or
`new`; the selected `file`/`side`/`line` must exist in the unified diff.
`id` is optional; provide it when diagrams, flows, or minimap markers link to the case.
Every validation case must have a single-entry `finding_ids` array and is rendered only
inside those findings, never in a mixed global table.

## Reference snippets

```json
{
  "id": "ref-restore-helper",
  "finding_ids": ["finding-stale-restore"],
  "path": "path/to/file.go",
  "lines": "920–930",
  "summary": "The gate checks only the first preserved cache.",
  "code": "func restorePreserved(...) {\n    before()\n    decisiveCheck()\n    after()\n}",
  "highlight_lines": [3],
  "back_target": "flow-live-source"
}
```

`highlight_lines` contains one-based line numbers within `code`, not source-file
line numbers. Include real source context on both sides of the highlighted lines;
normally two to four lines per side are enough. The renderer requires at least
one highlighted line and visible context before and after the highlighted range.
The dashboard inserts `summary` immediately after the last highlighted line in a
neutral gray inline annotation card. This is supporting evidence, not a finding.
Every reference snippet must have a single-entry `finding_ids` array; it appears in the
collapsed reference-evidence section of each related finding.

## Output behavior

The renderer derives additions, deletions, changed files, diff line anchors, and file navigation from the unified diff. It rejects:

- malformed hunk headers;
- duplicate IDs;
- a missing information-flow component diagram, malformed diagram topology, or links to unknown diagram targets;
- findings whose file/side/line is absent from the diff;
- API-object findings without a linked manifest example;
- test-matrix rows without an exact file/side/line diff anchor;
- reference snippets without valid highlighted lines and surrounding context;
- code blocks with invalid ranges or unsupported heat or smell levels;
- links to unknown spec IDs;
- unsupported severities, sides, marker kinds, or test tones.

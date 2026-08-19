// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-agent code review for a diff. Runs a high-level adversarial framing
// pass, fans out across a pluggable set of expert reviewers, adversarially
// verifies every finding, and synthesizes a ranked report.
//
// EXTENDING: add an expert by dropping a markdown file in
// `.agents/workflows/reviewers/<key>.md` (YAML frontmatter `key` + optional
// `applies_to` globs; the body is the review focus). The loader merges those
// with the built-in experts below by `key` (a same-key file overrides). No
// change to this orchestration is needed. See `reviewers/README.md`.
//
// Run with the workflow runner. Optional args:
//   { base?: string,                 // ref to diff against (default "origin/main")
//     reviewers?: [{key,focus}],     // explicit expert set (skips file loading/gating)
//     votes?: number,                // adversarial refuters per finding (default 2)
//     exhaustive?: boolean }         // keep nit-severity findings too (default false)
//
// The branch under review must be checked out; the workflow reviews HEAD vs
// `base`. Agents run git / read files themselves (the script has no shell access).

export const meta = {
  name: 'code-review',
  description: 'Multi-agent review of a diff: adversarial framing pass, fan out across a pluggable set of expert reviewers, adversarially verify every finding, and synthesize a ranked report.',
  whenToUse: 'Before opening or merging a PR, for a thorough self-verifying review of the changes against Dynamo\'s Rust/systems standards and comment hygiene.',
  phases: [
    { title: 'Scout', detail: 'list changed files; load + gate the relevant expert reviewers' },
    { title: 'Frame', detail: 'adversarial pass on the premise, approach, and validation' },
    { title: 'Review', detail: 'one agent per active expert over the diff' },
    { title: 'Verify', detail: 'adversarially refute each finding; drop the ones that fall' },
    { title: 'Synthesize', detail: 'dedupe, filter nits, and rank the survivors' },
  ],
}

// The runner may deliver `args` as a parsed object or as a JSON string; normalize both.
const A = (() => {
  if (typeof args === 'string') { try { return JSON.parse(args) } catch (e) { return {} } }
  return (args && typeof args === 'object') ? args : {}
})()
const BASE = A.base || 'origin/main'
const VOTES = A.votes || 2
const EXHAUSTIVE = !!A.exhaustive
// Scout resolves this to the concrete git diff spec that actually shows the
// changes (three-dot BASE...HEAD, else two-dot BASE for uncommitted work), so
// the framing/review/verify agents can't diff a different range than Scout did.
let DIFF = `${BASE}...HEAD`

// Shared discipline injected into every reviewer: material, changed-line
// concerns only — no nit padding or speculation.
const MATERIALITY =
  'Report only material concerns on lines this change touches. Do NOT report: cosmetic nits, ' +
  'style-only preferences, speculative concerns without evidence, tradeoffs that are clearly ' +
  'intended and acceptable, tiny overhead in obviously cold code, or generic "add tests" notes ' +
  'not tied to a concrete risk. Do not hunt pre-existing issues in surrounding code unless they ' +
  'are severe or this change amplifies them. If your expert lens has no meaningful surface here, ' +
  'return an empty findings array in one pass — do not invent work.'

// Built-in experts. `reviewers/*.md` files are merged on top of these by key.
// `distributed` and `concurrency` matter most for a datacenter-scale framework.
const BUILTIN_REVIEWERS = [
  { key: 'correctness', focus: 'logic bugs, unhandled errors, edge cases, off-by-one, wrong types, incorrect Result/Option handling, API/schema drift, unwrap()/expect() in non-test code' },
  { key: 'concurrency', focus: 'async/tokio: locks held across .await, blocking work on the executor, spawned-task shutdown/cancellation, tight loops without yield, channel backpressure, unbounded channels, shared-state races' },
  { key: 'distributed', focus: 'idempotency, crash/failover recovery gaps, state divergence across replicas, atomicity across layers, missing backpressure, stale versioning, invariants that break mid-rollout' },
  { key: 'simplicity', focus: 'over-engineering, needless abstraction, duplicated logic, one-hop wrappers, layering violations, premature generalization, dead code, unnecessary clone()/Arc/Mutex, reflexive Arc<Mutex<...>>, scope creep' },
  { key: 'perf', focus: 'hot-path heap allocation, boxing/allocation churn, avoidable clones/copies, intermediate collections, per-token or per-request work that should be hoisted, logging in hot paths' },
  { key: 'comment-hygiene', focus: 'comments in ANY file type (Rust, Dockerfile, YAML, shell) that restate the line, enumerate mechanical detail a reader can look up, or decoratively label structure; keep only a non-obvious why. Also AI-smell/obvious comments and historical/changelog comments.' },
  { key: 'tests-and-api', focus: 'behavior coverage vs line coverage, missing pytest markers, over-enumerated AI-style tests, breaking changes without migration; misleading names, unjustified interface/error-message churn' },
]

const FINDINGS = {
  type: 'object',
  additionalProperties: false,
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'line', 'severity', 'claim', 'why'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocking', 'major', 'minor', 'nit'] },
          claim: { type: 'string' },
          why: { type: 'string', description: 'concrete failure scenario or the rule violated' },
          suggestion: { type: 'string' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['refuted', 'reason'],
  properties: {
    refuted: { type: 'boolean', description: 'true if the finding is wrong, out of scope, immaterial, or not on a changed line' },
    reason: { type: 'string' },
  },
}

const EXPERT = { type: 'object', additionalProperties: false, required: ['key', 'focus'], properties: { key: { type: 'string' }, focus: { type: 'string' } } }

// Spawn VOTES independent skeptics, each told to refute; survive on majority-can't-refute.
const verifyFinding = (f, lens) =>
  parallel(Array.from({ length: VOTES }, (_, i) => () =>
    agent(
      `Adversarially verify this ${lens} finding. Default to refuted=true unless, by inspecting the diff (git --no-pager diff ${DIFF}) at ${f.file}:${f.line}, you can confirm it is a real, material defect ON A CHANGED LINE and in scope for this change.\n\nFINDING [${f.severity}]: ${f.claim}\nWHY: ${f.why}`,
      { label: `verify:${lens}:${f.file}:${f.line}#${i}`, phase: 'Verify', schema: VERDICT }
    )
  )).then(votes => {
    const live = votes.filter(Boolean)
    const refutes = live.filter(v => v.refuted).length
    return { ...f, lens, survived: live.length > 0 && refutes < Math.ceil(live.length / 2) }
  })

phase('Scout')
const scout = await agent(
  `List the source/config files changed vs ${BASE} (\`git --no-pager diff --name-only ${BASE}...HEAD\`, or \`... ${BASE}\` if that range is empty; exclude lockfiles, generated code, binary/data).\n\n` +
  `Built-in reviewer experts (JSON):\n${JSON.stringify(BUILTIN_REVIEWERS)}\n\n` +
  `Also load additional experts from \`.agents/workflows/reviewers/*.md\` if that directory exists — each file has YAML frontmatter with \`key\` (required) and optional \`applies_to\` (comma/space-separated globs); the markdown body is that expert's review focus. Merge them with the built-ins by \`key\` (a same-key file overrides the built-in focus).\n\n` +
  `From the merged set, select only the experts RELEVANT to this diff: honor each expert's \`applies_to\` globs when present, and skip experts with no meaningful surface for these files (e.g. skip concurrency/distributed/perf for a docs-only change). Return the changed paths, the selected experts as {key, focus} (focus verbatim from its source), and diff_spec = the exact git diff argument that shows these files: ${BASE}...HEAD if that range is non-empty, else ${BASE}.`,
  { label: 'scout:load+gate', phase: 'Scout', schema: { type: 'object', additionalProperties: false, required: ['files', 'experts', 'diff_spec'], properties: { files: { type: 'array', items: { type: 'string' } }, experts: { type: 'array', items: EXPERT }, diff_spec: { type: 'string' } } } }
)

const files = (scout && scout.files) || []
if (!files.length) {
  log(`No changed files vs ${BASE} — nothing to review.`)
  return { base: BASE, files: [], findings: [] }
}
DIFF = (scout && scout.diff_spec) || DIFF
// args.reviewers overrides file loading/gating; else use the loader's selection; else fall back to built-ins.
const override = Array.isArray(A.reviewers) ? A.reviewers.filter(e => e && e.key && e.focus) : []
let active = override.length ? override : ((scout && scout.experts) || [])
if (!active.length) active = BUILTIN_REVIEWERS
const fileList = files.join(', ')
log(`Reviewing ${files.length} changed file(s) vs ${BASE} with ${active.length} expert(s): ${active.map(e => e.key).join(', ')}.`)

// Framing pass and per-expert review run concurrently; each self-verifies.
const [framed, reviewed] = await parallel([
  () => agent(
    `Adversarially review the CHANGE vs ${BASE} at the design level (files: ${fileList}). Inspect it with \`git --no-pager diff ${DIFF}\`. Challenge: is the problem framed correctly and does the approach address the real outcome? What hidden assumptions, missing dependencies, or rollout constraints does it carry? Can local checks pass while the end-to-end system still fails? Is there a materially simpler or safer alternative? Does the validation give false confidence? ${MATERIALITY} Anchor each finding to a file:line in the diff.`,
    { label: 'frame:adversarial', phase: 'Frame', schema: FINDINGS }
  ).then(r => parallel(((r && r.findings) || []).map(f => () => verifyFinding(f, 'framing')))),

  () => pipeline(
    active,
    e => agent(
      `You are a strict Dynamo code reviewer. Apply the \`graham-code-review\` skill's philosophy and rules. Review ONLY the diff vs ${BASE} for these files: ${fileList} — inspect the actual changes with \`git --no-pager diff ${DIFF} -- <file>\`.\n\nEXPERT LENS: ${e.key} — ${e.focus}\n\n${MATERIALITY}`,
      { label: `review:${e.key}`, phase: 'Review', schema: FINDINGS }
    ),
    (review, e) => parallel(((review && review.findings) || []).map(f => () => verifyFinding(f, e.key)))
  ),
])

phase('Synthesize')
const all = [...(framed || []), ...((reviewed || []).flat())].filter(Boolean)
const seen = new Set()
const confirmed = []
for (const f of all) {
  if (!f.survived) continue
  // Drop nit-severity noise by default — but never for comment-hygiene, the lens
  // we most want surfaced, so it can't be silenced as a "nit".
  if (!EXHAUSTIVE && f.severity === 'nit' && f.lens !== 'comment-hygiene') continue
  const key = `${f.file}:${f.line}:${f.claim}`
  if (seen.has(key)) continue
  seen.add(key)
  confirmed.push(f)
}
const rank = { blocking: 0, major: 1, minor: 2, nit: 3 }
confirmed.sort((a, b) => (rank[a.severity] ?? 9) - (rank[b.severity] ?? 9))

log(`${confirmed.length} confirmed finding(s) after adversarial verification (${all.length - confirmed.length} refuted, filtered, or duplicate).`)
return { base: BASE, files, experts: active.map(e => e.key), findings: confirmed }

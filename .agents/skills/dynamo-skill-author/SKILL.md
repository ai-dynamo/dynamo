---
name: dynamo-skill-author
description: >-
  Author a new Dynamo agent skill, or refresh an existing one for a new
  Dynamo release, following the conventions established for the seven
  lifecycle skills in this directory. Covers naming, scaffolding, body
  authoring, and pre-PR validation. Trigger phrases include "add a new
  dynamo skill", "new dynamo skill", "author a skill", "scaffold a skill",
  "refresh skill for release", and "extend the dynamo skill set".
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - meta
  - authoring
  - skill-development
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "add a new dynamo skill",
  "new dynamo skill", "author a skill", "scaffold a skill", "refresh skill
  for release", "extend the dynamo skill set"
- Level 2 (this file): 4-phase workflow with decision branches
- Level 3: references/ for frontmatter and body shape, known authoring issues
           scripts/  for deterministic scaffold and validate helpers

Scope: authoring lifecycle for a skill under .agents/skills/dynamo-<name>/ —
       naming, scaffolding, body authoring, validation, PR submission, and
       per-release refresh of an existing skill.
Out of scope: authoring skills for non-Dynamo projects (the conventions
              here are Dynamo-specific extensions of the ai-infra-agent
              methodology), publishing to the public NVIDIA/skills catalog
              (that mirror is automated; nothing to author there directly).
-->

# Dynamo Skill Author

Author a new Dynamo agent skill, or refresh an existing one for a new Dynamo
release, following the conventions established for the seven lifecycle skills
in this directory. The skill is meta — it teaches an agent (or a human reader)
how to add another skill to this same directory with the same rigor floor.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Gather    → name, lifecycle stage, scope, collision check, sibling pick
Phase 2: Scaffold  → directory layout, frontmatter, body skeleton from a sibling
Phase 3: Author    → body sections, command-safety tiers, decision points,
                     refusal conditions, references/, scripts/
Phase 4: Validate  → shellcheck, YAML parse, cross-link audit, NV-ACES Tier 1,
                     pre-MR checklist, PR submission
```

---

## Command Safety

Authoring a skill is mostly file creation in the contributor's working tree.
Cluster-mutating operations do not arise. The safety classes below cover the
workflow's actual write surface; the Human-in-the-Loop contract still applies
to every decision point.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| Overwrite of an existing skill directory under `.agents/skills/` | Loses an in-flight skill. Confirm the directory is yours to overwrite. Prefer a fresh directory name. |
| `git push --force` against the PR branch | Drops review state. Use `--force-with-lease` only on personal branches and only if you understand the consequence. |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `bash scripts/scaffold-skill.sh ...` | Creates a new directory under `.agents/skills/dynamo-<name>/` and writes `SKILL.md`, `references/`, `scripts/`. Idempotent: refuses to overwrite. |
| `git commit -s -m "..."` | Adds your changes to the local history. Reversible until pushed. |
| `git push -u origin <branch>` | Publishes the branch. Reversible (delete branch + force-push), but reviewers may have already started. |

### SAFE — no confirmation needed

```
bash scripts/validate-skill.sh -d .agents/skills/dynamo-<name>
bash scripts/check-frontmatter.sh .agents/skills/dynamo-<name>/SKILL.md
shellcheck .agents/skills/dynamo-<name>/scripts/*.sh
ls .agents/skills/
cat .agents/skills/dynamo-<name>/SKILL.md
git status / git diff / git log
```

---

## Phase 1: Gather

**Goal.** Decide what to build before writing anything. A skill that is too
narrow forces the user to chain to a sibling mid-workflow; a skill that is
too broad bundles unrelated lifecycle stages.

**Inputs.**

| Input | Source |
|---|---|
| Proposed name | Contributor; convention is `dynamo-<verb>` for new lifecycle skills, product-name first for external NVIDIA brands with independent identity. |
| Lifecycle stage | Contributor; one of plan / optimize / serve / deploy / frontend / troubleshoot / benchmark, or a new stage. |
| Scope statement | Contributor; one sentence on what the skill owns and one on what it does not. |
| Sibling reference skill | Pick the existing skill closest in shape — `dynamo-deploy` for K8s-centric, `dynamo-serve` for local-run, `dynamo-troubleshoot` for day-2, `dynamo-benchmark` for measurement, `dynamo-plan` for decision-heavy. |

**Commands** (SAFE):

```bash
# Collision check.
ls .agents/skills/ | grep -i "<your-name>" || echo "no collision"

# Open PRs against this directory.
gh pr list --repo ai-dynamo/dynamo --search ".agents/skills/" --state open
```

**Decision points.**

- **Is this a new skill or a section in an existing one?** Cross-cutting
  integrations (Grove, KAI, GAIE, NIXL, model caching, observability) belong
  inside the relevant lifecycle skill — they are not standalone skills. A
  new skill is justified only when the workflow shape, decision points, and
  refusal conditions differ enough that they would crowd a sibling.
- **Which sibling do I copy?** Pick the one whose 4-phase shape applies
  most directly. `dynamo-troubleshoot` uses Triage → Inspect → Diagnose →
  Remediate; the rest use the install-oriented variant (Pre-Check → Configure
  / Author → Run / Validate → Verify / Apply). Match the shape, do not
  invent a new one.
- **What lifecycle stage tag goes in the frontmatter?** If the stage is new,
  flag the survey may need a refresh — see the directory README.

**Refusal conditions.**

The skill refuses to scaffold if:

- The proposed name matches an existing directory under `.agents/skills/`.
- The proposed name claims a Dynamo tool or surface without source-tree
  evidence. Example: a `dynamo-run` skill would refuse — no such CLI exists
  in the Dynamo source. Run `git grep -l "<tool name>"` against the Dynamo
  tree before naming.
- The scope statement is more than one sentence per direction (owned and
  not-owned). A multi-sentence scope is a signal the skill is too broad.

**Verification gate.** Advance only when: name passes collision check, scope
fits in one sentence per direction, sibling skill is named, lifecycle stage
is identified.

---

## Phase 2: Scaffold

**Goal.** Create the directory layout and a body skeleton derived from the
chosen sibling, so the contributor edits a working file rather than a blank
page.

**Inputs.** Name and sibling from Phase 1.

**Commands** (MUTATING):

```bash
# Scaffold from a sibling. Idempotent (refuses to overwrite).
bash scripts/scaffold-skill.sh \
    -n dynamo-<your-name> \
    -s dynamo-<sibling-name> \
    -v 1.2.0

# Confirm layout.
tree .agents/skills/dynamo-<your-name>/
```

The scaffold copies the sibling's `SKILL.md`, `references/`, and `scripts/`
into the new directory with placeholders inserted at the spots a contributor
must edit. Frontmatter `name`, `description`, and `tags` reset; body content
is preserved as a starting shape.

**Decision points.**

- **Description shape.** YAML folded scalar (`>-`). Opens with a verb. Lists
  trigger phrases inline. No angle-bracket placeholders inside the
  description — they get parsed as XML by NV-ACES and fail evaluation.
- **`version` value.** Matches the Dynamo release line you are authoring
  against (e.g. `1.2.0`). Not arbitrary semver.
- **`tags`.** Always include `dynamo` plus the lifecycle stage. Add up to
  three more relevant tags.

**Verification gate.** Advance only when: directory layout exists, frontmatter
parses, all placeholders are clearly marked (`<EDIT: ...>`), the scaffold
output is committed to a working branch so the diff against the sibling is
reviewable.

---

## Phase 3: Author

**Goal.** Replace the sibling-derived body with skill-specific content.
Preserve the structural sections; edit the substance.

**Body section order** (mandatory, in this sequence):

```
# <Skill Title>
<one-paragraph mission statement; <=4 lines>

## Workflow            — 4-phase block in a fenced code block
## Command Safety      — DESTRUCTIVE / MUTATING / SAFE tables
## Phase 1: <Name>     — Goal, Inputs, Commands, Decision points,
## Phase 2: <Name>       Refusal conditions, Verification gate
## Phase 3: <Name>
## Phase 4: <Name>
## Prerequisites
## Limitations
## Troubleshooting
## Available Scripts   — table with run_script() example invocations
## Refusal Conditions
## Cross-Skill Referencing
## References and Scripts
```

See `references/body-shape.md` for the full specification of each section.

**Decision points.**

- **4-phase shape variance.** The install-oriented Pre-Check → Configure →
  Run → Verify is the default. Day-2 skills substitute Triage → Inspect →
  Diagnose → Remediate. Planning skills substitute Gather → Pick → Execute →
  Capture. Pick one shape and stay in it — do not mix.
- **Which commands belong in which safety tier?** Read
  `references/body-shape.md` §Command-Safety Tiers; in doubt, default to
  the stricter tier.
- **When to add a `references/<name>.md` file.** Add one when a section
  would otherwise exceed ~150 lines, when the content is reused by another
  skill, or when the content has a structured shape (annotated YAML field
  reference, command matrix, known-issue catalog). See
  `references/body-shape.md` §When to Add a References File.
- **When to add a `scripts/<name>.sh`.** Add one when the skill needs a
  deterministic check, validation, or evidence-collection step the agent
  should be able to invoke without re-reading the skill body. Every script
  follows the `pass / fail / warn` (pre-operation) or `check()` (post-
  operation) pattern; see `references/body-shape.md` §Script Patterns.

**Refusal conditions.**

The skill refuses to advance to Phase 4 if:

- Any claim about a Dynamo tool, CLI flag, CRD field, or container image is
  not verifiable against the Dynamo source tree. Run `git grep` against the
  release branch you are authoring against and confirm the claim before
  embedding it.
- Any cluster-mutating command is missing from a DESTRUCTIVE / MUTATING
  table.
- A decision point has no explicit pause-and-ask prompt.

**Verification gate.** Advance only when: all required sections are present,
every claim is source-verified, command-safety tables are complete, decision
points and refusal conditions are explicit.

---

## Phase 4: Validate

**Goal.** Run every deterministic check before opening the PR. Fix every
failure. A failure here is cheaper to fix than a CI failure after the PR is
open.

**Inputs.** The completed skill directory from Phase 3.

**Commands** (SAFE):

```bash
# 1. Frontmatter parses and required fields present.
bash scripts/check-frontmatter.sh .agents/skills/dynamo-<your-name>/SKILL.md

# 2. Every script passes shellcheck with no warnings.
shellcheck .agents/skills/dynamo-<your-name>/scripts/*.sh

# 3. Cross-link audit — every references/ and scripts/ link resolves.
# 4. Length budget — SKILL.md within 200-600 lines, references files
#    within 300-700 lines each.
# Both run via:
bash scripts/validate-skill.sh -d .agents/skills/dynamo-<your-name>

# 5. (recommended) NV-ACES Tier 1 deterministic-scoring evaluator.
#    NVIDIA-internal; available via the standard NVIDIA Artifactory.
astra-skill-eval evaluate .agents/skills/dynamo-<your-name> --static
# Target overall >= 90; zero errors.
```

If `astra-skill-eval` is unavailable in your environment, the project's daily
CI runs it on every PR — read the CI output and iterate. The other four
checks must pass locally before pushing.

**Decision points.**

- **NV-ACES score below 90.** Read the failing rubric items and fix; do not
  override. The common-pitfall list in `references/known-issues.md`
  catalogues the typical lifts (e.g. adding `## Available Scripts` with a
  `run_script()` example).
- **PR description shape.** The Dynamo repo's PR template
  (`.github/pull_request_template.md`) has Overview / Details / Where to
  start / Related Issues. Fill all four. Include NV-ACES Tier 1 score,
  shellcheck-clean, YAML parse, cross-link audit results in Details.

**Refusal conditions.**

The skill refuses to open the PR if:

- Any of the four local validators fails.
- Frontmatter `version` is not aligned with the Dynamo release line you are
  authoring against.
- DCO sign-off is missing from any commit (`git log --format='%(trailers:key=Signed-off-by)' | grep -v Signed-off-by` returns lines).

**Verification gate.** Advance to PR submission only when all five
validators pass and the PR description fills the template.

---

## Prerequisites

The contributor's workstation needs:

- Python 3.10+ with `pyyaml` (`pip install pyyaml`).
- `shellcheck` (`brew install shellcheck` on macOS).
- `gh` CLI authenticated against GitHub.
- (Recommended) `astra-skill-eval` from the NVIDIA Artifactory, for local
  NV-ACES Tier 1 runs.

A Dynamo source checkout is required for source-verification of any tool,
CLI flag, CRD field, or container image claim:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git -C dynamo checkout release/1.2.0  # or the release line you target
```

## Limitations

- Authoring conventions in this skill are Dynamo-specific extensions of the
  `ai-infra-agent` methodology. They do not directly apply to skills for
  unrelated NVIDIA projects, though the structural rigor (4-phase workflow,
  DESTRUCTIVE / MUTATING / SAFE rubric, Human-in-the-Loop contract, known-
  issue 6-element shape) is portable.
- The public NVIDIA/skills catalog mirror is automated. Authoring there
  directly is not a supported path — author here, the mirror picks up the
  change in the next daily sync.
- Naming a new lifecycle stage (e.g. `dynamo-cache`, `dynamo-observe`) may
  require refreshing the architectural survey that backs the skill set; see
  the directory README §Per-Release Refresh.

## Troubleshooting

- **Frontmatter parse failure.** Most often the YAML folded scalar contains
  `<placeholder>` angle brackets that get treated as XML. Fix per
  `references/known-issues.md` §XML-Tag Pitfall.
- **shellcheck warning `SC2086`.** Unquoted variable expansion in a script.
  Quote it: `"$VAR"` not `$VAR`.
- **shellcheck warning `SC2329`.** Function reported as unused when invoked
  via `trap`. Add `# shellcheck disable=SC2329` directly above the function
  definition.
- **Cross-link audit failure.** A `references/<x>.md` or `scripts/<x>.sh`
  link in `SKILL.md` does not resolve on disk. Either create the file or
  fix the link.
- **NV-ACES score below 90.** See `references/known-issues.md` §NV-ACES
  Score Below 90 for the standard 15-point lift via `## Workflow` and
  `## Available Scripts` headers.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/scaffold-skill.sh` | Create a new skill directory under `.agents/skills/dynamo-<name>/` by copying from a sibling skill with frontmatter reset and placeholders inserted. | `-n <new-name>`, `-s <sibling>`, `-v <version>` (default `1.2.0`), `-d <skills-root>` (default `.agents/skills`) |
| `scripts/validate-skill.sh` | Run shellcheck, YAML parse, cross-link audit, and length budget against an authored skill directory. Exits non-zero on any failure. | `-d <skill-dir>` |
| `scripts/check-frontmatter.sh` | Strict frontmatter validator. Verifies required fields, valid `version`, no XML tags in `description`, valid `tags` set. | `<path-to-SKILL.md>` |

Invocation via the agentskills.io `run_script` protocol:

```python
run_script("scripts/scaffold-skill.sh", args=["-n", "dynamo-install", "-s", "dynamo-deploy"])
run_script("scripts/validate-skill.sh", args=["-d", ".agents/skills/dynamo-install"])
run_script("scripts/check-frontmatter.sh", args=[".agents/skills/dynamo-install/SKILL.md"])
```

Equivalent direct invocation:

```bash
bash .agents/skills/dynamo-skill-author/scripts/scaffold-skill.sh \
    -n dynamo-install -s dynamo-deploy

bash .agents/skills/dynamo-skill-author/scripts/validate-skill.sh \
    -d .agents/skills/dynamo-install

bash .agents/skills/dynamo-skill-author/scripts/check-frontmatter.sh \
    .agents/skills/dynamo-install/SKILL.md
```

---

## Refusal Conditions

In addition to the per-phase refusals listed above, this skill declines to
proceed if:

- The contributor requests skipping any of the four phases. Each phase has
  a verification gate that must pass before the next phase begins.
- The contributor requests authoring a skill that duplicates an existing
  skill's scope. The decision tree in Phase 1 §Is This a New Skill resolves
  to either a section addition in the existing skill or a clear scope
  differentiation; there is no third option.
- The contributor requests bypassing the source-verification rule. Any
  claim about Dynamo tools or surfaces must trace to a path or symbol in
  the Dynamo source tree.

## Cross-Skill Referencing

| Question | Skill to read |
|---|---|
| What does an actual Dynamo lifecycle skill look like end-to-end? | `dynamo-deploy/SKILL.md` (most cross-cutting; canonical worked example). |
| What does a non-install 4-phase shape look like? | `dynamo-troubleshoot/SKILL.md` (Triage → Inspect → Diagnose → Remediate). |
| What does the decision-heavy shape look like? | `dynamo-plan/SKILL.md` (Gather → Pick → Execute → Capture). |
| What does the directory-level philosophy and per-release refresh model look like? | `README.md` (the directory README). |

## References and Scripts

- [references/frontmatter-shape.md](references/frontmatter-shape.md) — required and optional frontmatter fields, valid values, common pitfalls (XML-tag, version-line alignment, tag set).
- [references/body-shape.md](references/body-shape.md) — section structure, command-safety tier rubric, decision-point and refusal-condition conventions, script patterns (`pass / fail / warn`, `check()`), known-issue 6-element shape, references file conventions.
- [references/known-issues.md](references/known-issues.md) — authoring pitfalls observed during the first seven skills (XML-tag pitfall, non-existent feature claim, premature meta-skill expansion, NV-ACES sub-90 lift).
- [scripts/scaffold-skill.sh](scripts/scaffold-skill.sh) — scaffold a new skill directory from a sibling.
- [scripts/validate-skill.sh](scripts/validate-skill.sh) — run all four local validators against an authored skill.
- [scripts/check-frontmatter.sh](scripts/check-frontmatter.sh) — strict frontmatter validator.

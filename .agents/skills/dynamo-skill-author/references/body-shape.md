<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Body Shape

Strict spec for the `SKILL.md` body — the prose after the frontmatter.
Authoring rule: every required section must appear in the order listed
below. Optional sections may be omitted but must keep the same name when
they do appear.

## Required Section Order

```
1. Progressive Disclosure block (HTML comment, immediately after frontmatter)
2. # <Skill Title>
3. <mission paragraph> (<= 4 lines)
4. ## Workflow                  — 4-phase block in a fenced code block
5. ## Command Safety            — DESTRUCTIVE / MUTATING / SAFE
6. ## Phase 1: <Name>           — Goal, Inputs, Commands, Decision points,
7. ## Phase 2: <Name>             Refusal conditions, Verification gate
8. ## Phase 3: <Name>
9. ## Phase 4: <Name>
10. ## Prerequisites
11. ## Limitations
12. ## Troubleshooting
13. ## Available Scripts        — table with run_script() example
14. ## Refusal Conditions
15. ## Cross-Skill Referencing
16. ## References and Scripts
```

A skill omitting any of sections 4-15 will fail NV-ACES Tier 1 unless an
explicit justification is added in the disclosure block.

## Workflow Section

A fenced code block with four phases. The contents differ by skill shape:

```
Phase 1: <Verb>  → <one-line summary>
Phase 2: <Verb>  → <one-line summary>
Phase 3: <Verb>  → <one-line summary>
Phase 4: <Verb>  → <one-line summary>
```

Three shapes are documented in this skill set; pick one and stay in it.
Mixing shapes is the most common authoring error after the XML-tag pitfall.

| Shape | When to use | Phase verbs |
|---|---|---|
| Install-oriented | Configuration / setup / deployment workflows | Pre-Check → Configure (or Author) → Run (or Validate) → Verify (or Apply) |
| Day-2 operations | Troubleshooting / incident response / observability | Triage → Inspect → Diagnose → Remediate |
| Decision-heavy | Planning / strategy / option selection | Gather → Pick (or Choose) → Execute → Capture |

## Command-Safety Tiers

Every cluster-mutating, file-mutating, or process-mutating command appears
in exactly one of three tier tables. The Human-in-the-Loop contract applies
to every DESTRUCTIVE and MUTATING command.

### DESTRUCTIVE

Commands that cause immediate, hard-to-reverse impact. The agent states
what the command destroys, asks the explicit confirmation prompt including
"you are responsible for the outcome", and waits for explicit yes.

Examples that always appear here:

- `kubectl delete crd ...`
- `kubectl delete ns ...`
- `helm uninstall ...`
- `git push --force` against shared branches
- Any operation that drops persistent state without a recovery path

### MUTATING

Commands that change state but are reversible or follow a normal lifecycle.
The agent states what changes, notes any disruption, and asks "Should I
proceed?"

Examples that typically appear here:

- `kubectl apply -f ...`
- `kubectl patch ... --type=merge ...`
- `helm upgrade ...`
- `git commit`, `git push` to a personal branch

### SAFE

Read-only inspection. No confirmation prompt. Listed as a code block, not
a table.

Examples:

- `kubectl get / describe / logs ...`
- `helm status / get values ...`
- `git status / diff / log`
- Local `python3 ...` against a checked-out source tree

### Tier-Selection Rules

- Default to the stricter tier when in doubt.
- A command that mutates only on the agent's working tree (not the cluster
  or remote systems) goes in MUTATING by default; the tier table notes the
  scope.
- Any command involving `--force`, `--no-verify`, `--ignore-errors`, or
  similar safety-bypass flags goes in DESTRUCTIVE regardless of the base
  command.

## Per-Phase Section Shape

Each phase header (`## Phase N: <Name>`) is followed by these six
sub-sections in this order:

```
**Goal.**           <one sentence>
**Inputs.**         <table: input | source>
**Commands.**       <SAFE-tier commands first; MUTATING after, gated on
                     decision-point confirmation>
**Decision points.** <bulleted list; each item ends with the explicit
                      pause-and-ask prompt>
**Refusal conditions.** <bulleted list; conditions under which the skill
                         declines even with user confirmation>
**Verification gate.** <one sentence describing what must be true to
                        advance to the next phase>
```

A phase missing **Decision points** or **Refusal conditions** indicates the
skill is too mechanical to need a Human-in-the-Loop pause — which is rarely
true. Most Dynamo workflows have at least one decision per phase.

## Available Scripts Section

A table with three columns: Script | Purpose | Arguments.

Followed by:

- A `run_script()` example block showing the agentskills.io canonical
  invocation for the agent consumer.
- An equivalent direct `bash` / `python3` invocation block for human
  readers debugging from the command line.

NV-ACES Tier 1 scans for the `run_script(` substring in this section.
Omitting it is a known sub-90 score cause.

## References and Scripts Section

A bulleted list, one bullet per file under `references/` and `scripts/`.
Format:

```
- [references/<file>.md](references/<file>.md) — <one-line description>
- [scripts/<file>.sh](scripts/<file>.sh) — <one-line description>
```

Every link must resolve on disk. The cross-link audit (Phase 4) checks
this mechanically.

## When to Add a `references/<name>.md` File

Add one when:

- A section in `SKILL.md` would otherwise exceed ~150 lines.
- The content is reused (or expected to be reused) by another skill.
- The content has a structured shape an LLM consumer benefits from
  parsing: an annotated YAML field reference, a command matrix, a known-
  issue catalog, a per-backend launch-flag table.
- The content shifts on a different cadence than the main body (e.g.
  per-release known issues vs the per-architecture workflow).

Length budget per file: 300-700 lines. Split by topic when longer
(`references/aggregated.md`, `references/disaggregated.md`).

## When to Add a `scripts/<name>.sh` File

Add one when:

- The skill needs a deterministic check the agent should be able to invoke
  without re-reading the body.
- The skill needs a pre-operation validator (a "is the cluster ready"
  check) — use the `pass / fail / warn` pattern.
- The skill needs a post-operation verifier (a "did the apply land" check)
  — use the `check()` pattern.
- The skill needs an evidence collector (a passive log-and-config bundle
  for day-2 troubleshooting).

## Script Patterns

### `pass / fail / warn` (Pre-Operation Validators)

```bash
PASS=0; FAIL=0; WARN=0
RESULTS=()

pass() { PASS=$((PASS+1)); RESULTS+=("PASS|$1|$2"); }
fail() { FAIL=$((FAIL+1)); RESULTS+=("FAIL|$1|$2"); }
warn() { WARN=$((WARN+1)); RESULTS+=("WARN|$1|$2"); }

# Example check.
if kubectl get crd dynamographdeployments.nvidia.com &>/dev/null; then
    pass "CRD present" "dynamographdeployments.nvidia.com"
else
    fail "CRD present" "not installed"
fi

# Summary.
echo
echo "===== Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo "Passed: $PASS  Failed: $FAIL  Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
```

### `check()` (Post-Operation Verifiers)

```bash
PASS=0; FAIL=0

check() {
    local desc="$1" cmd="$2" pattern="$3"
    local out
    if out=$(bash -c "$cmd" 2>&1) && echo "$out" | grep -q "$pattern"; then
        PASS=$((PASS+1)); echo "PASS: $desc"
    else
        FAIL=$((FAIL+1)); echo "FAIL: $desc"
    fi
}

check "Frontend reachable" \
      "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/v1/models" \
      "200"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
```

### Output Schema

Stdout lines follow the format:

```
PASS|<check name>|<detail>
FAIL|<check name>|<detail>
WARN|<check name>|<detail>
```

Free-form prose goes to stderr to keep stdout machine-parseable by the
agent.

## Known-Issue Entry Shape

When authoring a `references/known-issues.md`, every entry follows a strict
6-element format so LLM consumers can extract structured records:

```markdown
### <Symptom title>

**Symptom:** <one-sentence user-visible failure>

**Root cause:** <why it happens>

**Affected:** <versions, backends, environments>

**Fix:** <command or YAML the user runs>

**Verify:** <command that confirms the fix worked>
```

Optional 7th element: `**Reference:** <link to upstream issue or doc>`
when one exists.

## Cross-Skill Referencing Section

A two-column table:

```markdown
| Question | Skill to read |
|---|---|
| <question that this skill explicitly does not answer> | `<sibling-name>/SKILL.md` |
```

Use this to redirect the agent to the correct sibling skill instead of
attempting an out-of-scope task. Each sibling reference is a single path,
no deep linking.

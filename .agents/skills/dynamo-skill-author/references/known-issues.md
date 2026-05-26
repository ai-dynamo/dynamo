<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Known Authoring Issues

Pitfalls observed during the initial seven-skill authoring pass and the
mitigations that resolve them. Each entry follows the 6-element shape
documented in `body-shape.md` §Known-Issue Entry Shape.

---

### XML-Tag Pitfall in YAML Description

**Symptom:** `astra-skill-eval evaluate ... --static` fails with an XML
parse error against the `description:` field, or `check-frontmatter.py`
rejects the description with "XML-like placeholder detected".

**Root cause:** Angle-bracket placeholders inside the YAML folded scalar
(`<backend>`, `<model>`, `<name>`) get parsed as XML tags by the evaluator
even though they read like prose to a human author.

**Affected:** Every skill at first pass. Hit on `dynamo-serve` and
`dynamo-deploy` during authoring; fixed before merge.

**Fix:** Replace angle-bracket placeholders with lowercase placeholder
words in the description. Acceptable everywhere else in the skill body
(code blocks, tables) — only the YAML folded scalar is XML-parsed.

```yaml
# BAD
description: >-
  Deploy a Dynamo workload <backend> for model <name> ...

# GOOD
description: >-
  Deploy a Dynamo workload for a chosen backend and model ...
```

**Verify:**

```bash
python3 scripts/check-frontmatter.py path/to/SKILL.md
astra-skill-eval evaluate path/to/skill-dir --static  # if available
```

---

### Non-Existent Feature Claim

**Symptom:** Agent attempts to invoke a Dynamo tool, CLI flag, or surface
during a skill run and the command fails with "command not found" or "no
such argument". The skill body cites a feature that does not exist in the
Dynamo source.

**Root cause:** Authoring from training-data memory rather than
source-verifying every named tool against the active Dynamo release line.
Generic LLM documentation about distributed inference often mentions tool
names that were never built or were renamed.

**Affected:** Almost happened on the planning skill. A draft cited
`dynamo-run` as the local-run CLI; grep against the Dynamo source returned
no matches and no compiled binary. Fixed before the draft was committed —
the canonical local-run invocation is `python3 -m dynamo.<backend>`.

**Fix:** Before naming any Dynamo tool, CLI, CRD field, or container image
in a skill body, run a source-verification check against the release line
you target:

```bash
git -C ~/dynamo checkout release/1.2.0
git -C ~/dynamo grep -l "<feature-name>" -- '*.rs' '*.py' '*.yaml'
```

Empty output is the signal to stop and reconsider. The architectural
survey in the methodology backing this skill set catalogues the verified
inventory; consult it when unsure.

**Verify:**

```bash
# After the source-verification grep returns matches, confirm the symbol
# is actually invokable.
git -C ~/dynamo show release/1.2.0:<path-to-file> | head -50
```

---

### NV-ACES Score Below 90

**Symptom:** `astra-skill-eval evaluate <skill> --static` returns a score
below the 90 floor. Common when authoring a first skill before applying
the standard-headers extension documented in the directory README §6.

**Root cause:** Missing one or more of the headers the NV-ACES rubric
scans for: `## Workflow`, `## Available Scripts`, `## Prerequisites`,
`## Limitations`, `## Troubleshooting`. The deterministic-scoring
evaluator awards points for each header found and content matching its
expected vocabulary.

**Affected:** Early drafts of all seven existing skills. Average lifted
from 76.3 (Grade C) to 92.1 (Grade A-) after the standard-headers patch.

**Fix:**

1. Add `## Workflow` immediately after the mission paragraph, with the
   4-phase block inside.
2. Add `## Available Scripts` with a Script | Purpose | Arguments table
   and a `run_script()` example block. The evaluator scans for the
   `run_script(` substring.
3. Add `## Prerequisites`, `## Limitations`, `## Troubleshooting` even if
   short. Empty sections will not pass; populate each with at least one
   bullet pointing to the relevant skill content or to a sibling.

**Verify:**

```bash
astra-skill-eval evaluate path/to/skill-dir --static
# Overall score >= 90; zero errors.
```

---

### Premature Meta-Skill Expansion

**Symptom:** Contributor proposes a new lifecycle skill that has no Dynamo
source equivalent — a skill that would orchestrate other skills, or that
would document conventions only. Reviews fail because the skill has no
real workflow to validate.

**Root cause:** Meta-content (orchestration, conventions, documentation)
belongs in `README.md` or in this `dynamo-skill-author` skill, not in a
new lifecycle skill. The seven lifecycle skills each own a concrete
Dynamo workflow with verifiable inputs and outputs.

**Affected:** Considered during the initial scope exercise. A
`dynamo-orchestrate` meta-skill that would route between siblings was
proposed and rejected — the routing belongs in the agent's loader (which
reads frontmatter descriptions) and in the directory README's catalog,
not in a skill.

**Fix:** If the proposed work is orchestration or convention documentation,
write it into the directory `README.md` or into this skill. If the
proposed work is a real Dynamo workflow not covered by the existing seven,
proceed with full Phase 1 of `dynamo-skill-author`.

**Verify:** Phase 1 §Decision Points: the "new skill vs section in
existing skill" question resolves to a clear scope differentiation when
the work is genuine lifecycle content. If the answer is "it just routes
between other skills", it is not a skill.

---

### `shellcheck` SC2086 Unquoted Variable

**Symptom:** `shellcheck scripts/<name>.sh` reports SC2086 on a line like
`kubectl get pods -n $NAMESPACE`.

**Root cause:** Unquoted variable expansion can break on values
containing whitespace or shell metacharacters. The check is conservative
but worth respecting — quoting is cheap.

**Affected:** Hit on `dynamo-deploy/scripts/verify-platform.sh` and
`dynamo-benchmark/scripts/precheck-target.sh` during the initial pass.

**Fix:** Quote the variable: `kubectl get pods -n "$NAMESPACE"`.

**Verify:**

```bash
shellcheck path/to/script.sh
# No SC2086 reported.
```

---

### `shellcheck` SC2329 Function Reported Unused

**Symptom:** `shellcheck` reports SC2329 on a function definition; the
function is actually invoked via `trap` or via a name passed to another
function.

**Root cause:** `shellcheck` cannot statically detect indirect invocations.

**Affected:** Hit on `dynamo-benchmark/scripts/precheck-target.sh`
(`cleanup_pf` function invoked via `trap`).

**Fix:** Add a single-line directive above the function definition:

```bash
# shellcheck disable=SC2329
cleanup_pf() {
    ...
}
```

The disable scope is the next line only.

**Verify:**

```bash
shellcheck path/to/script.sh
# No SC2329 reported.
```

---

### Cross-Link Audit Failure

**Symptom:** `validate-skill.sh` reports `MISS: references/<x>.md` or
`MISS: scripts/<x>.sh` for a link that appears in `SKILL.md`.

**Root cause:** The link was added to `SKILL.md` (typically in the
References and Scripts section) before the file was created, or the file
was renamed without updating the link.

**Affected:** Common during iterative authoring.

**Fix:** Either create the missing file or fix the link to match an
existing file. Run the validator after every authoring pass, not only at
the end.

**Verify:**

```bash
bash scripts/validate-skill.sh -d path/to/skill-dir
# No MISS reported.
```

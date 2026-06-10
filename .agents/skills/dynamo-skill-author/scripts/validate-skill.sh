#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the local validators against a Dynamo skill directory:
#   1. shellcheck on every scripts/*.sh
#   2. YAML frontmatter parse + required field check (check-frontmatter.py)
#   3. Cross-link audit: every references/*.md and scripts/*.sh link in
#      SKILL.md resolves on disk
#   4. Length budget: SKILL.md 200-600 lines, references/*.md 150-700 lines
#   5. Publication-leak guard: internal authoring scaffolding (row-ID
#      citation markers, the citation manifest, the architectural survey,
#      the derive-public pipeline) must never reach published skill content
#   6. Citation resolution (optional, needs a Dynamo checkout via -r or
#      $DYNAMO): every Dynamo doc/source path the skill cites resolves, and
#      no claim is cited to an internal-only artifact (check-citations.py)
#
# Exits 0 on all-pass, non-zero on any failure. Output uses the
# PASS / FAIL / WARN line format so the agent can parse stdout.
#
# Usage:
#   bash validate-skill.sh -d <skill-dir> [-r <dynamo-checkout>]
#
# Example:
#   bash validate-skill.sh -d .agents/skills/dynamo-install -r ~/dynamo

set -euo pipefail

SKILL_DIR=""
REPO=""

usage() {
    cat >&2 <<USAGE
Usage: $0 -d <skill-dir> [-r <dynamo-checkout>]

Required:
  -d <skill-dir>    Path to the skill directory under .agents/skills/.

Optional:
  -r <dynamo-repo>  Path to a Dynamo checkout. Enables citation resolution
                    (check 6). Falls back to \$DYNAMO if unset.
USAGE
    exit 2
}

while getopts "d:r:h" opt; do
    case "$opt" in
        d) SKILL_DIR="$OPTARG" ;;
        r) REPO="$OPTARG" ;;
        h|*) usage ;;
    esac
done

[ -z "$SKILL_DIR" ] && { echo "ERROR: -d <skill-dir> is required" >&2; usage; }
[ -d "$SKILL_DIR" ] || { echo "ERROR: not a directory: $SKILL_DIR" >&2; exit 1; }

SKILL_MD="$SKILL_DIR/SKILL.md"
[ -f "$SKILL_MD" ] || { echo "ERROR: SKILL.md not found: $SKILL_MD" >&2; exit 1; }

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { PASS=$((PASS+1)); RESULTS+=("PASS|$1|$2"); }
fail() { FAIL=$((FAIL+1)); RESULTS+=("FAIL|$1|$2"); }
warn() { WARN=$((WARN+1)); RESULTS+=("WARN|$1|$2"); }

# 1. shellcheck on every scripts/*.sh
if compgen -G "$SKILL_DIR/scripts/*.sh" > /dev/null; then
    if command -v shellcheck >/dev/null 2>&1; then
        for script in "$SKILL_DIR"/scripts/*.sh; do
            if shellcheck "$script" >/dev/null 2>&1; then
                pass "shellcheck" "$(basename "$script")"
            else
                fail "shellcheck" "$(basename "$script") (run shellcheck $script to see warnings)"
            fi
        done
    else
        warn "shellcheck" "shellcheck not installed; skipping. Install via: brew install shellcheck"
    fi
else
    warn "shellcheck" "no scripts to check"
fi

# 2. Frontmatter parse + required fields.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_FM="$SCRIPT_DIR/check-frontmatter.py"
if [ -f "$CHECK_FM" ]; then
    if python3 "$CHECK_FM" "$SKILL_MD" >/dev/null 2>&1; then
        pass "frontmatter" "parses; required fields present"
    else
        fail "frontmatter" "validation failed (run: python3 '$CHECK_FM' '$SKILL_MD' for details)"
    fi
else
    warn "frontmatter" "check-frontmatter.py not found alongside this script"
fi

# 3. Cross-link audit. Matches actual markdown links and `bash ./scripts/...`
# / `python3 ./scripts/...` invocations referring to this skill's own files.
# Bare prose mentions of sibling-skill paths are deliberately not checked.
LINK_MISS=0
LINK_RE='\((references|scripts)/[a-z0-9_-]+\.(md|sh|py)\)'
INVOKE_RE='(bash|python3)[[:space:]]+(\./)?scripts/[a-z0-9_-]+\.(sh|py)'

extract() {
    grep -oE "$1" "$SKILL_MD" 2>/dev/null \
        | grep -oE '(references|scripts)/[a-z0-9_-]+\.(md|sh|py)' \
        | sort -u || true
}

LINKS=$( { extract "$LINK_RE"; extract "$INVOKE_RE"; } | sort -u )

while IFS= read -r path; do
    [ -z "$path" ] && continue
    if [ ! -f "$SKILL_DIR/$path" ]; then
        fail "cross-link" "missing in this skill: $path"
        LINK_MISS=$((LINK_MISS+1))
    fi
done <<<"$LINKS"

if [ "$LINK_MISS" -eq 0 ]; then
    pass "cross-link" "all references/ and scripts/ links resolve"
fi

# 4. Length budget.
SKILL_LINES=$(wc -l < "$SKILL_MD" | tr -d ' ')
if [ "$SKILL_LINES" -lt 200 ]; then
    warn "length" "SKILL.md is $SKILL_LINES lines (target 200-600)"
elif [ "$SKILL_LINES" -gt 600 ]; then
    warn "length" "SKILL.md is $SKILL_LINES lines (target 200-600; consider moving deep content to references/)"
else
    pass "length" "SKILL.md is $SKILL_LINES lines (target 200-600)"
fi

if compgen -G "$SKILL_DIR/references/*.md" > /dev/null; then
    for ref in "$SKILL_DIR"/references/*.md; do
        REF_LINES=$(wc -l < "$ref" | tr -d ' ')
        if [ "$REF_LINES" -gt 700 ]; then
            warn "length" "$(basename "$ref") is $REF_LINES lines (target <= 700; consider splitting by topic)"
        else
            pass "length" "$(basename "$ref") is $REF_LINES lines"
        fi
    done
fi

# 5. Publication-leak guard. Internal authoring scaffolding must never reach
# published skill content: row-ID citation markers ([A4], [F2], ...), the
# internal citation manifest (citations.md), the architectural survey
# (DYNAMO_REPO_SURVEY), the master->public derivation pipeline
# (derive-public, MANUAL_REVIEW), and internal repo paths (dynamo-skills/docs).
# These reference files that do not ship and are meaningless to public readers.
# The cross-link audit above only checks this skill's own references/ and
# scripts/ links, so these leaks slip past it.
LEAK_RE='\[[A-G][0-9]+\]|citations\.md|PLAN\.md|HANDOFF\.md|SKILL_AUTHORING|DYNAMO_REPO_SURVEY|derive-public|MANUAL_REVIEW|dynamo-skills/'
LEAK_HITS=0
while IFS= read -r leakfile; do
    [ -f "$leakfile" ] || continue
    # The skill's own authoring-tooling directory (where these validators live)
    # legitimately contains the leak patterns in its detection logic; skip it so
    # the meta-skill that ships the validators still self-validates clean.
    [ "$(cd "$(dirname "$leakfile")" && pwd)" = "$SCRIPT_DIR" ] && continue
    while IFS=: read -r leakline _; do
        [ -z "$leakline" ] && continue
        fail "leak" "$(basename "$leakfile"):$leakline (internal scaffolding in published content)"
        LEAK_HITS=$((LEAK_HITS+1))
    done < <(grep -nE "$LEAK_RE" "$leakfile" 2>/dev/null || true)
done < <(find "$SKILL_DIR" -type f \( -name '*.md' -o -name '*.sh' -o -name '*.py' \) | sort)
[ "$LEAK_HITS" -eq 0 ] && pass "leak" "no internal authoring scaffolding in published content"

# 6. Citation resolution (optional). The skills are distillations of the
# Dynamo docs and source, so every cited doc or source path should resolve in a real
# checkout and no claim should be cited to an internal-only artifact. Needs a
# Dynamo checkout via -r or $DYNAMO; skipped with a WARN if neither is set.
CHECK_CITES="$SCRIPT_DIR/check-citations.py"
CITE_REPO="${REPO:-${DYNAMO:-}}"
if [ ! -f "$CHECK_CITES" ]; then
    warn "citations" "check-citations.py not found alongside this script"
elif [ -z "$CITE_REPO" ] || { [ ! -d "$CITE_REPO/deploy" ] && [ ! -d "$CITE_REPO/components" ]; }; then
    warn "citations" "skipped; pass -r <dynamo-checkout> or set \$DYNAMO to resolve citations"
elif python3 "$CHECK_CITES" --repo "$CITE_REPO" "$SKILL_DIR" >/dev/null 2>&1; then
    pass "citations" "all Dynamo doc/source citations resolve in $CITE_REPO"
else
    fail "citations" "unresolved or internal-only citations (run: python3 '$CHECK_CITES' --repo '$CITE_REPO' '$SKILL_DIR')"
fi

# Summary.
echo
echo "===== Validate Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS  Failed: $FAIL  Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0

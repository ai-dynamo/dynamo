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
#
# Exits 0 on all-pass, non-zero on any failure. Output uses the
# PASS / FAIL / WARN line format so the agent can parse stdout.
#
# Usage:
#   bash validate-skill.sh -d <skill-dir>
#
# Example:
#   bash validate-skill.sh -d .agents/skills/dynamo-install

set -euo pipefail

SKILL_DIR=""

usage() {
    cat >&2 <<USAGE
Usage: $0 -d <skill-dir>

Required:
  -d <skill-dir>    Path to the skill directory under .agents/skills/.
USAGE
    exit 2
}

while getopts "d:h" opt; do
    case "$opt" in
        d) SKILL_DIR="$OPTARG" ;;
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

# Summary.
echo
echo "===== Validate Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS  Failed: $FAIL  Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0

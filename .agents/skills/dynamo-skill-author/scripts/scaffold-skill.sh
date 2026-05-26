#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scaffold a new Dynamo skill directory by copying from a sibling skill.
# Resets frontmatter and inserts <EDIT: ...> placeholders at the spots a
# contributor must edit. Idempotent: refuses to overwrite an existing skill.
#
# Usage:
#   bash scaffold-skill.sh -n <new-name> -s <sibling-name> [-v <version>] [-d <skills-root>]
#
# Example:
#   bash scaffold-skill.sh -n dynamo-install -s dynamo-deploy

set -euo pipefail

NEW_NAME=""
SIBLING=""
VERSION="1.2.0"
SKILLS_ROOT=".agents/skills"

usage() {
    cat >&2 <<USAGE
Usage: $0 -n <new-name> -s <sibling-name> [-v <version>] [-d <skills-root>]

Required:
  -n <new-name>     Name of the new skill (must begin with dynamo-).
  -s <sibling>      Existing skill to scaffold from (e.g. dynamo-deploy).

Optional:
  -v <version>      Frontmatter version field (default: $VERSION).
  -d <root>         Skills root directory (default: $SKILLS_ROOT).
USAGE
    exit 2
}

while getopts "n:s:v:d:h" opt; do
    case "$opt" in
        n) NEW_NAME="$OPTARG" ;;
        s) SIBLING="$OPTARG" ;;
        v) VERSION="$OPTARG" ;;
        d) SKILLS_ROOT="$OPTARG" ;;
        h|*) usage ;;
    esac
done

[ -z "$NEW_NAME" ] && { echo "ERROR: -n <new-name> is required" >&2; usage; }
[ -z "$SIBLING" ]  && { echo "ERROR: -s <sibling-name> is required" >&2; usage; }

case "$NEW_NAME" in
    dynamo-*) ;;
    *) echo "ERROR: new-name must begin with 'dynamo-'; got '$NEW_NAME'" >&2; exit 1 ;;
esac

SIBLING_DIR="$SKILLS_ROOT/$SIBLING"
NEW_DIR="$SKILLS_ROOT/$NEW_NAME"

[ -d "$SIBLING_DIR" ] || { echo "ERROR: sibling not found: $SIBLING_DIR" >&2; exit 1; }
[ -e "$NEW_DIR" ] && { echo "ERROR: target already exists: $NEW_DIR (refusing to overwrite)" >&2; exit 1; }

echo "Scaffolding $NEW_DIR from $SIBLING_DIR ..."

mkdir -p "$NEW_DIR/references" "$NEW_DIR/scripts"

# Copy references and scripts directories verbatim.
if [ -d "$SIBLING_DIR/references" ]; then
    cp -r "$SIBLING_DIR/references/." "$NEW_DIR/references/"
fi
if [ -d "$SIBLING_DIR/scripts" ]; then
    cp -r "$SIBLING_DIR/scripts/." "$NEW_DIR/scripts/"
fi

# Generate a fresh SKILL.md from the sibling with frontmatter reset.
SIBLING_BODY=$(awk 'NR > 1 && /^---$/{p=1; next} p{print}' "$SIBLING_DIR/SKILL.md")

cat > "$NEW_DIR/SKILL.md" <<SKILL_MD
---
name: $NEW_NAME
description: >-
  <EDIT: one-paragraph description. Opens with a verb. Lists trigger
  phrases inline so the agent loader matches the user's prompt. 50-500
  chars. No angle-bracket placeholders.>
version: $VERSION
author: NVIDIA
tags:
  - dynamo
  - <EDIT: lifecycle-stage>
tools:
  - Shell
  - Read
  - Write
---
$SIBLING_BODY
SKILL_MD

cat <<SUMMARY
PASS|scaffold|created $NEW_DIR
PASS|references|copied from $SIBLING_DIR/references/
PASS|scripts|copied from $SIBLING_DIR/scripts/
PASS|SKILL.md|frontmatter reset; body copied from $SIBLING (edit the body for the new skill)
INFO|next|edit frontmatter description and tags, then replace sibling-specific body content
INFO|next|run: bash $(dirname "$0")/validate-skill.sh -d $NEW_DIR
SUMMARY

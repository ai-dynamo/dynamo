#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Link Cursor personal skills from the agent store into ~/.cursor/skills so
# skill-relative scripts (e.g. scripts/render.sh) resolve consistently.

set -euo pipefail

STORE="${CURSOR_USER_SKILLS_STORE:-/cursor/stores/user/skills}"
DEST="${HOME}/.cursor/skills"

if [[ ! -d "$STORE" ]]; then
  exit 0
fi

mkdir -p "$DEST"

for skill_dir in "$STORE"/*/; do
  [[ -d "$skill_dir" ]] || continue
  name="$(basename "$skill_dir")"
  if [[ -f "$skill_dir/SKILL.md" ]]; then
    ln -sfn "$skill_dir" "$DEST/$name"
  elif [[ -f "$skill_dir/skill-snapshot/SKILL.md" ]]; then
    ln -sfn "$skill_dir/skill-snapshot" "$DEST/$name"
  fi
done

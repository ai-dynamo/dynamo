#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check by default. No downloads, dependency upgrades, builds, or server launches.
set -euo pipefail

mode=--check
if [[ ${1:-} == --check || ${1:-} == --apply ]]; then
  mode=$1
  shift
fi
if [[ $# != 1 ]]; then
  echo 'Usage: bash apply.sh [--check|--apply] /path/to/sglang-checkout' >&2
  exit 2
fi
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
target=$(cd -- "$1" && pwd -P)
expected=e00930c5489053f26d86b179cee0d087f846acbb
[[ $(git -C "$target" rev-parse --show-toplevel) == "$target" ]] || {
  echo 'Target must be the SGLang repository root.' >&2; exit 1;
}
[[ $(git -C "$target" rev-parse HEAD) == "$expected" ]] || {
  echo "Refusing wrong source revision; expected $expected" >&2; exit 1;
}
if ! git -C "$target" diff --quiet || ! git -C "$target" diff --cached --quiet; then
  echo 'Refusing dirty tracked worktree/index (or already-applied overlay).' >&2
  exit 1
fi
(cd "$bundle" && sha256sum --check SHA256SUMS)
patches=()
while IFS= read -r name; do
  [[ $name =~ ^[0-9]{4}-[a-z0-9-]+\.patch$ ]] || {
    echo 'Invalid series entry.' >&2; exit 1;
  }
  patches+=("$bundle/$name")
done < "$bundle/series"
[[ ${#patches[@]} == 6 ]] || { echo 'Expected six patches.' >&2; exit 1; }
# Check the complete series against the clean index, including added-file collisions.
git -C "$target" apply --check --index "${patches[@]}"
if [[ $mode == --apply ]]; then
  # No --reject/--3way: fail rather than leave partially applied hunks.
  git -C "$target" apply "${patches[@]}"
  echo 'Applied all six patches; HEAD and index are unchanged.'
else
  echo 'All six patches apply cleanly; target unchanged.'
fi

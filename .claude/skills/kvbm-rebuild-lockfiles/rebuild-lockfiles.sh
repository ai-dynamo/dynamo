#!/usr/bin/env bash
# Finds all Cargo.lock files from the repo root, deletes each one,
# then regenerates it by running `cargo generate-lockfile` in that directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Searching for Cargo.lock files under $REPO_ROOT ..."

while IFS= read -r lockfile; do
    dir="$(dirname "$lockfile")"
    echo ""
    echo "==> Removing $lockfile"
    rm "$lockfile"
    echo "    Regenerating lockfile in $dir ..."
    (cd "$dir" && cargo generate-lockfile)
    echo "    Done: $lockfile"
done < <(find "$REPO_ROOT" -name "Cargo.lock" -not -path "*/.git/*")

echo ""
echo "All Cargo.lock files rebuilt."

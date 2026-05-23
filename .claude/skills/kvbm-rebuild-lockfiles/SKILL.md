---
name: kvbm-rebuild-lockfiles
description: Regenerate all Cargo.lock files in the repo (wraps .sandbox/rebuild-lockfiles.sh)
user-invocable: true
disable-model-invocation: true
---

# KVBM Rebuild Lockfiles

Regenerate every `Cargo.lock` file under the repo. Wraps `.sandbox/rebuild-lockfiles.sh`, which finds every `Cargo.lock` (excluding `.git/`), deletes it, and runs `cargo generate-lockfile` in that directory.

## When To Use

- After adding/removing a workspace member (e.g. the `kvbm-scheduler` add/remove churn from ACTIVE_PLAN phase 0).
- After bumping a shared dependency that several nested workspaces consume.
- After resolving a merge conflict in a `Cargo.lock` where the right answer is "regenerate, don't hand-merge".
- Before a clean `cargo check --all-features --all-targets` to get a reproducible dep set.

**Not needed for** routine code changes. A single stale lockfile can be regenerated with `cargo update -p <crate>` in its directory — that's faster and a smaller diff.

## Arguments

`/dynamo:kvbm:rebuild-lockfiles [--dry-run]`

- **--dry-run**: Find and list the lockfiles without deleting or regenerating.

## Step 1: Preflight

```bash
test -x .sandbox/rebuild-lockfiles.sh || { echo "script missing: .sandbox/rebuild-lockfiles.sh"; exit 1; }
which cargo || { echo "cargo not on PATH"; exit 1; }
```

Show the user the set of lockfiles that will be touched (always do this, not just in dry-run):

```bash
find . -name Cargo.lock -not -path './.git/*' -not -path './target/*'
```

Warn if the list is unexpectedly long (>5) — that might indicate the sandbox venv accidentally has target/ build artifacts.

## Step 2: Confirm With The User

Show:

```
Rebuild Cargo.lock files
────────────────────────
Files that will be deleted and regenerated:
  <list>

This WILL change Cargo.lock content (intentionally). Make sure you have a
clean git state or are OK committing the churn.

git status for context:
  <git status -s output>
```

Confirm before proceeding unless `--dry-run`.

## Step 3: Run

```bash
bash .sandbox/rebuild-lockfiles.sh
```

Stream output. Each file is reported as:

```
==> Removing ./path/Cargo.lock
    Regenerating lockfile in ./path ...
    Done: ./path/Cargo.lock
```

## Step 4: Post-Run Verification

```bash
# Any unexpected leftovers?
find . -name 'Cargo.lock.bak' -o -name 'Cargo.lock.orig' 2>/dev/null

# Diff summary so the user can see what churned
git diff --stat -- '**/Cargo.lock'

# Smoke check: at least the root workspace still compiles its dep graph
cargo check --workspace --offline 2>&1 | tail -5 || echo "offline check failed — expected if new deps resolved online"
```

## Step 5: Next Steps

Tell the user:

```
Lockfiles regenerated. Follow-ups:

  1. Review the diff:      git diff -- '**/Cargo.lock' | head -100
  2. Run cargo check:      cargo check --all-features --all-targets
  3. Rebuild kvbm-py3:     /dynamo:kvbm:maturin-dev --clean
  4. Commit (if green):    git add '**/Cargo.lock' && git commit -m "rebuild lockfiles"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cargo generate-lockfile` fails with "failed to resolve" | Workspace member references a missing crate (e.g. `kvbm-scheduler` removed but still listed in root `Cargo.toml`) | Fix the workspace Cargo.toml first, then re-run |
| Lockfiles regenerate but `cargo check` still fails | Feature drift in a transitive dep | Check `cargo tree` for the conflicting version |
| Way more Cargo.lock files than expected | `target/` builds leaked one | Add `-not -path './target/*'` in the find command (already filtered in this skill, but check the underlying script) |
| Network timeouts during regenerate | Offline / registry proxy issue | Set `CARGO_NET_OFFLINE=false` and retry; or pre-warm the registry |

## Reference: What The Script Does

`.sandbox/rebuild-lockfiles.sh` is ~20 lines:

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while IFS= read -r lockfile; do
    dir="$(dirname "$lockfile")"
    rm "$lockfile"
    (cd "$dir" && cargo generate-lockfile)
done < <(find "$REPO_ROOT" -name "Cargo.lock" -not -path "*/.git/*")
```

It does **not** exclude `target/`, so if you've got stray target builds anywhere, they'll be touched too. Clean your target dirs first if that's a concern.

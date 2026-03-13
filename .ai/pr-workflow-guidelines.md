# PR Workflow Guidelines

Conventions for keeping pull requests healthy and reviewable.

## Keep Your Branch Close to Main

Be aware of PRs targeting `main` that are more than ~25 commits behind.
Stacked PRs targeting another branch are exempt.

### 1. Slower CI builds

CI builds start from a base image matching a recent `main` commit. The closer
your branch is to `main`, the more cache hits you get. We use BuildKit with
remote cache (`--cache-from`), but BuildKit layers are still content-addressed
-- if a `COPY` brings in source that changed since your branch point, that
layer and everything downstream rebuilds regardless. When your branch is far
behind:

- **Docker layer cache misses**: Changed source or Dockerfile instructions
  since your branch point invalidate cached layers, forcing rebuilds
  downstream.
- **Rust compilation cache misses**: More crate sources differ from what's
  cached, so more crates recompile from scratch.
- **Dependency drift**: `Cargo.lock` and `requirements.txt` evolve on `main`.
  Stale branches pull older versions that aren't cached, triggering fresh
  downloads and builds.

A branch too far behind `main` can trigger a near-cold build that takes
45-60 minutes. A branch close to `main` reuses most of the cache and builds
in a fraction of that time.

### 2. Reviewer burden

Merge commits from main pollute the diff with unrelated changes. Rebasing
produces a clean, linear history where every commit is the PR author's.

```bash
# BAD -- merge pulls in unrelated files; reviewer has to filter them out
git merge main

# GOOD -- clean diff showing only your work
git fetch origin && git rebase origin/main
```

### Guidance

- **Rebase when you are more than ~25 commits behind main.**
- **Prefer `rebase` over `merge`** for linear history. Force-push after
  rebasing (`git push --force-with-lease`).
- **Before requesting review**, check your distance from main:
  ```bash
  git fetch origin
  git rev-list --count HEAD..origin/main
  ```
  More than 25? Rebase first.
- **If CI is slow**, check your base commit age before debugging other causes.
- **Stacked PRs** are exempt. If a PR targets another branch in a stack,
  distance from `main` is expected and not a problem until the stack lands.

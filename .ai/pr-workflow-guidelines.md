# PR Workflow Guidelines

Conventions for keeping pull requests healthy, CI builds happy, and reviewable.

## Prefer Rebase Over Merge

If a PR contains 2-3 or more merge commits from `main`, flag it. Repeated
merges signal the branch is drifting and each one:

- **Slows CI** -- merge commits pull in source changes that invalidate Docker
  layer caches and Rust compilation caches, pushing builds toward cold
  rebuilds (45-60 min instead of minutes).
- **Pollutes the diff** with unrelated changes, making review harder.

Suggest the author rebase instead:

```bash
git fetch origin && git rebase origin/main
git push --force-with-lease
```

Stacked PRs targeting another branch (not `main`) are exempt.

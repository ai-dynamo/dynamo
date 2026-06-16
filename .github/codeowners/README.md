# CODEOWNERS as code

This directory generates the repository's root `CODEOWNERS` file from one
declarative source. **Do not hand-edit `CODEOWNERS`** - it is generated, and CI
fails if it drifts from the source here. Change `areas.yaml` and regenerate.

## Who reviews my change?

Usually nothing to do: when you open a PR, GitHub auto-requests the team that
owns the files you touched. To check explicitly, before pushing:

```bash
# the teams that will be auto-requested on your PR (union over changed files)
python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed --base main

# owners of specific paths
python .github/codeowners/who_owns.py --codeowners CODEOWNERS lib/llm/foo.rs deploy/operator/bar.go
```

A line with more than one team is co-ownership: under "any one approves," any one
of them satisfies the gate, so co-ownership adds review *visibility* without
adding required approvals.

## Files

| File | What it is |
|------|------------|
| `areas.yaml` | The single source of truth: path globs to GitHub team, by subsystem. **Edit this.** |
| `codeowners_match.py` | Shared matcher + resolution pipeline. Build, emit, and who_owns all call into this -- one matcher, one resolver, no drift. |
| `build_codeowners.py` | Resolves `areas.yaml` against the tree and validates 100% coverage (CI gate). |
| `emit_codeowners.py` | Generates the root `CODEOWNERS` (a minimal, per-area-grouped last-match cover). |
| `who_owns.py` | Answers "who reviews this?" for a path or a whole PR. |
| `test_codeowners.py` | Unit tests for the canonical matcher and the min-cost cover. |

## Change ownership

1. Edit `areas.yaml`: add a glob to an area, add a new area, or adjust the
   `shared` (co-own a path by two teams) block.
2. Regenerate from the repo root:

   ```bash
   pip install pyyaml
   python .github/codeowners/build_codeowners.py \
     --areas .github/codeowners/areas.yaml --repo . --strict
   python .github/codeowners/emit_codeowners.py \
     --areas .github/codeowners/areas.yaml --repo . \
     --out CODEOWNERS
   ```

3. Commit `areas.yaml` and `CODEOWNERS` together.

## How it stays correct (CI)

`.github/workflows/codeowners.yml` runs on every PR and fails if:

- any tracked file falls through to no owner (**coverage gate**) - a new
  directory no area claims blocks the PR until `areas.yaml` is updated; or
- the committed `CODEOWNERS` differs from what `areas.yaml` produces
  (**drift check**) - so the output always matches its source.

## Notes

- Owners are GitHub **teams**, never individuals. Team membership (who is on each
  team) is managed separately and is not part of this directory.
- The generated `CODEOWNERS` carries a top-of-file legend (area to team) and is
  grouped per area for the rare manual read; the machine is the real consumer.

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

External contributors: every PR also gets a "Who reviews this PR" report in
the `codeowners-reviewers` workflow's run summary (Actions tab), expanded to
member logins when the repo's org-read token is configured -- no org
membership or local tooling needed.

## Files

| File | What it is |
|------|------------|
| `areas.yaml` | The single source of truth: path globs to GitHub team, by subsystem. **Edit this.** |
| `external_contributors.yaml` | External individuals granted area-scoped codeownership. Attaches a person to an area **label** (not a copy of its globs); drives the `@handle` co-owner lines and `CONTRIBUTORS.md`. **Edit this.** |
| `codeowners_match.py` | Shared matcher + resolution pipeline. Build, emit, and who_owns all call into this -- one matcher, one resolver, no drift. |
| `build_codeowners.py` | Resolves `areas.yaml` against the tree and validates 100% coverage (CI gate). |
| `emit_codeowners.py` | Generates the root `CODEOWNERS` (a minimal, per-area-grouped last-match cover) and `CONTRIBUTORS.md`. |
| `who_owns.py` | Answers "who reviews this?" for a path or a whole PR. |
| `test_codeowners.py` | Unit tests for the canonical matcher, the min-cost cover, and external-contributor co-ownership. |

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

## External contributors

An external individual who has earned ownership of an area is granted it by
attaching them to that area's **label** in `external_contributors.yaml` -- never
by copying its globs. The generator then appends their `@handle` as a co-owner
on every `CODEOWNERS` line that area's team owns (base rules, path overrides,
shared, and file-type rows), so they inherit exactly the team's paths. Add a
glob to the area in `areas.yaml` and the contributor picks it up automatically.

```yaml
# external_contributors.yaml
contributors:
  - name: Jane Doe
    github: janedoe          # -> @janedoe on their CODEOWNERS lines
    level: maintainer        # contributor | trusted_contributor | maintainer | core_maintainer
    affiliation: Example Org
    areas: [router]          # area labels from areas.yaml
```

`level` is standing metadata shown in the generated `CONTRIBUTORS.md`; it does
not change routing (co-ownership is granted by `areas`). Ownership is
co-ownership: the area team stays on the line and the contributor is appended,
so under "any one approves" either can satisfy the gate.

Regenerating `CODEOWNERS` also regenerates `CONTRIBUTORS.md`; the
`emit_codeowners.py` command above already reads `external_contributors.yaml`
from the same directory by default. Commit the two source files and both
generated outputs together.

## How it stays correct (CI)

`.github/workflows/codeowners.yml` runs on every PR and fails if:

- any tracked file falls through to no owner (**coverage gate**) - a new
  directory no area claims blocks the PR until `areas.yaml` is updated; or
- the committed `CODEOWNERS` or `CONTRIBUTORS.md` differs from what the sources
  produce (**drift check**) - so the outputs always match their sources.

## Notes

- Base area owners are GitHub **teams**. The one exception is
  `external_contributors.yaml`, which appends named individuals as area-scoped
  **co-owners** (never sole owners); team membership is otherwise managed
  separately and is not part of this directory.
- The generated `CODEOWNERS` carries a top-of-file legend (area to team) and is
  grouped per area for the rare manual read; the machine is the real consumer.

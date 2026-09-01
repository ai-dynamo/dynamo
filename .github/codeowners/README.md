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
| `external_contributors.yaml` | External contributors with their governance standing (`level`) and areas. Maintainer-level entries drive individual `@handle` co-owner lines in `CODEOWNERS`; every entry lands in `CONTRIBUTORS.md`. **Edit this.** |
| `codeowners_match.py` | Shared matcher + policy resolver. Build, emit, and who_owns share its CODEOWNERS semantics. |
| `build_codeowners.py` | Validates the resolved policy against the tracked tree for 100% explicit coverage, and checks the `SIGS.md` roster maps every non-program area to a SIG (CI gate). |
| `emit_codeowners.py` | Generates root `CODEOWNERS` directly from declared policy globs, plus `CONTRIBUTORS.md`; it never reads the git tree. |
| `who_owns.py` | Answers "who reviews this?" for a path or a whole PR. |
| `test_codeowners.py` | Unit tests for matching, policy resolution, deterministic emission, precedence, and external-contributor co-ownership. |

## Change ownership

1. Edit `areas.yaml`: add a glob to an area, add a new area, or adjust the
   `shared` (co-own a path by two teams) block.
2. Regenerate from the repo root:

   ```bash
   pip install pyyaml
   python .github/codeowners/build_codeowners.py \
     --areas .github/codeowners/areas.yaml --repo . --strict
   python .github/codeowners/emit_codeowners.py \
     --areas .github/codeowners/areas.yaml --out CODEOWNERS
   ```

3. Commit `areas.yaml` and `CODEOWNERS` together.

## External contributors

Externals cannot join the `dynamo-<area>-codeowners` org teams, so an external
Maintainer who has earned ownership of an area is granted it by attaching them
to that area's **label** in `external_contributors.yaml` -- never by copying
its globs. The generator then appends their `@handle` as a co-owner on every
`CODEOWNERS` line that area's team owns (base rules, path overrides, shared,
and file-type rows), so they inherit exactly the team's paths. Add a glob to
the area in `areas.yaml` and the contributor picks it up automatically.

```yaml
# external_contributors.yaml
contributors:
  - name: Jane Doe
    github: janedoe          # -> @janedoe on their CODEOWNERS lines
    level: maintainer        # contributor | trusted_contributor | maintainer | core_maintainer
    affiliation: Example Org
    areas: [router]          # area labels from areas.yaml
```

`level` gates emission: only `maintainer` and `core_maintainer` reach
`CODEOWNERS`, because a code owner's review satisfies the required-review gate
and that is merge authority. A `trusted_contributor` may review and approve but
cannot merge (see `GOVERNANCE.md`), so lower levels are recorded in
`CONTRIBUTORS.md` only, and `areas` decides which lines a maintainer co-owns.
Ownership is co-ownership: the area team stays on the line and the contributor
is appended, so under "any one approves" either can satisfy the gate.

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

- Emission is a pure function of `areas.yaml` and
  `external_contributors.yaml`. Adding, deleting, or moving unrelated tracked
  files cannot rewrite the generated rules.
- Rule precedence follows CODEOWNERS last-match-wins semantics: catch-all and
  base area rules, file-type defaults, nested path overrides, then explicit
  `shared` rules. Overlapping file-type rules retain their YAML declaration
  order, so later entries refine earlier ones.
- Legacy `classify.keyword_rules` are rejected because auto-classification and
  keyword co-ownership required a live tree. Declare ownership with explicit
  area `path_globs` or `shared` entries.
- Legacy `advisory` is rejected wherever it appears — a top-level `advisory:`
  block, an `advisory` key on a `shared` entry, or one on a filetype rule.
  Every owner in CODEOWNERS blocks, so there is no non-blocking spelling left;
  ignoring the key would silently turn a rule its author marked non-blocking
  into a required approver. Rejection is on key **presence**, so `advisory:
  false` and `advisory: []` fail too. To keep a reviewer, move the entry to
  `shared` and list **all** intended owners: `shared` rules are emitted last
  and win by last-match, so a single-owner `shared` entry replaces the current
  owner rather than adding to it.
- Base area owners are GitHub **teams**. The one exception is
  `external_contributors.yaml`, which appends named individuals as area-scoped
  **co-owners** (never sole owners); team membership is otherwise managed
  separately and is not part of this directory.
- The generated `CODEOWNERS` carries a top-of-file legend (area to team) and is
  grouped per area for the rare manual read; the machine is the real consumer.

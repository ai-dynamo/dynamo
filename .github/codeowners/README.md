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
| `external_contributors.yaml` | External individuals granted area-scoped codeownership. Attaches a person to an area **label** (not a copy of its globs); drives the `@handle` co-owner lines and `CONTRIBUTORS.md`. **Edit this.** |
| `codeowners_match.py` | Shared matcher + policy resolver. Build, emit, and who_owns share its CODEOWNERS semantics. |
| `build_codeowners.py` | Validates the resolved policy against the tracked tree for 100% explicit coverage (CI gate). |
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

### Approval of ownership changes

`meta.delegated_policy_reviews: true` delegates exactly `areas.yaml` and root
`CODEOWNERS` to the required `ownership-approvals` check. The generated file ends
with exact ownerless rules for those two paths; all other files retain native
CODEOWNERS review. The existing `codeowners` check continues to validate coverage,
contracts, tests and generated-file drift.

The approval check compares effective owners before and after the proposed merge,
using the canonical last-match-wins matcher. For every changed owner set, each
previous and proposed owner must approve:

| Ownership change | Required approvals |
| --- | --- |
| Ops → Ops + Runtime | Ops **and** Runtime |
| Docs → Operator | Docs **and** Operator |
| Ops + Runtime → Runtime | Ops **and** Runtime |
| A + B → A + C | A, B **and** C |

One eligible reviewer can represent multiple groups. The PR author and bots cannot
approve; current team membership and the latest decisive review are used. Comments
do not revoke approvals, but dismissals and changes-requested reviews do. Named
individual co-owners must approve personally; a sole individual owner who authors
the PR cannot self-approve. Email-only owners cannot be resolved by this check;
use GitHub team or user handles for delegated policy changes.

Approvals count only for the current PR head commit. Every push or rebase requires
fresh approvals. Changes to enforcement Python or its workflows additionally
require Ops and Process. Contract/default/delegation-setting changes, added or
removed directory/wildcard scopes, and reordered broad scopes also require Process:
they can affect future files absent from the tracked tree. Reassigning an existing
scope without reordering or adding an exact-file override uses the affected owners
directly. Adding a shared directory rule can move it past other broad scopes and
therefore require Process.

The trusted workflow reads policy blobs as data and never checks out or executes
PR code with membership/write credentials. Its separately named check cannot be
satisfied by rerunning an old structural `codeowners` job. PR updates and
submitted/dismissed reviews refresh the live PR; base-policy/tree changes refresh
open policy PRs in one runner. Source-only base edits need no refresh. Missing
API access or memberships blocks approval. Ownership-changing PRs sharing a head
are blocked until they use distinct commits, because checks are commit-scoped.

#### Rollout

1. Provide `ORG_READ_TOKEN`, limited to organization **Members: read**, to the
   trusted workflow. Publish checks with `GITHUB_TOKEN` (`checks: write`).
2. Require **`ownership-approvals` from GitHub Actions (App ID 15368)** on `main`,
   keeping `codeowners` required. The activation guard rejects delegation until
   both checks are required.
3. Merge after native Ops/Process review. The bootstrap supplies the new check
   only while live `main` disables delegation. Other open PRs may wait for it.
4. Existing PRs are **not automatically migrated**. Re-run their `codeowners`
   workflow or dispatch the new workflow with a single `pr` number. The same
   single-PR dispatch refreshes membership changes without a new commit.

Branch-rule administration and secret provisioning are separate from changing
repository files. Do not remove native protection or the activation guard before
the required check is configured.

All repository CI workflows remain under their existing Ops ownership. This
design trusts that review boundary: App-ID pinning distinguishes GitHub Actions
from user-posted statuses, but does not distinguish two workflows run by the same
App. Someone authorized to change repository CI can change CI enforcement; an
independent GitHub App or ruleset-required workflow would be needed to remove that
existing trust relationship.

### Add owners without dropping existing accountability

GitHub uses last-match-wins semantics, so a narrower row replaces the complete
owner set from earlier rows. A `shared` entry therefore lists its **complete**
owner set -- the enclosing owners it keeps, then the ones being added:

```yaml
shared:
  - glob: deploy/inference-gateway/epp/Dockerfile
    owners: [epp, ops, operator]   # epp+ops kept, operator added
```

The strict gate checks that restatement, so omitting `epp` above fails the
build rather than quietly taking the file from them. Enclosure is resolved, not
tiered: a shared row is measured against whoever owns the path immediately
before it, which may have been decided by an ancestor several levels up or by
an intermediate rule that already reassigned it. The catch-all never counts --
every explicit row exists to replace it.

That check covers `shared` rows only. An area `path_globs` entry nested inside
another area's directory still replaces the outer owner silently. Whether an
owner should inherit down a subtree at all is an open question about the
ownership model; until it is settled, a reviewer reading the `areas.yaml` diff
is the control there.

For an invariant that should not emit another CODEOWNERS row, use
`required_owners`. The strict gate checks every matching tracked file against
the final generated artifact:

```yaml
required_owners:
  - glob: deploy/operator/
    owners: [operator]
```

This is appropriate for parent-team guarantees. A narrower specialist rule
under `deploy/operator/` may add GMS or Observability, but it cannot silently
remove Operator. Each `required_owners` glob must match at least one tracked
path; strict validation rejects stale contracts after their final path is
deleted.

Removing a `required_owners` entry lifts the requirement -- deliberately, or a
contract could never be retired. The edit is itself a policy change: it is
judged full-tree and reviewed by this directory's owners, so lifting a
guarantee is always a visible, owned decision.

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
- any declared ownership glob matches no tracked file (**stale policy gate**) -
  a PR that deletes or renames a glob's final matching path must prune the
  declaration in the same PR (that edit makes it a policy change, so the PR
  is then judged full-tree like any routing change). Staleness a PR merely
  inherited from its base is a warning there and blocks only policy PRs and
  full-tree runs, so base churn never red-Xes unrelated work; or
- final last-match resolution removes an owner promised by `required_owners`,
  or a blocking file-type declaration (**ownership contract gate**) - coverage
  by the wrong team no longer counts as success; or
- a `shared` row drops an owner granted by the row it overrides (**shared
  additivity gate**) - the restatement `shared` requires is machine-checked,
  not trusted; or
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

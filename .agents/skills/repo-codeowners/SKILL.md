---
name: repo-codeowners
description: Works with Dynamo's generated CODEOWNERS - finds out who reviews a change, fixes a failing codeowners CI check, changes review routing, or grants an external contributor area-scoped ownership. Use when the codeowners check fails on a PR, a new directory is unclaimed, someone asks who reviews a path or PR, review routing needs to change, or a contributor should be added as a code owner.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - codeowners
    - review-routing
    - dev-workflow
---

# Skill: CODEOWNERS Operations

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

The root `CODEOWNERS` is a build artifact generated from
`.github/codeowners/areas.yaml` (one entry per subsystem area mapping path
globs to a GitHub team). Never hand-edit `CODEOWNERS` - CI regenerates it and
fails on any drift. Every change goes through `areas.yaml` (or
`external_contributors.yaml`) followed by regeneration.

Pick the flow that matches the situation.

## Flow 1: Who reviews this change?

```bash
# owners of your working tree's changed files (union, as GitHub will request)
python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed

# owners of specific paths
python .github/codeowners/who_owns.py --codeowners CODEOWNERS <path> [<path> ...]
```

Add `--people` to expand each team to its member logins. This works only for
NVIDIA org members with an authenticated `gh` - GitHub does not show team
membership to non-members. Without it (or when the lookup fails), the team
handles are the answer; the actual reviewers appear on the PR once it opens.
For an open PR, the `codeowners-reviewers` workflow also posts this table
(with member names when the org-read secret is configured) to its run summary
in the Actions tab, visible to external contributors.

## Flow 2: The `codeowners` CI check failed

This is the coverage gate doing its job: the PR adds at least one file that no
area claims. Nothing ships unowned. On pull requests the gate is diff-aware:
only files YOUR PR adds or changes block it; unowned paths inherited from the
base branch are a non-fatal warning. (A PR that edits ownership policy itself
is judged against the full tree, since a policy edit can orphan any path.)

1. Read the failing job log. The gate prints the exact uncovered files:
   `catch-all-only sample (add an explicit glob to cover these): [...]`.
2. Decide which area owns the new path. Match it to the subsystem whose code it
   extends (the PR that introduced `examples/custom_encoder/` was a multimodal
   feature, so the claim went under the `multimodal` area).
3. Add ONE line to `.github/codeowners/areas.yaml` under that area's
   `path_globs` (directory claims end with `/`). There is no keyword-based
   auto-classification - every claim is an explicit glob (or a `shared:`
   entry for multi-team paths).
4. Regenerate and verify:
   ```bash
   python .github/codeowners/build_codeowners.py \
       --areas .github/codeowners/areas.yaml --repo . --strict
   python .github/codeowners/emit_codeowners.py \
       --areas .github/codeowners/areas.yaml --out CODEOWNERS
   ```
   The strict run must report 100% coverage.
5. Commit `areas.yaml` and `CODEOWNERS` together (same commit), signed
   (`git commit -s`).

Removals fail differently: deleting a directory never fails coverage (it
counts files, and a claim matching nothing is not an error), but regeneration
drops the directory's rule, so the **drift check** fails until the
regenerated `CODEOWNERS` is committed with the deletion. Run step 4 and
commit both files. While there, prune the now-dead glob from `areas.yaml`;
the `build_codeowners.py` report lists globs that no longer match any file.

## Flow 3: Change review routing

Edit `.github/codeowners/areas.yaml` - move a glob between areas, add a
`shared:` entry (multi-team co-ownership), or adjust `classify.filetype_rules`.
Then regenerate exactly as in Flow 2 step 4 and commit both files. A routing
PR auto-requests the ops team (which owns `.github/codeowners/`) and the
process team (which owns the generated root `CODEOWNERS`); either review
covers its half.

Semantics worth knowing before you file one:

- **Any one owner approves.** GitHub combines all owners on a CODEOWNERS line
  with OR: for a co-owned file, one approval from any listed team satisfies
  the branch protection. Co-ownership adds review visibility, not extra
  required approvals.
- **The PR UI can mislead.** When a reviewer belongs to more than one owning
  team, GitHub's reviewers panel may drop a codeowner entry that is actually
  satisfied (or pending). Branch protection still enforces correctly - trust
  the merge gate, not the panel.
- **Last match wins.** A more specific rule later in the generated file
  replaces earlier owners for its paths - so a nested override can
  intentionally narrow a parent's co-ownership. Paths whose joint ownership
  must never be dropped are declared under `required_owners:` in
  `areas.yaml`; the CI gate fails any policy change that removes a declared
  owner from a matching path.

## Flow 4: Grant an external contributor area-scoped ownership

Individuals who have earned ownership of an area are attached to the area's
label - never to a copy of its globs - in
`.github/codeowners/external_contributors.yaml`:

```yaml
contributors:
  - name: Jane Doe
    github: janedoe          # -> @janedoe appended to the area's lines
    level: maintainer        # contributor | trusted_contributor | maintainer | core_maintainer
    affiliation: Example Org
    areas: [router]          # area labels from areas.yaml
```

Regeneration (Flow 2 step 4) appends the handle as a co-owner on every line
the area's team owns and rebuilds `CONTRIBUTORS.md`. Commit all three files
(`external_contributors.yaml`, `CODEOWNERS`, `CONTRIBUTORS.md`) together.
`level` is standing metadata shown in `CONTRIBUTORS.md`; routing is granted by
`areas`.

## Reference

- Schema, the last-match-wins model, and the change process:
  `.github/codeowners/README.md`
- The gate and drift check run in `.github/workflows/codeowners.yml` on every
  PR; both must pass before merge.

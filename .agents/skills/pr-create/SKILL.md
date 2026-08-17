---
name: pr-create
description: Opens a pull request from the current branch against ai-dynamo/dynamo, including branch and remote checks, DCO verification, a repository-compliant Conventional Commit title, a complete PR body, push, and gh pr create. Use when a Dynamo change is committed on a branch and the user asks to open, create, submit, or draft the upstream pull request.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - pull-request
    - contribution
---

# Create an Upstream Pull Request

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Open a pull request from the current branch to `ai-dynamo/dynamo`. Do not implement unrelated
changes or create an issue as part of this workflow.

## 1. Inspect the Branch

Run from the repository root:

```bash
git status --short --branch

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
  echo 'Detached HEAD: check out a named branch before opening a pull request.' >&2
  exit 1
fi
if [ "$branch" = 'main' ]; then
  echo 'Open the pull request from a non-main branch.' >&2
  exit 1
fi

git remote -v
git fetch upstream main
git log --oneline upstream/main..HEAD
git diff --stat upstream/main...HEAD
git diff --check upstream/main...HEAD
```

Reuse `$branch` for every later step instead of re-reading the current branch.

Stop and explain the problem when:

- the branch has no commits beyond `upstream/main`;
- tracked changes are uncommitted;
- the branch contains changes unrelated to the requested pull request; or
- `upstream` does not resolve to `ai-dynamo/dynamo`.

Ignore unrelated untracked files. Never add, commit, delete, or include them.

## 2. Verify Every Commit

Dynamo requires a Developer Certificate of Origin trailer on every commit. Verify each branch commit
carries a `Signed-off-by:` trailer matching that commit's author identity:

```bash
while read -r commit; do
  author="$(git show -s --format='%an <%ae>' "$commit")"
  signoffs="$(git show -s --format='%(trailers:key=Signed-off-by,valueonly,unfold)' "$commit")"

  if ! printf '%s\n' "$signoffs" | grep -Fxqi -- "$author"; then
    echo "$commit is missing a Signed-off-by trailer matching $author" >&2
    exit 1
  fi
done < <(git rev-list upstream/main..HEAD)
```

The repository's `scripts/dco_check.py` commit-msg hook only enforces that a `Signed-off-by:` line is
present, so the loop above is the stricter local check; report a mismatch rather than treating hook
success as sufficient. A cryptographic GPG or SSH signature is optional and does not replace DCO
sign-off.

When sign-off is missing, stop and show the appropriate repair command, such as
`git commit --amend --no-edit -s` for the last commit or
`git rebase --signoff upstream/main` for a range; do not rewrite published history without explicit
approval.

Review the actual patch before drafting the pull request:

```bash
git diff upstream/main...HEAD
```

Use the validation results already produced for the branch. Do not claim a check was run unless its
result is known.

## 3. Draft the Title

Read the current rules in `AGENTS.md` and `.github/workflows/lint-pr-title.yaml`. Use:

```text
type(scope): imperative summary
```

Choose one allowed type from the workflow. Typical choices are `docs`, `fix`, `feat`, `test`,
`refactor`, `perf`, `ci`, `build`, or `chore`. Choose a short scope that names the affected area,
such as `skills`, `router`, `frontend`, `vllm`, or `operator`.

The title must describe the whole PR, not merely the last commit. Keep it concise, lowercase after
the colon, and omit a trailing period. Example:

```text
feat(skills): add upstream submission workflows
```

## 4. Draft the Body

Read `.github/pull_request_template.md` before drafting because it can change. The current template
requires these sections:

```markdown
## Overview:

<what changed and why>

## Details:

<implementation details, including validation such as `<command>` and its result>

## Where should the reviewer start?

<the files or entry points to read first>

## Related Issues

- Closes #<issue>
```

Report validation inside `## Details:` rather than adding a separate `Validation` heading, so the
body stays template-compatible while still recording which checks ran.

`## Related Issues` takes exactly one path. Use `Closes #<issue>` when the pull request should close
the issue, `Relates to #<issue>` when it references, depends on, or partially addresses one, or the
template's no-issue confirmation when there is none. Delete the paths that do not apply:

```markdown
## Related Issues

- [x] Confirmed — no related issue
```

Use `Not run (<reason>)` for relevant checks that were not run. Do not include placeholder text such
as `#XXXX`.

Write the final body to a unique private temporary file so quoting and Markdown remain intact:

```bash
pr_body_file="$(mktemp "${TMPDIR:-/tmp}/dynamo-pr-body.XXXXXX")"
chmod 600 "$pr_body_file"
trap 'rm -f "$pr_body_file"' EXIT

cat >"$pr_body_file" <<'EOF'
<final pull request body>
EOF
```

## 5. Push and Open

Confirm GitHub authentication and derive the head repository from the exact `origin` remote rather
than from the current directory's default repository:

```bash
gh auth status

origin_repo="$(gh repo view "$(git remote get-url origin)" \
  --json nameWithOwner,isFork,parent \
  --jq '.nameWithOwner')"
origin_owner="${origin_repo%%/*}"
```

`origin` is a valid head repository when it is `ai-dynamo/dynamo` itself — a feature branch pushed
directly to upstream is legitimate — or a fork whose parent is `ai-dynamo/dynamo`. Stop when it is
neither:

```bash
gh repo view "$(git remote get-url origin)" --json nameWithOwner,isFork,parent \
  --jq 'select(.nameWithOwner == "ai-dynamo/dynamo"
    or (.isFork and .parent.nameWithOwner == "ai-dynamo/dynamo"))
    | .nameWithOwner'
```

Push the captured branch to `origin` without force:

```bash
git push -u origin "$branch"
```

If an open pull request already exists for the branch, report it instead of creating a duplicate:

```bash
gh pr list --repo ai-dynamo/dynamo --head "$origin_owner:$branch" --state open
```

Open the pull request explicitly against upstream `main`, reusing the same derived owner, captured
branch, and quoted title and body file:

```bash
title='<type(scope): summary>'

gh pr create \
  --repo ai-dynamo/dynamo \
  --base main \
  --head "$origin_owner:$branch" \
  --title "$title" \
  --body-file "$pr_body_file"
```

Add `--draft` only when the user requested a draft or the change is intentionally not ready for
review. Leave maintainer edits enabled by default.

## 6. Report

Return the pull request number and URL, title, head/base branches, and validation performed. Remind
the user that full CI may require a maintainer comment of `/ok to test <short-sha>` when applicable.

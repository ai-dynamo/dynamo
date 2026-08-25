---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DCO and Licensing
subtitle: Sign, verify, and repair commits for contribution acceptance
---

Dynamo requires every commit to include a Developer Certificate of Origin (DCO) sign-off. The
sign-off certifies that you have the right to submit the contribution under the project's
[Apache 2.0 License](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE).

## Create a Signed-off Commit

Configure your Git identity with the name and email you intend to use:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

Add the sign-off trailer with `git commit -s`:

```bash
git commit -s -m "fix(component): describe the change"
```

The resulting commit message ends with:

```text
Signed-off-by: Your Name <you@example.com>
```

Use your real name. The trailer's name and email must match the commit author identity.

## Verify the Latest Commit

Display the complete commit message:

```bash
git show -s --format='%B' HEAD
```

Confirm that it contains the expected `Signed-off-by` line.

## Verify Every Commit

Check all commits on your branch relative to upstream `main`:

```bash
git log --format='%h %s%n%(trailers:key=Signed-off-by)' upstream/main..HEAD
```

Each listed commit must have a sign-off trailer.

## Repair the Latest Commit

If only the latest commit is missing its sign-off, amend it:

```bash
git commit --amend --signoff --no-edit
git push --force-with-lease
```

Amending changes the commit SHA. Use `--force-with-lease`, not `--force`, when updating a published
branch.

## Repair Multiple Commits

Start an interactive rebase that includes the unsigned commits:

```bash
git rebase -i upstream/main
```

Mark each unsigned commit as `edit`. For each stop, run:

```bash
git commit --amend --signoff --no-edit
git rebase --continue
```

After the rebase, verify every commit again, then update the remote branch:

```bash
git log --format='%h %s%n%(trailers:key=Signed-off-by)' upstream/main..HEAD
git push --force-with-lease
```

For additional recovery options, see the repository's
[DCO troubleshooting guide](https://github.com/ai-dynamo/dynamo/blob/main/DCO.md).

## Verified Commit Signatures for Fork Pull Requests

DCO sign-off alone is not enough to get automatic trusted-CI approval on a pull request from a
fork. The
[fork CI approval workflow](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/trigger-ci-approval-flow.yml)
posts `/ok to test` for a fork PR's head commit only when every commit in the PR also carries a
cryptographic signature that GitHub reports as `Verified`. This is a separate requirement from the
DCO `Signed-off-by` trailer: DCO sign-off is a text trailer you add yourself, while a verified
signature is cryptographic proof, checked by GitHub, that the commit was made by the key owner.

This requirement applies only to the automatic trusted-CI approval path for fork PRs. It does not
apply to branches pushed directly to this repository, which use the normal internal-contributor CI
path.

If any commit in a fork PR is unsigned or its signature does not show as `Verified` on GitHub, the
workflow does not post `/ok to test`. A maintainer can still approve the PR manually, but signing
your commits lets trusted CI start automatically.

To have your commits verified, configure commit signing (GPG, SSH, or S/MIME) and add the
corresponding public key to your GitHub account, then commit as usual. See GitHub's
[Signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
guide for setup instructions and
[Checking your commit and tag signature verification status](https://docs.github.com/en/authentication/managing-commit-signature-verification/checking-your-commit-and-tag-signature-verification-status)
to confirm a commit will show as `Verified`.

## Licensing

By contributing, you agree that your contribution is licensed under the
[Apache 2.0 License](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE). All participation is
also governed by the
[Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md).

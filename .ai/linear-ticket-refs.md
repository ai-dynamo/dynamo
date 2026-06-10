<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Linear Ticket References in Source Code

Source code in this repository must not carry references to internal Linear
tickets (`DIS-XXXX`, `DYN-XXXX`, or any `<PROJECT>-NNNN` Linear ID). Linear is
NVIDIA-internal, so those IDs are unresolvable for external contributors and
public users.

## What to do when one is flagged

1. **Look for an existing GitHub issue** that already tracks the same work —
   the GitHub issue search on the repo's *Issues* tab is usually enough.
2. **If none exists, file a new GitHub issue** with a public-friendly summary.
   Don't include the Linear URL or ID in the public issue body; assign it to
   yourself when you file it.
3. **Replace the Linear ref in the source** with the GitHub issue number
   (`GH-NNNN` in identifiers and code comments, or `#NNNN` where prose allows).
4. **Carry both refs in the PR description** (`Closes #NNNN`, plus a short
   note that an internal Linear ticket also exists) and cross-link the GitHub
   issue back on the Linear ticket so they stay connected.

## Scope

Applies to the working tree — anything that ships to `main`. Does **not** apply
to commit-message bodies, branch names, or PR titles/descriptions.

Markdown documentation (`*.md`) may reference Linear tickets for internal
context, but prefer to also include the matching GitHub link when one exists.

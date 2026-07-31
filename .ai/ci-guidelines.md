<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI Guidelines

## Validating test fixes in the same PR

Pre-merge CI only runs tests marked `pre_merge`. A fix to a `post_merge`,
`nightly`, `weekly`, or `release` test is otherwise unverified until the PR
has already merged.

When the PR exists to fix a failing or quarantined test:

1. **Promote** the test's scheduling marker to `pytest.mark.pre_merge`,
   replacing the original. If the test is quarantined via
   `@pytest.mark.skip(...)` / `@pytest.mark.skipif(...)`, remove (or narrow)
   the skip in the same commit — otherwise the promotion is a no-op.
2. **Annotate** with a `TODO` next to the marker:
   ```python
   # TODO: revert to pytest.mark.post_merge after pre_merge validation
   # on this PR (see .ai/ci-guidelines.md).
   pytest.mark.pre_merge,
   ```
3. **Verify** the promoted test passed in the PR's CI.
4. **Revert** the marker back (and reinstate any quarantine skip that
   should stay) in a follow-up commit on the same PR before merging. Both
   commits ship in one PR — splitting them loses the signal.

Skip this for unrelated refactors that incidentally touch a non-pre-merge
test (formatting, imports, type hints).

### Reviewer checklist

- Non-`pre_merge` test touched → temporary promotion present with `TODO`.
- A revert commit on this same PR restores the original marker (and any
  quarantine skip) before merge.
- The promoted run was green on the commit immediately before the revert
  (by construction, the SHA being merged no longer carries the promotion).
- Withhold final approval until the revert commit lands — or re-approve
  explicitly after it does. Squash-merging an already-approved PR with the
  promotion still active reintroduces exactly the gap this flow closes.

### Future hardening

A lint check that fails when `pytest.mark.pre_merge` appears next to a
`# TODO: revert to` comment would make step 4 enforceable rather than
process-dependent. Out of scope for this doc; track as a follow-up.

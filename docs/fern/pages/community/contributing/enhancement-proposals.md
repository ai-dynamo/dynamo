---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Enhancement Proposals
subtitle: Take an architectural change from idea to agreed design
---

A Dynamo Enhancement Proposal, or DEP, is how the project reaches agreement on a design before
anyone writes the code. It exists so that a change large enough to be hard to reverse gets argued
while it is still cheap to change.

A DEP is two artifacts, and you need both:

- **The proposal body** lives in [ai-dynamo/enhancements](https://github.com/ai-dynamo/enhancements)
  as a markdown file under `deps/`, reviewed as a pull request. Writing the design as a file lets
  reviewers comment on the specific paragraph they disagree with, and leaves the agreed design
  readable once the discussion is over.
- **A tracking issue** in [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/issues/new?template=dep.yml)
  carries the state. It holds the `dep:` label, records the area that determines who is assigned,
  and is what people watch to follow the proposal. Open it from the DEP template and link the
  proposal pull request from it.

Keep the design in the file, not in the issue. The issue points at the proposal and tracks where it
has got to; the file is the proposal.

## When a DEP Is Required, and When You Might Want One

A DEP is required when a change:

- affects multiple areas
- introduces or modifies a public API
- alters communication plane architecture
- affects backend integration contracts

Nothing smaller is required to have one, but anything may. A DEP is for reaching
agreement on a design before the code exists, and that is worth doing whenever
the cost of being wrong is high: a change to how the project works rather than
what it does, a convention everyone will have to follow, a feature whose shape
several people disagree about. The DEP process itself is a DEP, `0000-dep-process`. A change over 100 core lines needs a
[Contribution Request](contribution-flow.mdx). A CR is permission to build, where a DEP is agreement
on the design. An architectural change needs the DEP, not both: agreeing the design settles whether
the work is wanted.

If you are unsure, open a Contribution Request and ask. Being told a DEP is unnecessary costs you an
afternoon, and writing one nobody needed costs a week.

## Sponsors

Every DEP is sponsored by a [Special Interest Group](../governance/sigs.md). The SIG covering the
affected areas takes it on, acting through one of its Co-Leads or an area Maintainer, and that
person is your Sponsor: they host the design discussion, identify the reviewers the design needs,
agree a timeline with you, call the approval vote once review concludes, and merge the proposal.

So start by taking the idea to the SIG covering the area, before you write the full proposal. Every
area maps to a SIG in [SIGS.md](https://github.com/ai-dynamo/dynamo/blob/main/SIGS.md), so there is
always a group to bring it to. If a change spans several SIGs, the one covering the largest share of
it sponsors, and the others send reviewers.

A design that no SIG will carry is usually a design nobody has agreed is worth building, and finding
that out in a meeting is cheaper than finding it out in review.

## DEP Lifecycle

1. **Draft.** Open the tracking issue from the DEP template, then branch or fork `enhancements` and
   copy the limited or complete template to `deps/NNNN-my-feature.md`, keeping `NNNN` as a
   placeholder: the number is assigned at merge, so do not pick one. Start from the limited template
   and pull sections from the complete one as the design demands them.
2. **Discussion.** Bring it to the SIG covering the affected areas. This is where the design gets
   pressure-tested, and it is much cheaper than discovering the objection in review.
3. **Proposed.** Open the pull request against `enhancements` and link it from the tracking issue.
   Your Sponsor adds the required reviewers.
4. **Review.** Iterate on the pull request, where line-level comments belong. The tracking issue
   holds the open-ended design questions that outlive any single revision.
5. **Approved.** The Maintainers of every area the DEP touches are your reviewers, including areas
   outside the sponsoring SIG. Once their review has concluded, approval is an explicit vote of the
   Core Maintainers rather than lazy consensus, and it takes a two-thirds supermajority. Your
   Sponsor calls it. A decision comes within 30 days of the proposal being marked ready for review.
   The Sponsor then merges it and assigns the DEP its number, and the outcome is recorded on the
   tracking issue.
6. **Implementing, then completed.** Track the work with issues or pull requests linked from the
   DEP. Update its status as the work lands. Those pull requests do not need their own Contribution
   Requests: link the DEP instead. Review is unchanged, so each one still needs its code owners and
   the usual two approvals.

A proposal that stalls is marked deferred. One replaced by a later design is marked superseded, with
a pointer to whatever replaced it.

The states are tracked with `dep:` labels on the tracking issue, and these seven are the whole set: `dep:draft`,
`dep:proposed`, `dep:approved`, `dep:implementing`, `dep:completed`, `dep:deferred`, and
`dep:superseeded`. Discussion and review are activities, not states, and carry no label.

## Change a DEP After Approval

Small corrections, clarified wording, and fixed examples go in as ordinary pull requests to the DEP.

A change that alters what was agreed needs the reviewers who approved it to look again. If you find
yourself explaining why the new version is really the same design, it is not, and it needs another
review round.

## Required Sections

The templates carry the required sections, but the ones that decide whether a DEP is useful are:

- **Motivation.** The problem, stated so that someone who disagrees with your solution still agrees
  the problem is real.
- **Alternate solutions.** What else you considered and why each one loses. A DEP with no
  alternatives reads as a decision already made.
- **Requirements.** What the design has to satisfy, written so a reviewer can check the proposal
  against it.

Write the motivation before the proposal. A design that arrives before its problem is the most
common reason a DEP gets sent back.

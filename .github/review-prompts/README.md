<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codeowner review prompts

The [frontend](frontend.md) and [runtime](runtime.md) prompts describe review
concerns observed repeatedly in comments submitted by members of the corresponding
CODEOWNERS groups. They guide independent reviewers toward concrete defects in the
current change. They do not replace human codeowner review or establish new API
contracts from historical feedback.

The review bot loads both files from one fetched commit on `ai-dynamo/dynamo`'s
`main` branch. Changes to these prompts in a pull request take effect after they
reach `main`; the bot never uses the pull request's copies as review instructions.
Both prompts receive the same reviewed checkout, base and head commits, architecture
context, and complete PR history as the existing reviewers. They return only a JSON
object containing a `findings` array, with `path`, `start_line`, `end_line`, an empty
`test_ids` array, `body`, and `ai_fix` in each finding. Line ranges must contain only
added RIGHT-side lines. An empty findings array means that no new defect qualifies.

Two independent research agents collected the source discussions on September 10,
2026. The frontend sample contains 40 selected PRs and their complete 617 inline
comments, 568 reviews, and 312 general comments. Its recurring concerns include
request-field propagation, streaming state and identity, typed errors, accounting,
transport behavior, and repeated work on request paths. Examples include
[parser configuration reaching its consumer](https://github.com/ai-dynamo/dynamo/pull/12541#discussion_r3866837382),
[streamed item ordering](https://github.com/ai-dynamo/dynamo/pull/11604#discussion_r3909782413),
and [preserving HTTP/2 negotiation](https://github.com/ai-dynamo/dynamo/pull/14173#discussion_r3974898972).

The runtime sample contains 266 selected PRs and their complete 4,443 inline
comments, 3,632 reviews, and 1,951 general comments. Its recurring concerns include
error classification across transports, lifecycle transitions, supported callers'
allocation costs, bindings and configuration, outbound HTTP behavior, and telemetry
consistency. Examples include
[overload becoming a retryable connection error](https://github.com/ai-dynamo/dynamo/pull/14369#discussion_r3963382083),
[explicit shutdown versus object destruction](https://github.com/ai-dynamo/dynamo/pull/13321#discussion_r3834765070),
and [metadata surviving a rejected metric family](https://github.com/ai-dynamo/dynamo/pull/14071#discussion_r3971232936).

These are bounded, deliberately selected samples with overlapping team memberships,
not estimates of group-wide prevalence or independent consensus. Membership was
retrieved at collection time. Some source reviews explicitly disclose AI assistance;
account attribution does not establish unaided human authorship. Replies, corrections,
withdrawn findings, and staged scope constrain how a historical concern should be
applied. The prompts therefore require inspection of the current code and discussion
before reporting a defect, and exclude speculative, pre-existing, duplicate, stylistic,
and missing-test-only findings.

Keep these prompts in plain English and preserve the shared output contract when
updating them. Add a concern when multiple source discussions substantiate it, retain
the concrete behavioral distinction, and avoid turning a historical fix into an
unconditional requirement for unrelated changes.

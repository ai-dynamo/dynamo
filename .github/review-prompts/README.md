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

Use either prompt directly with a coding assistant and identify the change to review,
such as a pull request, commit range, or local diff. The assistant needs access to
the diff and surrounding source. Include existing review discussion when available
so it can avoid repeating reported defects. The prompts describe what to investigate
and what makes a finding actionable; they do not require a particular review tool
or response schema.

The [review bot](https://github.com/NVIDIA-dev/dynamo-review-agent) also uses these
prompts. It loads them from one fetched commit on Dynamo main and adds its own
execution and output instructions. Those integration details belong in the bot.

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

Keep these prompts in plain English and usable directly when updating them. Add a
concern when multiple source discussions substantiate it, retain
the concrete behavioral distinction, and avoid turning a historical fix into an
unconditional requirement for unrelated changes.

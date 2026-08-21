---
name: reviewers-slack
description: Turns the canonical reviewers plan for an ai-dynamo/dynamo pull request linked from a Slack thread into a stable, group-scoped Slack request, without re-notifying existing reviewers, and marks requested groups complete as reviews arrive. Use when invoked as /reviewers-slack with a Slack thread URL, including DM-only test runs before live thread posting is enabled.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - slack
    - pull-request
    - reviewers
---

# Slack PR Reviewers

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Turn the repository's canonical `reviewers` plan into a stable Slack request. Reuse
that skill for CODEOWNERS scope, review coverage, candidate ranking, and path
summaries; own only Slack discovery, identity mapping, rendering, and persisted state
here.

## Deployment guard

Operate in **DM-only test mode** until the user explicitly authorizes live thread
writes in a later request.

- Read the supplied thread when access permits, but never write to it.
- Send the rendered request to the authenticated user's self-DM.
- Render real Slack group and user mentions exactly as they would appear live.
- On repeated runs, edit the existing managed DM message instead of sending another.
- Allow `--pr <GitHub PR URL>` only in test mode when the source thread is inaccessible.
- Do not infer live authorization from the slash-command invocation.

## Input and source thread

Accept:

```text
/reviewers-slack <Slack thread URL>
/reviewers-slack <Slack thread URL> --pr <GitHub PR URL>
```

Parse the permalink with `scripts/reviewers_slack_state.py parse-url`. Require a
message permalink containing a channel ID and parent timestamp; reject channel-only
links.

Read the parent and every reply, then extract
`https://github.com/ai-dynamo/dynamo/pull/<N>` links:

- Use the only open Dynamo PR when exactly one is present.
- If multiple open Dynamo PRs are present, ask which one is in scope.
- If the thread is inaccessible, report the Slack error. Continue only in test mode
  with an explicit `--pr` override.
- Never use a PR override for a live thread write.

Resolve the authenticated Slack profile. In test mode, find the real self-DM
conversation (`D...`) from visible IM conversations; never construct a DM ID.

## Managed message

Read destination history fresh and find a message authored by the authenticated user
containing either marker form:

```text
reviewers-slack:v1 repo=ai-dynamo/dynamo pr=<N> scope=<12 hex chars>
reviewers-slack:v1:repo=ai-dynamo/dynamo:pr=<N>:scope=<12 hex chars>
```

Treat the marker as managed state. Never edit another user's message. Stop if more
than one matching managed message exists.

Write every managed message in English.

## Canonical reviewer plan

Use the `reviewers` skill to produce the canonical plan for the resolved PR. Compute
the scope fingerprint from its sorted `scope` records:

```json
{
  "repo": "ai-dynamo/dynamo",
  "pr": 11946,
  "files": [
    {"path": ".github/workflows/foo.yml", "owners": ["@ai-dynamo/dynamo-ops-codeowners"]}
  ]
}
```

```bash
python3 scripts/reviewers_slack_state.py fingerprint scope.json
```

The fingerprint excludes the head SHA and patch contents. New commits touching the
same owned paths do not churn reviewer assignments; changed paths or CODEOWNERS
mappings do.

Apply the plan according to managed state:

- **No managed message:** omit groups already covered. Use the canonical reviewers
  and path summaries only for uncovered groups. If all groups are covered, write
  nothing.
- **Same fingerprint:** preserve every existing requested block, reviewer list, and
  path wording. Use current plan coverage only to refresh completion styling.
- **Different fingerprint:** remove no-longer-required groups. Preserve existing
  groups and update their paths and completion. Add only newly required, uncovered
  groups using the canonical plan. Replace the message in place.

Never select or rank reviewers independently in this wrapper.

## Resolve Slack mentions

For each new visible group:

- Resolve the same-named Slack user group live and render `<!subteam^S...>`.
- Resolve each canonical GitHub reviewer to `<@U...>` through a verified mapping,
  exact corporate email, or unambiguous exact-name lookup.
- Skip an unresolved candidate and take the next eligible candidate from the
  canonical ranking. Never invent a Slack handle or render a bare GitHub login.
- Select up to three people; keep two when later candidates cannot be resolved or
  lack meaningful evidence.
- Repeat the same person normally under every applicable group.

## Render new request blocks

For each visible group render:

1. the Slack group mention followed once by two or three user mentions, with no label;
2. one or two canonical path summaries directly below it, never more than three.

```text
<!subteam^S123> <@U123> <@U456>
- `components/src/dynamo/frontend/` — Adds separate HTTP and TCP TLS options.
- `components/src/dynamo/common/configuration/` — Propagates TLS settings through runtime configuration.
```

Use 4–8 word implementation summaries. Do not describe review concerns or desired
verification.

Render only Slack mentions for individuals; do not show GitHub logins.

## Persist reviewer mappings

Store the GitHub-to-Slack mapping for visible groups in ranked order:

```json
{"groups":{"dynamo-runtime-codeowners":[{"github":"nnshah1","slack_id":"U123"}]}}
```

Encode it with:

```bash
python3 scripts/reviewers_slack_state.py encode-reviewers reviewers.json
```

Append exactly one seed link as the final nonblank line, separated from the final
block by a blank line. Use a single period as its complete visible label:

```text
[.](https://github.com/ai-dynamo/dynamo/pull/10921#reviewers-slack:v1:repo=ai-dynamo/dynamo:pr=10921:scope=0123456789ab:state=<encoded-state>)
```

Keep the link outside all group blocks and strikethrough. On updates, preserve the
`.` label and replace its target. Remove legacy visible request headings or state
prose when migrating a managed message.

In test mode, begin with `TEST — would update <thread permalink>`.

## Update completion

Coverage comes from the current canonical plan, not the stored reviewer list. A
counted review by any current member of an exact owner team completes its visible
block.

Retain completed requested blocks, strike through every visible line separately, and
append `✅` to the struck-through heading:

```text
~~<!subteam^S123> <@U123> <@U456>~~ ✅
- ~~`components/src/dynamo/frontend/` — Adds separate HTTP and TCP TLS options.~~
```

Remove stale completion styling when current coverage disappears. Completion means a
counted review exists, not necessarily approval. Never add a completed block that was
omitted from the initial request.

## Write safely

Immediately before writing, reread the active Slack outgoing-message formatting rules.

- **First run:** send one managed message containing only uncovered groups to the
  self-DM. Write nothing when all groups are covered.
- **Repeated run:** edit the existing managed message by its real `D...` conversation
  ID and timestamp.
- Never write to the source thread while the deployment guard is active.
- Return the Slack permalink or message ID and state whether the message was created
  or updated.

Use `scripts/reviewers_slack_state.py` for permalink parsing, deterministic scope
hashing, marker parsing, and reviewer-state encoding.

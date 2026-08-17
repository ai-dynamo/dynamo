---
name: issue-create
description: Opens an issue against ai-dynamo/dynamo using the repository's current GitHub issue forms, including bug reports, feature requests, contribution requests, full DEPs, and lightweight DEPs. Use when the user asks to file, create, submit, or draft an upstream Dynamo issue and the title, labels, and body must align with .github/ISSUE_TEMPLATE.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - issue
    - contribution
---

# Create an Upstream Issue

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Create a public issue in `ai-dynamo/dynamo` that matches a current repository issue form. Do not use
a blank issue when `.github/ISSUE_TEMPLATE/config.yml` disables blank issues.

## 1. Read the Current Forms

Treat the repository files as the source of truth; do not rely on remembered fields:

```bash
find .github/ISSUE_TEMPLATE -maxdepth 1 -type f -print
sed -n '1,240p' .github/ISSUE_TEMPLATE/config.yml
sed -n '1,260p' .github/ISSUE_TEMPLATE/*.yml
```

For each form, extract:

- `name`, `description`, title prefix, and automatic labels;
- every field label in body order;
- dropdown options; and
- which fields are required.

If the local checkout might be stale, fetch the templates from upstream before creating the issue:

```bash
git fetch upstream main
git show upstream/main:.github/ISSUE_TEMPLATE/<template>.yml
```

## 2. Select One Template

Choose the form whose stated purpose matches the user's request:

- **Bug Report** for reproducible incorrect behavior.
- **Feature Request** for a desired capability when the user is not proposing to implement it.
- **Contribution Request** when the user intends to implement a bug fix, feature, refactor, or
  performance improvement and wants maintainer agreement.
- **Lightweight DEP** for a smaller design or process proposal that needs Summary, Motivation, and
  Proposal.
- **Dynamo Enhancement Proposal (DEP)** for a substantial feature, architecture change, public API
  change, integration contract, or cross-component design.

If two forms are genuinely plausible and choosing incorrectly would change the review workflow, ask
the user which intent applies. Otherwise choose the best match and state it.

## 3. Build the Issue Exactly From the Form

Use the form's title prefix verbatim and replace any placeholder suffix with a concise description.
Apply all labels declared by the selected template. Do not invent labels or omit automatic labels
when creating the issue through `gh`.

Render every required field as a Markdown `##` heading using the form's visible `label`, in the same
order as the YAML. Include optional fields only when useful. For dropdowns, copy exactly one listed
option.

For example, a bug body follows the current visible labels rather than its internal YAML IDs:

```markdown
## Describe the Bug

<description>

## Steps to Reproduce

1. <step>

## Expected Behavior

<expected result>

## Actual Behavior

<actual result>

## Environment

<environment details>
```

Never submit placeholders such as `TBD`, `Enter bug title`, or `XXXX`. Do not fabricate reproduction
steps, environment data, alternatives, PR size, or affected components. Ask for missing required
facts when they cannot be established from the conversation or repository.

Every issue created by this skill is public, so review the body of any template before submission.
Inspect the title, body, and any referenced artifact for credentials, tokens, private URLs, personal
data, non-public organization or customer names, and sensitive portions of logs, stack traces, or
environment output. Redact only the sensitive portions and keep the remaining diagnostics, because
logs and environment details are required fields on several forms. Include an intentionally public
name only when the user explicitly confirms it is already public and should appear.

For DEPs, use an area exactly from the DEP form's dropdown and apply it as a label in addition to the
form's DEP labels.

Before submission, show or internally review the final template choice, title, labels, and body for
completeness and sensitive information.

## 4. Check for Duplicates

Search open and closed issues using distinctive terms from the proposed title:

```bash
gh issue list --repo ai-dynamo/dynamo --state all --search '<distinctive terms>' --limit 20
gh search issues --repo ai-dynamo/dynamo --match title,body '<distinctive terms>'
```

If a likely duplicate exists, report it and do not create another issue unless the user explicitly
asks to proceed.

## 5. Create the Issue

Write the reviewed body to a unique private temporary file and use non-interactive arguments. Keep
user-derived values in quoted shell variables and build the label list as an array; never pass them
through `eval` or embed them directly in the command:

```bash
gh auth status

body_file="$(mktemp "${TMPDIR:-/tmp}/dynamo-issue-body.XXXXXX")"
chmod 600 "$body_file"
trap 'rm -f "$body_file"' EXIT

# Write the reviewed issue body, then set the reviewed title and template labels.
cat >"$body_file" <<'EOF'
<reviewed issue body>
EOF

title='<template prefix><concise title>'
labels=('<template label>')

args=(--repo ai-dynamo/dynamo --title "$title" --body-file "$body_file")
for label in "${labels[@]}"; do
  args+=(--label "$label")
done

gh issue create "${args[@]}"
```

Quoting the title this way keeps values such as `Fix user's cache` intact, and the array supports any
number of labels.

For a DEP, add the selected area label to `labels`. Prefer explicit title, labels, and body over
`--template` so the submitted issue is deterministic and can be reviewed before the external write.

Create the issue only when the user has asked to file or open it. If the user asked only for a draft,
return the proposed title, labels, and body without calling `gh issue create`.

## 6. Report

Return the issue number and URL, selected template, title, and labels. For Contribution Requests,
mention that implementation should wait for maintainer approval. For DEPs, mention the initial DEP
lifecycle labels applied by the template.

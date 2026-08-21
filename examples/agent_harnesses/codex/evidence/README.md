<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex qualification evidence

Keep one dated Markdown summary per qualification run. Store bulky JSONL and request traces outside git unless a reviewer specifically requests a redacted fixture.

Record:

- Absolute date, operator, Dynamo commit/image, frontend type, model, Codex CLI version, ACP adapter pin, and driver commit.
- Whether the endpoint was stock Dynamo or ThunderAgent. Never describe stock behavior as lifecycle-qualified.
- The exact command with credentials removed.
- Two prompt results, the stable `ready.session_id`, observed tool/result behavior, and targeted validation result.
- Redacted trace counts grouped by `agent_context.session_id` and `input_trigger`.
- Process cleanup status. For ThunderAgent, include the single `session_final` result and independent proof that no program remains.
- Test, lint, and `git diff --check` results.
- Unresolved failures and untested claims.

Do not store API keys, authorization headers, prompts containing proprietary content, home-directory paths, kubeconfigs, or unredacted model responses. Use a unique output path for each run so existing evidence is never truncated accidentally.

The local lifecycle evidence for this branch is in [qualification-2026-08-20.md](qualification-2026-08-20.md).

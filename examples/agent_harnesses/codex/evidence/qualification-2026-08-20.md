<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex local and NScale stock qualification — 2026-08-20

## Scope

This record covers deterministic lifecycle verification and a live stock-Dynamo run of the Codex driver on branch `codex-well-lit-path`. ThunderAgent lifecycle proof is recorded separately after the sequential router arm is deployed.

## Frozen inputs

- Dynamo base: `a6261680a974ca7c74dcf49592a7376d7de99380`
- Pre-polish integration commit: `2180a3278b9239412ab4b260aacbaccc87a6c09d`
- ACP Python client: `agent-client-protocol==0.12.0`
- Codex ACP adapter: `@agentclientprotocol/codex-acp@1.1.14`
- Installed Codex CLI: `codex-cli 0.147.0`
- Installed Node.js: `v25.3.0`
- Installed uv: `0.10.0`

## Result

`uv run --no-project --with agent-client-protocol==0.12.0 python -m unittest discover -v -s .agents/skills/dynamo-agent-harness/scripts -p 'test_drive_harness.py'` passed 12 of 12 tests with the pinned ACP client installed.

The suite proves that the child receives only allowlisted runtime values, the selected Dynamo credential, and fresh Codex configuration; ambient GitHub, Kubernetes, and Codex values are excluded. It also proves that ThunderAgent finalization runs once after ACP process close on normal completion, failed turns, and `KeyboardInterrupt`; stock mode never sends the terminal request; an empty ACP session ID prevents finalization; terminal transport failure fails closed; and a failed turn remains the primary error when finalization also fails.

## NScale stock result

- Backend: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`, `Qwen/Qwen3-0.6B`, 32,768-token context, Dynamo-native `hermes` tool parser and `qwen3` reasoning parser, one exact-count DRA B200 GPU, stock round-robin frontend, and no endpoint authentication.
- Verify session: ACP returned Codex thread/session `01a022d7-8303-7ac0-b411-001e17c1f180`. A requested shell action was denied by the `verify` permission policy and ended `cancelled`; a second no-tool prompt on the same thread returned `CODEX_STOCK_SESSION_OK` with `end_turn`. Explicit close exited zero and emitted no stock lifecycle envelope.
- Isolated act session: ACP returned `01a022d9-5bee-7f12-a090-4385a067f73d` in a disposable `/tmp` workspace. The model invoked a shell tool, but twice reported `pwd` as the file's first line instead of following the requested `sed` command. This proves the tool transport and approval path ran, but the 0.6B smoke model failed the task oracle; no correctness claim is made. The disposable file and directory were removed.
- Cleanup: No `drive_harness.py` or `codex-acp` process remained after either close. The project cluster footprint remained one GPU and two total nodes; no shared taint, label, secret, PVC, or other namespace was touched.

The first pre-parser attempt used a 4,096-token backend and was rejected because Codex's bootstrap exceeded 17k tokens. Raising only the project worker to the model-supported 32,768-token context resolved that protocol gate. The backend independently returned a correctly parsed required tool call, so the remaining act-task miss is attributed to this tiny model's agent quality rather than Dynamo's parser or the ACP transport.

## Remaining live evidence

ThunderAgent qualification still needs one successful terminal record and independent program-state evidence after normal close. Failed-turn and real-signal live cleanup remain valuable follow-up cases beyond the deterministic local coverage. A stronger coding model is required before this path can satisfy a task-correctness gate; Qwen3-0.6B is retained only as a low-cost protocol and lifecycle smoke model.

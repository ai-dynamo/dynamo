<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Unified ProRL → Miles → Dynamo NVFP4 plan

Run ID: `prorl-miles-nvfp4-m6-f049b16-20260826t052909z`

## Frozen boundaries

| Repository | Commit | Responsibility |
|---|---|---|
| Dynamo | `578fbde869` | Serve OpenAI chat, preserve SGLang routed experts and `weight_version`, and accept Miles weight updates. |
| ProRL | `0ee1d48f65` | Own the coding-agent runtime, root/subagent trajectories, patch extraction, and SWE-Gym reward. |
| Miles | `bc1341fd5` | Own rollout scheduling, Qwen3 NVFP4 trainer numerics, routing replay, optimizer state, and Dynamo weight synchronization. |
| SWE-Gym harness | `16dd480cce9b27bf111a362d280881c6def5d2a7` | Generate the official eval script and grade `FAIL_TO_PASS` plus `PASS_TO_PASS`. |
| Miles image | `radixark/miles@sha256:39b5227d8b8cc9997d0777d72b555ea1f1f0dc4c1027649f8c44c6ce88f51859` | Pin Megatron-LM `235952df60` and Transformer Engine 2.17.0. |

Miles carries one isolated two-file overlay: bridge mode propagates an explicitly selected standard-GPT YaRN configuration into the Megatron provider, plus a focused regression test proving YaRN overrides and ordinary bridge preservation. The trainer manifest hash-pins and tests both files before allocating the training process.

Integration-only manifests and validators stay in this directory. The task change does not add policy to Miles or Dynamo, and the SWE-Gym harness is staged as a source archive rather than added to Miles dependencies.

The sequence boundary is 131,072 tokens end to end. Dynamo SGLang uses the Qwen static YaRN override (factor 4.0, original maximum 32,768), explicitly permits the served context to exceed the checkpoint-derived length, and must answer a live 120,000-token OpenAI request before rollouts start. Claude Code reserves its 4,096-token response budget by using a 126,976-token auto-compact window with a 95% trigger against the same 131,072-token model ceiling; Miles sets the same rollout context, Megatron sequence length and standard-GPT YaRN parameters; the rollout validator rejects every returned trace longer than 131,072; and the rollout pod fails if ProRL reports dropping any raw trace at that boundary. Miles trains with TP1/CP4 across four B200s and a per-GPU packing budget of 32,768 tokens, giving the same 131,072-token per-sample admission ceiling while distributing a full-length sequence across all four actors. This is an admission and packing ceiling, not a promise that every trajectory consumes the full window.

## Real training task

Use ProRL's bundled `examples/swegym_slime_grpo/swegym_train_293.jsonl@[0:1]` row:

- Instance: `getmoto__moto-7365`
- Repository: `getmoto/moto` at `7f6c9cb1deafb280fe7fcc7551c38e397f11a706`
- Bug: DynamoDB `ADD` performs float arithmetic instead of `Decimal` arithmetic.
- Regression: `test_update_item_add_float` must change from fail to pass.
- Preservation: `test_update_different_map_elements_in_single_request` must remain passing.
- Reward: ProRL extracts the agent's git patch, applies it to a fresh official task image, runs the commit-pinned SWE-Gym eval script, and returns 1 only when both conditions hold.

The official image baseline was reproduced on-cluster as `1 failed, 1 passed`; the held-out reference patch produced `2 passed`. The model receives the issue text and workspace, not the reference patch.

## Unified path

```text
Miles: one prompt group × four sampled trajectories
  -> slime_bridge.generate_rollout_polar_async
  -> ProRL: four real Claude Code sessions
       -> Agent subagent for bug localization
       -> root Read/Edit/Bash turns against /testbed copy
       -> Dynamo /v1/chat/completions
       -> ThunderAgent -> SGLang TP2/EP2 NVFP4
       <- exact tokens, logprobs, routed experts, weight version
       -> fresh task container + SWE-Gym FAIL_TO_PASS/PASS_TO_PASS grader
  -> variable trace siblings grouped as four trajectory units
  -> Miles routing replay + one Qwen3-30B-A3B NVFP4 optimizer step on 4 B200s
  -> post-step Dynamo weight synchronization
```

## Milestones

- [x] R0: select a bundled ProRL task with a programmatic reward and prove its baseline and reference outcomes in the official image.
- [x] R1: replace the synthetic session-completion task with the pinned SWE-Gym task, four attempts, full coding tools, an Agent subagent, and fresh-runtime grading.
- [ ] R2: prove a live 120,000-token prompt reaches SGLang, then run four live multiturn trajectories and require four distinct root sessions, four child links, real tool turns, aligned logprobs/routed experts/weight versions, non-empty patches, and at least one solved trajectory.
- [ ] R3: rerun four fresh trajectories through Miles and require exact raw-to-train token/version joins, a nonzero finite NVFP4 gradient, `valid_step=true`, checkpoint, and post-step Dynamo sync.
- [ ] R4: independently audit source boundaries and live evidence, fix any failed gate, commit only the minimal integration overlay, and remove the exact M6 Kubernetes resources.

## Stop conditions

Continue through R4. Stop only for lost cluster access, unavailable B200 capacity, exhausted shared storage, or a preserved source/runtime incompatibility.

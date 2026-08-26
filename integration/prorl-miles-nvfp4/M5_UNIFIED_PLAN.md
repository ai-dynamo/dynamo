<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# M5 unified ProRL → Miles → Dynamo plan

Run ID: `prorl-miles-nvfp4-m5-0ee1d48-20260826t033310z`

## Frozen code boundaries

| Repository | Commit | M5 responsibility |
|---|---|---|
| Dynamo | `578fbde869` | Serve chat completions and preserve SGLang `weight_version` alongside routed experts in `nvext.engine_data`. |
| ProRL | `0ee1d48f65` | Run the real coding-agent harness, capture root/subagent calls, and convert traces into real Miles samples. |
| Miles | `bc1341fd5` | Own Qwen NVFP4 trainer numerics, routing replay, optimizer state, and Dynamo worker weight synchronization. |
| Miles runtime | `radixark/miles@sha256:39b5227d8b8cc9997d0777d72b555ea1f1f0dc4c1027649f8c44c6ce88f51859` | Pin Megatron-LM `235952df60` and Transformer Engine 2.17.0 used by the routed-replay and NVFP4 trainer path. |

The deployment overlay stays local. No ProRL policy enters Miles or Dynamo, and no NVFP4 implementation enters ProRL.

The serving baseline is produced by Miles at `bc1341fd5`:

```bash
python3 tools/convert_hf_to_nvfp4.py \
  --model-dir /shared/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --save-dir /shared/test-artifacts/prorl-miles-nvfp4-f7de8eb-20260825t073154z/models/Qwen3-30B-A3B-NVFP4-baseline
```

## One codepath

```text
Miles rollout manager
  -> slime_bridge.rollout.generate_rollout_polar_async
  -> ProRL rollout API
  -> ProRL gateway + real Claude Code Agent subagent runtime
  -> Dynamo /v1/chat/completions
  -> ThunderAgent -> native SGLang TP2/EP2 NVFP4
  <- exact tokens, logprobs, routed experts, weight version
  -> real miles.utils.types.Sample siblings
  -> Miles routing replay + Qwen NVFP4 optimizer on 4 B200s
  -> dedicated Dynamo /engine weight update
```

## Milestones

- [x] U0: clean source commits and cross-repository unit/compatibility tests.
- [x] U1: stage immutable source archives, Dynamo Python overlay, configs, and manifests.
- [x] U2: start a dedicated 2×B200 Dynamo worker and prove chat plus engine-control health.
- [x] U3: run the public ProRL bridge with a real Claude Code `Agent` subagent (the name in pinned Claude Code 2.1.139); require root/child lineage, exact sample alignment, routed experts `[tokens-1,48,8]`, and a weight version.
- [x] U4: run one maintained Qwen3-30B-A3B NVFP4 optimizer step on 4×B200 using those ProRL samples.
- [x] U5: join raw ProRL token hashes to Miles trainer inputs; require nonzero advantage and gradient, `valid_step=true`, checkpoint, and post-step Dynamo sync.
- [x] U6: independent codepath and live-evidence audits; fix and rerun any failed gate.

## Preserved live evidence

- Bridge: explicit root → child → resumed-root Dynamo lineage and three rewarded ProRL/Miles samples; `evidence/bridge/bridge-summary.json` SHA-256 `750517ff9d26476ff6cc3956e56829668cd4b6dc61fa8d1083d92df115b535b5`.
- Trainer: the fresh optimizer rollout independently had explicit root → child → resumed-root Dynamo lineage; its three raw trace token/version pairs joined exactly to optimizer input, routing replay was consumed, advantages and gradient were nonzero (`17.306922912597656`), `valid_step=true`, checkpoint iteration 0 was written, and post-step Dynamo sync passed; `evidence/train/train-summary.json` SHA-256 `4bae9ebe039401f25592bf7350363389d6eff1926811f52694c555a1929e32e1`.
- Post-sync: public chat returned exactly `POST_SYNC_OK`; `evidence/post-sync-chat.json` SHA-256 `7c4f1ee146ae4a6533631a205b500773fce23bda42b80fed14cb188cb74f65af`. The CPU-only `m5-train-evidence` Job completed 1/1 against the preserved artifacts.
- Run root: `/shared/test-artifacts/prorl-miles-nvfp4-m5-0ee1d48-20260826t033310z`.

## Stop conditions

Continue until U5 and U6 pass. Stop only for lost cluster access, unavailable B200 capacity, exhausted shared storage, or a demonstrated source/runtime incompatibility with a preserved reproducer.

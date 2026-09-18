<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Networked frontend-to-vLLM smoke

This experiment exercises lifecycle headers through Dynamo's public HTTP frontend, post-selection KV hint policy, distributed worker endpoint, Python vLLM handler, and vLLM request-completion action.

The experiment-only activation hook is in `lib/bindings/python/rust/llm/entrypoint.rs` and is disabled unless `DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY` is set. The linked policy implementation is under `../policy/`.

The exact configuration, measured results, and limitations are recorded in `dynamo-workflows/dynamo/routing/agentic-kv-management/experiments/continuum-kv-hints/report.md`.

Run the constrained-cache matrix against a pinned experiment image:

```bash
export CONTINUUM_IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-6d7cf575cb-vllm-4091050295

for case in baseline evict retain retain-final; do
  bash experiments/continuum-kv-hints/networked-e2e/run_constrained_cache.sh \
    "$case" "/tmp/continuum-kv-hints/$case"
done
```

`baseline` and `retain` apply cache pressure before resuming the target session. `evict` marks an unpressured resume as final and then replays it, while `retain-final` combines cache pressure, retained reuse, final-session eviction, and a post-final replay.

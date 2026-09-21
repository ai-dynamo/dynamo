<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo vLLM constrained-cache matrix

This experiment exercises lifecycle headers through Dynamo's public HTTP frontend, post-selection KV hint policy, worker endpoint, Python vLLM handler, and vLLM request-completion action.

The experiment-only activation hook is in `lib/bindings/python/rust/llm/entrypoint.rs` and is disabled unless `DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY` is set. The linked policy implementation is under `../policy/`.

Results and limitations are recorded in `dynamo-workflows/dynamo/routing/agentic-kv-management/experiments/continuum-kv-hints/report.md`.

Run the constrained-cache matrix against a pinned experiment image:

```bash
export CONTINUUM_IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-3c5a01b513-vllm-9b6e116be2

for case in baseline evict retain retain-final; do
  bash experiments/continuum-kv-hints/dynamo-vllm-e2e-hints-microbench/run_constrained_cache.sh \
    "$case" "/tmp/continuum-kv-hints/$case"
done
```

`baseline` and `retain` apply cache pressure before resuming the target session. `evict` marks an unpressured resume as final and then replays it, while `retain-final` combines cache pressure, retained reuse, final-session eviction, and a post-final replay.

Run the TTL-expiry and shared-prefix cases separately:

```bash
bash experiments/continuum-kv-hints/dynamo-vllm-e2e-hints-microbench/run_constrained_cache.sh \
  retain-expired /tmp/continuum-kv-hints/retain-expired
bash experiments/continuum-kv-hints/dynamo-vllm-e2e-hints-microbench/run_constrained_cache.sh \
  shared-evict /tmp/continuum-kv-hints/shared-evict
```

`retain-expired` waits two seconds after creating a one-second retention lease before applying pressure. `shared-evict` associates the same prefix with two sessions on one worker, evicts the first session, and then measures cache reuse and indexed lineage for the second.

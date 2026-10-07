<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang-cadence mocker (experiment only, pending ledger D7)

Dynamo's mocker wraps `aisimulate-core`, so the SGLang KV-event cadence fix lives in
ai-dynamo/aisimulate: commit `7fd984e805ae` ("fix: emit SGLang-mode KV events at SGLang's insert
points and cadence", local branch `rupei/sglang-kv-event-cadence` on aisimulate `e45612e1`, not
published). This campaign pins `aisimulate-core = 0.13.0-dev.202609300000000061`, which is
aisimulate `e66bd87deb10` (`.cargo_vcs_info.json`). `aisimulate-core-sglang-cadence.patch` is that
commit backported onto `e66bd87deb10`, with paths relative to the crate root.

Backport notes (conflicts were test-only plus one field):

- The two `admission_undo_*` tests are dropped: they exercise the undo log of aisimulate
  `d1e3cd84`, which `e66bd87deb10` predates. Its checkpointed admission is already present.
- `test_cache_materialization_processes_only_newly_completed_blocks` flushes the event queue
  before it swaps in the test sink, so the first store is not merged into the measured one.
- `merge_into_tail` compares only the DP rank: `KvEvent` has no `tier` field at this version
  (every SGLang-mode event is device tier).

Validation: `cargo test -p aisimulate-core --lib` on the backport passes (1,722 passed,
1 ignored), including the commit's three cadence tests. The patched crate's `src/` equals the
tested tree.

Known gaps (from the commit's review; not changed here):

- `prefill_first_tokens` returns early when `reserve_decode_pages` fails, which skips the
  post-prefill prompt insert (SGLang inserts unconditionally). Rare: it needs KV so tight that
  eviction cannot free one decode slot per running request.
- Removals still come about 4x as often as real SGLang's (about 2.0 vs 0.49 per request): the
  cache evicts pages, SGLang whole radix leaves.

## Build

```bash
SGL=$(lib/e2e-indexer-tools/sglang-cadence-mocker/prepare.sh /tmp/aisim-sglang-cadence)
CARGO_TARGET_DIR=target-sglang-cadence cargo bench --no-run -p dynamo-bench \
  --no-default-features --features mooncake --bench mooncake_bench "$SGL"
git checkout -- Cargo.lock   # cargo rewrote it for the patched source
```

`prepare.sh` checks the `.crate` against the `Cargo.lock` checksum before patching. The default
build (no `--config`) keeps the published crate and the legacy cadence, so both cadences come
from one tree.

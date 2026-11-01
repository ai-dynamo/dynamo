<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hints

KV transfer hints, for both ends of the wire: the hint message and its versioned actions with the runtime-config keys a worker advertises the capability under (`wire`), the source choice a selection makes for a target worker (`transfer`), and a worker's hashed hint identity built once from its runtime config (`keys`). The router's kernel produces hints with this crate; engines, their sidecars, and `components/src/dynamo/*/kv_hints.py` consume them. It is the counterpart of `dynamo-kv-events`, which is what engines publish to the router.

Dependencies: `dynamo-router-protocol` (worker identities, the residency owner key, block-hash newtypes), serde, serde_json, rustc-hash. No tokio, no transport crates, no stage logic: `KvTransferCandidates` and `CacheOwnerKvTransferKeys` stay in `dynamo-kv-indexer` (they carry the indexer's residency routing snapshot), and `resolve_transfer_sources`, the `WorkerConfigLike` adapter, and the attach point on the selected request stay in `dynamo-scheduler`. The layering gate (`scripts/router_layering_check.sh`) checks the dependency set; `scripts/router_public_api_check.sh` keeps the public surface in `public-api.txt`.

`label_key` hashes with `DefaultHasher::new()`: deterministic within one toolchain, not a cross-version contract. Both sides of every comparison are computed in one process today.

Part of the router split (see `lib/router/README.md`).

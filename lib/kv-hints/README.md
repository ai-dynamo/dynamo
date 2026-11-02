<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hints

KV transfer hints, for both ends of the wire: the hint message and its versioned actions with the runtime-config keys a worker advertises the capability under (`wire`), the source choice a selection makes for a target worker (`transfer`), and a worker's hashed hint identity built once from its runtime config (`keys`). The router's kernel produces hints with this crate; engines, their sidecars, and `components/src/dynamo/*/kv_hints.py` consume them. It is the counterpart of `dynamo-kv-events`, which is what engines publish to the router.

Dependencies: `dynamo-router-protocol` (worker identities, block-hash newtypes), `dynamo-kv-indexer` (the transfer candidate vocabulary `KvTransferCandidateSource`, `label_key`, `topology_keys`, re-exported here at their old paths, and the residency owner key behind `KvTransferCandidateSource::CacheOwner`), serde, serde_json, rustc-hash. The edge points one way: a feature crate depends on the stage that produces its types, and the indexer never names this crate. No transport crates, no stage logic of its own: `KvTransferCandidates` and `CacheOwnerKvTransferKeys` stay in `dynamo-kv-indexer` (they carry the indexer's residency routing snapshot), and `resolve_transfer_sources`, the `WorkerConfigLike` adapter, and the attach point on the selected request stay in `dynamo-scheduler`. The scheduler kernel, `dynamo-kv-router`, and the hosts consume this crate; no stage depends on it. The layering gate (`scripts/router_layering_check.sh`) checks the dependency set; `scripts/router_public_api_check.sh` keeps the public surface in `public-api.txt`.

Part of the router split (see `lib/router/README.md`).

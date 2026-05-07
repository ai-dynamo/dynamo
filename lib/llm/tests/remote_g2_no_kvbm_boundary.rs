// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

#[test]
fn remote_g2_router_contract_does_not_use_forbidden_kvbm_boundary() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        manifest_dir.join("../kv-router/src/remote_g2_plan.rs"),
        manifest_dir.join("src/kv_router.rs"),
        manifest_dir.join("src/protocols/common/preprocessor.rs"),
    ];
    let forbidden = [
        "dynamo_kvbm",
        "kvbm::",
        "kvbm-",
        "velo",
        "TransferManager",
        "kvbm_connector",
        "KVBMConnector",
    ];

    for file in files {
        let contents =
            std::fs::read_to_string(&file).unwrap_or_else(|err| panic!("read {file:?}: {err}"));
        for &needle in &forbidden {
            assert!(
                !contents.contains(needle),
                "{file:?} contains forbidden v1 remote G2 dependency marker {needle:?}"
            );
        }
    }
}

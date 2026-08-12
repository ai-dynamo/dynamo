// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const BLOCK_POOL: &str = include_str!("../cache/vllm_block_pool.rs");
const VLLM_BACKEND: &str = include_str!("vllm_backend.rs");

fn assert_absent(label: &str, source: &str, forbidden: &[&str]) {
    let source = source.to_ascii_lowercase();
    let found = forbidden
        .iter()
        .copied()
        .filter(|token| source.contains(token))
        .collect::<Vec<_>>();

    assert!(
        found.is_empty(),
        "{label} crosses its source firewall with: {}",
        found.join(", ")
    );
}

#[test]
fn vllm_block_pool_is_a_leaf_core() {
    assert_absent(
        "vllm_block_pool.rs",
        BLOCK_POOL,
        &[
            "kvbm",
            "offload",
            "scheduler",
            "crate::",
            "dynamo_kv_router",
            "kveventpublishers",
            "kvcacheevent",
            "rawkvevent",
            "kv_cache_trace",
        ],
    );
}

#[test]
fn vllm_backend_has_no_kvbm_or_legacy_g1_dependencies() {
    assert_absent(
        "vllm_backend.rs",
        VLLM_BACKEND,
        &[
            "kvbm",
            "moveblock",
            "positionallineagehash",
            "plh",
            "g1acquire",
            "g1backend",
            "offload",
            "swapin",
            "swap_in",
            "immutableblock",
        ],
    );
}

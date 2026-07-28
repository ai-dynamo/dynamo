// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const BLOCK_POOL: &str = include_str!("../cache/vllm_block_pool.rs");
const VLLM_BACKEND: &str = include_str!("vllm_backend.rs");

/// Every KV cache group implementation, by the label a failure should name.
const VLLM_GROUPS: [(&str, &str); 2] = [
    ("vllm_groups.rs", include_str!("vllm_groups.rs")),
    (
        "vllm_groups/full_attention.rs",
        include_str!("vllm_groups/full_attention.rs"),
    ),
];

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
    for (label, source) in [("vllm_backend.rs", VLLM_BACKEND)]
        .into_iter()
        .chain(VLLM_GROUPS)
    {
        assert_absent(
            label,
            source,
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
}

/// A group implementation must never reach the typed event sink: `KvCacheEvent`
/// has no group field, so the router would read another group's hash as an
/// attention block. Only the coordinator publishes.
#[test]
fn groups_never_publish_kv_events() {
    for (label, source) in VLLM_GROUPS {
        assert_absent(
            label,
            source,
            &["kveventpublishers", "kvcacheevent", "rawkvevent", "publish"],
        );
    }
}

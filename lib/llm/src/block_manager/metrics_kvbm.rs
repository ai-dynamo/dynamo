// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::metrics::MetricsRegistry;
use prometheus::IntCounter;

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    // number of offload requests
    pub offload_requests: IntCounter,

    // number of blocks offloaded from device to host
    pub offload_blocks_d2h: IntCounter,

    // number of onboard requests
    pub onboard_requests: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of save kv layer requests
    pub save_kv_layer_requests: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,
}

impl KvbmMetrics {
    pub fn new(mr: &dyn MetricsRegistry) -> Self {
        let offload_requests = mr
            .create_intcounter("offload_requests", "The number of offload requests", &[])
            .unwrap();
        let offload_blocks_d2h = mr
            .create_intcounter(
                "offload_blocks_d2h",
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let onboard_requests = mr
            .create_intcounter("onboard_requests", "The number of onboard requests", &[])
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                "onboard_blocks_h2d",
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                "onboard_blocks_d2d",
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let save_kv_layer_requests = mr
            .create_intcounter(
                "save_kv_layer_requests",
                "The number of save kv layer requests",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter("matched_tokens", "The number of matched tokens", &[])
            .unwrap();
        Self {
            offload_requests,
            offload_blocks_d2h,
            onboard_requests,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            save_kv_layer_requests,
            matched_tokens,
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the KV Event Consolidator

use serde::{Deserialize, Serialize};

/// Configuration for the KV Event Consolidator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEventConsolidatorConfig {
    /// ZMQ endpoint to subscribe to vLLM events (e.g., "tcp://localhost:5557")
    pub vllm_event_endpoint: String,

    /// ZMQ endpoint to publish consolidated events (e.g., "tcp://*:5558")
    pub consolidated_event_endpoint: String,
}

impl Default for KvEventConsolidatorConfig {
    fn default() -> Self {
        Self {
            vllm_event_endpoint: "tcp://localhost:5557".to_string(),
            consolidated_event_endpoint: "tcp://*:5558".to_string(),
        }
    }
}

impl KvEventConsolidatorConfig {
    pub fn new(vllm_event_endpoint: String, consolidated_event_endpoint: String) -> Self {
        Self {
            vllm_event_endpoint,
            consolidated_event_endpoint,
        }
    }
}

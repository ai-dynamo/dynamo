// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the KV Event Consolidator

use serde::{Deserialize, Serialize};

use super::tracker::EventSource;

/// Configuration for the KV Event Consolidator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEventConsolidatorConfig {
    /// ZMQ endpoint to subscribe to engine events (vLLM or TensorRT-LLM) (e.g., "tcp://localhost:5557")
    pub engine_event_endpoint: String,

    /// ZMQ endpoint to publish consolidated events (e.g., "tcp://*:5558")
    /// Worker-side publishers subscribe to this and add worker_id before forwarding to NATS
    pub consolidated_event_endpoint: String,

    /// Engine source for events (vLLM or TensorRT-LLM)
    pub engine_source: EventSource,

    /// Data parallel rank for this consolidator instance.
    /// Used by KVBM events (which don't carry dp_rank) to match the engine's dp_rank.
    /// With attention DP, each rank has its own consolidator.
    pub data_parallel_rank: Option<i32>,
}

impl Default for KvEventConsolidatorConfig {
    fn default() -> Self {
        Self {
            engine_event_endpoint: "tcp://localhost:5557".to_string(),
            consolidated_event_endpoint: "tcp://*:5558".to_string(),
            engine_source: EventSource::Vllm,
            data_parallel_rank: None,
        }
    }
}

impl KvEventConsolidatorConfig {
    pub fn new(
        engine_event_endpoint: String,
        consolidated_event_endpoint: String,
        engine_source: EventSource,
    ) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source,
            data_parallel_rank: None,
        }
    }

    /// Create config for vLLM
    pub fn new_vllm(engine_event_endpoint: String, consolidated_event_endpoint: String) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source: EventSource::Vllm,
            data_parallel_rank: None,
        }
    }

    /// Create config for TensorRT-LLM
    pub fn new_trtllm(engine_event_endpoint: String, consolidated_event_endpoint: String) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source: EventSource::Trtllm,
            data_parallel_rank: None,
        }
    }

    /// Set the data parallel rank for this consolidator instance
    pub fn with_data_parallel_rank(mut self, dp_rank: Option<i32>) -> Self {
        self.data_parallel_rank = dp_rank;
        self
    }
}

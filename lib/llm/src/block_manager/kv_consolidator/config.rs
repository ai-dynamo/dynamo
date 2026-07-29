// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the KV Event Consolidator

use serde::{Deserialize, Serialize};

use super::tracker::EventSource;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvEventConsolidationMode {
    #[default]
    Dedup,
    Passthrough,
}

impl KvEventConsolidationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dedup => "dedup",
            Self::Passthrough => "passthrough",
        }
    }
}

impl std::str::FromStr for KvEventConsolidationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dedup" => Ok(Self::Dedup),
            "passthrough" => Ok(Self::Passthrough),
            _ => Err(format!("Unknown KV event consolidator mode: {s}")),
        }
    }
}

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

    /// How the consolidator should process store/remove events.
    pub mode: KvEventConsolidationMode,
}

impl Default for KvEventConsolidatorConfig {
    fn default() -> Self {
        Self {
            engine_event_endpoint: "tcp://localhost:5557".to_string(),
            consolidated_event_endpoint: "tcp://*:5558".to_string(),
            engine_source: EventSource::Vllm,
            mode: KvEventConsolidationMode::Dedup,
        }
    }
}

impl KvEventConsolidatorConfig {
    pub fn new(
        engine_event_endpoint: String,
        consolidated_event_endpoint: String,
        engine_source: EventSource,
        mode: KvEventConsolidationMode,
    ) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source,
            mode,
        }
    }

    /// Create config for vLLM
    pub fn new_vllm(
        engine_event_endpoint: String,
        consolidated_event_endpoint: String,
        mode: KvEventConsolidationMode,
    ) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source: EventSource::Vllm,
            mode,
        }
    }

    /// Create config for TensorRT-LLM
    pub fn new_trtllm(
        engine_event_endpoint: String,
        consolidated_event_endpoint: String,
        mode: KvEventConsolidationMode,
    ) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source: EventSource::Trtllm,
            mode,
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the KV Event Consolidator

use serde::{Deserialize, Serialize};

use super::tracker::EventSource;

/// Output transport type for consolidated events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsolidatorOutputTransport {
    /// Publish consolidated events to ZMQ (for vLLM)
    Zmq,
    /// Publish consolidated events to NATS (for TensorRT-LLM)
    Nats,
}

/// Configuration for the KV Event Consolidator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEventConsolidatorConfig {
    /// ZMQ endpoint to subscribe to engine events (vLLM or TensorRT-LLM) (e.g., "tcp://localhost:5557")
    pub engine_event_endpoint: String,

    /// Output endpoint/identifier for consolidated events
    /// For ZMQ: endpoint to publish (e.g., "tcp://*:5558")
    /// For NATS: placeholder (not used, kept for API compatibility)
    pub consolidated_event_endpoint: String,

    /// Engine source for events (vLLM or TensorRT-LLM)
    /// The output transport is inferred from this: Vllm -> Zmq, Trtllm -> Nats
    pub engine_source: EventSource,
}

impl Default for KvEventConsolidatorConfig {
    fn default() -> Self {
        Self {
            engine_event_endpoint: "tcp://localhost:5557".to_string(),
            consolidated_event_endpoint: "tcp://*:5558".to_string(),
            engine_source: EventSource::Vllm,
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
        }
    }

    /// Infer the output transport from the engine source
    /// - Vllm -> Zmq (vLLM router expects ZMQ)
    /// - Trtllm -> Nats (TensorRT-LLM router expects NATS RouterEvents)
    ///
    /// Note: EventSource::Kvbm is never used as engine_source in KvEventConsolidatorConfig.
    /// KVBM events are sent directly to the consolidator handle.
    pub fn output_transport(&self) -> ConsolidatorOutputTransport {
        match self.engine_source {
            EventSource::Vllm => ConsolidatorOutputTransport::Zmq,
            EventSource::Trtllm => ConsolidatorOutputTransport::Nats,
            EventSource::Kvbm => {
                // This case should never be reached - KvEventConsolidatorConfig is only created
                // via new_vllm() or new_trtllm(), which set engine_source to Vllm or Trtllm.
                unreachable!("KvEventConsolidatorConfig should never have engine_source=Kvbm")
            }
        }
    }

    /// Create config for vLLM (ZMQ output)
    pub fn new_vllm(engine_event_endpoint: String, consolidated_event_endpoint: String) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint,
            engine_source: EventSource::Vllm,
        }
    }

    /// Create config for TensorRT-LLM (NATS output)
    pub fn new_trtllm(engine_event_endpoint: String) -> Self {
        Self {
            engine_event_endpoint,
            consolidated_event_endpoint: "nats".to_string(), // Placeholder
            engine_source: EventSource::Trtllm,
        }
    }
}

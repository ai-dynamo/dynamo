// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_tokens::Token;

pub use aisimulate_core::engine::{G2Scope, NativeHostOffloadConfig};

/// Trait for publishing KV cache events.
/// This abstracts the runtime dependency so mocker components can remain generic.
pub trait KvCacheEventSink: Send + Sync {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()>;

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        _storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.publish(event)
    }

    /// Publishes events that share one source visibility boundary.
    ///
    /// Implementations that do not have a native batch representation retain
    /// singleton delivery semantics by default.
    fn publish_batch_with_storage_tiers(
        &self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> anyhow::Result<()> {
        let mut first_error = None;
        for (event, storage_tier) in events {
            if let Err(error) = self.publish_with_storage_tier(event, storage_tier) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Raw KV event payload used by transport-specific publishers such as the
/// vLLM-native ZMQ event stream.
#[derive(Debug, Clone)]
pub struct RawKvEvent {
    pub event: KvCacheEvent,
    pub block_token_ids: Option<Vec<Vec<u32>>>,
    pub storage_tier: StorageTier,
}

/// Trait for publishing transport-specific raw KV event payloads.
pub trait RawKvEventSink: Send + Sync {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()>;

    /// Publishes raw events that share one source visibility boundary.
    ///
    /// Implementations that do not have a native batch representation retain
    /// singleton delivery semantics by default.
    fn publish_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        let mut first_error = None;
        for event in events {
            if let Err(error) = self.publish(event) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Shared KV event publisher bundle used by schedulers and KV managers.
#[derive(Clone, Default)]
pub struct KvEventPublishers {
    event_sink: Option<Arc<dyn KvCacheEventSink>>,
    raw_sink: Option<Arc<dyn RawKvEventSink>>,
}

impl KvEventPublishers {
    pub fn new(
        event_sink: Option<Arc<dyn KvCacheEventSink>>,
        raw_sink: Option<Arc<dyn RawKvEventSink>>,
    ) -> Self {
        Self {
            event_sink,
            raw_sink,
        }
    }

    pub fn raw_enabled(&self) -> bool {
        self.raw_sink.is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.event_sink.is_none() && self.raw_sink.is_none()
    }

    pub fn publish(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.publish_with_storage_tier(event, block_token_ids, StorageTier::Device)
    }

    pub fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish_with_storage_tier(event.clone(), storage_tier)?;
        }

        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish(RawKvEvent {
                event,
                block_token_ids: block_token_ids.map(|token_ids| token_ids.to_vec()),
                storage_tier,
            })?;
        }

        Ok(())
    }

    /// Publishes normal KV events without also forwarding them to a raw sink.
    ///
    /// Deferred live-scheduler forwarding uses this to preserve its source
    /// visibility boundary for normal and raw sinks independently.
    pub(crate) fn publish_event_sink_batch_only(
        &self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> anyhow::Result<()> {
        if let Some(sink) = self.event_sink.as_ref() {
            sink.publish_batch_with_storage_tiers(events)?;
        }
        Ok(())
    }

    /// Publishes raw events as one source visibility boundary.
    pub(crate) fn publish_raw_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        if let Some(sink) = self.raw_sink.as_ref() {
            sink.publish_batch(events)?;
        }
        Ok(())
    }
}

/// Replay-neutral per-pass metrics shared by offline and Live Mocker drivers.
pub use aisimulate_core::replay::ForwardPassSnapshot;

/// Trait for publishing forward pass metrics snapshots.
/// This abstracts the FPM publishing pipeline so mocker schedulers remain generic.
pub trait FpmSink: Send + Sync {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()>;
}

/// Optional FPM sink used by schedulers.
/// Wraps `Option<Arc<dyn FpmSink>>` for ergonomic passing and no-op default behavior.
#[derive(Clone, Default)]
pub struct FpmPublisher {
    sink: Option<Arc<dyn FpmSink>>,
}

impl FpmPublisher {
    pub fn new(sink: Option<Arc<dyn FpmSink>>) -> Self {
        Self { sink }
    }

    pub fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()> {
        if let Some(sink) = &self.sink {
            sink.publish(snapshot)?;
        }
        Ok(())
    }
}

/// Replay-owned request DTO shared by Dynamo's compatibility and Live Mocker
/// drivers. The type remains provider-neutral; Dynamo-specific metadata is
/// interpreted only by Dynamo adapters.
pub use aisimulate_core::replay::DirectRequest;

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_id: Option<Token>,
    /// Terminal flag: the request's lifecycle has ended. Replay drivers free
    /// resources and advance/notify on this.
    pub completed: bool,
    /// Set with `completed` when the request was rejected without ever running
    /// (its footprint exceeds the whole KV pool); drivers free/advance but
    /// exclude it from token/latency/throughput stats.
    #[serde(default)]
    pub rejected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_delay_ms: Option<f64>,
    /// Prompt tokens served from KV cache at admission (scheduler truth,
    /// post-eviction). Set once, on the request's first output signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
}

pub use crate::config::MockerConfig;
pub use aisimulate_core::engine::{
    Backend as EngineType, PreemptionMode, TransferTimingMode as KvTransferTimingMode, WorkerType,
};

/// Configuration for reasoning/thinking token output in the mocker.
///
/// When set, the mocker wraps the first portion of each response in thinking
/// boundary tokens: `[start_token, random..., end_token, random...]`.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReasoningConfig {
    pub start_thinking_token_id: u32,
    pub end_thinking_token_id: u32,
    #[validate(range(min = 0.0, max = 1.0))]
    pub thinking_ratio: f64,
}

impl ReasoningConfig {
    /// Number of thinking tokens (including start/end boundaries) for a given osl.
    /// Returns 0 if osl < 2 (thinking disabled). Otherwise clamps to [2, osl].
    pub fn num_thinking_tokens(&self, max_output_tokens: usize) -> usize {
        if max_output_tokens < 2 {
            return 0;
        }
        let raw = (max_output_tokens as f64 * self.thinking_ratio).floor() as usize;
        if raw == 0 {
            return 0;
        }
        raw.max(2).min(max_output_tokens)
    }

    /// Number of response tokens after the thinking block.
    pub fn num_response_tokens(&self, max_output_tokens: usize) -> usize {
        max_output_tokens.saturating_sub(self.num_thinking_tokens(max_output_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;
    #[derive(Default)]
    struct FailingRawSink {
        attempts: Mutex<Vec<u64>>,
    }

    impl RawKvEventSink for FailingRawSink {
        fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
            self.attempts.lock().unwrap().push(event.event.event_id);
            if event.event.event_id == 2 {
                anyhow::bail!("injected raw sink failure");
            }
            Ok(())
        }
    }

    #[test]
    fn raw_sink_batch_fallback_attempts_later_events_after_failure() {
        let sink = FailingRawSink::default();
        let error = sink
            .publish_batch(
                (1..=3)
                    .map(|event_id| RawKvEvent {
                        event: KvCacheEvent {
                            event_id,
                            data: dynamo_kv_router::protocols::KvCacheEventData::Cleared,
                            dp_rank: 0,
                        },
                        block_token_ids: None,
                        storage_tier: StorageTier::Device,
                    })
                    .collect(),
            )
            .unwrap_err();

        assert_eq!(error.to_string(), "injected raw sink failure");
        assert_eq!(*sink.attempts.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn direct_request_priorities_are_backward_compatible() {
        let legacy = json!({
            "tokens": [1, 2],
            "max_output_tokens": 3,
            "uuid": null,
            "dp_rank": 0,
            "arrival_timestamp_ms": null
        });
        let request: DirectRequest = serde_json::from_value(legacy).unwrap();
        assert_eq!(request.priority, 0);
        assert_eq!(request.strict_priority, 0);
        assert_eq!(request.router_priorities(), (0.0, 0));

        let rendered = serde_json::to_value(&request).unwrap();
        assert!(rendered.get("priority").is_none());
        assert!(rendered.get("strict_priority").is_none());
    }

    #[test]
    fn direct_request_derives_router_priorities() {
        let negative: DirectRequest = serde_json::from_value(json!({
            "tokens": [1],
            "max_output_tokens": 1,
            "uuid": null,
            "dp_rank": 0,
            "arrival_timestamp_ms": null,
            "priority": -7,
            "strict_priority": 4
        }))
        .unwrap();
        assert_eq!(negative.router_priorities(), (0.0, 4));

        let positive = DirectRequest {
            priority: 9,
            strict_priority: 5,
            ..negative
        };
        assert_eq!(positive.router_priorities(), (9.0, 5));
        let rendered = serde_json::to_value(&positive).unwrap();
        assert_eq!(rendered["priority"], 9);
        assert_eq!(rendered["strict_priority"], 5);
    }
}

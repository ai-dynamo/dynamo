// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Event Consolidator queue and tier normalization.
//!
//! The consolidator no longer deduplicates or rewrites block identities across
//! sources. Instead it:
//! - receives raw engine events
//! - normalizes their storage tier metadata
//! - merges them into a single ordered stream
//! - republishes them unchanged to the downstream Dynamo publisher

use dynamo_kv_router::protocols::StorageTier as RouterStorageTier;
use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent};

/// Event source for KV cache events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EventSource {
    /// Events from vLLM worker (G1/GPU)
    Vllm,
    /// Events from TensorRT-LLM worker (G1/GPU)
    Trtllm,
    /// Events from KVBM
    Kvbm,
}

impl std::str::FromStr for EventSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vllm" | "VLLM" | "GPU" => Ok(EventSource::Vllm),
            "trtllm" | "TRTLLM" | "TensorRT-LLM" => Ok(EventSource::Trtllm),
            "kvbm" | "KVBM" => Ok(EventSource::Kvbm),
            _ => Err(format!("Unknown event source: {}", s)),
        }
    }
}

impl EventSource {
    /// Convert to string representation.
    pub fn to_str(&self) -> &'static str {
        match self {
            EventSource::Vllm => "vllm",
            EventSource::Trtllm => "trtllm",
            EventSource::Kvbm => "kvbm",
        }
    }

    fn default_storage_tier(self) -> StorageTier {
        match self {
            EventSource::Vllm | EventSource::Trtllm | EventSource::Kvbm => StorageTier::Device,
        }
    }
}

/// Storage tier carried in the raw KV event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Device,
    HostPinned,
    Disk,
}

impl StorageTier {
    /// Parse from KV event `medium` strings.
    pub fn from_vllm_medium(s: &str) -> Option<Self> {
        RouterStorageTier::from_kv_medium(s).map(Into::into)
    }

    /// Convert to KV event `medium` strings.
    pub fn to_vllm_medium(self) -> &'static str {
        match self {
            StorageTier::Device => "GPU",
            StorageTier::HostPinned => "CPU_TIER1",
            StorageTier::Disk => "CPU_TIER2",
        }
    }

    /// Convert to string representation.
    pub fn to_str(self) -> &'static str {
        match self {
            StorageTier::Device => "device",
            StorageTier::HostPinned => "host_pinned",
            StorageTier::Disk => "disk",
        }
    }
}

impl From<RouterStorageTier> for StorageTier {
    fn from(value: RouterStorageTier) -> Self {
        match value {
            RouterStorageTier::Device => Self::Device,
            RouterStorageTier::HostPinned => Self::HostPinned,
            RouterStorageTier::Disk | RouterStorageTier::External => Self::Disk,
        }
    }
}

/// Legacy type alias for backward compatibility.
#[deprecated(note = "Use StorageTier instead")]
pub type StorageMedium = StorageTier;

/// One batch in the consolidated outbound stream.
#[derive(Debug, Clone)]
pub struct ConsolidatedEventBatch {
    pub events: Vec<RawKvEvent>,
    pub data_parallel_rank: Option<i32>,
}

/// Simple event queue used by the consolidator.
#[derive(Debug, Default)]
pub struct CacheStatusTracker {
    event_queue: Vec<ConsolidatedEventBatch>,
}

impl CacheStatusTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a single-block STORE event from KVBM.
    #[allow(clippy::too_many_arguments)]
    pub fn handle_store(
        &mut self,
        block_hash: String,
        source: EventSource,
        token_ids: Vec<u32>,
        parent_hash: Option<String>,
        block_size: usize,
        lora_name: Option<String>,
        tier: Option<StorageTier>,
        data_parallel_rank: Option<i32>,
    ) -> bool {
        let parsed_block_hash = match parse_hash(&block_hash, "block_hash") {
            Some(hash) => hash,
            None => return false,
        };

        let parsed_parent_hash = match parent_hash {
            Some(parent_hash) => match parse_hash(&parent_hash, "parent_hash") {
                Some(hash) => Some(BlockHashValue::Unsigned(hash)),
                None => return false,
            },
            None => None,
        };

        let tier = tier.unwrap_or_else(|| source.default_storage_tier());
        self.enqueue_batch(
            vec![RawKvEvent::BlockStored {
                block_hashes: vec![BlockHashValue::Unsigned(parsed_block_hash)],
                parent_block_hash: parsed_parent_hash,
                token_ids,
                block_size,
                medium: Some(tier.to_vllm_medium().to_string()),
                lora_name,
                block_mm_infos: None,
                is_eagle: None,
            }],
            data_parallel_rank,
        );
        true
    }

    /// Queue a single-block REMOVE event from KVBM.
    pub fn handle_remove(
        &mut self,
        block_hash: &str,
        source: EventSource,
        tier: Option<StorageTier>,
        data_parallel_rank: Option<i32>,
    ) -> bool {
        let parsed_block_hash = match parse_hash(block_hash, "block_hash") {
            Some(hash) => hash,
            None => return false,
        };

        let tier = tier.unwrap_or_else(|| source.default_storage_tier());
        self.enqueue_batch(
            vec![RawKvEvent::BlockRemoved {
                block_hashes: vec![BlockHashValue::Unsigned(parsed_block_hash)],
                medium: Some(tier.to_vllm_medium().to_string()),
            }],
            data_parallel_rank,
        );
        true
    }

    /// Queue a batch of raw engine events after normalizing tier metadata.
    pub fn handle_batch(
        &mut self,
        events: Vec<RawKvEvent>,
        data_parallel_rank: Option<i32>,
        source: EventSource,
    ) {
        let default_tier = source.default_storage_tier();
        let normalized_events = events
            .into_iter()
            .map(|event| normalize_event_tier(event, default_tier))
            .collect();
        self.enqueue_batch(normalized_events, data_parallel_rank);
    }

    /// Queue a global clear event.
    pub fn handle_clear_all(&mut self) {
        self.enqueue_batch(vec![RawKvEvent::AllBlocksCleared], None);
    }

    /// Drain all queued batches.
    pub fn drain_events(&mut self) -> Vec<ConsolidatedEventBatch> {
        let events = std::mem::take(&mut self.event_queue);
        if !events.is_empty() {
            tracing::debug!(
                "Draining {} pending consolidated batch(es) for publishing",
                events.len()
            );
        }
        events
    }

    #[cfg(test)]
    fn num_pending_batches(&self) -> usize {
        self.event_queue.len()
    }

    fn enqueue_batch(&mut self, events: Vec<RawKvEvent>, data_parallel_rank: Option<i32>) {
        if events.is_empty() {
            return;
        }

        self.event_queue.push(ConsolidatedEventBatch {
            events,
            data_parallel_rank,
        });
    }
}

fn normalize_event_tier(event: RawKvEvent, default_tier: StorageTier) -> RawKvEvent {
    match event {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            medium,
            lora_name,
            block_mm_infos,
            is_eagle,
        } => RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            medium: Some(medium.unwrap_or_else(|| default_tier.to_vllm_medium().to_string())),
            lora_name,
            block_mm_infos,
            is_eagle,
        },
        RawKvEvent::BlockRemoved {
            block_hashes,
            medium,
        } => RawKvEvent::BlockRemoved {
            block_hashes,
            medium: Some(medium.unwrap_or_else(|| default_tier.to_vllm_medium().to_string())),
        },
        RawKvEvent::AllBlocksCleared => RawKvEvent::AllBlocksCleared,
    }
}

fn parse_hash(hash: &str, field_name: &str) -> Option<u64> {
    match hash.parse::<u64>() {
        Ok(hash) => Some(hash),
        Err(error) => {
            tracing::warn!(
                field_name = field_name,
                hash = hash,
                error = %error,
                "Failed to parse KV event hash"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kvbm_store_event_gets_medium_and_preserves_payload() {
        let mut tracker = CacheStatusTracker::new();

        assert!(tracker.handle_store(
            "100".to_string(),
            EventSource::Kvbm,
            vec![1, 2, 3, 4],
            Some("90".to_string()),
            4,
            Some("adapter".to_string()),
            Some(StorageTier::HostPinned),
            Some(7),
        ));

        assert_eq!(tracker.num_pending_batches(), 1);
        let batches = tracker.drain_events();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].data_parallel_rank, Some(7));

        match batches[0].events[0].clone() {
            RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                medium,
                lora_name,
                ..
            } => {
                assert_eq!(block_hashes.len(), 1);
                assert_eq!(block_hashes[0].into_u64(), 100);
                assert_eq!(parent_block_hash.map(BlockHashValue::into_u64), Some(90));
                assert_eq!(token_ids, vec![1, 2, 3, 4]);
                assert_eq!(block_size, 4);
                assert_eq!(medium.as_deref(), Some("CPU_TIER1"));
                assert_eq!(lora_name.as_deref(), Some("adapter"));
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
    }

    #[test]
    fn kvbm_remove_event_gets_medium() {
        let mut tracker = CacheStatusTracker::new();

        assert!(tracker.handle_remove("123", EventSource::Kvbm, Some(StorageTier::Disk), None,));

        let batches = tracker.drain_events();
        assert_eq!(batches.len(), 1);
        match batches[0].events[0].clone() {
            RawKvEvent::BlockRemoved {
                block_hashes,
                medium,
            } => {
                assert_eq!(block_hashes.len(), 1);
                assert_eq!(block_hashes[0].into_u64(), 123);
                assert_eq!(medium.as_deref(), Some("CPU_TIER2"));
            }
            other => panic!("expected BlockRemoved, got {other:?}"),
        }
    }

    #[test]
    fn raw_engine_batches_are_preserved_and_default_tier_is_added() {
        let mut tracker = CacheStatusTracker::new();

        tracker.handle_batch(
            vec![RawKvEvent::BlockStored {
                block_hashes: vec![BlockHashValue::Unsigned(11), BlockHashValue::Unsigned(12)],
                parent_block_hash: Some(BlockHashValue::Unsigned(10)),
                token_ids: vec![1, 2, 3, 4],
                block_size: 2,
                medium: None,
                lora_name: Some("adapter".to_string()),
                block_mm_infos: None,
                is_eagle: Some(true),
            }],
            Some(3),
            EventSource::Trtllm,
        );

        let batches = tracker.drain_events();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].data_parallel_rank, Some(3));

        match batches[0].events[0].clone() {
            RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                medium,
                lora_name,
                is_eagle,
                ..
            } => {
                assert_eq!(block_hashes.len(), 2);
                assert_eq!(block_hashes[0].into_u64(), 11);
                assert_eq!(block_hashes[1].into_u64(), 12);
                assert_eq!(parent_block_hash.map(BlockHashValue::into_u64), Some(10));
                assert_eq!(token_ids, vec![1, 2, 3, 4]);
                assert_eq!(block_size, 2);
                assert_eq!(medium.as_deref(), Some("GPU"));
                assert_eq!(lora_name.as_deref(), Some("adapter"));
                assert_eq!(is_eagle, Some(true));
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
    }

    #[test]
    fn explicit_medium_is_preserved() {
        let mut tracker = CacheStatusTracker::new();

        tracker.handle_batch(
            vec![RawKvEvent::BlockRemoved {
                block_hashes: vec![BlockHashValue::Unsigned(22)],
                medium: Some("CPU_TIER2".to_string()),
            }],
            None,
            EventSource::Vllm,
        );

        let batches = tracker.drain_events();
        assert_eq!(batches.len(), 1);

        match batches[0].events[0].clone() {
            RawKvEvent::BlockRemoved {
                block_hashes,
                medium,
            } => {
                assert_eq!(block_hashes[0].into_u64(), 22);
                assert_eq!(medium.as_deref(), Some("CPU_TIER2"));
            }
            other => panic!("expected BlockRemoved, got {other:?}"),
        }
    }

    #[test]
    fn clear_all_queues_single_batch() {
        let mut tracker = CacheStatusTracker::new();

        tracker.handle_clear_all();

        let batches = tracker.drain_events();
        assert_eq!(batches.len(), 1);
        assert!(matches!(
            batches[0].events.as_slice(),
            [RawKvEvent::AllBlocksCleared]
        ));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalized events emitted by the host-offload core.

use serde::{Deserialize, Serialize};

/// Framework-neutral identity for one logical KV block.
///
/// The token digest is scoped by its logical parent and KV group so identical
/// token chunks in different prefixes or groups do not alias. It does not
/// expose framework block ids or physical cache locations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct HostBlockKey {
    pub group_index: u32,
    pub parent: Option<[u8; 32]>,
    pub token_digest: [u8; 32],
}

impl HostBlockKey {
    pub fn new(group_index: u32, parent: Option<[u8; 32]>, token_digest: [u8; 32]) -> Self {
        Self {
            group_index,
            parent,
            token_digest,
        }
    }
}

/// Opaque location of a source or destination block in G1.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct G1Location(u64);

impl G1Location {
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Opaque id used to correlate queued and completed transfers.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TransferId(u64);

impl TransferId {
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Direction of a simulated host transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferDirection {
    G1ToG2,
    G2ToG1,
}

/// Scheduler action that had to wait for a pending G1-to-G2 store.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFenceReason {
    SourceReuse,
    RequestPreemption,
}

/// Stable event vocabulary shared by offline replay and real-engine parity
/// traces. Times are relative simulation times in milliseconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum HostOffloadEvent {
    StorePrepared {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
        source: G1Location,
        bytes: usize,
    },
    StoreQueued {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
        source: G1Location,
        bytes: usize,
    },
    StoreCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
    },
    LoadQueued {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
        destination: G1Location,
        bytes: usize,
    },
    LoadCompleted {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
    },
    G2Evicted {
        at_ms: f64,
        key: HostBlockKey,
    },
    CapacityRetry {
        at_ms: f64,
        key: HostBlockKey,
    },
    SourceFenced {
        at_ms: f64,
        transfer_id: TransferId,
        key: HostBlockKey,
        source: G1Location,
        reason: SourceFenceReason,
    },
}

impl HostOffloadEvent {
    pub fn at_ms(&self) -> f64 {
        match self {
            Self::StorePrepared { at_ms, .. }
            | Self::StoreQueued { at_ms, .. }
            | Self::StoreCompleted { at_ms, .. }
            | Self::LoadQueued { at_ms, .. }
            | Self::LoadCompleted { at_ms, .. }
            | Self::G2Evicted { at_ms, .. }
            | Self::CapacityRetry { at_ms, .. }
            | Self::SourceFenced { at_ms, .. } => *at_ms,
        }
    }

    pub fn key(&self) -> HostBlockKey {
        match self {
            Self::StorePrepared { key, .. }
            | Self::StoreQueued { key, .. }
            | Self::StoreCompleted { key, .. }
            | Self::LoadQueued { key, .. }
            | Self::LoadCompleted { key, .. }
            | Self::G2Evicted { key, .. }
            | Self::CapacityRetry { key, .. }
            | Self::SourceFenced { key, .. } => *key,
        }
    }

    pub fn transfer_direction(&self) -> Option<TransferDirection> {
        match self {
            Self::StorePrepared { .. }
            | Self::StoreQueued { .. }
            | Self::StoreCompleted { .. }
            | Self::SourceFenced { .. } => Some(TransferDirection::G1ToG2),
            Self::LoadQueued { .. } | Self::LoadCompleted { .. } => Some(TransferDirection::G2ToG1),
            Self::G2Evicted { .. } | Self::CapacityRetry { .. } => None,
        }
    }
}

/// Destination for normalized host-offload events.
pub trait HostOffloadEventSink: Send + Sync {
    fn record(&self, event: &HostOffloadEvent);
}

/// Default sink for callers that do not request host-offload tracing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopHostOffloadEventSink;

impl HostOffloadEventSink for NoopHostOffloadEventSink {
    fn record(&self, _event: &HostOffloadEvent) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn key(group_index: u32) -> HostBlockKey {
        HostBlockKey::new(group_index, Some([1; 32]), [2; 32])
    }

    #[test]
    fn logical_key_includes_group_and_parent() {
        let base = key(0);
        let another_group = key(1);
        let another_parent = HostBlockKey::new(0, Some([3; 32]), [2; 32]);

        assert_ne!(base, another_group);
        assert_ne!(base, another_parent);
    }

    #[test]
    fn typed_events_report_time_key_and_direction() {
        let event = HostOffloadEvent::StoreQueued {
            at_ms: 12.5,
            transfer_id: TransferId::new(7),
            key: key(0),
            source: G1Location::new(11),
            bytes: 4096,
        };

        assert_eq!(event.at_ms(), 12.5);
        assert_eq!(event.key(), key(0));
        assert_eq!(event.transfer_direction(), Some(TransferDirection::G1ToG2));

        let eviction = HostOffloadEvent::G2Evicted {
            at_ms: 15.0,
            key: key(0),
        };
        assert_eq!(eviction.transfer_direction(), None);
    }

    #[test]
    fn event_serialization_has_stable_discriminator_and_opaque_ids() {
        let event = HostOffloadEvent::LoadQueued {
            at_ms: 2.0,
            transfer_id: TransferId::new(4),
            key: key(0),
            destination: G1Location::new(9),
            bytes: 1024,
        };
        let value = serde_json::to_value(event).unwrap();

        assert_eq!(value["event"], "load_queued");
        assert_eq!(value["transfer_id"], 4);
        assert_eq!(value["destination"], 9);
        assert_eq!(value["bytes"], 1024);
    }

    #[test]
    fn noop_sink_is_object_safe() {
        let sink: Arc<dyn HostOffloadEventSink> = Arc::new(NoopHostOffloadEventSink);
        let event = HostOffloadEvent::CapacityRetry {
            at_ms: 3.0,
            key: key(0),
        };

        sink.record(&event);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Builder for lightweight connector metadata exchanged with inference workers.
#[derive(Debug, Default)]
pub struct ConnectorMetadataBuilder {
    protocol_version: u32,
    slot_creates: Vec<SlotCreate>,
    forward_events: BTreeMap<String, BTreeMap<u32, String>>,
    slot_deletes: Vec<String>,
    transactions: Vec<Transaction>,
}

impl ConnectorMetadataBuilder {
    pub fn new(protocol_version: u32) -> Self {
        Self {
            protocol_version,
            slot_creates: Vec::new(),
            forward_events: BTreeMap::new(),
            slot_deletes: Vec::new(),
            transactions: Vec::new(),
        }
    }

    pub fn queue_slot_create(
        &mut self,
        request_id: impl Into<String>,
        create_event: impl Into<String>,
    ) {
        self.slot_creates.push(SlotCreate {
            request_id: request_id.into(),
            create_event: create_event.into(),
        });
    }

    pub fn queue_forward_event(
        &mut self,
        request_id: impl Into<String>,
        rank: u32,
        event: impl Into<String>,
    ) {
        let entry = self.forward_events.entry(request_id.into()).or_default();
        entry.insert(rank, event.into());
    }

    pub fn queue_slot_delete(&mut self, request_id: impl Into<String>) {
        self.slot_deletes.push(request_id.into());
    }

    pub fn queue_transaction(&mut self, request_id: impl Into<String>, payload: Value) {
        self.transactions.push(Transaction {
            request_id: request_id.into(),
            payload,
        });
    }

    pub fn reset(&mut self) {
        self.slot_creates.clear();
        self.forward_events.clear();
        self.slot_deletes.clear();
        self.transactions.clear();
    }

    pub fn build_bytes(&mut self) -> Result<Vec<u8>> {
        if self.slot_creates.is_empty()
            && self.forward_events.is_empty()
            && self.slot_deletes.is_empty()
            && self.transactions.is_empty()
        {
            return Ok(Vec::new());
        }

        let forward_events: Vec<ForwardEvent> = self
            .forward_events
            .iter()
            .map(|(request_id, events)| ForwardEvent {
                request_id: request_id.clone(),
                worker_events: events
                    .iter()
                    .map(|(rank, event)| WorkerEvent {
                        rank: *rank,
                        event: event.clone(),
                    })
                    .collect(),
            })
            .collect();

        let payload = ConnectorMetadataPayload {
            protocol_version: self.protocol_version,
            slot_creates: &self.slot_creates,
            forward_events,
            slot_deletes: self
                .slot_deletes
                .iter()
                .map(|request_id| SlotDelete {
                    request_id: request_id.clone(),
                })
                .collect(),
            transactions: &self.transactions,
        };

        let bytes = serde_json::to_vec(&payload).unwrap_or_else(|err| {
            panic!("failed to serialize connector metadata payload: {err}");
        });

        self.reset();

        Ok(bytes)
    }
}

#[derive(Serialize)]
struct ConnectorMetadataPayload<'a> {
    protocol_version: u32,
    slot_creates: &'a [SlotCreate],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    forward_events: Vec<ForwardEvent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    slot_deletes: Vec<SlotDelete>,
    #[serde(skip_serializing_if = "slice_is_empty")]
    transactions: &'a [Transaction],
}

#[derive(Debug, Serialize)]
struct SlotCreate {
    request_id: String,
    create_event: String,
}

#[derive(Serialize)]
struct ForwardEvent {
    request_id: String,
    worker_events: Vec<WorkerEvent>,
}

#[derive(Serialize)]
struct WorkerEvent {
    rank: u32,
    event: String,
}

#[derive(Serialize)]
struct SlotDelete {
    request_id: String,
}

#[derive(Debug, Serialize)]
struct Transaction {
    request_id: String,
    payload: Value,
}

fn slice_is_empty<T>(slice: &[T]) -> bool {
    slice.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_and_clears_builder() {
        let mut builder = ConnectorMetadataBuilder::new(1);
        assert!(builder.build_bytes().unwrap().is_empty());

        builder.queue_slot_create("req-a", "evt-create");
        builder.queue_forward_event("req-a", 0, "evt-forward-0");
        builder.queue_forward_event("req-a", 1, "evt-forward-1");
        builder.queue_slot_delete("req-b");
        builder.queue_transaction("req-x", Value::String("payload".into()));

        let bytes = builder.build_bytes().unwrap();
        assert!(!bytes.is_empty());

        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["protocol_version"], 1);
        assert_eq!(value["slot_creates"][0]["request_id"], "req-a");
        assert_eq!(
            value["forward_events"][0]["worker_events"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(value["slot_deletes"][0]["request_id"], "req-b");
        assert_eq!(value["transactions"][0]["request_id"], "req-x");

        assert!(builder.build_bytes().unwrap().is_empty());
    }
}

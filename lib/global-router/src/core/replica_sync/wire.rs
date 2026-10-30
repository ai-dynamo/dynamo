// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversions between the wire messages and the core's event types.
//!
//! Each wire event carries the [`RoutingPartitionId`] of the core partition
//! it belongs to, since the core's host-mode replica channels are per
//! partition. The core's `token_sequence` is never carried: a pool has no
//! KV blocks the core could match, so [`from_wire_event`] always books
//! without it.

use std::time::Duration;

use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, PrefillLoadHint, WorkerWithDpRank,
};
use dynamo_kv_router::services::selection::AffinityBindingEvent;
use thiserror::Error;

use super::proto;
use super::proto::active_sequence_event::Data;

/// Why a wire message does not convert to a core event.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WireError {
    /// `ActiveSequenceEvent.data` is unset: the sender used a variant this
    /// receiver does not know, or sent none.
    #[error("active sequence event {request_id} has no data")]
    MissingData { request_id: String },
    /// An `AddRequest` gives `expected_prefill_duration_ms` without
    /// `initial_effective_prefill_tokens`, which a [`PrefillLoadHint`] requires.
    #[error("active sequence event {request_id} has a prefill duration but no prefill tokens")]
    IncompleteHint { request_id: String },
    /// A string id the core requires is empty.
    #[error("{field} is empty")]
    EmptyId { field: &'static str },
}

/// The wire form of a core event in `partition`.
pub fn to_wire_event(
    partition: &RoutingPartitionId,
    event: &ActiveSequenceEvent,
) -> proto::ActiveSequenceEvent {
    let data = match &event.data {
        ActiveSequenceEventData::AddRequest {
            token_sequence: _,
            track_prefill_tokens,
            expected_output_tokens,
            prefill_load_hint,
        } => Data::Add(proto::AddRequest {
            expected_output_tokens: *expected_output_tokens,
            track_prefill_tokens: *track_prefill_tokens,
            initial_effective_prefill_tokens: prefill_load_hint
                .map(|hint| hint.initial_effective_prefill_tokens as u64),
            expected_prefill_duration_ms: prefill_load_hint
                .and_then(|hint| hint.expected_prefill_duration)
                .map(|duration| duration.as_millis() as u64),
        }),
        ActiveSequenceEventData::MarkPrefillCompleted => {
            Data::MarkPrefillCompleted(proto::Empty {})
        }
        ActiveSequenceEventData::Free => Data::Free(proto::Empty {}),
    };
    proto::ActiveSequenceEvent {
        model_name: partition.model_name.clone(),
        routing_group: partition.routing_group.clone(),
        request_id: event.request_id.clone(),
        worker_id: event.worker.worker_id,
        dp_rank: event.worker.dp_rank,
        router_id: event.router_id,
        lora_name: event.lora_name.clone(),
        data: Some(data),
    }
}

/// The core event a wire event carries, and the partition it belongs to.
pub fn from_wire_event(
    event: proto::ActiveSequenceEvent,
) -> Result<(RoutingPartitionId, ActiveSequenceEvent), WireError> {
    let proto::ActiveSequenceEvent {
        model_name,
        routing_group,
        request_id,
        worker_id,
        dp_rank,
        router_id,
        lora_name,
        data,
    } = event;
    let partition = partition(model_name, routing_group)?;
    if request_id.is_empty() {
        return Err(WireError::EmptyId {
            field: "request_id",
        });
    }
    let data = match data {
        Some(Data::Add(add)) => {
            let prefill_load_hint = match (
                add.initial_effective_prefill_tokens,
                add.expected_prefill_duration_ms,
            ) {
                (None, None) => None,
                (None, Some(_)) => return Err(WireError::IncompleteHint { request_id }),
                (Some(tokens), duration_ms) => Some(PrefillLoadHint {
                    initial_effective_prefill_tokens: tokens as usize,
                    expected_prefill_duration: duration_ms.map(Duration::from_millis),
                }),
            };
            ActiveSequenceEventData::AddRequest {
                token_sequence: None,
                track_prefill_tokens: add.track_prefill_tokens,
                expected_output_tokens: add.expected_output_tokens,
                prefill_load_hint,
            }
        }
        Some(Data::MarkPrefillCompleted(_)) => ActiveSequenceEventData::MarkPrefillCompleted,
        Some(Data::Free(_)) => ActiveSequenceEventData::Free,
        None => return Err(WireError::MissingData { request_id }),
    };
    Ok((
        partition,
        ActiveSequenceEvent {
            request_id,
            worker: WorkerWithDpRank { worker_id, dp_rank },
            data,
            router_id,
            lora_name,
        },
    ))
}

/// The wire form of a session binding.
pub fn to_wire_binding(binding: &AffinityBindingEvent) -> proto::AffinityBinding {
    proto::AffinityBinding {
        model_name: binding.partition.model_name.clone(),
        routing_group: binding.partition.routing_group.clone(),
        session_id: binding.session_id.clone(),
        worker_id: binding.worker_id,
        dp_rank: binding.dp_rank,
        sequence: binding.sequence,
        writer_id: binding.writer_id,
    }
}

/// The session binding a wire binding carries.
pub fn from_wire_binding(
    binding: proto::AffinityBinding,
) -> Result<AffinityBindingEvent, WireError> {
    let proto::AffinityBinding {
        model_name,
        routing_group,
        session_id,
        worker_id,
        dp_rank,
        sequence,
        writer_id,
    } = binding;
    let partition = partition(model_name, routing_group)?;
    if session_id.is_empty() {
        return Err(WireError::EmptyId {
            field: "session_id",
        });
    }
    Ok(AffinityBindingEvent {
        partition,
        session_id,
        worker_id,
        dp_rank,
        sequence,
        writer_id,
    })
}

fn partition(model_name: String, routing_group: String) -> Result<RoutingPartitionId, WireError> {
    if model_name.is_empty() {
        return Err(WireError::EmptyId {
            field: "model_name",
        });
    }
    if routing_group.is_empty() {
        return Err(WireError::EmptyId {
            field: "routing_group",
        });
    }
    Ok(RoutingPartitionId::new(model_name, routing_group))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::MAX_REPLICA_BATCH_EVENTS;

    fn partition() -> RoutingPartitionId {
        RoutingPartitionId::new("Qwen/Qwen3-0.6B", "global")
    }

    fn event(data: ActiveSequenceEventData, lora_name: Option<&str>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: "req-1".to_string(),
            worker: WorkerWithDpRank {
                worker_id: 42,
                dp_rank: 0,
            },
            data,
            router_id: 7,
            lora_name: lora_name.map(str::to_string),
        }
    }

    fn add(prefill_load_hint: Option<PrefillLoadHint>) -> ActiveSequenceEventData {
        ActiveSequenceEventData::AddRequest {
            token_sequence: Some(vec![1, 2, 3]),
            track_prefill_tokens: true,
            expected_output_tokens: Some(128),
            prefill_load_hint,
        }
    }

    fn round_trip(data: ActiveSequenceEventData, lora_name: Option<&str>) -> ActiveSequenceEvent {
        let original = event(data, lora_name);
        let wire = to_wire_event(&partition(), &original);
        let (got_partition, got) = from_wire_event(wire).unwrap();
        assert_eq!(got_partition, partition());
        assert_eq!(got.request_id, original.request_id);
        assert_eq!(got.worker, original.worker);
        assert_eq!(got.router_id, original.router_id);
        assert_eq!(got.lora_name, original.lora_name);
        got
    }

    #[test]
    fn add_request_round_trips_without_hint_or_lora() {
        let got = round_trip(add(None), None);
        match got.data {
            ActiveSequenceEventData::AddRequest {
                token_sequence,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
            } => {
                assert_eq!(token_sequence, None, "token_sequence is never carried");
                assert!(track_prefill_tokens);
                assert_eq!(expected_output_tokens, Some(128));
                assert_eq!(prefill_load_hint, None);
            }
            other => panic!("expected AddRequest, got {other:?}"),
        }
    }

    #[test]
    fn add_request_round_trips_with_hint_and_lora() {
        let hint = PrefillLoadHint {
            initial_effective_prefill_tokens: 512,
            expected_prefill_duration: Some(Duration::from_millis(250)),
        };
        let got = round_trip(add(Some(hint)), Some("adapter-a"));
        match got.data {
            ActiveSequenceEventData::AddRequest {
                prefill_load_hint, ..
            } => assert_eq!(prefill_load_hint, Some(hint)),
            other => panic!("expected AddRequest, got {other:?}"),
        }
    }

    #[test]
    fn add_request_round_trips_hint_without_duration() {
        let hint = PrefillLoadHint {
            initial_effective_prefill_tokens: 64,
            expected_prefill_duration: None,
        };
        let got = round_trip(add(Some(hint)), None);
        match got.data {
            ActiveSequenceEventData::AddRequest {
                prefill_load_hint, ..
            } => assert_eq!(prefill_load_hint, Some(hint)),
            other => panic!("expected AddRequest, got {other:?}"),
        }
    }

    #[test]
    fn mark_prefill_completed_round_trips() {
        let got = round_trip(ActiveSequenceEventData::MarkPrefillCompleted, None);
        assert!(matches!(
            got.data,
            ActiveSequenceEventData::MarkPrefillCompleted
        ));
    }

    #[test]
    fn free_round_trips() {
        let got = round_trip(ActiveSequenceEventData::Free, Some("adapter-b"));
        assert!(matches!(got.data, ActiveSequenceEventData::Free));
    }

    #[test]
    fn event_without_data_is_rejected() {
        let mut wire = to_wire_event(&partition(), &event(ActiveSequenceEventData::Free, None));
        wire.data = None;
        assert_eq!(
            from_wire_event(wire).unwrap_err(),
            WireError::MissingData {
                request_id: "req-1".to_string()
            }
        );
    }

    #[test]
    fn event_with_duration_but_no_tokens_is_rejected() {
        let mut wire = to_wire_event(&partition(), &event(add(None), None));
        if let Some(Data::Add(add)) = wire.data.as_mut() {
            add.expected_prefill_duration_ms = Some(10);
        }
        assert_eq!(
            from_wire_event(wire).unwrap_err(),
            WireError::IncompleteHint {
                request_id: "req-1".to_string()
            }
        );
    }

    #[test]
    fn event_with_empty_ids_is_rejected() {
        let wire = to_wire_event(&partition(), &event(ActiveSequenceEventData::Free, None));
        let mut no_model = wire.clone();
        no_model.model_name.clear();
        assert_eq!(
            from_wire_event(no_model).unwrap_err(),
            WireError::EmptyId {
                field: "model_name"
            }
        );
        let mut no_request = wire;
        no_request.request_id.clear();
        assert_eq!(
            from_wire_event(no_request).unwrap_err(),
            WireError::EmptyId {
                field: "request_id"
            }
        );
    }

    fn binding(dp_rank: Option<u32>) -> AffinityBindingEvent {
        AffinityBindingEvent {
            partition: partition(),
            session_id: "session-1".to_string(),
            worker_id: 42,
            dp_rank,
            sequence: 9,
            writer_id: 7,
        }
    }

    fn assert_binding_round_trips(original: AffinityBindingEvent) {
        let got = from_wire_binding(to_wire_binding(&original)).unwrap();
        assert_eq!(got.partition, original.partition);
        assert_eq!(got.session_id, original.session_id);
        assert_eq!(got.worker_id, original.worker_id);
        assert_eq!(got.dp_rank, original.dp_rank);
        assert_eq!(got.sequence, original.sequence);
        assert_eq!(got.writer_id, original.writer_id);
    }

    #[test]
    fn binding_round_trips_with_dp_rank() {
        assert_binding_round_trips(binding(Some(3)));
    }

    #[test]
    fn binding_round_trips_without_dp_rank() {
        assert_binding_round_trips(binding(None));
    }

    #[test]
    fn binding_with_empty_session_is_rejected() {
        let mut wire = to_wire_binding(&binding(None));
        wire.session_id.clear();
        assert_eq!(
            from_wire_binding(wire).unwrap_err(),
            WireError::EmptyId {
                field: "session_id"
            }
        );
    }

    #[test]
    fn batch_holds_the_core_batch_limit() {
        let events = (0..MAX_REPLICA_BATCH_EVENTS)
            .map(|_| to_wire_event(&partition(), &event(ActiveSequenceEventData::Free, None)))
            .collect();
        let batch = proto::ReplicaEventBatch {
            events,
            bindings: Vec::new(),
        };
        assert_eq!(batch.events.len(), MAX_REPLICA_BATCH_EVENTS);
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversions between the wire messages and the core's session bindings.
//!
//! Each wire binding carries the [`RoutingPartitionId`] of the core
//! partition it belongs to, which the core's session table is keyed by.

use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::services::selection::AffinityBindingEvent;
use thiserror::Error;

use super::proto;

/// Why a wire message does not convert to a core binding.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WireError {
    /// A string id the core requires is empty.
    #[error("{field} is empty")]
    EmptyId { field: &'static str },
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

    fn partition() -> RoutingPartitionId {
        RoutingPartitionId::new("Qwen/Qwen3-0.6B", "global")
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
    fn binding_with_empty_partition_ids_is_rejected() {
        let mut no_model = to_wire_binding(&binding(None));
        no_model.model_name.clear();
        assert_eq!(
            from_wire_binding(no_model).unwrap_err(),
            WireError::EmptyId {
                field: "model_name"
            }
        );
        let mut no_group = to_wire_binding(&binding(None));
        no_group.routing_group.clear();
        assert_eq!(
            from_wire_binding(no_group).unwrap_err(),
            WireError::EmptyId {
                field: "routing_group"
            }
        );
    }
}

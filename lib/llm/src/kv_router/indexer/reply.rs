// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-typed wrappers for the indexer's request-plane responses.
//!
//! The response DTOs live in the router crates, which do not depend on `dynamo-runtime`.
//! These `#[serde(transparent)]` newtypes carry the runtime `MaybeError` contract (the
//! `Error(message)` variant is the error state) so the responses can flow through
//! `PushRouter`, `AddressedPushRouter`, and `Ingress` with the same wire encoding as the
//! bare enums.

use dynamo_kv_indexer::host::{
    IndexerQueryResponse, IndexerRecordRoutingDecisionResponse, WorkerKvQueryResponse,
};
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use serde::{Deserialize, Serialize};

/// [`WorkerKvQueryResponse`] as a request-plane stream item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerKvQueryReply(pub WorkerKvQueryResponse);

impl From<WorkerKvQueryResponse> for WorkerKvQueryReply {
    fn from(response: WorkerKvQueryResponse) -> Self {
        Self(response)
    }
}

impl From<WorkerKvQueryReply> for WorkerKvQueryResponse {
    fn from(reply: WorkerKvQueryReply) -> Self {
        reply.0
    }
}

impl MaybeError for WorkerKvQueryReply {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self(WorkerKvQueryResponse::Error(err.to_string()))
    }

    fn err(&self) -> Option<DynamoError> {
        match &self.0 {
            WorkerKvQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

/// [`IndexerQueryResponse`] as a request-plane stream item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IndexerQueryReply(pub IndexerQueryResponse);

impl From<IndexerQueryResponse> for IndexerQueryReply {
    fn from(response: IndexerQueryResponse) -> Self {
        Self(response)
    }
}

impl From<IndexerQueryReply> for IndexerQueryResponse {
    fn from(reply: IndexerQueryReply) -> Self {
        reply.0
    }
}

impl MaybeError for IndexerQueryReply {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self(IndexerQueryResponse::Error(err.to_string()))
    }

    fn err(&self) -> Option<DynamoError> {
        match &self.0 {
            IndexerQueryResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

/// [`IndexerRecordRoutingDecisionResponse`] as a request-plane stream item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IndexerRecordRoutingDecisionReply(pub IndexerRecordRoutingDecisionResponse);

impl From<IndexerRecordRoutingDecisionResponse> for IndexerRecordRoutingDecisionReply {
    fn from(response: IndexerRecordRoutingDecisionResponse) -> Self {
        Self(response)
    }
}

impl From<IndexerRecordRoutingDecisionReply> for IndexerRecordRoutingDecisionResponse {
    fn from(reply: IndexerRecordRoutingDecisionReply) -> Self {
        reply.0
    }
}

impl MaybeError for IndexerRecordRoutingDecisionReply {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self(IndexerRecordRoutingDecisionResponse::Error(err.to_string()))
    }

    fn err(&self) -> Option<DynamoError> {
        match &self.0 {
            IndexerRecordRoutingDecisionResponse::Error(msg) => Some(DynamoError::msg(msg.clone())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_indexer::host::WireTieredMatchDetails;
    use dynamo_router_protocol::residency::ResetScope;

    #[test]
    fn worker_kv_query_reply_is_wire_transparent() {
        let response = WorkerKvQueryResponse::TreeDump {
            events: Vec::new(),
            last_event_id: 42,
            reset_scope: ResetScope::All,
        };
        let bare = serde_json::to_vec(&response).unwrap();
        let wrapped = serde_json::to_vec(&WorkerKvQueryReply(response)).unwrap();
        assert_eq!(bare, wrapped);
        let decoded: WorkerKvQueryReply = serde_json::from_slice(&bare).unwrap();
        assert!(matches!(
            decoded.0,
            WorkerKvQueryResponse::TreeDump {
                last_event_id: 42,
                ..
            }
        ));
    }

    #[test]
    fn indexer_replies_are_wire_transparent() {
        let query = IndexerQueryResponse::TieredScores(WireTieredMatchDetails::default());
        assert_eq!(
            serde_json::to_vec(&query).unwrap(),
            serde_json::to_vec(&IndexerQueryReply(query.clone())).unwrap()
        );
        let record = IndexerRecordRoutingDecisionResponse::Recorded;
        assert_eq!(
            serde_json::to_vec(&record).unwrap(),
            serde_json::to_vec(&IndexerRecordRoutingDecisionReply(record.clone())).unwrap()
        );
    }

    #[test]
    fn error_variant_is_the_error_state() {
        let ok = WorkerKvQueryReply(WorkerKvQueryResponse::InvalidRange {
            start_id: 3,
            end_id: 1,
        });
        assert!(ok.err().is_none());
        let failed = WorkerKvQueryReply::from_err(std::io::Error::other("boom"));
        // `DynamoError::msg` prefixes its class ("Internal: ..."), as the removed impls did.
        assert!(failed.err().unwrap().to_string().ends_with("boom"));
        assert!(matches!(failed.0, WorkerKvQueryResponse::Error(msg) if msg == "boom"));

        let failed = IndexerQueryReply(IndexerQueryResponse::Error("down".to_string()));
        assert!(failed.err().unwrap().to_string().ends_with("down"));
        assert!(
            IndexerRecordRoutingDecisionReply(IndexerRecordRoutingDecisionResponse::Recorded)
                .err()
                .is_none()
        );
    }
}

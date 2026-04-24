// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-local session abstraction for conditional disaggregation.
//!
//! The real implementation is expected to sit on top of the two-stream velo
//! session work. This trait keeps the decode coordinator independent from that
//! transport while preserving the important contract: session-held pins are
//! released only after an explicit unpin request and acknowledgement.

use std::pin::Pin;

use anyhow::Result;
use futures::{Stream, future::BoxFuture};
use kvbm_disagg_protocol::{
    DescriptorRequest, DescriptorResponse, DisaggSequenceHash, HashSelection, SessionEndpoint,
    SessionId, UnpinAck, UnpinRequest,
};
use kvbm_logical::blocks::ImmutableBlock;

use crate::{G2, InstanceId, SequenceHash};

/// Stream of session events consumed by a coordinator monitor task.
pub type SessionEventStream = Pin<Box<dyn Stream<Item = SessionEvent> + Send + 'static>>;

/// Convert native KVBM sequence hashes to the JSON-safe protocol form.
pub fn hash_to_wire(hash: SequenceHash) -> DisaggSequenceHash {
    hash.as_u128().to_string()
}

pub fn hashes_to_wire(hashes: impl IntoIterator<Item = SequenceHash>) -> Vec<DisaggSequenceHash> {
    hashes.into_iter().map(hash_to_wire).collect()
}

/// Blocks used to seed a decode-side remote-prefill session.
#[derive(Debug, Clone, Default)]
pub struct SessionBlocks {
    /// Blocks already ready in local G2 and available for descriptor requests.
    pub ready_g2: Vec<ImmutableBlock<G2>>,
    /// Hashes known to this side but not yet ready as G2 descriptors.
    pub pending_hashes: Vec<SequenceHash>,
}

impl SessionBlocks {
    pub fn new(ready_g2: Vec<ImmutableBlock<G2>>, pending_hashes: Vec<SequenceHash>) -> Self {
        Self {
            ready_g2,
            pending_hashes,
        }
    }

    pub fn ready_hashes_wire(&self) -> Vec<DisaggSequenceHash> {
        self.ready_g2
            .iter()
            .map(|block| hash_to_wire(block.sequence_hash()))
            .collect()
    }

    pub fn pending_hashes_wire(&self) -> Vec<DisaggSequenceHash> {
        hashes_to_wire(self.pending_hashes.iter().copied())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    Attached {
        peer_instance_id: InstanceId,
    },
    DescriptorRequest(DescriptorRequest),
    UnpinRequested(UnpinRequest),
    UnpinAcked(UnpinAck),
    BlocksAdded {
        blocks: Vec<kvbm_disagg_protocol::DisaggBlockRef>,
    },
    Detached {
        reason: Option<String>,
    },
    Failed {
        reason: String,
    },
}

/// Abstraction over a bidirectional decode/prefill session.
pub trait PrefillSession: Send + Sync {
    fn session_id(&self) -> SessionId;

    fn endpoint(&self) -> Option<SessionEndpoint>;

    fn add_ready_blocks(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()>;

    fn add_pending_hashes(&self, hashes: Vec<SequenceHash>) -> Result<()>;

    fn subscribe(&self) -> SessionEventStream;

    fn respond_to_descriptor_request(&self, response: DescriptorResponse) -> Result<()>;

    /// Release session-owned pins matching `selection`. This must not release
    /// coordinator-owned references to the same blocks.
    fn release_session_pins(&self, selection: &HashSelection) -> Result<Vec<DisaggSequenceHash>>;

    fn ack_unpin(&self, ack: UnpinAck) -> Result<()>;

    /// Local-initiated unpin protocol. Included now so future decode output
    /// pull paths can require a peer ack before releasing session pins.
    fn request_unpin(&self, request: UnpinRequest) -> BoxFuture<'static, Result<UnpinAck>>;

    fn close(&self, reason: Option<String>);
}

pub trait PrefillSessionFactory: Send + Sync {
    fn create_decode(&self, session_id: SessionId) -> Result<std::sync::Arc<dyn PrefillSession>>;
}

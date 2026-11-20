// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV hints a worker-selection policy sends with the worker it books.
//!
//! A policy that wants the booked worker to act on its KV cache, for example to fetch the prefix
//! it was credited for copying, attaches a [`HintEmitter`] with
//! [`WorkerSelectionPolicy::with_hints`]. After the router books the worker the policy picked, it
//! calls [`HintEmitter::emit`] once with a [`HintContext`] and sends the returned actions to the
//! worker as one `KvHint`, numbered in order. The router sends hints only to workers that
//! advertise the `router_hint` capability, and only on the KV path, which is the one that books.
//!
//! Hints never change which worker is selected; the policy's filters, scorers, and picker do.
//!
//! [`WorkerSelectionPolicy::with_hints`]: crate::WorkerSelectionPolicy::with_hints

use dynamo_kv_hints::{KvCopyLocations, KvCopySource, KvHintAction};
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;

use crate::WorkerInputs;

/// Builds the KV actions for a booked worker. Attached to a policy with
/// [`WorkerSelectionPolicy::with_hints`](crate::WorkerSelectionPolicy::with_hints).
pub trait HintEmitter: Send + Sync {
    /// The optional worker inputs the emitter reads from the [`HintContext`]. Only
    /// [`WorkerInputs::KV_COPY`] is readable at booking: declaring it makes the router compute
    /// the booked worker's copy source and locations for [`HintContext::kv_copy`] and
    /// [`HintContext::kv_copy_locations`]. Without it both are `None`.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// The actions for the worker `context` describes, in the engine's format and in the order
    /// they should carry. An empty list sends no hint. An error fails the booking, which the
    /// router releases; return `Ok(Vec::new())` for a hint that is merely unavailable.
    fn emit(&self, context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintError>;
}

/// What an emitter knows about the booking it builds hints for.
pub struct HintContext<'a> {
    request_id: &'a str,
    worker: WorkerWithDpRank,
    isl_tokens: usize,
    block_size: u32,
    session_context: Option<&'a SessionContext>,
    kv_copy: Option<&'a KvCopySource>,
    kv_copy_locations: Option<&'a KvCopyLocations<'a>>,
}

impl<'a> HintContext<'a> {
    /// A context without copy data. The router builds it; tests and benches may too.
    pub fn new(
        request_id: &'a str,
        worker: WorkerWithDpRank,
        isl_tokens: usize,
        block_size: u32,
        session_context: Option<&'a SessionContext>,
    ) -> Self {
        Self {
            request_id,
            worker,
            isl_tokens,
            block_size,
            session_context,
            kv_copy: None,
            kv_copy_locations: None,
        }
    }

    /// Attach the booked worker's copy source, and where its blocks are when the source is
    /// still reachable.
    pub fn with_kv_copy(
        mut self,
        copy: Option<&'a KvCopySource>,
        locations: Option<&'a KvCopyLocations<'a>>,
    ) -> Self {
        self.kv_copy = copy;
        self.kv_copy_locations = locations;
        self
    }

    /// The request's ID, which the hint's `message_id` echoes.
    pub fn request_id(&self) -> &'a str {
        self.request_id
    }

    /// The booked worker.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// The prompt length, in tokens.
    pub fn isl_tokens(&self) -> usize {
        self.isl_tokens
    }

    /// The partition's KV block size, in tokens.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The request's session, when the host manages one.
    pub fn session_context(&self) -> Option<&'a SessionContext> {
        self.session_context
    }

    /// The copy the booked worker can make, for an emitter that declares
    /// [`WorkerInputs::KV_COPY`]: the source selection chose for it, when some source holds a
    /// longer prefix than the worker does. `None` when no source does, when the partition cannot
    /// compute copies, or when the emitter did not declare the input.
    pub fn kv_copy(&self) -> Option<&'a KvCopySource> {
        self.kv_copy
    }

    /// Where the blocks of [`Self::kv_copy`] are. `None` when there is no copy or its source is
    /// no longer reachable; a fetch is then not possible.
    pub fn kv_copy_locations(&self) -> Option<&'a KvCopyLocations<'a>> {
        self.kv_copy_locations
    }
}

/// An emitter's failure, which fails the booking it was asked about.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct HintError {
    message: String,
}

impl HintError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

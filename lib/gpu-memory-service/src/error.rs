// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the GPU Memory Service.

use crate::state::{ServerState, StateEvent};

/// Errors that can occur in the GPU Memory Service.
#[derive(thiserror::Error, Debug)]
pub enum GmsError {
    /// Allocation not found by ID.
    #[error("allocation not found: {0}")]
    AllocationNotFound(String),

    /// Invalid state machine transition.
    #[error("invalid transition: {event:?} from state {from:?}")]
    InvalidTransition {
        /// Current state.
        from: ServerState,
        /// Attempted event.
        event: StateEvent,
    },

    /// Operation not allowed in current state.
    #[error("{op} not allowed in state {state:?}")]
    OperationNotAllowed {
        /// Operation name.
        op: &'static str,
        /// Current state.
        state: ServerState,
    },

    /// CUDA driver error.
    #[error("CUDA error: {0}")]
    Cuda(#[from] dynamo_memory::StorageError),

    /// Transport/IO error.
    #[error("transport error: {0}")]
    Transport(#[from] anyhow::Error),

    /// Lock acquisition timed out.
    #[error("lock acquisition timed out")]
    LockTimeout,

    /// RPC response timed out.
    #[error("RPC response timed out")]
    RpcTimeout,

    /// Memory layout hash mismatch.
    #[error("stale memory layout: expected {expected}, got {actual}")]
    StaleMemoryLayout {
        /// Expected hash.
        expected: String,
        /// Actual hash.
        actual: String,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Protocol encoding/decoding error.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Session not found.
    #[error("session not found: {0}")]
    SessionNotFound(u64),
}

/// Result type alias for GMS operations.
pub type GmsResult<T> = std::result::Result<T, GmsError>;

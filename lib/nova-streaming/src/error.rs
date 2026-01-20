// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for nova-streaming.

use crate::protocol::AnchorId;
use thiserror::Error;

/// Error when attaching to a stream anchor.
#[derive(Debug, Error)]
pub enum AttachError {
    /// Handle is not in a state that allows attachment.
    #[error("handle is not attachable: {0}")]
    NotAttachable(String),

    /// Anchor not found or already finalized.
    #[error("anchor not found: {0}")]
    AnchorNotFound(AnchorId),

    /// Connection to anchor owner failed.
    #[error("connection failed: {0}")]
    ConnectionFailed(#[from] anyhow::Error),

    /// Attach request timed out.
    #[error("attach timed out")]
    Timeout,

    /// Anchor rejected the attachment.
    #[error("attach rejected: {0}")]
    Rejected(String),

    /// Worker ID could not be resolved to instance ID.
    /// This means the target instance is either not discoverable or has gone away.
    #[error("unknown worker: {0}")]
    UnknownWorker(dynamo_identity::WorkerId),

    /// Feature not yet implemented.
    #[error("not implemented")]
    NotImplemented,
}

/// Error when sending data through a stream.
#[derive(Debug, Error)]
pub enum SendError {
    /// Stream has been closed.
    #[error("stream closed")]
    Closed,

    /// Stream has been cancelled.
    #[error("stream cancelled")]
    Cancelled,

    /// Send operation timed out.
    #[error("send timed out")]
    Timeout,

    /// Serialization failed.
    #[error("serialization failed: {0}")]
    Serialization(String),

    /// Transport error.
    #[error("transport error: {0}")]
    Transport(#[from] anyhow::Error),
}

/// Error when detaching from a stream anchor.
#[derive(Debug, Error)]
pub enum DetachError {
    /// Not currently attached.
    #[error("not attached")]
    NotAttached,

    /// Detach request failed.
    #[error("detach failed: {0}")]
    Failed(#[from] anyhow::Error),
}

/// Error when finalizing a stream.
#[derive(Debug, Error)]
pub enum FinalizeError {
    /// Stream already finalized.
    #[error("already finalized")]
    AlreadyFinalized,

    /// Finalize request failed.
    #[error("finalize failed: {0}")]
    Failed(#[from] anyhow::Error),
}

/// General stream error for receiver side.
#[derive(Debug, Error)]
pub enum StreamError {
    /// Received an error frame from sender.
    #[error("sender error: {0}")]
    SenderError(String),

    /// Deserialization failed.
    #[error("deserialization failed: {0}")]
    Deserialization(String),

    /// Attach error.
    #[error("attach error: {0}")]
    Attach(#[from] AttachError),

    /// Send error.
    #[error("send error: {0}")]
    Send(#[from] SendError),

    /// Detach error.
    #[error("detach error: {0}")]
    Detach(#[from] DetachError),

    /// Finalize error.
    #[error("finalize error: {0}")]
    Finalize(#[from] FinalizeError),
}

/// Error when an anchor slot cannot be allocated.
#[derive(Debug, Error)]
pub enum AnchorRegistryError {
    /// No free slots available.
    #[error("no free anchor slots available")]
    Exhausted,

    /// Slot generation has been exceeded.
    #[error("anchor slot generation exhausted")]
    GenerationExhausted,
}

/// Error when cancelling a stream.
#[derive(Debug, Error)]
pub enum CancelError {
    /// Stream already closed.
    #[error("stream already closed")]
    AlreadyClosed,

    /// Failed to notify sender of cancellation.
    #[error("failed to notify sender: {0}")]
    NotifyFailed(#[from] anyhow::Error),
}

/// Error from a non-blocking try_send operation.
///
/// This is used by the hot path to avoid heap allocations. If the internal
/// buffer is full, the frame is returned so the caller can retry or apply
/// backpressure.
#[derive(Debug)]
pub enum TrySendError<T> {
    /// Internal buffer is full. The frame is returned for retry.
    Full(T),

    /// A background send failed. The stream should be considered broken.
    BackgroundError(anyhow::Error),

    /// Stream has been closed.
    Closed,

    /// Stream was cancelled by the receiver.
    Cancelled(String),

    /// Serialization failed.
    Serialization(String),
}

impl<T> std::fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full(_) => write!(f, "buffer full"),
            Self::BackgroundError(e) => write!(f, "background error: {}", e),
            Self::Closed => write!(f, "stream closed"),
            Self::Cancelled(reason) => write!(f, "stream cancelled: {}", reason),
            Self::Serialization(e) => write!(f, "serialization failed: {}", e),
        }
    }
}

impl<T: std::fmt::Debug> std::error::Error for TrySendError<T> {}

impl<T> TrySendError<T> {
    /// Returns true if this is a Full error (buffer full, can retry).
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full(_))
    }

    /// Returns true if this is a fatal error (stream broken).
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::BackgroundError(_) | Self::Closed | Self::Cancelled(_))
    }

    /// Extract the frame from a Full error for retry.
    pub fn into_inner(self) -> Option<T> {
        match self {
            Self::Full(frame) => Some(frame),
            _ => None,
        }
    }
}

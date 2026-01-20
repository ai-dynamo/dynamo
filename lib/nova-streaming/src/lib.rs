// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

mod channel;
mod error;
mod protocol;
mod streaming;

// =============================================================================
// Primary Public API - NovaStreaming-centric
// =============================================================================

/// Re-export the main entry point for stream-based communication.
pub use streaming::{
    ArmedHandle, NovaStreaming, NovaStreamingBuilder, ReceiverRegistry, SendMode, StreamHandle,
    StreamSender,
    // Handler names (useful for wait_for_handler)
    STREAM_ATTACH_HANDLER, STREAM_CANCEL_HANDLER, STREAM_DATA_HANDLER,
};

/// Protocol types for stream communication.
pub use protocol::{AnchorId, AttachResponse, StreamFrame};

/// Error types for stream operations.
pub use error::{
    AttachError, CancelError, DetachError, FinalizeError, SendError, StreamError, TrySendError,
};

// =============================================================================
// Channel Types - StreamReceiver
// =============================================================================

/// Re-export channel types for stream receivers.
pub use channel::{
    ReceiverState, StreamReceiver, StreamReceiverBuilder, StreamReceiverContext,
    StreamReceiverContextWeak,
};

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the distributed registry.

use std::fmt;

/// Errors that can occur during registry operations.
#[derive(Debug)]
pub enum RegistryError {
    /// Failed to encode a message.
    EncodeError { context: &'static str },

    /// Failed to decode a message.
    DecodeError {
        context: &'static str,
        expected: String,
        got: String,
    },

    /// Protocol version mismatch.
    VersionMismatch { expected: u8, got: u8 },

    /// Invalid message type.
    InvalidMessageType { got: u8 },

    /// Message too short.
    MessageTooShort { expected: usize, got: usize },

    /// Transport error.
    TransportError { message: String },

    /// Hub disconnected.
    Disconnected,

    /// Request timed out.
    Timeout { duration_ms: u64 },

    /// Lease conflict - key is leased by another client.
    LeaseConflict { key_debug: String },

    /// Storage error.
    StorageError { message: String },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EncodeError { context } => {
                write!(f, "encode error: {}", context)
            }
            Self::DecodeError {
                context,
                expected,
                got,
            } => {
                write!(
                    f,
                    "decode error in {}: expected {}, got {}",
                    context, expected, got
                )
            }
            Self::VersionMismatch { expected, got } => {
                write!(
                    f,
                    "protocol version mismatch: expected {}, got {}",
                    expected, got
                )
            }
            Self::InvalidMessageType { got } => {
                write!(f, "invalid message type: {}", got)
            }
            Self::MessageTooShort { expected, got } => {
                write!(
                    f,
                    "message too short: expected {} bytes, got {}",
                    expected, got
                )
            }
            Self::TransportError { message } => {
                write!(f, "transport error: {}", message)
            }
            Self::Disconnected => {
                write!(f, "hub disconnected")
            }
            Self::Timeout { duration_ms } => {
                write!(f, "request timed out after {}ms", duration_ms)
            }
            Self::LeaseConflict { key_debug } => {
                write!(f, "lease conflict for key: {}", key_debug)
            }
            Self::StorageError { message } => {
                write!(f, "storage error: {}", message)
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result type for registry operations.
pub type RegistryResult<T> = Result<T, RegistryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RegistryError::DecodeError {
            context: "register message",
            expected: "4 bytes for count".to_string(),
            got: "2 bytes".to_string(),
        };
        assert!(err.to_string().contains("register message"));
        assert!(err.to_string().contains("4 bytes"));
    }

    #[test]
    fn test_version_mismatch() {
        let err = RegistryError::VersionMismatch {
            expected: 1,
            got: 2,
        };
        assert!(err.to_string().contains("version mismatch"));
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error classification for etcd operations.
//!
//! Categorizes etcd errors into reconnectable, expected, or fatal conditions
//! to enable smart retry logic.

use std::fmt;
use tonic::Code;

/// Errors that indicate a connection issue requiring reconnection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconnectableError {
    /// Connection to etcd server was closed
    ConnectionClosed,
    /// Operation timed out
    Timeout,
    /// Service unavailable (etcd server down or unreachable)
    Unavailable,
    /// Lease was not found (may have expired during disconnect)
    LeaseNotFound,
}

impl fmt::Display for ReconnectableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionClosed => write!(f, "connection closed"),
            Self::Timeout => write!(f, "operation timed out"),
            Self::Unavailable => write!(f, "service unavailable"),
            Self::LeaseNotFound => write!(f, "lease not found"),
        }
    }
}

/// Classification of etcd errors for determining retry strategy.
#[derive(Debug)]
pub(crate) enum EtcdErrorClass {
    /// Error should trigger reconnection and retry
    Reconnectable(ReconnectableError),
    /// Expected condition (key not found) - not an error
    NotFound,
    /// Fatal error that cannot be recovered by reconnecting
    Fatal(anyhow::Error),
}

/// Classify an etcd error to determine appropriate handling.
///
/// # Classification Strategy
///
/// - **Reconnectable**: Connection/transport errors that can be fixed by reconnecting
/// - **NotFound**: Key doesn't exist (expected condition for queries)
/// - **Fatal**: All other errors (permissions, invalid request, etc.)
pub(crate) fn classify_error(err: etcd_client::Error) -> EtcdErrorClass {
    // Use structured error matching instead of fragile string matching
    match err {
        etcd_client::Error::GRpcStatus(status) => {
            // Classify based on gRPC status code
            match status.code() {
                Code::NotFound => {
                    // Check if it's a lease not found or key not found
                    let msg = status.message().to_lowercase();
                    if msg.contains("lease") {
                        EtcdErrorClass::Reconnectable(ReconnectableError::LeaseNotFound)
                    } else {
                        // Key not found is expected, not an error
                        EtcdErrorClass::NotFound
                    }
                }
                Code::Unavailable => EtcdErrorClass::Reconnectable(ReconnectableError::Unavailable),
                Code::DeadlineExceeded => {
                    EtcdErrorClass::Reconnectable(ReconnectableError::Timeout)
                }
                Code::Cancelled | Code::Aborted => {
                    // Connection-related cancellations
                    EtcdErrorClass::Reconnectable(ReconnectableError::ConnectionClosed)
                }
                _ => {
                    // All other gRPC errors are fatal
                    EtcdErrorClass::Fatal(anyhow::anyhow!(
                        "gRPC error: {} (code: {:?})",
                        status.message(),
                        status.code()
                    ))
                }
            }
        }
        etcd_client::Error::TransportError(_) => {
            // Transport errors are reconnectable
            EtcdErrorClass::Reconnectable(ReconnectableError::Unavailable)
        }
        etcd_client::Error::IoError(_) => {
            // I/O errors are reconnectable
            EtcdErrorClass::Reconnectable(ReconnectableError::ConnectionClosed)
        }
        _ => {
            // All other errors (LeaseKeepAliveError, etc.) are fatal
            EtcdErrorClass::Fatal(err.into())
        }
    }
}

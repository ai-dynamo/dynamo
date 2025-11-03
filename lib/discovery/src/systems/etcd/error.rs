// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error classification for etcd operations.
//!
//! Categorizes etcd errors into reconnectable, expected, or fatal conditions
//! to enable smart retry logic.

use std::fmt;

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
    // Convert error to string for pattern matching
    // etcd_client::Error doesn't expose structured error types,
    // so we use string matching on the error message
    let err_str = err.to_string().to_lowercase();

    // Check for connection/transport errors
    if err_str.contains("unavailable")
        || err_str.contains("connection refused")
        || err_str.contains("connection reset")
        || err_str.contains("broken pipe")
        || err_str.contains("not connected")
    {
        return EtcdErrorClass::Reconnectable(ReconnectableError::Unavailable);
    }

    // Check for connection closed
    if err_str.contains("connection closed")
        || err_str.contains("connection error")
        || err_str.contains("stream closed")
        || err_str.contains("channel closed")
    {
        return EtcdErrorClass::Reconnectable(ReconnectableError::ConnectionClosed);
    }

    // Check for timeout
    if err_str.contains("timeout") || err_str.contains("deadline exceeded") {
        return EtcdErrorClass::Reconnectable(ReconnectableError::Timeout);
    }

    // Check for lease not found
    if err_str.contains("lease not found") || err_str.contains("requested lease not found") {
        return EtcdErrorClass::Reconnectable(ReconnectableError::LeaseNotFound);
    }

    // Check for key not found (this is expected, not an error)
    if err_str.contains("key not found") {
        return EtcdErrorClass::NotFound;
    }

    // Everything else is considered fatal
    EtcdErrorClass::Fatal(err.into())
}

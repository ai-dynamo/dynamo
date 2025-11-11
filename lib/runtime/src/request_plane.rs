// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for request plane transport mode

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Request plane transport mode configuration
///
/// This determines how requests are distributed from routers to workers:
/// - `Nats`: Use NATS for request distribution (default, legacy)
/// - `Http`: Use HTTP/2 for request distribution (recommended)
/// - `Tcp`: Use raw TCP for request distribution
/// - `Zmq`: Use ZeroMQ for request distribution
/// - `Uds`: Use Unix Domain Sockets for request distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestPlaneMode {
    /// Use NATS for request plane (default for backward compatibility)
    Nats,
    /// Use HTTP/2 for request plane (recommended)
    Http,
    /// Use raw TCP for request plane
    Tcp,
    /// Use ZeroMQ for request plane
    Zmq,
    /// Use Unix Domain Sockets for request plane (local only)
    Uds,
}

impl Default for RequestPlaneMode {
    fn default() -> Self {
        Self::Nats
    }
}

impl fmt::Display for RequestPlaneMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nats => write!(f, "nats"),
            Self::Http => write!(f, "http"),
            Self::Tcp => write!(f, "tcp"),
            Self::Zmq => write!(f, "zmq"),
            Self::Uds => write!(f, "uds"),
        }
    }
}

impl FromStr for RequestPlaneMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nats" => Ok(Self::Nats),
            "http" | "http2" => Ok(Self::Http),
            "tcp" => Ok(Self::Tcp),
            "zmq" | "zeromq" => Ok(Self::Zmq),
            "uds" | "unix" => Ok(Self::Uds),
            _ => Err(anyhow::anyhow!(
                "Invalid request plane mode: '{}'. Valid options are: 'nats', 'http', 'tcp', 'zmq', 'uds'",
                s
            )),
        }
    }
}

impl RequestPlaneMode {
    /// Get the request plane mode from environment variable
    ///
    /// Reads from `DYN_REQUEST_PLANE` environment variable.
    /// Falls back to default (NATS) if not set or invalid.
    pub fn from_env() -> Self {
        std::env::var("DYN_REQUEST_PLANE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default()
    }

    /// Get the request plane mode from environment variable or default
    ///
    /// Reads from `DYN_REQUEST_PLANE` environment variable.
    /// If not set, uses the provided default.
    pub fn from_env_or(default: Self) -> Self {
        std::env::var("DYN_REQUEST_PLANE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_plane_mode_display() {
        assert_eq!(RequestPlaneMode::Nats.to_string(), "nats");
        assert_eq!(RequestPlaneMode::Http.to_string(), "http");
        assert_eq!(RequestPlaneMode::Tcp.to_string(), "tcp");
        assert_eq!(RequestPlaneMode::Zmq.to_string(), "zmq");
        assert_eq!(RequestPlaneMode::Uds.to_string(), "uds");
    }

    #[test]
    fn test_request_plane_mode_default() {
        assert_eq!(RequestPlaneMode::default(), RequestPlaneMode::Nats);
    }

}

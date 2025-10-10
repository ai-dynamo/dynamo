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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestPlaneMode {
    /// Use NATS for request plane (default for backward compatibility)
    Nats,
    /// Use HTTP/2 for request plane (recommended)
    Http,
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
        }
    }
}

impl FromStr for RequestPlaneMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nats" => Ok(Self::Nats),
            "http" | "http2" => Ok(Self::Http),
            _ => Err(anyhow::anyhow!(
                "Invalid request plane mode: '{}'. Valid options are: 'nats', 'http'",
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

    /// Check if this mode is NATS
    pub fn is_nats(&self) -> bool {
        matches!(self, Self::Nats)
    }

    /// Check if this mode is HTTP
    pub fn is_http(&self) -> bool {
        matches!(self, Self::Http)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_plane_mode_from_str() {
        assert_eq!("nats".parse::<RequestPlaneMode>().unwrap(), RequestPlaneMode::Nats);
        assert_eq!("http".parse::<RequestPlaneMode>().unwrap(), RequestPlaneMode::Http);
        assert_eq!("http2".parse::<RequestPlaneMode>().unwrap(), RequestPlaneMode::Http);
        assert_eq!("NATS".parse::<RequestPlaneMode>().unwrap(), RequestPlaneMode::Nats);
        assert_eq!("HTTP".parse::<RequestPlaneMode>().unwrap(), RequestPlaneMode::Http);
        assert!("invalid".parse::<RequestPlaneMode>().is_err());
    }

    #[test]
    fn test_request_plane_mode_display() {
        assert_eq!(RequestPlaneMode::Nats.to_string(), "nats");
        assert_eq!(RequestPlaneMode::Http.to_string(), "http");
    }

    #[test]
    fn test_request_plane_mode_default() {
        assert_eq!(RequestPlaneMode::default(), RequestPlaneMode::Nats);
    }

    #[test]
    fn test_request_plane_mode_is_methods() {
        assert!(RequestPlaneMode::Nats.is_nats());
        assert!(!RequestPlaneMode::Nats.is_http());
        assert!(RequestPlaneMode::Http.is_http());
        assert!(!RequestPlaneMode::Http.is_nats());
    }
}


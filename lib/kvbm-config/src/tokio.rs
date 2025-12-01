// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio runtime configuration.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Tokio runtime configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct TokioConfig {
    /// Number of async worker threads.
    /// If None, uses the number of logical CPUs.
    #[validate(range(min = 1))]
    pub worker_threads: Option<usize>,

    /// Maximum number of blocking threads.
    /// If None, uses Tokio's default (512).
    #[validate(range(min = 1))]
    pub max_blocking_threads: Option<usize>,
}

impl TokioConfig {
    /// Build a Tokio runtime from this configuration.
    pub fn build_runtime(&self) -> std::io::Result<::tokio::runtime::Runtime> {
        let mut builder = ::tokio::runtime::Builder::new_multi_thread();

        if let Some(threads) = self.worker_threads {
            builder.worker_threads(threads);
        }

        if let Some(blocking) = self.max_blocking_threads {
            builder.max_blocking_threads(blocking);
        }

        builder.enable_all().build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TokioConfig::default();
        assert!(config.worker_threads.is_none());
        assert!(config.max_blocking_threads.is_none());
    }

    #[test]
    fn test_build_runtime_with_defaults() {
        let config = TokioConfig::default();
        let runtime = config.build_runtime().expect("Failed to build runtime");
        drop(runtime);
    }

    #[test]
    fn test_build_runtime_with_custom_threads() {
        let config = TokioConfig {
            worker_threads: Some(2),
            max_blocking_threads: Some(4),
        };
        let runtime = config.build_runtime().expect("Failed to build runtime");
        drop(runtime);
    }
}

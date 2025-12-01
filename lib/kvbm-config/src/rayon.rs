// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rayon thread pool configuration.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Rayon thread pool configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct RayonConfig {
    /// Number of threads in the Rayon thread pool.
    /// If None, uses the number of logical CPUs.
    #[validate(range(min = 1))]
    pub num_threads: Option<usize>,
}

#[cfg(feature = "rayon")]
impl RayonConfig {
    /// Build a Rayon thread pool from this configuration.
    pub fn build_pool(&self) -> Result<::rayon::ThreadPool, ::rayon::ThreadPoolBuildError> {
        let mut builder = ::rayon::ThreadPoolBuilder::new();

        if let Some(threads) = self.num_threads {
            builder = builder.num_threads(threads);
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RayonConfig::default();
        assert!(config.num_threads.is_none());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_build_pool() {
        let config = RayonConfig {
            num_threads: Some(2),
        };
        let pool = config.build_pool().expect("Failed to build pool");
        assert_eq!(pool.current_num_threads(), 2);
    }
}

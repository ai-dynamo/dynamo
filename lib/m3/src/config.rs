// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytesize::ByteSize;
use derive_builder::Builder;
use dynamo_memory::nixl::NixlAgent;
use std::sync::Arc;

/// Helper type for accepting both string sizes ("100GiB") and raw bytes (usize)
#[derive(Debug, Clone)]
pub enum SizeInput {
    Bytes(usize),
    Str(String),
}

impl SizeInput {
    /// Convert to bytes, parsing string if needed
    pub fn to_bytes(&self) -> Result<usize, String> {
        match self {
            Self::Bytes(b) => Ok(*b),
            Self::Str(s) => s
                .parse::<ByteSize>()
                .map(|bs| bs.as_u64() as usize)
                .map_err(|e| format!("invalid size format '{}': {}", s, e)),
        }
    }
}

impl From<usize> for SizeInput {
    fn from(v: usize) -> Self {
        Self::Bytes(v)
    }
}

impl From<&str> for SizeInput {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

impl From<String> for SizeInput {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

/// Configuration for M3Store
#[derive(Debug, Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct M3Config {
    /// Total size of disk tier storage
    #[builder(private, setter(name = "_disk_tier_size"))]
    pub disk_tier_size: usize,

    /// Size of fast tier (in-memory cache)
    #[builder(private, default, setter(name = "_fast_tier_size"))]
    pub fast_tier_size: Option<usize>,

    /// Chunk size for splitting objects in disk tier (must be power of 2, 1KB-32MB)
    #[builder(private, setter(name = "_chunk_size"))]
    pub chunk_size: usize,

    /// Whether to enable fast tier caching
    /// None = auto-default to true if fast_tier_size >= 1GiB
    #[builder(setter(strip_option), default)]
    pub enable_fast_tier_caching: Option<bool>,

    /// Threshold (0.0-1.0) at which to trigger eager eviction
    #[builder(default = "0.85")]
    pub eager_eviction_threshold: f64,

    /// NIXL agent for fast tier registration
    pub nixl_agent: Arc<NixlAgent>,

    /// Path for RocksDB storage
    #[builder(default = "std::path::PathBuf::from(\"/tmp/m3_rocksdb\")")]
    pub db_path: std::path::PathBuf,
}

impl M3ConfigBuilder {
    /// Set disk tier size (accepts "100GiB" or raw usize)
    pub fn disk_tier_size(&mut self, size: impl Into<SizeInput>) -> &mut Self {
        match size.into().to_bytes() {
            Ok(bytes) => {
                self.disk_tier_size = Some(bytes);
            }
            Err(e) => {
                tracing::warn!("Invalid disk_tier_size: {}", e);
                self.disk_tier_size = Some(0); // Will fail validation
            }
        }
        self
    }

    /// Set fast tier size (accepts "4GiB" or raw usize)
    pub fn fast_tier_size(&mut self, size: impl Into<SizeInput>) -> &mut Self {
        match size.into().to_bytes() {
            Ok(bytes) => {
                self.fast_tier_size = Some(Some(bytes));
            }
            Err(e) => {
                tracing::warn!("Invalid fast_tier_size: {}", e);
            }
        }
        self
    }

    /// Set chunk size (accepts "8MiB" or raw usize)
    pub fn chunk_size(&mut self, size: impl Into<SizeInput>) -> &mut Self {
        match size.into().to_bytes() {
            Ok(bytes) => {
                self.chunk_size = Some(bytes);
            }
            Err(e) => {
                tracing::warn!("Invalid chunk_size: {}", e);
                self.chunk_size = Some(0); // Will fail validation
            }
        }
        self
    }

    fn validate(&self) -> Result<(), String> {
        // Validate disk_tier_size
        let disk_size = self.disk_tier_size.ok_or("disk_tier_size is required")?;
        if disk_size == 0 {
            return Err("disk_tier_size must be greater than 0".into());
        }

        // Validate chunk_size
        let chunk_size = self.chunk_size.ok_or("chunk_size is required")?;
        if chunk_size == 0 {
            return Err("chunk_size must be greater than 0".into());
        }
        if !chunk_size.is_power_of_two() {
            return Err(format!("chunk_size must be power of 2, got {}", chunk_size));
        }
        if !(1024..=(32 << 20)).contains(&chunk_size) {
            return Err(format!(
                "chunk_size must be between 1KB and 32MB, got {}",
                chunk_size
            ));
        }

        // Apply smart default for enable_fast_tier_caching
        let enable_caching = match self.enable_fast_tier_caching {
            Some(Some(explicit)) => explicit,
            Some(None) | None => {
                // Auto-default: enable if fast_tier_size >= 1GiB
                self.fast_tier_size
                    .flatten()
                    .map(|bytes| bytes >= (1 << 30))
                    .unwrap_or(false)
            }
        };

        // If caching enabled, require fast_tier_size
        if enable_caching {
            match self.fast_tier_size.flatten() {
                None => {
                    return Err(
                        "enable_fast_tier_caching=true requires fast_tier_size to be set".into(),
                    );
                }
                Some(0) => return Err("fast_tier_size must be greater than 0".into()),
                _ => {}
            }
        }

        // Validate eager_eviction_threshold
        if let Some(threshold) = self.eager_eviction_threshold
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(format!(
                "eager_eviction_threshold must be between 0.0 and 1.0, got {}",
                threshold
            ));
        }

        // Validate nixl_agent
        if self.nixl_agent.is_none() {
            return Err("nixl_agent is required".into());
        }

        Ok(())
    }
}

impl M3Config {
    /// Get effective enable_fast_tier_caching value after smart defaulting
    pub fn is_fast_tier_enabled(&self) -> bool {
        self.enable_fast_tier_caching.unwrap_or_else(|| {
            self.fast_tier_size
                .map(|bytes| bytes >= (1 << 30))
                .unwrap_or(false)
        })
    }

    /// Get fast tier size, or error if fast tier is enabled but size not set
    pub fn get_fast_tier_size(&self) -> Result<usize, String> {
        if self.is_fast_tier_enabled() {
            self.fast_tier_size
                .ok_or_else(|| "fast tier enabled but size not set".into())
        } else {
            Err("fast tier not enabled".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_agent() -> Arc<NixlAgent> {
        Arc::new(NixlAgent::new("test_agent").expect("failed to create test agent"))
    }

    #[test]
    fn test_size_input_from_usize() {
        let input: SizeInput = 1024.into();
        assert_eq!(input.to_bytes().unwrap(), 1024);
    }

    #[test]
    fn test_size_input_from_str() {
        let input: SizeInput = "1GiB".into();
        assert_eq!(input.to_bytes().unwrap(), 1 << 30);

        let input: SizeInput = "100MB".into();
        assert_eq!(input.to_bytes().unwrap(), 100_000_000);

        let input: SizeInput = "8MiB".into();
        assert_eq!(input.to_bytes().unwrap(), 8 << 20);
    }

    #[test]
    fn test_config_builder_with_strings() {
        let config = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .fast_tier_size("4GiB")
            .chunk_size("8MiB")
            .nixl_agent(create_test_agent())
            .build()
            .expect("failed to build config");

        assert_eq!(config.disk_tier_size, 100 << 30);
        assert_eq!(config.fast_tier_size, Some(4 << 30));
        assert_eq!(config.chunk_size, 8 << 20);
        assert!(config.is_fast_tier_enabled()); // >= 1GiB, so defaults to true
    }

    #[test]
    fn test_config_builder_with_usize() {
        let config = M3ConfigBuilder::default()
            .disk_tier_size(100 << 30)
            .fast_tier_size(4 << 30)
            .chunk_size(8 << 20)
            .nixl_agent(create_test_agent())
            .build()
            .expect("failed to build config");

        assert_eq!(config.disk_tier_size, 100 << 30);
        assert_eq!(config.fast_tier_size, Some(4 << 30));
        assert_eq!(config.chunk_size, 8 << 20);
    }

    #[test]
    fn test_config_smart_default_fast_tier_enabled() {
        // >= 1GiB, should default to true
        let config = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .fast_tier_size("2GiB")
            .chunk_size("8MiB")
            .nixl_agent(create_test_agent())
            .build()
            .expect("failed to build config");

        assert!(config.is_fast_tier_enabled());
    }

    #[test]
    fn test_config_smart_default_fast_tier_disabled() {
        // < 1GiB, should default to false
        let config = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .fast_tier_size("512MiB")
            .chunk_size("8MiB")
            .nixl_agent(create_test_agent())
            .build()
            .expect("failed to build config");

        assert!(!config.is_fast_tier_enabled());
    }

    #[test]
    fn test_config_explicit_override() {
        // < 1GiB but explicitly enabled
        let config = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .fast_tier_size("512MiB")
            .chunk_size("8MiB")
            .enable_fast_tier_caching(true)
            .nixl_agent(create_test_agent())
            .build()
            .expect("failed to build config");

        assert!(config.is_fast_tier_enabled());
    }

    #[test]
    fn test_config_invalid_chunk_size_not_power_of_two() {
        let result = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .chunk_size(1000) // Not power of 2
            .nixl_agent(create_test_agent())
            .build();

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("chunk_size must be power of 2")
        );
    }

    #[test]
    fn test_config_invalid_chunk_size_too_small() {
        let result = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .chunk_size(512) // < 1KB
            .nixl_agent(create_test_agent())
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_chunk_size_too_large() {
        let result = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .chunk_size(64 << 20) // > 32MB
            .nixl_agent(create_test_agent())
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_missing_fast_tier_size_when_enabled() {
        let result = M3ConfigBuilder::default()
            .disk_tier_size("100GiB")
            .chunk_size("8MiB")
            .enable_fast_tier_caching(true) // Explicitly enabled but no size
            .nixl_agent(create_test_agent())
            .build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires fast_tier_size"));
    }
}

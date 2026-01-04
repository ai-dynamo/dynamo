// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use blake3::hash as blake3_hash;
use std::path::PathBuf;

use dynamo_runtime::config::environment_names::llm;

#[derive(Clone)]
pub struct LoRACache {
    cache_root: PathBuf,
}

impl LoRACache {
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Get cache path from DYN_LORA_PATH environment variable.
    /// Defaults to `$HOME/.cache/dynamo_loras` if not set.
    pub fn from_env() -> Result<Self> {
        let cache_root = std::env::var(llm::DYN_LORA_PATH).unwrap_or_else(|_| {
            // Use $HOME/.cache/dynamo_loras as default, fallback to /tmp if HOME is not set
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home)
                .join(".cache")
                .join("dynamo_loras")
                .to_string_lossy()
                .to_string()
        });
        Ok(Self::new(PathBuf::from(cache_root)))
    }

    /// Get local cache path for LoRA ID
    pub fn get_cache_path(&self, lora_id: &str) -> PathBuf {
        self.cache_root.join(lora_id)
    }

    /// Check if LoRA is cached
    pub fn is_cached(&self, lora_id: &str) -> bool {
        self.get_cache_path(lora_id).exists()
    }

    /// Convert a LoRA URI to a cache key.
    /// This is a static method to ensure consistent cache key generation
    /// across Rust and Python code.
    pub fn uri_to_cache_key(uri: &str) -> String {
        // Collision-resistant and filesystem-safe.
        //
        // Format: {scheme}_{hash32}
        // - scheme: derived from the URI scheme (e.g. s3, gcs, http), sanitized to [a-z0-9_]
        // - hash32: first 32 hex chars (128-bit) of blake3(uri)
        let scheme = uri
            .split("://")
            .next()
            .unwrap_or("uri")
            .to_ascii_lowercase()
            .chars()
            .map(|c| {
                if c.is_ascii_lowercase() || c.is_ascii_digit() {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();

        let hash = blake3_hash(uri.as_bytes()).to_string();
        let hash32 = &hash[..32];
        format!("{scheme}_{hash32}")
    }

    /// Validate cached LoRA has required files
    /// TODO: Add support for other weight file formats supported by trtllm
    pub fn validate_cached(&self, lora_id: &str) -> Result<bool> {
        let path = self.get_cache_path(lora_id);
        if !path.exists() {
            return Ok(false);
        }

        // Check for at least adapter_config.json
        let config_path = path.join("adapter_config.json");
        if !config_path.exists() {
            return Ok(false);
        }

        // Check for at least one weight file
        // TODO: Add support for other weight file formats supported by trtllm
        let has_weights = path.join("adapter_model.safetensors").exists()
            || path.join("adapter_model.bin").exists()
            || path.join("model.lora_weights.npy").exists();

        Ok(has_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().to_path_buf());
        assert_eq!(cache.cache_root, temp_dir.path());
    }

    #[test]
    fn test_get_cache_path() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().to_path_buf());
        let lora_path = cache.get_cache_path("my-lora");
        assert_eq!(lora_path, temp_dir.path().join("my-lora"));
    }

    #[test]
    fn test_is_cached() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().to_path_buf());

        // Create a lora directory
        let lora_dir = temp_dir.path().join("test-lora");
        fs::create_dir(&lora_dir).unwrap();

        assert!(cache.is_cached("test-lora"));
        assert!(!cache.is_cached("non-existent"));
    }

    #[test]
    fn test_validate_cached() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().to_path_buf());

        // Create a lora directory with required files
        let lora_dir = temp_dir.path().join("valid-lora");
        fs::create_dir(&lora_dir).unwrap();
        fs::write(lora_dir.join("adapter_config.json"), "{}").unwrap();
        fs::write(lora_dir.join("adapter_model.safetensors"), "").unwrap();

        assert!(cache.validate_cached("valid-lora").unwrap());

        // Test missing weight file
        let lora_dir2 = temp_dir.path().join("invalid-lora");
        fs::create_dir(&lora_dir2).unwrap();
        fs::write(lora_dir2.join("adapter_config.json"), "{}").unwrap();

        assert!(!cache.validate_cached("invalid-lora").unwrap());
    }

    #[test]
    fn test_uri_to_cache_key() {
        let s3_key = LoRACache::uri_to_cache_key("s3://bucket/path/to/lora");
        assert!(s3_key.starts_with("s3_"));
        assert_eq!(s3_key.len(), "s3_".len() + 32);

        let file_key = LoRACache::uri_to_cache_key("file:///local/path");
        assert!(file_key.starts_with("file_"));
        assert_eq!(file_key.len(), "file_".len() + 32);

        // Deterministic
        assert_eq!(
            LoRACache::uri_to_cache_key("s3://bucket/path/to/lora"),
            s3_key
        );
    }
}

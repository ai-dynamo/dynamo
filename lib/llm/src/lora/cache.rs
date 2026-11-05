// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

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

    /// Convert the exact LoRA URI to a bounded filesystem cache key shared by
    /// Rust and Python. Legacy underscore-separated keys are ambiguous and
    /// cannot be reused without source-URI provenance; leave them untouched.
    pub fn uri_to_cache_key(uri: &str) -> String {
        format!("lora-v2-{:x}", Sha256::digest(uri.as_bytes()))
    }

    /// Validate cached LoRA has required files
    /// TODO: Add support for other weight file formats supported by trtllm
    pub fn validate_cached(&self, lora_id: &str) -> Result<bool> {
        let path = self.get_cache_path(lora_id);
        Self::validate_path(&path)
    }

    /// Validate a LoRA directory, including source-owned caches such as a
    /// Hugging Face Hub snapshot outside `DYN_LORA_PATH`.
    pub fn validate_path(path: &Path) -> Result<bool> {
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
    fn test_validate_external_snapshot_path() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("adapter_config.json"), "{}").unwrap();
        fs::write(temp_dir.path().join("adapter_model.safetensors"), "").unwrap();

        assert!(LoRACache::validate_path(temp_dir.path()).unwrap());
    }

    #[test]
    fn test_uri_to_cache_key() {
        assert_eq!(
            LoRACache::uri_to_cache_key("s3://bucket/path/to/lora"),
            "lora-v2-76606d1d1fa1089ee492b990f060b13ad4e07a09d9a6f69dc823ff405ab67dc4"
        );
    }

    #[test]
    fn cache_keys_preserve_uri_distinctions() {
        let uris = [
            "s3://bucket/adapter.v1",
            "s3://bucket/adapter_v1",
            "s3://bucket/team/adapter",
            "s3://bucket/team_adapter",
            "custom://adapter?revision=v1",
            "custom://adapter?revision=v2",
        ];
        let keys: std::collections::HashSet<_> =
            uris.into_iter().map(LoRACache::uri_to_cache_key).collect();
        assert_eq!(keys.len(), uris.len());
    }

    #[test]
    fn long_uri_uses_one_bounded_cache_component() {
        let uri = format!("s3://bucket/{}", "adapter/".repeat(100));
        let key = LoRACache::uri_to_cache_key(&uri);
        assert_eq!(key.len(), "lora-v2-".len() + 64);
        assert_eq!(Path::new(&key).components().count(), 1);
        assert!(
            key["lora-v2-".len()..]
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::path::PathBuf;

use dynamo_runtime::config::environment_names::llm;

const DEFAULT_LORA_CACHE_DIR: &str = "/tmp/dynamo_loras";

#[derive(Clone)]
pub struct LoRACache {
    cache_root: PathBuf,
}

impl LoRACache {
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Get cache path from DYN_LORA_PATH environment variable
    pub fn from_env() -> Result<Self> {
        let cache_root = std::env::var(llm::DYN_LORA_PATH)
            .unwrap_or_else(|_| DEFAULT_LORA_CACHE_DIR.to_string());
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
}

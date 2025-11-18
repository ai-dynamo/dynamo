// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for LoRA downloading and caching

#[cfg(test)]
mod tests {
    use dynamo_llm::lora::{LocalLoRASource, LoRACache, LoRADownloader, LoRASourceTrait};
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_lora_source_download() {
        let temp_dir = TempDir::new().unwrap();
        let lora_dir = temp_dir.path().join("test-lora");
        fs::create_dir(&lora_dir).unwrap();
        fs::write(lora_dir.join("adapter_config.json"), "{}").unwrap();
        fs::write(lora_dir.join("adapter_model.safetensors"), "weights").unwrap();

        let source = LocalLoRASource::new();
        let lora_uri = format!("file://{}", lora_dir.display());

        // Test exists
        assert!(source.exists(&lora_uri).await.unwrap());

        // Test download (should return original path for local files)
        let downloaded_path = source
            .download(&lora_uri, &temp_dir.path().join("ignored"))
            .await
            .unwrap();
        assert_eq!(downloaded_path, lora_dir);
    }

    #[tokio::test]
    async fn test_local_lora_source_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let lora_dir = temp_dir.path().join("test-lora");
        fs::create_dir(&lora_dir).unwrap();
        fs::write(lora_dir.join("adapter_config.json"), "{}").unwrap();
        fs::write(lora_dir.join("adapter_model.safetensors"), "weights").unwrap();

        let source = LocalLoRASource::new();
        let lora_uri = format!("file://{}", lora_dir.display());

        let metadata = source.metadata(&lora_uri).await.unwrap();
        assert!(metadata.is_some());

        let meta = metadata.unwrap();
        assert_eq!(meta["file_count"], 2);
        assert!(meta["total_size_bytes"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_cache_validation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().to_path_buf());

        // Create a valid cached LoRA
        let lora_dir = temp_dir.path().join("valid-lora");
        fs::create_dir(&lora_dir).unwrap();
        fs::write(lora_dir.join("adapter_config.json"), "{}").unwrap();
        fs::write(lora_dir.join("adapter_model.safetensors"), "").unwrap();

        assert!(cache.validate_cached("valid-lora").unwrap());

        // Create an invalid cached LoRA (missing weights)
        let invalid_dir = temp_dir.path().join("invalid-lora");
        fs::create_dir(&invalid_dir).unwrap();
        fs::write(invalid_dir.join("adapter_config.json"), "{}").unwrap();

        assert!(!cache.validate_cached("invalid-lora").unwrap());

        // Non-existent LoRA
        assert!(!cache.validate_cached("non-existent").unwrap());
    }

    #[tokio::test]
    async fn test_downloader_with_local_source() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LoRACache::new(temp_dir.path().join("cache"));

        // Create a local LoRA
        let lora_dir = temp_dir.path().join("local-lora");
        fs::create_dir(&lora_dir).unwrap();
        fs::write(lora_dir.join("adapter_config.json"), "{}").unwrap();
        fs::write(lora_dir.join("adapter_model.safetensors"), "weights").unwrap();

        let source: Arc<dyn LoRASourceTrait> = Arc::new(LocalLoRASource::new());
        let downloader = LoRADownloader::new(vec![source], cache);

        let lora_uri = format!("file://{}", lora_dir.display());
        let downloaded_path = downloader.download_if_needed(&lora_uri).await.unwrap();

        // For local files, should return original path
        assert_eq!(downloaded_path, lora_dir);
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let temp_dir = TempDir::new().unwrap();
        let cache_root = temp_dir.path().join("cache");
        fs::create_dir(&cache_root).unwrap();

        let cache = LoRACache::new(cache_root.clone());

        // Pre-populate cache with a LoRA at path "my-loras/test-lora"
        // This matches the cache key that will be derived from "s3://bucket/my-loras/test-lora"
        let cached_lora = cache_root.join("my-loras").join("test-lora");
        fs::create_dir_all(&cached_lora).unwrap();
        fs::write(cached_lora.join("adapter_config.json"), "{}").unwrap();
        fs::write(
            cached_lora.join("adapter_model.safetensors"),
            "cached_weights",
        )
        .unwrap();

        // Create downloader with empty sources - no network fetch possible
        let downloader = LoRADownloader::new(vec![], cache);

        // Use an S3 URI that maps to the pre-populated cache entry
        let lora_uri = "s3://bucket/my-loras/test-lora";

        // Call download_if_needed - should hit cache without attempting any downloads
        let result_path = downloader.download_if_needed(lora_uri).await.unwrap();

        // Assert the returned path equals the cached path
        assert_eq!(result_path, cached_lora);

        // Verify no new downloads occurred by checking the weights file content is unchanged
        let weights_content = fs::read_to_string(result_path.join("adapter_model.safetensors"))
            .unwrap();
        assert_eq!(weights_content, "cached_weights");

        // Verify the cache directory structure is unchanged
        assert!(result_path.join("adapter_config.json").exists());
        assert!(result_path.join("adapter_model.safetensors").exists());
    }
}


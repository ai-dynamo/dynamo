// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

use hf_hub::Cache;
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::download as mx;

use dynamo_runtime::config::environment_names::model as env_model;

/// Check if a model is already cached in the HuggingFace hub cache directory.
/// Returns the path to the cached model directory if found, None otherwise.
///
/// Uses hf-hub's Cache API to check for cached files. For tokenizer-only downloads
/// (ignore_weights=true), we check for config.json and tokenizer files.
/// For full downloads, we also require weight files to be present.
fn get_cached_model_path(model_name: &str, ignore_weights: bool) -> Option<PathBuf> {
    let cache = Cache::new(get_model_express_cache_dir());
    let repo = cache.model(model_name.to_string());

    // Check for required config file
    let config_path = repo.get("config.json")?;

    // Check for tokenizer files (at least one must exist)
    let has_tokenizer = repo.get("tokenizer.json").is_some()
        || repo.get("tokenizer_config.json").is_some()
        || repo.get("tiktoken.model").is_some()
        || has_tiktoken_file(config_path.parent()?);

    if !has_tokenizer {
        return None;
    }

    // For full downloads, check for weight files
    if !ignore_weights {
        // Check common weight file patterns - at least one must exist
        let has_weights = repo.get("model.safetensors").is_some()
            || repo.get("pytorch_model.bin").is_some()
            || repo.get("model.safetensors.index.json").is_some()
            || repo.get("pytorch_model.bin.index.json").is_some();

        if !has_weights {
            return None;
        }
    }

    // Return the parent directory (snapshot dir) containing the model files
    let snapshot_path = config_path.parent()?.to_path_buf();
    tracing::info!("Found cached model '{model_name}' at {snapshot_path:?}, skipping download");
    Some(snapshot_path)
}

/// Check if the snapshot directory contains any `*.tiktoken` file (e.g. `qwen.tiktoken`).
fn has_tiktoken_file(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|ext| ext == "tiktoken"))
}

/// Check if offline mode is enabled via HF_HUB_OFFLINE environment variable.
fn is_offline_mode() -> bool {
    env::var(env_model::huggingface::HF_HUB_OFFLINE)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Check if shared-storage mode is disabled via MODEL_EXPRESS_NO_SHARED_STORAGE.
/// When true, the Model Express client streams files from the server over gRPC
/// instead of relying on a shared filesystem path. This is required when the
/// server and worker pods do not share a filesystem (e.g. RWO PVCs, cross-namespace
/// deployments).
fn is_no_shared_storage() -> bool {
    env::var(env_model::model_express::MODEL_EXPRESS_NO_SHARED_STORAGE)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Download a model using ModelExpress client. Returns the cached path
/// immediately if the model is already in the HF cache; otherwise requests it
/// from the server and falls back to direct download on server failure.
/// If ignore_weights is true, model weight files will be skipped.
/// Returns the path to the model files.
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    // Cache-first: skip the remote round-trip when the snapshot is already on disk.
    if let Some(cached_path) = get_cached_model_path(&model_name, ignore_weights) {
        tracing::info!("Using cached model '{model_name}' at {cached_path:?}");
        return Ok(cached_path);
    }

    if is_offline_mode() {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let mut config: MxClientConfig = MxClientConfig::default();
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }

    let result = match MxClient::new(config).await {
        Ok(mut client) => {
            tracing::info!("Successfully connected to ModelExpress server");
            match client
                .request_model_with_provider_and_fallback(
                    &model_name,
                    MxModelProvider::HuggingFace,
                    ignore_weights,
                )
                .await
            {
                Ok(()) => {
                    tracing::info!("Server download succeeded for model: {model_name}");
                    match client
                        .get_model_path(&model_name, MxModelProvider::HuggingFace)
                        .await
                    {
                        Ok(path) => Ok(path),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to resolve local model path after server download for '{model_name}': {e}. \
                                Falling back to direct download."
                            );
                            mx_download_direct(&model_name, ignore_weights).await
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Server download failed for model '{model_name}': {e}. Falling back to direct download."
                    );
                    mx_download_direct(&model_name, ignore_weights).await
                }
            }
        }
        Err(e) => {
            tracing::warn!("Cannot connect to ModelExpress server: {e}. Using direct download.");
            mx_download_direct(&model_name, ignore_weights).await
        }
    };

    match result {
        Ok(path) => {
            tracing::info!("ModelExpress download completed successfully for model: {model_name}");
            Ok(path)
        }
        Err(e) => {
            tracing::warn!("ModelExpress download failed for model '{model_name}': {e}");
            Err(e)
        }
    }
}

// Direct download using the ModelExpress client.
async fn mx_download_direct(model_name: &str, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir();
    mx::download_model(
        model_name,
        MxModelProvider::HuggingFace,
        Some(cache_dir),
        ignore_weights,
    )
    .await
}

// TODO: remove in the future. This is a temporary workaround to find common
// cache directory between client and server.
fn get_model_express_cache_dir() -> PathBuf {
    // Check HF_HUB_CACHE environment variable
    // reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubcache
    if let Ok(cache_path) = env::var(env_model::huggingface::HF_HUB_CACHE) {
        return PathBuf::from(cache_path);
    }

    // Check HF_HOME environment variable (standard Hugging Face cache directory)
    // reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome
    if let Ok(hf_home) = env::var(env_model::huggingface::HF_HOME) {
        return PathBuf::from(hf_home).join("hub");
    }

    if let Ok(cache_path) = env::var(env_model::model_express::MODEL_EXPRESS_CACHE_PATH) {
        return PathBuf::from(cache_path);
    }

    let home = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    PathBuf::from(home).join(".cache/huggingface/hub")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_from_hf_with_model_express() {
        let test_path = PathBuf::from("test-model");
        let _result: anyhow::Result<PathBuf> = from_hf(test_path, false).await;
    }

    #[test]
    fn test_get_model_express_cache_dir() {
        let cache_dir = get_model_express_cache_dir();
        assert!(!cache_dir.to_string_lossy().is_empty());
        assert!(cache_dir.is_absolute() || cache_dir.starts_with("."));
    }

    #[serial_test::serial]
    #[test]
    fn test_get_model_express_cache_dir_with_hf_home() {
        // Test that HF_HOME is respected when set
        unsafe {
            // Clear other cache env vars to ensure HF_HOME is tested
            env::remove_var(env_model::huggingface::HF_HUB_CACHE);
            env::remove_var(env_model::model_express::MODEL_EXPRESS_CACHE_PATH);
            env::set_var(env_model::huggingface::HF_HOME, "/custom/cache/path");
            let cache_dir = get_model_express_cache_dir();
            assert_eq!(cache_dir, PathBuf::from("/custom/cache/path/hub"));

            // Clean up
            env::remove_var(env_model::huggingface::HF_HOME);
        }
    }

    /// Materialize a minimal HF cache snapshot at `cache_root` for `repo_id`,
    /// optionally including a fake weight file. Local writes only, no network.
    fn write_fake_snapshot(cache_root: &Path, repo_id: &str, include_weights: bool) -> PathBuf {
        let folder = format!("models--{}", repo_id.replace('/', "--"));
        let repo_dir = cache_root.join(&folder);
        let sha = "deadbeefcafebabe1234567890abcdef12345678";
        let snapshot_dir = repo_dir.join("snapshots").join(sha);
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), sha).unwrap();
        std::fs::write(snapshot_dir.join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        std::fs::write(snapshot_dir.join("tokenizer.json"), "{}").unwrap();
        if include_weights {
            std::fs::write(snapshot_dir.join("model.safetensors"), b"fake-weights").unwrap();
        }
        snapshot_dir
    }

    /// Set HF_HUB_CACHE to `path` and clear sibling vars that would otherwise take
    /// precedence in `get_model_express_cache_dir`.
    fn force_hf_hub_cache(path: &Path) {
        unsafe {
            env::remove_var(env_model::huggingface::HF_HOME);
            env::remove_var(env_model::model_express::MODEL_EXPRESS_CACHE_PATH);
            env::set_var(env_model::huggingface::HF_HUB_CACHE, path);
        }
    }

    fn clear_hf_hub_cache() {
        unsafe {
            env::remove_var(env_model::huggingface::HF_HUB_CACHE);
        }
    }

    #[serial_test::serial]
    #[test]
    fn test_get_cached_model_path_finds_populated_snapshot() {
        let temp = tempfile::TempDir::new().unwrap();
        let snapshot = write_fake_snapshot(temp.path(), "test-org/test-model", true);
        force_hf_hub_cache(temp.path());

        let resolved = get_cached_model_path("test-org/test-model", false);

        clear_hf_hub_cache();

        let resolved = resolved.expect("expected cache hit on fully populated snapshot");
        assert_eq!(resolved, snapshot);
    }

    #[serial_test::serial]
    #[test]
    fn test_get_cached_model_path_missing_model_returns_none() {
        let temp = tempfile::TempDir::new().unwrap();
        force_hf_hub_cache(temp.path());

        let resolved = get_cached_model_path("nonexistent-org/nonexistent-model", false);

        clear_hf_hub_cache();

        assert!(
            resolved.is_none(),
            "expected no cache hit when model is absent"
        );
    }

    #[serial_test::serial]
    #[test]
    fn test_get_cached_model_path_weights_requirement() {
        // Snapshot has config + tokenizer but no weight files. Required-weights
        // mode misses; ignore_weights mode hits.
        let temp = tempfile::TempDir::new().unwrap();
        write_fake_snapshot(temp.path(), "test-org/tokenizer-only", false);
        force_hf_hub_cache(temp.path());

        let weights_required = get_cached_model_path("test-org/tokenizer-only", false);
        let weights_optional = get_cached_model_path("test-org/tokenizer-only", true);

        clear_hf_hub_cache();

        assert!(
            weights_required.is_none(),
            "ignore_weights=false should miss when weight files are absent",
        );
        assert!(
            weights_optional.is_some(),
            "ignore_weights=true should hit on config + tokenizer alone",
        );
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn test_from_hf_returns_cached_path_without_network() {
        // Made-up repo that doesn't exist on HF — Ok result proves cache-first bypassed the network.
        let temp = tempfile::TempDir::new().unwrap();
        let snapshot = write_fake_snapshot(temp.path(), "test-org/cache-first-only", true);
        force_hf_hub_cache(temp.path());

        let resolved = from_hf("test-org/cache-first-only", false).await;

        clear_hf_hub_cache();

        let path = resolved.expect("expected cache-first to succeed without network");
        assert_eq!(path, snapshot);
    }
}

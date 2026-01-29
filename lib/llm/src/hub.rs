// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::download as mx;

use dynamo_runtime::config::environment_names::model as env_model;

/// Check if a model is already cached in the HuggingFace hub cache directory.
/// Returns the path to the cached model if found, None otherwise.
///
/// For tokenizer-only downloads (ignore_weights=true), we check for tokenizer files.
/// For full downloads, we also check for weight files (*.safetensors or *.bin).
fn get_cached_model_path(model_name: &str, ignore_weights: bool) -> Option<PathBuf> {
    let cache_dir = get_model_express_cache_dir();

    // HuggingFace hub cache structure: models--{org}--{model}/snapshots/{sha}/
    let safe_name = model_name.replace('/', "--");
    let model_dir = cache_dir.join(format!("models--{}", safe_name));

    let snapshots_dir = model_dir.join("snapshots");
    if !snapshots_dir.exists() {
        return None;
    }

    // Find the most recent snapshot (by modification time)
    let mut snapshots: Vec<_> = fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .collect();

    if snapshots.is_empty() {
        return None;
    }

    // Sort by modification time (most recent first)
    snapshots.sort_by(|a, b| {
        let a_time = a.metadata().and_then(|m| m.modified()).ok();
        let b_time = b.metadata().and_then(|m| m.modified()).ok();
        b_time.cmp(&a_time)
    });

    for snapshot in snapshots {
        let snapshot_path = snapshot.path();

        // Check for required tokenizer files
        let has_tokenizer = snapshot_path.join("tokenizer.json").exists()
            || snapshot_path.join("tokenizer_config.json").exists();
        let has_config = snapshot_path.join("config.json").exists();

        if !has_tokenizer || !has_config {
            continue;
        }

        // For full downloads, also check for weight files
        if !ignore_weights {
            let has_weights = fs::read_dir(&snapshot_path).ok().is_some_and(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.ends_with(".safetensors")
                        || name.ends_with(".bin")
                        || name.ends_with(".pt")
                })
            });
            if !has_weights {
                continue;
            }
        }

        tracing::info!("Found cached model '{model_name}' at {snapshot_path:?}, skipping download");
        return Some(snapshot_path);
    }

    None
}

/// Check if offline mode is enabled via HF_HUB_OFFLINE environment variable.
fn is_offline_mode() -> bool {
    env::var(env_model::huggingface::HF_HUB_OFFLINE)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Download a model using ModelExpress client. The client first requests for the model
/// from the server and fallbacks to direct download in case of server failure.
/// If ignore_weights is true, model weight files will be skipped
/// Returns the path to the model files
///
/// If HF_HUB_OFFLINE=1 is set and the model is already cached, returns the cached
/// path without making any API calls to HuggingFace.
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    // In offline mode, check cache first and return immediately if found
    if is_offline_mode() {
        if let Some(cached_path) = get_cached_model_path(&model_name, ignore_weights) {
            tracing::info!(
                "Offline mode: using cached model '{model_name}' without API validation"
            );
            return Ok(cached_path);
        }
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let mut config: MxClientConfig = MxClientConfig::default();
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
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
                    match client.get_model_path(&model_name).await {
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
}

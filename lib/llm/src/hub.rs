// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unexpected_cfgs)]

use hf_hub::api::tokio::ApiBuilder;
use std::env;
use std::path::{Path, PathBuf};

#[cfg(feature = "model-express")]
use model_express_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
#[cfg(feature = "model-express")]
use model_express_common::download as mx;

const MODEL_EXPRESS_ENDPOINT_ENV_VAR: &str = "MODEL_EXPRESS_URL";
const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";

/// Checks if a file is a model weight file
fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

/// Attempt to download a model from Hugging Face using ModelExpress client
/// Only called when model-express feature is enabled, otherwise it will fall back to homonymous hf-hub function
/// Returns the directory it is in
/// If ignore_weights is true, model weight files will be skipped
#[cfg(feature = "model-express")]
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    // Only use ModelExpress if the environment variable is explicitly set
    if let Ok(endpoint) = env::var(MODEL_EXPRESS_ENDPOINT_ENV_VAR) {
        tracing::info!(
            "ModelExpress endpoint configured, attempting to use ModelExpress for model: {model_name}"
        );

        let config: MxClientConfig = MxClientConfig::default().with_endpoint(endpoint.clone());

        let result = match MxClient::new(config.clone()).await {
            Ok(mut client) => {
                tracing::info!("Successfully connected to ModelExpress server");
                match client
                    .request_model_with_provider_and_fallback(
                        &model_name,
                        MxModelProvider::HuggingFace,
                    )
                    .await
                {
                    Ok(()) => {
                        tracing::info!("Server download succeeded for model: {model_name}");
                        get_mx_model_path_from_cache(&model_name)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Server download failed for model '{model_name}': {e}. Falling back to direct download."
                        );
                        mx_download_direct(&model_name).await
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Cannot connect to ModelExpress server: {e}. Using direct download."
                );
                mx_download_direct(&model_name).await
            }
        };

        match result {
            Ok(path) => {
                tracing::info!(
                    "ModelExpress download completed successfully for model: {model_name}"
                );
                return Ok(path);
            }
            Err(e) => {
                tracing::warn!(
                    "ModelExpress download failed for model '{model_name}': {e}. Falling back to hf-hub."
                );
            }
        }
    }

    tracing::info!("Using hf-hub for model: {model_name}");
    download_with_hf_hub(&model_name, ignore_weights).await
}

/// Attempt to download a model from Hugging Face using hf-hub directly
/// Called when model-express feature is not enabled
/// Returns the directory it is in
/// If ignore_weights is true, model weight files will be skipped
#[cfg(not(feature = "model-express"))]
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    if env::var(MODEL_EXPRESS_ENDPOINT_ENV_VAR).is_ok() {
        tracing::warn!(
            "ModelExpress endpoint configured but model-express feature not enabled. Using hf-hub."
        );
    }

    tracing::info!("Using hf-hub for model: {model_name}");
    download_with_hf_hub(&model_name, ignore_weights).await
}

// Direct download using the ModelExpress client.
#[cfg(feature = "model-express")]
async fn mx_download_direct(model_name: &str) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir();
    mx::download_model(model_name, MxModelProvider::HuggingFace, Some(cache_dir)).await
}

/// Attempt to download a model from Hugging Face with hf-hub
/// Returns the directory it is in
/// If ignore_weights is true, model weight files will be skipped
async fn download_with_hf_hub(model_name: &str, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let token = env::var(HF_TOKEN_ENV_VAR).ok();

    let api = ApiBuilder::from_env()
        .with_progress(true)
        .with_token(token)
        .high()
        .build()?;

    let repo = api.model(model_name.to_string());

    let info = repo.info().await
        .map_err(|e| anyhow::anyhow!("Failed to fetch model '{model_name}' from HuggingFace: {e}. Is this a valid HuggingFace ID?"))?;

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{model_name}' exists but contains no downloadable files."
        ));
    }

    let mut model_path = PathBuf::new();
    let mut files_downloaded = false;

    for sibling in info.siblings {
        if is_ignored_file(&sibling.rfilename) || is_image_file(&sibling.rfilename) {
            continue;
        }

        if ignore_weights && is_weight_file(&sibling.rfilename) {
            continue;
        }

        match repo.get(&sibling.rfilename).await {
            Ok(path) => {
                model_path = path;
                files_downloaded = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download file '{}' from model '{model_name}': {e}",
                    sibling.rfilename
                ));
            }
        }
    }

    if !files_downloaded {
        let file_type = if ignore_weights {
            "non-weight"
        } else {
            "valid"
        };
        return Err(anyhow::anyhow!(
            "No {file_type} files found for model '{model_name}'."
        ));
    }

    match model_path.parent() {
        Some(path) => Ok(path.to_path_buf()),
        None => Err(anyhow::anyhow!(
            "Invalid HF cache path: {}",
            model_path.display()
        )),
    }
}

fn is_ignored_file(filename: &str) -> bool {
    const IGNORED_FILES: [&str; 5] = [
        ".gitattributes",
        "LICENSE",
        "LICENSE.txt",
        "README.md",
        "USE_POLICY.md",
    ];
    IGNORED_FILES.contains(&filename)
}

fn is_image_file(filename: &str) -> bool {
    filename.ends_with(".png")
        || filename.ends_with("PNG")
        || filename.ends_with(".jpg")
        || filename.ends_with("JPG")
        || filename.ends_with(".jpeg")
        || filename.ends_with("JPEG")
}

#[cfg(feature = "model-express")]
fn get_mx_model_path_from_cache(model_name: &str) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir();
    let model_dir = cache_dir.join(model_name);

    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "Model '{model_name}' was downloaded but directory not found at expected location: {}",
            model_dir.display()
        ));
    }

    Ok(model_dir)
}

#[cfg(feature = "model-express")]
fn get_model_express_cache_dir() -> PathBuf {
    if let Ok(cache_path) = env::var("HF_HUB_CACHE") {
        return PathBuf::from(cache_path);
    }

    if let Ok(cache_path) = env::var("MODEL_EXPRESS_PATH") {
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

    #[cfg(feature = "model-express")]
    #[test]
    fn test_get_model_express_cache_dir() {
        let cache_dir = get_model_express_cache_dir();
        assert!(!cache_dir.to_string_lossy().is_empty());
        assert!(cache_dir.is_absolute() || cache_dir.starts_with("."));
    }

    #[test]
    fn test_is_ignored_file() {
        assert!(is_ignored_file(".gitattributes"));
        assert!(is_ignored_file("LICENSE"));
        assert!(is_ignored_file("LICENSE.txt"));
        assert!(is_ignored_file("README.md"));
        assert!(is_ignored_file("USE_POLICY.md"));

        assert!(!is_ignored_file("model.bin"));
        assert!(!is_ignored_file("tokenizer.json"));
        assert!(!is_ignored_file("config.json"));
    }

    #[test]
    fn test_is_weight_file() {
        assert!(is_weight_file("model.bin"));
        assert!(is_weight_file("model.safetensors"));
        assert!(is_weight_file("model.h5"));
        assert!(is_weight_file("model.msgpack"));
        assert!(is_weight_file("model.ckpt.index"));

        assert!(!is_weight_file("tokenizer.json"));
        assert!(!is_weight_file("config.json"));
        assert!(!is_weight_file("README.md"));
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file("image.png"));
        assert!(is_image_file("image.PNG"));
        assert!(is_image_file("photo.jpg"));
        assert!(is_image_file("photo.JPG"));
        assert!(is_image_file("picture.jpeg"));
        assert!(is_image_file("picture.JPEG"));

        assert!(!is_image_file("model.bin"));
        assert!(!is_image_file("tokenizer.json"));
        assert!(!is_image_file("config.json"));
        assert!(!is_image_file("README.md"));
    }
}

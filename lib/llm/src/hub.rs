// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use model_express_client::{Client, ClientConfig, ModelProvider};
use model_express_common::download;
use hf_hub::api::tokio::ApiBuilder;
use std::env;
use std::path::{Path, PathBuf};

const MODEL_EXPRESS_ENDPOINT_ENV_VAR: &str = "MODEL_EXPRESS_ENDPOINT";
const DEFAULT_MODEL_EXPRESS_ENDPOINT: &str = "http://localhost:8001";
const HF_TOKEN_ENV_VAR: &str = "HF_TOKEN";

/// Checks if a file is a model weight file
fn is_weight_file(filename: &str) -> bool {
    filename.ends_with(".bin")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".h5")
        || filename.ends_with(".msgpack")
        || filename.ends_with(".ckpt.index")
}

/// Attempt to download a model from Hugging Face using ModelExpress client with hf-hub fallback
/// Returns the directory it is in
/// If ignore_weights is true, model weight files will be skipped
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();
    
    // Get ModelExpress server endpoint from environment or use default
    let endpoint = env::var(MODEL_EXPRESS_ENDPOINT_ENV_VAR)
        .unwrap_or_else(|_| DEFAULT_MODEL_EXPRESS_ENDPOINT.to_string());
    
    // Create client configuration
    let config = ClientConfig::for_testing(&endpoint);
    
    // First try to use the server with fallback
    let result = match Client::new(config.clone()).await {
        Ok(mut client) => {
            // Try server-based download first
            match client.request_model_with_provider_and_fallback(&model_name, ModelProvider::HuggingFace).await {
                Ok(()) => {
                    // Server download succeeded, now get the path
                    get_model_path_from_cache(&model_name)
                }
                Err(e) => {
                    // Server failed, fallback to direct download
                    tracing::warn!("Server download failed for model '{}': {}. Falling back to direct download.", model_name, e);
                    download_model_directly(&model_name).await
                }
            }
        }
        Err(e) => {
            // Can't connect to server, use direct download
            tracing::warn!("Cannot connect to ModelExpress server: {}. Using direct download.", e);
            download_model_directly(&model_name).await
        }
    };
    
    // If ModelExpress methods fail, try hf-hub as final fallback
    match result {
        Ok(path) => Ok(path),
        Err(e) => {
            tracing::warn!("ModelExpress download failed for model '{}': {}. Falling back to hf-hub.", model_name, e);
            download_with_hf_hub(&model_name, ignore_weights).await
        }
    }
}

/// Download model directly using ModelExpress download function
async fn download_model_directly(model_name: &str) -> anyhow::Result<PathBuf> {
    // Get cache directory
    let cache_dir = get_model_express_cache_dir();
    
    // Use ModelExpress download function directly
    download::download_model(model_name, ModelProvider::HuggingFace, Some(cache_dir)).await
}

/// Download model using hf-hub as final fallback
async fn download_with_hf_hub(model_name: &str, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    tracing::info!("Downloading model '{}' using hf-hub", model_name);
    
    // Get cache directory
    let cache_dir = get_model_express_cache_dir();
    
    // Get Hugging Face token if available
    let token = env::var(HF_TOKEN_ENV_VAR).ok();
    let api = ApiBuilder::from_env()
        .with_progress(true)
        .with_token(token)
        .high()
        .with_cache_dir(cache_dir)
        .build()?;
    
    let repo = api.model(model_name.to_string());
    
    // Get model info
    let info = repo.info().await
        .map_err(|e| anyhow::anyhow!("Failed to fetch model '{}' from HuggingFace: {}. Is this a valid HuggingFace ID?", model_name, e))?;
    
    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!("Model '{}' exists but contains no downloadable files.", model_name));
    }
    
    // Download files (excluding ignored files)
    let mut model_path = PathBuf::new();
    let mut files_downloaded = false;
    
    for sibling in info.siblings {
        if is_ignored_file(&sibling.rfilename) || is_image_file(&sibling.rfilename) {
            continue;
        }

        // If ignore_weights is true, skip weight files
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
                    "Failed to download file '{}' from model '{}': {}",
                    sibling.rfilename,
                    model_name,
                    e
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
            "No {} files found for model '{}'.",
            file_type,
            model_name
        ));
    }
    
    // Return the parent directory (model directory)
    match model_path.parent() {
        Some(path) => {
            tracing::info!("Successfully downloaded model '{}' using hf-hub to: {}", model_name, path.display());
            Ok(path.to_path_buf())
        }
        None => Err(anyhow::anyhow!("Invalid HF cache path: {}", model_path.display())),
    }
}

/// Check if a file should be ignored during download
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

/// Check if a file is an image file that should be ignored
fn is_image_file(filename: &str) -> bool {
    filename.ends_with(".png")
        || filename.ends_with("PNG")
        || filename.ends_with(".jpg")
        || filename.ends_with("JPG")
        || filename.ends_with(".jpeg")
        || filename.ends_with("JPEG")
}

/// Get the model path from cache after server download
fn get_model_path_from_cache(model_name: &str) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir();
    let model_dir = cache_dir.join(model_name);
    
    if !model_dir.exists() {
        return Err(anyhow::anyhow!(
            "Model '{}' was downloaded but directory not found at expected location: {}",
            model_name,
            model_dir.display()
        ));
    }
    
    Ok(model_dir)
}

/// Get the ModelExpress cache directory
fn get_model_express_cache_dir() -> PathBuf {
    // Try to get from environment variable first
    if let Ok(cache_path) = env::var("HF_HUB_CACHE") {
        return PathBuf::from(cache_path);
    }
    
    // Fall back to default Hugging Face cache location
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
        // This test verifies that the function compiles and can be called
        // We don't actually download a model in tests to avoid network dependencies
        
        // Test with a simple path
        let test_path = PathBuf::from("test-model");
        
        // The function should compile and be callable
        // In a real scenario, this would attempt to download the model
        // For testing purposes, we just verify the function signature is correct
        let _result: anyhow::Result<PathBuf> = from_hf(test_path, false).await;
        
        // If we get here, the function compiled and ran without panicking
        // The actual result will depend on whether ModelExpress server is available
        // and whether the test model exists
    }

    #[test]
    fn test_get_model_express_cache_dir() {
        let cache_dir = get_model_express_cache_dir();
        
        // Should return a valid path
        assert!(!cache_dir.to_string_lossy().is_empty());
        
        // Should be an absolute path
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
}

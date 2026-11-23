//! HuggingFace Hub integration for downloading model config files.
//!
//! When the model source is a HuggingFace model ID (e.g. `Qwen/Qwen3-VL-8B-Instruct`)
//! rather than a local directory, this module downloads `config.json` and
//! `preprocessor_config.json` / video processor config files to the local HF
//! cache and returns the resolved path.

use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::api::tokio::ApiBuilder;

const HF_TOKEN_ENV: &str = "HF_TOKEN";

/// Resolve a model source to a local directory containing `config.json`.
///
/// If the source is already a local directory with `config.json`, returns it as-is.
/// Otherwise, treats the source as a HuggingFace model ID and downloads
/// `config.json` plus optional processor configs to the local HF cache.
pub async fn resolve_model_config_dir(source: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(source);
    if path.join("config.json").exists() {
        return Ok(path.to_path_buf());
    }

    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Ok(token) = std::env::var(HF_TOKEN_ENV) {
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }
    }
    let api = builder
        .build()
        .context("Failed to build HuggingFace API client")?;
    let repo = api.model(source.to_string());

    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("Failed to download config.json for model '{source}'"))?;

    // Best-effort download of processor configs (not all models have them).
    let _ = repo.get("preprocessor_config.json").await;
    let _ = repo.get("video_preprocessor_config.json").await;
    let _ = repo.get("processor_config.json").await;

    config_path
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("Invalid HF cache path for model '{source}'"))
}

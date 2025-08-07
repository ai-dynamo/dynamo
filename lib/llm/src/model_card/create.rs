// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::model_card::model::ModelDeploymentCard;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::model_card::model::{ModelInfoType, PromptFormatterArtifact, TokenizerKind};

use super::model::GenerationConfig;

impl ModelDeploymentCard {
    /// Allow user to override the name we register this model under.
    /// Corresponds to vllm's `--served-model-name`.
    pub fn set_name(&mut self, name: &str) {
        self.display_name = name.to_string();
        self.service_name = name.to_string();
    }

    /// Build an in-memory ModelDeploymentCard from either:
    /// - a folder containing config.json, tokenizer.json and token_config.json
    /// - a GGUF file
    pub async fn load(config_path: impl AsRef<Path>) -> anyhow::Result<ModelDeploymentCard> {
        let config_path = config_path.as_ref();
        if config_path.is_dir() {
            Self::from_local_path(config_path).await
        } else {
            Self::from_gguf(config_path).await
        }
    }

    /// Creates a ModelDeploymentCard from a local directory path.
    ///
    /// Currently HuggingFace format is supported and following files are expected:
    /// - config.json: Model configuration in HuggingFace format
    /// - tokenizer.json: Tokenizer configuration in HuggingFace format
    /// - tokenizer_config.json: Optional prompt formatter configuration
    ///
    /// # Arguments
    /// * `local_root_dir` - Path to the local model directory
    ///
    /// # Errors
    /// Returns an error if:
    /// - The path doesn't exist or isn't a directory
    /// - The path contains invalid Unicode characters
    /// - Required model files are missing or invalid
    async fn from_local_path(local_root_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let local_root_dir = local_root_dir.as_ref();
        check_valid_local_repo_path(local_root_dir)?;
        let repo_id = local_root_dir
            .canonicalize()?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path contains invalid Unicode"))?
            .to_string();
        let model_name = local_root_dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid model directory name"))?;
        Self::from_repo(&repo_id, model_name).await
    }

    async fn from_gguf(gguf_file: &Path) -> anyhow::Result<Self> {
        let model_name = gguf_file
            .iter()
            .next_back()
            .map(|n| n.to_string_lossy().to_string());
        let Some(model_name) = model_name else {
            // I think this would only happy on an empty path
            anyhow::bail!(
                "Could not extract model name from path '{}'",
                gguf_file.display()
            );
        };

        // TODO: we do this in HFConfig also, unify
        let content = super::model::load_gguf(gguf_file)?;
        let context_length = content.get_metadata()[&format!("{}.context_length", content.arch())]
            .to_u32()
            .unwrap_or(0);
        tracing::debug!(context_length, "Loaded context length from GGUF");

        Ok(Self {
            display_name: model_name.to_string(),
            service_name: model_name.to_string(),
            model_info: Some(ModelInfoType::GGUF(gguf_file.to_path_buf())),
            tokenizer: Some(TokenizerKind::from_gguf(gguf_file)?),
            gen_config: None, // AFAICT there is no equivalent in a GGUF
            prompt_formatter: Some(PromptFormatterArtifact::GGUF(gguf_file.to_path_buf())),
            chat_template_file: None,
            prompt_context: None, // TODO - auto-detect prompt context
            revision: 0,
            last_published: None,
            context_length,
            kv_cache_block_size: 0,
            migration_limit: 0,
            user_data: None,
        })
    }

    #[allow(dead_code)]
    async fn from_ngc_repo(_: &str) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "ModelDeploymentCard::from_ngc_repo is not implemented"
        ))
    }

    async fn from_repo(repo_id: &str, model_name: &str) -> anyhow::Result<Self> {
        // This is usually the right choice
        let context_length = crate::file_json_field(
            &PathBuf::from(repo_id).join("config.json"),
            "max_position_embeddings",
        )
        // But sometimes this is
        .or_else(|_| {
            crate::file_json_field(
                &PathBuf::from(repo_id).join("tokenizer_config.json"),
                "model_max_length",
            )
        })
        // If neither of those are present let the engine default it
        .unwrap_or(0);

        Ok(Self {
            display_name: model_name.to_string(),
            service_name: model_name.to_string(),
            model_info: Some(ModelInfoType::from_repo(repo_id).await?),
            tokenizer: Some(TokenizerKind::from_repo(repo_id).await?),
            gen_config: GenerationConfig::from_repo(repo_id).await.ok(), // optional
            prompt_formatter: PromptFormatterArtifact::from_repo(repo_id).await?,
            chat_template_file: PromptFormatterArtifact::chat_template_from_repo(repo_id).await?,
            prompt_context: None, // TODO - auto-detect prompt context
            revision: 0,
            last_published: None,
            context_length,
            kv_cache_block_size: 0, // set later
            migration_limit: 0,
            user_data: None,
        })
    }
}

impl ModelInfoType {
    pub async fn from_repo(repo_id: &str) -> Result<Self> {
        Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract model info from repo {}", repo_id))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfConfigJson(
            check_for_file(repo, "config.json").await?,
        ))
    }
}

impl PromptFormatterArtifact {
    pub async fn from_repo(repo_id: &str) -> Result<Option<Self>> {
        // we should only error if we expect a prompt formatter and it's not found
        // right now, we don't know when to expect it, so we just return Ok(Some/None)
        Ok(Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract prompt format from repo {}", repo_id))
            .ok())
    }

    pub async fn chat_template_from_repo(repo_id: &str) -> Result<Option<Self>> {
        Ok(Self::chat_template_try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract prompt format from repo {}", repo_id))
            .ok())
    }

    async fn chat_template_try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfChatTemplate(
            check_for_file(repo, "chat_template.jinja").await?,
        ))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfTokenizerConfigJson(
            check_for_file(repo, "tokenizer_config.json").await?,
        ))
    }
}

impl TokenizerKind {
    pub async fn from_repo(repo_id: &str) -> Result<Self> {
        Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract tokenizer kind from repo {}", repo_id))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfTokenizerJson(
            check_for_file(repo, "tokenizer.json").await?,
        ))
    }
}

impl GenerationConfig {
    pub async fn from_repo(repo_id: &str) -> Result<Self> {
        Self::try_is_hf_repo(repo_id)
            .await
            .with_context(|| format!("unable to extract generation config from repo {repo_id}"))
    }

    async fn try_is_hf_repo(repo: &str) -> anyhow::Result<Self> {
        Ok(Self::HfGenerationConfigJson(
            check_for_file(repo, "generation_config.json").await?,
        ))
    }
}

/// Checks if the provided path contains the expected file.
async fn check_for_file(repo_id: &str, file: &str) -> anyhow::Result<String> {
    let p = PathBuf::from(repo_id).join(file);
    let name = p.display().to_string();
    if !p.exists() {
        anyhow::bail!("File not found: {name}")
    }
    Ok(name)
}

/// Checks if the provided path is a valid local repository path.
///
/// # Arguments
/// * `path` - Path to validate
///
/// # Errors
/// Returns an error if the path doesn't exist or isn't a directory
fn check_valid_local_repo_path(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Model path does not exist: {}",
            path.display()
        ));
    }

    if !path.is_dir() {
        return Err(anyhow::anyhow!(
            "Model path is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

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

use std::collections::HashMap;

use crate::model_card::model::ModelDeploymentCard;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::model_card::model::{ModelInfoType, PromptFormatterArtifact, TokenizerKind};

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
        Ok(Self {
            display_name: model_name.to_string(),
            service_name: model_name.to_string(),
            model_info: Some(ModelInfoType::GGUF(gguf_file.to_path_buf())),
            tokenizer: Some(TokenizerKind::from_gguf(gguf_file)?),
            prompt_formatter: Some(PromptFormatterArtifact::GGUF(gguf_file.to_path_buf())),
            prompt_context: None, // TODO - auto-detect prompt context
            revision: 0,
            last_published: None,
        })
    }

    /// TODO: This will be implemented after nova-hub is integrated with the model-card
    /// TODO: Attempt to auto-detect model type and construct an MDC from a NGC repo
    #[allow(dead_code)]
    async fn from_ngc_repo(_: &str) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "ModelDeploymentCard::from_ngc_repo is not implemented"
        ))
    }

    async fn from_repo(repo_id: &str, model_name: &str) -> anyhow::Result<Self> {
        Ok(Self {
            display_name: model_name.to_string(),
            service_name: model_name.to_string(),
            model_info: Some(ModelInfoType::from_repo(repo_id).await?),
            tokenizer: Some(TokenizerKind::from_repo(repo_id).await?),
            prompt_formatter: PromptFormatterArtifact::from_repo(repo_id).await?,
            prompt_context: None, // TODO - auto-detect prompt context
            revision: 0,
            last_published: None,
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

/// Checks if the provided path contains the expected file.
async fn check_for_file(repo_id: &str, file: &str) -> anyhow::Result<String> {
    let mut files = check_for_files(repo_id, vec![file.to_string()]).await?;
    let file = files
        .remove(file)
        .ok_or(anyhow::anyhow!("file {} not found", file))?;
    Ok(file)
}

async fn check_for_files(repo_id: &str, files: Vec<String>) -> Result<HashMap<String, String>> {
    let dir_entries =
        fs::read_dir(repo_id).with_context(|| format!("Failed to read directory: {}", repo_id))?;
    let mut found_files = HashMap::new();
    for entry in dir_entries {
        let entry =
            entry.with_context(|| format!("Failed to read directory entry in {}", repo_id))?;
        let path = entry.path();
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid file name in {}", repo_id))?;
        if files.contains(&file_name.to_string()) {
            found_files.insert(
                file_name.to_string(),
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path"))?
                    .to_string(),
            );
        }
    }
    Ok(found_files)
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

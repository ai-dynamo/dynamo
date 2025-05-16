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

//! # Model Deployment Card
//!
//! The ModelDeploymentCard (MDC) is the primary model configuration structure that will be available to any
//! component that needs to interact with the model or its dependent artifacts.
//!
//! The ModelDeploymentCard contains LLM model deployment configuration information:
//! - Display name and service name for the model
//! - Model information (ModelInfoType)
//! - Tokenizer configuration (TokenizerKind)
//! - Prompt formatter settings (PromptFormatterArtifact)
//! - Various metadata like revision, publish time, etc.

use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_runtime::slug::Slug;
use dynamo_runtime::transports::nats;
use either::Either;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;
use url::Url;

use crate::gguf::{Content, ContentConfig, ModelConfigLike};
use crate::key_value_store::Versioned;
use crate::protocols::TokenIdType;

/// Delete model deployment cards that haven't been re-published after this long.
/// Cleans up if the worker stopped.
pub const BUCKET_TTL: Duration = Duration::from_secs(5 * 60);

/// If a model deployment card hasn't been refreshed in this much time the worker is likely gone
const CARD_MAX_AGE: chrono::TimeDelta = chrono::TimeDelta::minutes(5);

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ModelInfoType {
    HfConfigJson(String),
    GGUF(PathBuf),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HfTokenizerJson(String),
    GGUF(Box<HfTokenizer>),
}

/// Supported types of prompt formatters.
///
/// We need a way to associate the prompt formatter template definition with an associated
/// data model which is expected for rendering.
///
/// All current prompt formatters are Jinja2 templates which use the OpenAI ChatCompletionRequest
/// format. However, we currently do not have a discovery path to know if the model supports tool use
/// unless we inspect the template.
///
/// TODO(): Add an enum for the PromptFormatDataModel with at minimum arms for:
/// - OaiChat
/// - OaiChatToolUse
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PromptFormatterArtifact {
    HfTokenizerConfigJson(String),
    GGUF(PathBuf),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

#[derive(Serialize, Deserialize, Clone, Debug, Builder, Default)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    /// Identifier to expect in OpenAI compatible HTTP request, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    /// This will get slugified for use in NATS.
    pub service_name: String,

    /// Model information
    pub model_info: Option<ModelInfoType>,

    /// Tokenizer configuration
    pub tokenizer: Option<TokenizerKind>,

    /// Prompt Formatter configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_formatter: Option<PromptFormatterArtifact>,

    /// Prompt Formatter Config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_context: Option<Vec<PromptContextMixin>>,

    /// When this card was last advertised by a worker. None if not yet published.
    pub last_published: Option<chrono::DateTime<chrono::Utc>>,

    /// Incrementing count of how many times we published this card
    #[serde(default, skip_serializing)]
    pub revision: u64,
}

impl ModelDeploymentCard {
    pub fn builder() -> ModelDeploymentCardBuilder {
        ModelDeploymentCardBuilder::default()
    }

    /// Create a ModelDeploymentCard where only the name is filled in.
    ///
    /// Single-process setups don't need an MDC to communicate model details, but it
    /// simplifies the code to assume we always have one. This is how you get one in those
    /// cases. A quasi-null object: <https://en.wikipedia.org/wiki/Null_object_pattern>
    pub fn with_name_only(name: &str) -> ModelDeploymentCard {
        ModelDeploymentCard {
            display_name: name.to_string(),
            service_name: Slug::slugify(name).to_string(),
            ..Default::default()
        }
    }

    /// A URL and NATS friendly and very likely unique ID for this model.
    /// Mostly human readable. a-z, 0-9, _ and - only.
    /// Pass the service_name.
    pub fn service_name_slug(s: &str) -> Slug {
        Slug::from_string(s)
    }

    /// How often we should check if a model deployment card expired because it's workers are gone
    pub fn expiry_check_period() -> Duration {
        match CARD_MAX_AGE.to_std() {
            Ok(duration) => duration / 3,
            Err(_) => {
                // Only happens if CARD_MAX_AGE is negative, which it isn't
                unreachable!("Cannot run card expiry watcher, invalid CARD_MAX_AGE");
            }
        }
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(file)?)?)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(json: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(json)?)
    }

    //
    // Methods
    //

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
    }

    pub fn set_service_name(&mut self, service_name: &str) {
        self.service_name = service_name.to_string();
    }

    pub fn slug(&self) -> Slug {
        ModelDeploymentCard::service_name_slug(&self.service_name)
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn mdcsum(&self) -> String {
        let json = self.to_json().unwrap();
        format!("{}", blake3::hash(json.as_bytes()))
    }

    /// Was this card last published a long time ago, suggesting the worker is gone?
    pub fn is_expired(&self) -> bool {
        if let Some(last_published) = self.last_published.as_ref() {
            chrono::Utc::now() - last_published > CARD_MAX_AGE
        } else {
            false
        }
    }

    /// Is this a full model card with tokenizer?
    /// There are cases where we have a placeholder card (see `with_name_only`).
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn tokenizer_hf(&self) -> anyhow::Result<HfTokenizer> {
        match &self.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => {
                HfTokenizer::from_file(file).map_err(anyhow::Error::msg)
            }
            Some(TokenizerKind::GGUF(t)) => Ok(*t.clone()),
            None => {
                anyhow::bail!("Blank ModelDeploymentCard does not have a tokenizer");
            }
        }
    }

    /// Move the files this MDC uses into the NATS object store.
    /// Updates the URI's to point to NATS.
    pub async fn move_to_nats(&mut self, nats_client: nats::Client) -> Result<()> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        tracing::debug!(
            nats_addr,
            %bucket_name,
            "Uploading model deployment card fields to NATS"
        );

        if let Some(ModelInfoType::HfConfigJson(ref src_file)) = self.model_info {
            if !nats::is_nats_url(src_file) {
                let target = format!("nats://{nats_addr}/{bucket_name}/config.json");
                nats_client
                    .object_store_upload(&PathBuf::from(src_file), Url::parse(&target)?)
                    .await?;
                self.model_info = Some(ModelInfoType::HfConfigJson(target));
            }
        }

        if let Some(PromptFormatterArtifact::HfTokenizerConfigJson(ref src_file)) =
            self.prompt_formatter
        {
            if !nats::is_nats_url(src_file) {
                let target = format!("nats://{nats_addr}/{bucket_name}/tokenizer_config.json");
                nats_client
                    .object_store_upload(&PathBuf::from(src_file), Url::parse(&target)?)
                    .await?;
                self.prompt_formatter =
                    Some(PromptFormatterArtifact::HfTokenizerConfigJson(target));
            }
        }

        if let Some(TokenizerKind::HfTokenizerJson(ref src_file)) = self.tokenizer {
            if !nats::is_nats_url(src_file) {
                let target = format!("nats://{nats_addr}/{bucket_name}/tokenizer.json");
                nats_client
                    .object_store_upload(&PathBuf::from(src_file), Url::parse(&target)?)
                    .await?;
                self.tokenizer = Some(TokenizerKind::HfTokenizerJson(target));
            }
        }

        Ok(())
    }

    /// Move the files this MDC uses from the NATS object store to local disk.
    /// Updates the URI's to point to the created files.
    ///
    /// The returned TempDir must be kept alive, it cleans up on drop.
    pub async fn move_from_nats(&mut self, nats_client: nats::Client) -> Result<tempfile::TempDir> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        let target_dir = tempfile::TempDir::with_prefix(bucket_name.to_string())?;
        tracing::debug!(
            nats_addr,
            %bucket_name,
            target_dir = %target_dir.path().display(),
            "Downloading model deployment card fields from NATS"
        );

        if let Some(ModelInfoType::HfConfigJson(ref src_url)) = self.model_info {
            if nats::is_nats_url(src_url) {
                let target = target_dir.path().join("config.json");
                nats_client
                    .object_store_download(Url::parse(src_url)?, &target)
                    .await?;
                self.model_info = Some(ModelInfoType::HfConfigJson(target.display().to_string()));
            }
        }

        if let Some(PromptFormatterArtifact::HfTokenizerConfigJson(ref src_url)) =
            self.prompt_formatter
        {
            if nats::is_nats_url(src_url) {
                let target = target_dir.path().join("tokenizer_config.json");
                nats_client
                    .object_store_download(Url::parse(src_url)?, &target)
                    .await?;
                self.prompt_formatter = Some(PromptFormatterArtifact::HfTokenizerConfigJson(
                    target.display().to_string(),
                ));
            }
        }

        if let Some(TokenizerKind::HfTokenizerJson(ref src_url)) = self.tokenizer {
            if nats::is_nats_url(src_url) {
                let target = target_dir.path().join("tokenizer.json");
                nats_client
                    .object_store_download(Url::parse(src_url)?, &target)
                    .await?;
                self.tokenizer = Some(TokenizerKind::HfTokenizerJson(target.display().to_string()));
            }
        }

        Ok(target_dir)
    }

    /// Delete this card from the key-value store and it's URLs from the object store
    pub async fn delete_from_nats(&mut self, nats_client: nats::Client) -> Result<()> {
        let nats_addr = nats_client.addr();
        let bucket_name = self.slug();
        tracing::trace!(
            nats_addr,
            %bucket_name,
            "Delete model deployment card from NATS"
        );
        nats_client
            .object_store_delete_bucket(bucket_name.as_ref())
            .await
    }
}

impl Versioned for ModelDeploymentCard {
    fn revision(&self) -> u64 {
        self.revision
    }

    fn set_revision(&mut self, revision: u64) {
        self.last_published = Some(chrono::Utc::now());
        self.revision = revision;
    }
}

impl fmt::Display for ModelDeploymentCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.slug())
    }
}
pub trait ModelInfo: Send + Sync {
    /// Model type
    fn model_type(&self) -> String;

    /// Token ID for the beginning of sequence
    fn bos_token_id(&self) -> TokenIdType;

    /// Token ID for the end of sequence
    fn eos_token_ids(&self) -> Vec<TokenIdType>;

    /// Maximum position embeddings / max sequence length
    fn max_position_embeddings(&self) -> usize;

    /// Vocabulary size
    fn vocab_size(&self) -> usize;
}

impl ModelInfoType {
    pub async fn get_model_info(&self) -> Result<Arc<dyn ModelInfo>> {
        match self {
            Self::HfConfigJson(info) => HFConfig::from_json_file(info).await,
            Self::GGUF(path) => HFConfig::from_gguf(path),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFConfig {
    /// denotes the mixin to the flattened data model which can be present
    /// in the config.json file
    architectures: Vec<String>,

    /// general model type
    model_type: String,

    text_config: Option<HFTextConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFTextConfig {
    bos_token_id: TokenIdType,

    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<TokenIdType, Vec<TokenIdType>>,

    /// max sequence length
    max_position_embeddings: usize,

    /// number of layers in the model
    num_hidden_layers: usize,

    /// number of attention heads in the model
    num_attention_heads: usize,

    /// Vocabulary size
    vocab_size: usize,
}

impl HFConfig {
    async fn from_json_file(file: &str) -> Result<Arc<dyn ModelInfo>> {
        let contents = std::fs::read_to_string(file)?;
        let mut config: Self = serde_json::from_str(&contents)?;
        if config.text_config.is_none() {
            let text_config: HFTextConfig = serde_json::from_str(&contents)?;
            config.text_config = Some(text_config);
        }
        Ok(Arc::new(config))
    }
    fn from_gguf(gguf_file: &Path) -> Result<Arc<dyn ModelInfo>> {
        let content = load_gguf(gguf_file)?;
        let model_config_metadata: ContentConfig = (&content).into();
        let num_hidden_layers =
            content.get_metadata()[&format!("{}.block_count", content.arch())].to_u32()? as usize;

        let bos_token_id = content.get_metadata()["tokenizer.ggml.bos_token_id"].to_u32()?;
        let eos_token_id = content.get_metadata()["tokenizer.ggml.eos_token_id"].to_u32()?;

        // to_vec returns a Vec that's already there, so it's cheap
        let vocab_size = content.get_metadata()["tokenizer.ggml.tokens"]
            .to_vec()?
            .len();

        let arch = content.arch().to_string();
        Ok(Arc::new(HFConfig {
            architectures: vec![format!("{}ForCausalLM", capitalize(&arch))],
            // "general.architecture"
            model_type: arch,
            text_config: Some(HFTextConfig {
                bos_token_id,
                eos_token_id: Either::Left(eos_token_id),
                // "llama.context_length"
                max_position_embeddings: model_config_metadata.max_seq_len(),
                // "llama.block_count"
                num_hidden_layers,
                // "llama.attention.head_count"
                num_attention_heads: model_config_metadata.num_attn_heads(),
                // "tokenizer.ggml.tokens".len()
                vocab_size,
            }),
        }))
    }
}

impl ModelInfo for HFConfig {
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn bos_token_id(&self) -> TokenIdType {
        self.text_config.as_ref().unwrap().bos_token_id
    }

    fn eos_token_ids(&self) -> Vec<TokenIdType> {
        match &self.text_config.as_ref().unwrap().eos_token_id {
            Either::Left(eos_token_id) => vec![*eos_token_id],
            Either::Right(eos_token_ids) => eos_token_ids.clone(),
        }
    }

    fn max_position_embeddings(&self) -> usize {
        self.text_config.as_ref().unwrap().max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.text_config.as_ref().unwrap().vocab_size
    }
}

impl TokenizerKind {
    pub fn from_gguf(gguf_file: &Path) -> anyhow::Result<Self> {
        let content = load_gguf(gguf_file)?;
        let out = crate::gguf::convert_gguf_to_hf_tokenizer(&content)
            .with_context(|| gguf_file.display().to_string())?;
        Ok(TokenizerKind::GGUF(Box::new(out.tokenizer)))
    }
}

fn load_gguf(gguf_file: &Path) -> anyhow::Result<Content> {
    let filename = gguf_file.display().to_string();
    let mut f = File::open(gguf_file).with_context(|| filename.clone())?;
    // vec because GGUF can be split into multiple files (shards)
    let mut readers = vec![&mut f];
    crate::gguf::Content::from_readers(&mut readers).with_context(|| filename.clone())
}

fn capitalize(s: &str) -> String {
    s.chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 0 {
                c.to_uppercase().to_string()
            } else {
                c.to_lowercase().to_string()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::HFConfig;
    use std::path::Path;

    #[tokio::test]
    pub async fn test_config_json_llama3() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct/config.json");
        let config = HFConfig::from_json_file(&config_file.display().to_string()).await?;
        assert_eq!(config.bos_token_id(), 128000);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_config_json_llama4() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/Llama-4-Scout-17B-16E-Instruct/config.json");
        let config = HFConfig::from_json_file(&config_file.display().to_string()).await?;
        assert_eq!(config.bos_token_id(), 200000);
        Ok(())
    }
}

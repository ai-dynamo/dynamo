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
use either::Either;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

use crate::gguf::{Content, ContentConfig};
use crate::protocols::TokenIdType;

pub const BUCKET_NAME: &str = "mdc";

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

#[derive(Serialize, Deserialize, Clone, Debug, Builder)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    /// Identifier to expect in OpenAI compatible HTTP request, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    /// This will get slugified for use in NATS.
    pub service_name: String,

    /// Model information
    pub model_info: ModelInfoType,

    /// Tokenizer configuration
    pub tokenizer: TokenizerKind,

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

    /// Does this model expect preprocessing (tokenization, etc) to be already done?
    /// If this is true they get a BackendInput JSON. If this is false they get
    /// a ChatCompletionRequest JSON.
    #[serde(default)]
    pub requires_preprocessing: bool,
}

impl ModelDeploymentCard {
    pub fn builder() -> ModelDeploymentCardBuilder {
        ModelDeploymentCardBuilder::default()
    }

    /// A URL and NATS friendly and very likely unique ID for this model.
    /// Mostly human readable. a-z, 0-9, _ and - only.
    /// Pass the service_name.
    pub fn service_name_slug(s: &str) -> Slug {
        Slug::from_string(s)
    }

    pub fn set_service_name(&mut self, service_name: &str) {
        self.service_name = service_name.to_string();
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

    pub fn slug(&self) -> Slug {
        ModelDeploymentCard::service_name_slug(&self.service_name)
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        let mut card: ModelDeploymentCard = serde_json::from_str(&std::fs::read_to_string(file)?)?;
        card.requires_preprocessing = false;
        Ok(card)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(json: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
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

    pub fn tokenizer_hf(&self) -> anyhow::Result<HfTokenizer> {
        match &self.tokenizer {
            TokenizerKind::HfTokenizerJson(file) => {
                HfTokenizer::from_file(file).map_err(anyhow::Error::msg)
            }
            TokenizerKind::GGUF(t) => Ok(*t.clone()),
        }
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
    bos_token_id: TokenIdType,

    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<TokenIdType, Vec<TokenIdType>>,

    /// denotes the mixin to the flattened data model which can be present
    /// in the config.json file
    architectures: Vec<String>,

    /// general model type
    model_type: String,

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
    async fn from_json_file(file: &String) -> Result<Arc<dyn ModelInfo>> {
        let contents = std::fs::read_to_string(file)?;
        let config: Self = serde_json::from_str(&contents)?;
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
            bos_token_id,
            eos_token_id: Either::Left(eos_token_id),
            architectures: vec![format!("{}ForCausalLM", capitalize(&arch))],
            // "general.architecture"
            model_type: arch,
            // "llama.context_length"
            max_position_embeddings: model_config_metadata.max_seq_len(),
            // "llama.block_count"
            num_hidden_layers,
            // "llama.attention.head_count"
            num_attention_heads: model_config_metadata.num_attn_heads(),
            // "tokenizer.ggml.tokens".len()
            vocab_size,
        }))
    }
}

impl ModelInfo for HFConfig {
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn bos_token_id(&self) -> TokenIdType {
        self.bos_token_id
    }

    fn eos_token_ids(&self) -> Vec<TokenIdType> {
        match &self.eos_token_id {
            Either::Left(eos_token_id) => vec![*eos_token_id],
            Either::Right(eos_token_ids) => eos_token_ids.clone(),
        }
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
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

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::common::checked_file::CheckedFile;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_type::{ModelInput, ModelType};
use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_runtime::{slug::Slug, storage::key_value_store::Versioned};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

use crate::preprocessor::media::{MediaDecoder, MediaFetcher};
use crate::protocols::TokenIdType;

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "v1/mdc";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ModelInfoType {
    HfConfigJson(CheckedFile),
}

impl ModelInfoType {
    pub fn checksum(&self) -> String {
        match self {
            ModelInfoType::HfConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            ModelInfoType::HfConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            ModelInfoType::HfConfigJson(c) => c.update_dir(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HfTokenizerJson(CheckedFile),
}

impl TokenizerKind {
    pub fn checksum(&self) -> String {
        match self {
            TokenizerKind::HfTokenizerJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            TokenizerKind::HfTokenizerJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            TokenizerKind::HfTokenizerJson(c) => c.update_dir(dir),
        }
    }
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
    HfTokenizerConfigJson(CheckedFile),
    HfChatTemplate(CheckedFile),
}

impl PromptFormatterArtifact {
    pub fn checksum(&self) -> String {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.checksum().to_string(),
            PromptFormatterArtifact::HfChatTemplate(c) => c.checksum().to_string(),
        }
    }

    /// Is this file available locally
    pub fn is_local(&self) -> bool {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.is_local(),
            PromptFormatterArtifact::HfChatTemplate(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.update_dir(dir),
            PromptFormatterArtifact::HfChatTemplate(c) => c.update_dir(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConfig {
    HfGenerationConfigJson(CheckedFile),
}

impl GenerationConfig {
    pub fn checksum(&self) -> String {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.update_dir(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Builder, Default)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    // Cache the Slugified display_name so we can share references to it
    slug: Slug,

    /// Model information
    pub model_info: Option<ModelInfoType>,

    /// Tokenizer configuration
    pub tokenizer: Option<TokenizerKind>,

    /// Prompt Formatter configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_formatter: Option<PromptFormatterArtifact>,

    /// chat template may be stored as a separate file instead of in `prompt_formatter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_file: Option<PromptFormatterArtifact>,

    /// Generation config - default sampling params
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_config: Option<GenerationConfig>,

    /// Prompt Formatter Config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_context: Option<Vec<PromptContextMixin>>,

    /// Max context (in number of tokens) this model can handle
    pub context_length: u32,

    /// Size of a KV cache block - vllm only currently
    /// Passed to the engine and the KV router.
    pub kv_cache_block_size: u32,

    /// How many times a request can be migrated to another worker if the HTTP server lost
    /// connection to the current worker.
    pub migration_limit: u32,

    /// Specifies whether the model is a chat, completions, etc model.
    pub model_type: ModelType,

    /// Specifies the model input type.
    /// `Tokens` for engines that expect pre-processed input.
    /// `Text` for engines that take care of pre-processing themselves.
    pub model_input: ModelInput,

    /// User-defined metadata for custom worker behavior
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_data: Option<serde_json::Value>,

    #[serde(default)]
    pub runtime_config: ModelRuntimeConfig,

    /// Media decoding configuration
    #[serde(default)]
    pub media_decoder: Option<MediaDecoder>,

    /// Media fetching configuration
    #[serde(default)]
    pub media_fetcher: Option<MediaFetcher>,

    #[serde(skip, default)]
    checksum: OnceLock<String>,
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
            slug: Slug::from_string(name),
            ..Default::default()
        }
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(&file)?;
        Ok(serde_json::from_str(&contents).inspect_err(|err| {
            crate::log_json_err(&file.as_ref().display().to_string(), &contents, err)
        })?)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(contents: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(contents)
            .inspect_err(|err| crate::log_json_err("unknown", contents, err))?)
    }

    //
    // Methods
    //

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.display_name
    }

    #[inline]
    pub fn slug(&self) -> &Slug {
        &self.slug
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn mdcsum(&self) -> &str {
        self.checksum
            .get_or_init(|| {
                // Only include the important fields
                let mut bytes_to_hash: Vec<u8> = Vec::with_capacity(512);
                bytes_to_hash.extend(self.display_name.as_bytes());

                // The files can be either a URL or a local path, so we ignore that and hash their
                // checksum instead, which won't change wherever they are.

                if let Some(model_info) = self.model_info.as_ref() {
                    bytes_to_hash.extend(model_info.checksum().as_bytes());
                }
                if let Some(tokenizer) = self.tokenizer.as_ref() {
                    bytes_to_hash.extend(tokenizer.checksum().as_bytes());
                }
                if let Some(prompt_formatter) = self.prompt_formatter.as_ref() {
                    bytes_to_hash.extend(prompt_formatter.checksum().as_bytes());
                }
                if let Some(chat_template) = self.chat_template_file.as_ref() {
                    bytes_to_hash.extend(chat_template.checksum().as_bytes());
                }
                if let Some(gen_config) = self.gen_config.as_ref() {
                    bytes_to_hash.extend(gen_config.checksum().as_bytes());
                }

                if let Some(prompt_context_vec) = self.prompt_context.as_ref() {
                    // Paste it as the bytes of the debug format. It's a Vec of enum, so this should be
                    // fine. If the debug representation changes that only happens in a new release.
                    bytes_to_hash.extend(format!("{prompt_context_vec:?}").as_bytes());
                }
                bytes_to_hash.extend(self.context_length.to_be_bytes());
                bytes_to_hash.extend(self.kv_cache_block_size.to_be_bytes());

                // TODO: Do we want any of user_data or runtime_config?

                blake3::hash(&bytes_to_hash).to_string()
            })
            .as_ref()
    }

    /// Is this a full model card with tokenizer?
    /// There are cases where we have a placeholder card (see `with_name_only`).
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn tokenizer_hf(&self) -> anyhow::Result<HfTokenizer> {
        match &self.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(checked_file)) => {
                let p = checked_file.path().ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer is URL-backed ({:?})", checked_file.url())
                })?;
                HfTokenizer::from_file(p)
                    .inspect_err(|err| {
                        if let Some(serde_err) = err.downcast_ref::<serde_json::Error>()
                            && let Ok(contents) = std::fs::read_to_string(p)
                        {
                            crate::log_json_err(&p.display().to_string(), &contents, serde_err);
                        }
                    })
                    .map_err(anyhow::Error::msg)
                    .with_context(|| p.display().to_string())
            }
            None => {
                anyhow::bail!("Blank ModelDeploymentCard does not have a tokenizer");
            }
        }
    }

    /// Allow user to override the name we register this model under.
    /// Corresponds to vllm's `--served-model-name`.
    pub fn set_name(&mut self, name: &str) {
        self.display_name = name.to_string();
        self.slug = Slug::from_string(name);
    }

    /// Build an in-memory ModelDeploymentCard from a folder containing config.json,
    /// tokenizer.json and tokenizer_config.json (i.e. a huggingface repo checkout).
    /// Optional custom template.
    pub fn load_from_disk(
        config_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<ModelDeploymentCard> {
        Self::from_local_path(config_path.as_ref(), custom_template_path)
    }

    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_input, ModelInput::Tokens)
    }

    /// Download the files this card needs to work: config.json, tokenizer.json, etc.
    pub async fn download_config(&mut self) -> anyhow::Result<()> {
        if self.has_local_files() {
            tracing::trace!("All model config is local, not downloading");
            return Ok(());
        }

        let ignore_weights = true;
        let local_path = crate::hub::from_hf(&self.display_name, ignore_weights).await?;

        self.update_dir(&local_path);
        Ok(())
    }

    /// Are all the files we need (tokenizer.json, etc) available locally?
    fn has_local_files(&self) -> bool {
        let has_model_info = self
            .model_info
            .as_ref()
            .map(|p| p.is_local())
            .unwrap_or(true);
        let has_tokenizer = self
            .tokenizer
            .as_ref()
            .map(|p| p.is_local())
            .unwrap_or(true);
        let has_prompt_formatter = self
            .prompt_formatter
            .as_ref()
            .map(|p| p.is_local())
            .unwrap_or(true);
        let has_chat_template_file = self
            .chat_template_file
            .as_ref()
            .map(|p| p.is_local())
            .unwrap_or(true);
        let has_gen_config = self
            .gen_config
            .as_ref()
            .map(|p| p.is_local())
            .unwrap_or(true);

        has_model_info
            && has_tokenizer
            && has_prompt_formatter
            && has_chat_template_file
            && has_gen_config
    }

    /// Update the directory for files like tokenizer.json be in here.
    fn update_dir(&mut self, dir: &Path) {
        if let Some(model_info) = self.model_info.as_mut() {
            model_info.update_dir(dir);
        }
        if let Some(tk) = self.tokenizer.as_mut() {
            tk.update_dir(dir);
        }
        if let Some(pf) = self.prompt_formatter.as_mut() {
            pf.update_dir(dir);
        }
        if let Some(ct) = self.chat_template_file.as_mut() {
            ct.update_dir(dir);
        }
        if let Some(gc) = self.gen_config.as_mut() {
            gc.update_dir(dir);
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
    fn from_local_path(
        local_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        check_valid_local_repo_path(&local_path)?;
        Self::from_repo_checkout(&local_path, custom_template_path)
    }

    fn from_repo_checkout(
        local_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let local_path = local_path.as_ref();

        // This is usually the right choice
        let context_length =
            crate::file_json_field(&local_path.join("config.json"), "max_position_embeddings")
                // But sometimes this is
                .or_else(|_| {
                    crate::file_json_field(
                        &local_path.join("tokenizer_config.json"),
                        "model_max_length",
                    )
                })
                // If neither of those are present let the engine default it
                .unwrap_or(0);

        // Load chat template - either custom or from repo
        let chat_template_file = if let Some(template_path) = custom_template_path {
            if !template_path.exists() {
                anyhow::bail!(
                    "Custom template file does not exist: {}",
                    template_path.display()
                );
            }

            // Verify the file is readable
            let _template_content = std::fs::read_to_string(template_path).with_context(|| {
                format!(
                    "Failed to read custom template file: {}",
                    template_path.display()
                )
            })?;

            Some(PromptFormatterArtifact::HfChatTemplate(
                CheckedFile::from_disk(template_path)?,
            ))
        } else {
            PromptFormatterArtifact::chat_template_from_disk(local_path)?
        };

        let display_name = local_path.display().to_string();
        Ok(Self {
            slug: Slug::from_string(&display_name),
            display_name,
            model_info: Some(ModelInfoType::from_disk(local_path)?),
            tokenizer: Some(TokenizerKind::from_disk(local_path)?),
            gen_config: GenerationConfig::from_disk(local_path).ok(), // optional
            prompt_formatter: PromptFormatterArtifact::from_disk(local_path)?,
            chat_template_file,
            prompt_context: None, // TODO - auto-detect prompt context
            context_length,
            kv_cache_block_size: 0, // set later
            migration_limit: 0,
            model_type: Default::default(),  // set later
            model_input: Default::default(), // set later
            user_data: None,
            runtime_config: ModelRuntimeConfig::default(),
            media_decoder: None,
            media_fetcher: None,
            checksum: OnceLock::new(),
        })
    }
}

impl PartialEq for ModelDeploymentCard {
    fn eq(&self, other: &ModelDeploymentCard) -> bool {
        self.mdcsum() == other.mdcsum()
    }
}

/// A ModelDeploymentCard is published a single time per instance and never updated.
impl Versioned for ModelDeploymentCard {
    fn revision(&self) -> u64 {
        0
    }

    fn set_revision(&mut self, _revision: u64) {}
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
    /// TODO: This is only used in a single test, no other code. Remove?
    fn max_position_embeddings(&self) -> Option<usize>;

    /// Vocabulary size
    /// TODO: This is only used in a single test, no other code. Remove?
    fn vocab_size(&self) -> Option<usize>;
}

impl ModelInfoType {
    pub fn get_model_info(&self) -> Result<Arc<dyn ModelInfo>> {
        match self {
            Self::HfConfigJson(checked_file) => {
                let Some(path) = checked_file.path() else {
                    anyhow::bail!("model info is not a local path: {checked_file:?}");
                };
                Ok(HFConfig::from_json_file(path)?)
            }
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

    // Sometimes it's inside HFTextConfig, sometimes it's here
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFTextConfig {
    // It can take multiple attempts to load this, so Option
    bos_token_id: Option<TokenIdType>,

    // We set this once bos_token_id is loaded so we don't have to deal with Option
    #[serde(default)]
    final_bos_token_id: TokenIdType,

    eos_token_id: Option<serde_json::Value>,

    #[serde(default)]
    final_eos_token_ids: Vec<TokenIdType>,

    /// max sequence length
    max_position_embeddings: Option<usize>,

    /// number of layers in the model
    /// Optional because some multimodal models (e.g., LLaVA) don't include this in text_config
    num_hidden_layers: Option<usize>,

    /// number of attention heads in the model
    num_attention_heads: Option<usize>,

    /// Vocabulary size
    vocab_size: Option<usize>,
}

impl HFConfig {
    fn from_json_file<P: AsRef<Path>>(file: P) -> Result<Arc<dyn ModelInfo>> {
        let file_path = file.as_ref();
        let contents = std::fs::read_to_string(file_path)?;
        let mut config: Self = json_five::from_str(&contents)
            .inspect_err(|err| {
                tracing::error!(path=%file_path.display(), %err, "Failed to parse config.json as JSON5");
            })?;
        if config.text_config.is_none() {
            let text_config: HFTextConfig = json_five::from_str(&contents)
                .inspect_err(|err| {
                    tracing::error!(path=%file_path.display(), %err, "Failed to parse text config from config.json as JSON5");
                })?;
            config.text_config = Some(text_config);
        }

        // Sometimes bos_token_id is in generation_config.json not config.json
        let Some(text_config) = config.text_config.as_mut() else {
            anyhow::bail!(
                "Missing text config fields (model_type, eos_token_ids, etc) in config.json"
            );
        };

        let gencfg_path = file_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("generation_config.json");
        if text_config.bos_token_id.is_none() {
            let bos_token_id = crate::file_json_field::<TokenIdType>(&gencfg_path, "bos_token_id")
                .context(
                    "missing bos_token_id in generation_config.json and config.json, cannot load",
                )?;
            text_config.bos_token_id = Some(bos_token_id);
        }
        // Now that we have it for sure, set it in the non-Option field
        let final_bos_token_id = text_config.bos_token_id.take().unwrap();
        text_config.final_bos_token_id = final_bos_token_id;

        // TODO: refactor this when we switch to per-architecture tokenization
        let final_eos_token_ids: Vec<TokenIdType> = config
            .eos_token_id
            .as_ref()
            .or(text_config.eos_token_id.as_ref())
            .and_then(|v| {
                if v.is_number() {
                    v.as_number()
                        .and_then(|n| n.as_u64())
                        .map(|n| vec![n as TokenIdType])
                } else if v.is_array() {
                    let arr = v.as_array().unwrap(); // Safety: We just checked
                    Some(
                        arr.iter()
                            .filter_map(|inner_v| {
                                inner_v
                                    .as_number()
                                    .and_then(|n| n.as_u64())
                                    .map(|n| n as TokenIdType)
                            })
                            .collect(),
                    )
                } else {
                    tracing::error!(
                        ?v,
                        path = %file_path.display(),
                        "eos_token_id is not a number or an array, cannot use"
                    );
                    None
                }
            })
            .or_else(|| {
                // Maybe it's in generation_config.json
                crate::file_json_field::<serde_json::Value>(&gencfg_path, "eos_token_id")
                .inspect_err(
                    |err| tracing::warn!(%err, "Missing eos_token_id in generation_config.json"),
                )
                .ok()
                .and_then(|v| {
                    if v.is_number() {
                        v.as_number()
                            .and_then(|n| n.as_u64())
                            .map(|n| vec![n as TokenIdType])
                    } else if v.is_array() {
                        let arr = v.as_array().unwrap();
                        Some(
                            arr.iter()
                                .filter_map(|inner_v| {
                                    inner_v
                                        .as_number()
                                        .and_then(|n| n.as_u64())
                                        .map(|n| n as TokenIdType)
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing eos_token_id in config.json and generation_config.json, cannot load"
                )
            })?;
        text_config.final_eos_token_ids = final_eos_token_ids;

        Ok(Arc::new(config))
    }
}

impl ModelInfo for HFConfig {
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn bos_token_id(&self) -> TokenIdType {
        self.text_config.as_ref().unwrap().final_bos_token_id
    }

    fn eos_token_ids(&self) -> Vec<TokenIdType> {
        self.text_config
            .as_ref()
            .unwrap()
            .final_eos_token_ids
            .clone()
    }

    fn max_position_embeddings(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().max_position_embeddings
    }

    fn vocab_size(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().vocab_size
    }
}

impl ModelInfoType {
    pub fn from_disk(directory: &Path) -> Result<Self> {
        let f = CheckedFile::from_disk(directory.join("config.json")).with_context(|| {
            format!(
                "unable to extract config.json from directory {}",
                directory.display()
            )
        })?;
        Ok(Self::HfConfigJson(f))
    }
}

impl GenerationConfig {
    pub fn from_disk(directory: &Path) -> Result<Self> {
        let f = CheckedFile::from_disk(directory.join("generation_config.json")).with_context(
            || {
                format!(
                    "unable to extract generation_config from directory {}",
                    directory.display()
                )
            },
        )?;
        Ok(Self::HfGenerationConfigJson(f))
    }
}

impl PromptFormatterArtifact {
    pub fn from_disk(directory: &Path) -> Result<Option<Self>> {
        // we should only error if we expect a prompt formatter and it's not found
        // right now, we don't know when to expect it, so we just return Ok(Some/None)
        match CheckedFile::from_disk(directory.join("tokenizer_config.json")) {
            Ok(f) => Ok(Some(Self::HfTokenizerConfigJson(f))),
            Err(_) => Ok(None),
        }
    }

    pub fn chat_template_from_disk(directory: &Path) -> Result<Option<Self>> {
        match CheckedFile::from_disk(directory.join("chat_template.jinja")) {
            Ok(f) => Ok(Some(Self::HfChatTemplate(f))),
            Err(_) => Ok(None),
        }
    }
}

impl TokenizerKind {
    pub fn from_disk(directory: &Path) -> Result<Self> {
        let f = CheckedFile::from_disk(directory.join("tokenizer.json")).with_context(|| {
            format!(
                "unable to extract tokenizer kind from directory {}",
                directory.display()
            )
        })?;
        Ok(Self::HfTokenizerJson(f))
    }
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

#[cfg(test)]
mod tests {
    use super::HFConfig;
    use std::path::Path;

    #[test]
    pub fn test_config_json_llama3() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), 128000);
        Ok(())
    }

    #[test]
    pub fn test_config_json_llama4() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/Llama-4-Scout-17B-16E-Instruct/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), 200000);
        Ok(())
    }

    /// The Python JSON parser accepts `Infinity` as a numeric value. This is explicitly against the
    /// JSON spec, but inevitably people rely on it, so we have to allow it.
    /// We treat that file as JSON5 (a lenient superset of JSON) to be able to parse it.
    #[test]
    fn test_invalid_json_but_py_accepts_it() {
        dynamo_runtime::logging::init();
        let path = "tests/data/sample-models/NVIDIA-Nemotron-Nano-12B-v2-Base/config.json";
        let _ = HFConfig::from_json_file(path).unwrap();
    }
}

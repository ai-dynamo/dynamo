// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candle-based classifier for semantic routing
//!
//! Loads models directly from HuggingFace using safetensors format.
//! Pure Rust implementation with no Python dependencies.

use super::Classifier;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[cfg(feature = "candle-classifier")]
use candle_core::{DType, Device, IndexOp, Module, Tensor};
#[cfg(feature = "candle-classifier")]
use candle_nn::VarBuilder;
#[cfg(feature = "candle-classifier")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig, HiddenAct};
#[cfg(feature = "candle-classifier")]
use hf_hub::{api::sync::Api, Repo, RepoType};
#[cfg(feature = "candle-classifier")]
use std::path::PathBuf;
#[cfg(feature = "candle-classifier")]
use tokenizers::Tokenizer;

/// Candle-based classifier for CodeIsAbstract/ReasoningTextClassifier
///
/// This classifier loads ModernBERT-based models directly from HuggingFace
/// using safetensors format. It's a pure Rust implementation with excellent
/// CPU and GPU support.
#[cfg(feature = "candle-classifier")]
pub struct CandleClassifier {
    model: BertModel,
    classifier_head: candle_nn::Linear,
    tokenizer: Tokenizer,
    device: Device,
    max_len: usize,
    labels: Vec<String>,
}

#[cfg(feature = "candle-classifier")]
impl CandleClassifier {
    /// Create a new CandleClassifier by loading from HuggingFace
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "CodeIsAbstract/ReasoningTextClassifier")
    /// * `max_len` - Maximum sequence length for tokenization
    /// * `device` - Candle device (CPU or CUDA)
    pub fn from_pretrained(
        model_id: &str,
        max_len: usize,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Candle classifier from HuggingFace: {}", model_id);

        // Download model files from HuggingFace
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        tracing::debug!("Config: {:?}", config_path);
        tracing::debug!("Weights: {:?}", weights_path);
        tracing::debug!("Tokenizer: {:?}", tokenizer_path);

        // Load configuration
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        // Extract BERT config
        let bert_config = BertConfig {
            vocab_size: config["vocab_size"].as_u64().unwrap_or(50368) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: config["hidden_dropout_prob"].as_f64().unwrap_or(0.0) as f64,
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
            type_vocab_size: config["type_vocab_size"].as_u64().unwrap_or(1) as usize,
            initializer_range: config["initializer_range"].as_f64().unwrap_or(0.02) as f64,
            layer_norm_eps: config["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f64,
            pad_token_id: config["pad_token_id"].as_u64().unwrap_or(50283) as usize,
            position_embedding_type: candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };

        // Load model weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        // Load BERT encoder
        let model = BertModel::load(vb.pp("bert"), &bert_config)?;

        // Load classification head (linear layer)
        let num_labels = config["num_labels"].as_u64().unwrap_or(2) as usize;
        let classifier_vb = vb.pp("classifier");
        let classifier_head = candle_nn::linear(
            bert_config.hidden_size,
            num_labels,
            classifier_vb
        )?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Extract labels (default to binary for ReasoningTextClassifier)
        let labels = if let Some(id2label) = config.get("id2label") {
            let mut label_pairs: Vec<(usize, String)> = id2label
                .as_object()
                .ok_or_else(|| anyhow!("id2label is not an object"))?
                .iter()
                .map(|(k, v)| {
                    let idx = k.parse::<usize>().unwrap_or(0);
                    let label = v.as_str().unwrap_or("unknown").to_string();
                    (idx, label)
                })
                .collect();
            label_pairs.sort_by_key(|(idx, _)| *idx);
            label_pairs.into_iter().map(|(_, label)| label).collect()
        } else {
            vec!["non-reasoning".to_string(), "reasoning".to_string()]
        };

        tracing::info!(
            "Initialized Candle classifier: {} labels, max_len={}, device={:?}",
            labels.len(),
            max_len,
            device
        );

        Ok(Self {
            model,
            classifier_head,
            tokenizer,
            device,
            max_len,
            labels,
        })
    }

    /// Create from local paths (for testing or custom deployments)
    pub fn from_local(
        weights_path: PathBuf,
        config_path: PathBuf,
        tokenizer_path: PathBuf,
        max_len: usize,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Candle classifier from local files");

        // Load configuration
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        // Extract BERT config
        let bert_config = BertConfig {
            vocab_size: config["vocab_size"].as_u64().unwrap_or(50368) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: config["hidden_dropout_prob"].as_f64().unwrap_or(0.0) as f64,
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(8192) as usize,
            type_vocab_size: config["type_vocab_size"].as_u64().unwrap_or(1) as usize,
            initializer_range: config["initializer_range"].as_f64().unwrap_or(0.02) as f64,
            layer_norm_eps: config["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f64,
            pad_token_id: config["pad_token_id"].as_u64().unwrap_or(50283) as usize,
            position_embedding_type: candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };

        // Load model weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        let model = BertModel::load(vb.pp("bert"), &bert_config)?;

        let num_labels = config["num_labels"].as_u64().unwrap_or(2) as usize;
        let classifier_vb = vb.pp("classifier");
        let classifier_head = candle_nn::linear(
            bert_config.hidden_size,
            num_labels,
            classifier_vb
        )?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let labels = vec!["non-reasoning".to_string(), "reasoning".to_string()];

        tracing::info!(
            "Initialized Candle classifier from local: {} labels, max_len={}",
            labels.len(),
            max_len
        );

        Ok(Self {
            model,
            classifier_head,
            tokenizer,
            device,
            max_len,
            labels,
        })
    }
}

#[cfg(feature = "candle-classifier")]
impl Classifier for CandleClassifier {
    fn classify(&self, text: &str) -> Result<HashMap<String, f32>> {
        let text_preview = if text.len() > 50 {
            format!("{}...", &text[..50])
        } else {
            text.to_string()
        };
        tracing::info!("🔍 Candle classify() called with text: '{}'", text_preview);

        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        tracing::info!("Tokenized: {} tokens (before padding/truncation)", ids.len());

        // Truncate or pad to max_len
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
            mask.truncate(self.max_len);
        }
        if ids.len() < self.max_len {
            let pad = self.max_len - ids.len();
            ids.extend(std::iter::repeat(0).take(pad));
            mask.extend(std::iter::repeat(0).take(pad));
        }

        // Convert to tensors [1, seq_len]
        let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let mask_i64: Vec<i64> = mask.iter().map(|&x| x as i64).collect();

        let input_ids = Tensor::from_slice(&ids_i64[..], (1, self.max_len), &self.device)?;
        let attention_mask = Tensor::from_slice(&mask_i64[..], (1, self.max_len), &self.device)?;

        // Token type IDs (zeros for single sentence)
        let token_type_ids = Tensor::zeros((1, self.max_len), candle_core::DType::I64, &self.device)?;

        // Forward pass through BERT -> [batch, seq_len, hidden_size]
        let hidden = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        tracing::debug!("Encoder hidden shape: {:?}", hidden.shape());

        // Masked mean pooling across sequence (honors attention mask)
        // Convert mask to f32 and add dimension: [1, seq_len] -> [1, seq_len, 1]
        let mask_f32 = attention_mask.to_dtype(candle_core::DType::F32)?
            .unsqueeze(candle_core::D::Minus1)?;

        // Multiply hidden states by mask: [1, seq_len, hidden] * [1, seq_len, 1]
        let masked_hidden = hidden.broadcast_mul(&mask_f32)?;

        // Sum across sequence dimension and divide by mask sum
        let sum_hidden = masked_hidden.sum(candle_core::D::Minus2)?; // [1, hidden]
        let mask_sum = mask_f32.sum(candle_core::D::Minus2)?; // [1, 1]
        let mask_sum = mask_sum.clamp(1e-6, f32::MAX)?; // Avoid division by zero
        let pooled = sum_hidden.broadcast_div(&mask_sum)?; // [1, hidden]

        tracing::debug!("Pooled shape: {:?}", pooled.shape());

        // Classification head: [1, hidden] -> [1, num_labels]
        let logits = self.classifier_head.forward(&pooled)?;

        // Squeeze batch dimension: [1, num_labels] -> [num_labels]
        let logits = logits.squeeze(0)?;

        // Softmax to get probabilities
        let probs = candle_nn::ops::softmax(&logits, 0)?;
        let probs_vec = probs.to_vec1::<f32>()?;

        tracing::info!("Raw probabilities vector: {:?}", probs_vec);

        // Convert to HashMap
        if probs_vec.len() != self.labels.len() {
            return Err(anyhow!(
                "Label count mismatch: {} labels but {} probabilities",
                self.labels.len(),
                probs_vec.len()
            ));
        }

        let mut result = HashMap::with_capacity(self.labels.len());
        for (label, &prob) in self.labels.iter().zip(probs_vec.iter()) {
            result.insert(label.clone(), prob);
        }

        tracing::info!("Candle classifier final result: {:?}", result);
        Ok(result)
    }
}

// Placeholder implementations when feature is not enabled
#[cfg(not(feature = "candle-classifier"))]
pub struct CandleClassifier;

#[cfg(not(feature = "candle-classifier"))]
impl CandleClassifier {
    pub fn from_pretrained(
        _model_id: &str,
        _max_len: usize,
        _device: (),
    ) -> Result<Self> {
        Err(anyhow!("CandleClassifier requires 'candle-classifier' feature"))
    }

    pub fn from_local(
        _weights_path: std::path::PathBuf,
        _config_path: std::path::PathBuf,
        _tokenizer_path: std::path::PathBuf,
        _max_len: usize,
        _device: (),
    ) -> Result<Self> {
        Err(anyhow!("CandleClassifier requires 'candle-classifier' feature"))
    }
}

#[cfg(not(feature = "candle-classifier"))]
impl Classifier for CandleClassifier {
    fn classify(&self, _text: &str) -> Result<HashMap<String, f32>> {
        Err(anyhow!("CandleClassifier requires 'candle-classifier' feature"))
    }
}

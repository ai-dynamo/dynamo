// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use tokenizers::tokenizer::{AddedToken, Tokenizer as HfTokenizer};

use super::{
    Encoding, Error, Result, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
}

impl HuggingFaceTokenizer {
    /// Load from `tokenizer.json`, merging in special tokens declared only in
    /// a sibling `tokenizer_config.json`'s `added_tokens_decoder`. Without
    /// this, some releases (Qwen2-VL-2B's `<|image_pad|>`) BPE-shatter and
    /// silently break MM-aware routing. The merge is idempotent.
    pub fn from_file(model_name: &str) -> Result<Self> {
        let mut tokenizer = HfTokenizer::from_file(model_name)
            .map_err(|err| Error::msg(format!("Error loading tokenizer: {}", err)))?;

        if let Some(parent) = Path::new(model_name).parent() {
            merge_special_tokens_from_config(&mut tokenizer, parent);
        }

        Ok(HuggingFaceTokenizer { tokenizer })
    }

    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }

    /// Wrap an already-loaded `HfTokenizer` and merge in the sibling
    /// `tokenizer_config.json` special tokens; see [`Self::from_file`].
    pub fn from_tokenizer_with_model_dir(tokenizer: HfTokenizer, model_dir: &Path) -> Self {
        let mut tokenizer = tokenizer;
        merge_special_tokens_from_config(&mut tokenizer, model_dir);
        HuggingFaceTokenizer { tokenizer }
    }
}

/// Promote `tokenizer_config.json`'s `special: true` `added_tokens_decoder`
/// entries onto `tokenizer`. Missing-file / parse errors are swallowed since
/// the file is optional. See [`HuggingFaceTokenizer::from_file`].
fn merge_special_tokens_from_config(tokenizer: &mut HfTokenizer, model_dir: &Path) {
    let cfg_path = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
        return;
    };
    let cfg: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                target: "tokenizer",
                path = %cfg_path.display(),
                error = %e,
                "tokenizer_config.json parse failed; skipping special-token merge"
            );
            return;
        }
    };
    let Some(decoder) = cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) else {
        return;
    };

    let mut to_add: Vec<AddedToken> = Vec::new();
    for (_id, spec) in decoder {
        let obj = match spec.as_object() {
            Some(o) => o,
            None => continue,
        };
        // The id is informational — `add_special_tokens` reuses the existing
        // vocab id via `Model::token_to_id` on the content string.
        if obj.get("special").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        let Some(content) = obj.get("content").and_then(|v| v.as_str()) else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        let single_word = obj
            .get("single_word")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let lstrip = obj.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false);
        let rstrip = obj.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false);
        let normalized = obj
            .get("normalized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let token = AddedToken::from(content.to_string(), true)
            .single_word(single_word)
            .lstrip(lstrip)
            .rstrip(rstrip)
            .normalized(normalized);
        to_add.push(token);
    }

    if to_add.is_empty() {
        return;
    }
    // Dedups against existing added-tokens, so this is a no-op when
    // tokenizer.json already had them. Return value = net-new count.
    let added = tokenizer.add_special_tokens(&to_add);
    if added > 0 {
        tracing::debug!(
            target: "tokenizer",
            path = %cfg_path.display(),
            added,
            candidates = to_add.len(),
            "merged additional special tokens from tokenizer_config.json"
        );
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // This self.tokenizer is the library
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|err| Error::msg(format!("Error tokenizing input: {err}")))?;

        Ok(Encoding::Hf(Box::new(encoding)))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let hf_encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), false)
            .map_err(|err| Error::msg(format!("Error batch tokenizing input: {err}")))?;

        let encodings = hf_encodings
            .into_iter()
            .map(|enc| Encoding::Hf(Box::new(enc)))
            .collect();

        Ok(encodings)
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        // This calls into the library
        let text = self
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|err| Error::msg(format!("Error de-tokenizing input: {err}")))?;

        Ok(text.into())
    }
}

impl Tokenizer for HuggingFaceTokenizer {}

impl From<HfTokenizer> for HuggingFaceTokenizer {
    fn from(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }
}

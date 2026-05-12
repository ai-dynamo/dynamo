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
    /// Load a HuggingFace tokenizer from `tokenizer.json`.
    ///
    /// If a sibling `tokenizer_config.json` is present, its
    /// `added_tokens_decoder` entries with `"special": true` are merged in as
    /// special tokens. This mirrors what HuggingFace's Python
    /// `AutoTokenizer.from_pretrained()` does — some model releases (e.g.
    /// Qwen2-VL-2B-Instruct) declare special tokens only in
    /// `tokenizer_config.json`'s `added_tokens_decoder`, not in
    /// `tokenizer.json`'s `added_tokens`. Without this merge, the rust
    /// `tokenizers` crate BPE-shatters those tokens at encode time (e.g.
    /// `<|image_pad|>` for Qwen2-VL-2B encodes to 6 sub-tokens instead of
    /// the intended single id 151655), which silently breaks anything that
    /// relies on the token round-tripping (MM-aware KV routing, response
    /// post-processing keyed on special tokens, etc.).
    ///
    /// The merge is idempotent: tokens already registered in
    /// `tokenizer.json`'s added_tokens are detected via
    /// `Model::token_to_id` and reuse their existing id, so this is a no-op
    /// for the common case where both files agree (Qwen2.5-VL, Qwen3-VL,
    /// LLaVA, Phi-3, etc.).
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

    /// Wrap an already-loaded `HfTokenizer`, then merge any
    /// `tokenizer_config.json` special tokens from `model_dir`.
    ///
    /// Use this when the caller has its own reason to load the bare
    /// `HfTokenizer` first (e.g. so it can surface a precise JSON parse
    /// error with line context) but still wants `tokenizer.json` +
    /// `tokenizer_config.json` to be merged into the final tokenizer.
    /// Without this merge, models whose `<|image_pad|>`-style special
    /// tokens live only in `tokenizer_config.json` (e.g. Qwen2-VL-2B)
    /// silently encode their special tokens as multi-token BPE
    /// fragments, breaking MM-aware KV routing and any other path that
    /// depends on those tokens round-tripping. See
    /// [`HuggingFaceTokenizer::from_file`] for the full rationale.
    pub fn from_tokenizer_with_model_dir(tokenizer: HfTokenizer, model_dir: &Path) -> Self {
        let mut tokenizer = tokenizer;
        merge_special_tokens_from_config(&mut tokenizer, model_dir);
        HuggingFaceTokenizer { tokenizer }
    }
}

/// Read `tokenizer_config.json` from `model_dir` and register any
/// `special: true` entries from its `added_tokens_decoder` map as special
/// tokens on `tokenizer`. See `HuggingFaceTokenizer::from_file` for context.
///
/// Errors (missing file, parse failures, non-special entries) are swallowed
/// because the file is optional: tokenizer.json alone is a valid HF
/// tokenizer layout. We only log a debug line on parse failure so operators
/// can spot a malformed sibling config without breaking the load.
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
        // Only promote entries the model release explicitly marks as special.
        // The id field is informational here: the tokenizers crate's
        // `add_special_tokens` looks the content up in `Model::token_to_id`
        // and reuses the existing vocab id (e.g. 151655 for Qwen2-VL's
        // `<|image_pad|>`), so we don't need to plumb the id through.
        if obj.get("special").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        let Some(content) = obj.get("content").and_then(|v| v.as_str()) else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        // Preserve the per-token strip/normalize flags so the merged token
        // matches the same input forms HF Python would match.
        let single_word = obj
            .get("single_word")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let lstrip = obj.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false);
        let rstrip = obj.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false);
        // `added_tokens_decoder` defaults to `normalized: false` for special
        // tokens; treat a missing key the same way.
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
    // `add_special_tokens` dedups against the existing added-tokens set, so
    // tokens already present from `tokenizer.json` are no-ops. The return
    // value is the count of net-new registrations.
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

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
///
/// `pub` so downstream crates (e.g. `dynamo-llm`'s model_card) can apply
/// the same promotion before extracting the special-token boundary list
/// for the L1 prefix cache — otherwise the cache and the wrapped
/// tokenizer would disagree on which strings are atomic specials.
pub fn merge_special_tokens_from_config(tokenizer: &mut HfTokenizer, model_dir: &Path) {
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
        // Warn (not debug) when the merge actually promotes anything —
        // intentionally loud so accidental promotion of a previously-
        // non-special token shows up immediately in worker logs. Lists
        // the literal token strings so debugging doesn't require a
        // second pass through `added_tokens_decoder`.
        let promoted: Vec<&str> = to_add.iter().map(|t| t.content.as_str()).collect();
        tracing::warn!(
            target: "tokenizer",
            path = %cfg_path.display(),
            added,
            candidates = to_add.len(),
            promoted = ?promoted,
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

#[cfg(test)]
mod tests {
    //! The existing tool-calling / reasoning parser tests inject already-
    //! decoded text and never run `tokenizer.decode`, so they cannot
    //! catch the class of bug where a non-special marker gets accidentally
    //! promoted to "special" and silently disappears under
    //! `skip_special_tokens=True`. One unit test pins both halves of the
    //! gate (promote special:true / skip special:false) end-to-end through
    //! the actual encode -> decode round trip — the layer above which the
    //! parser tests cannot reach.
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn merge_gate_round_trips_through_decode() {
        // Minimal WordLevel `tokenizer.json`. `<|special_kept|>` and
        // `<|special_dropped|>` are deliberately NOT pre-declared as
        // special in `added_tokens` here — that's exactly the shape
        // `merge_special_tokens_from_config` is supposed to fix from
        // tokenizer_config.json.
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<|special_kept|>": 3, "<|special_dropped|>": 4},
                "unk_token": "<unk>"
            }
        }"#;

        // `<|special_kept|>` is `special: true` → must be promoted, and
        // therefore stripped under skip_special_tokens=true.
        // `<|special_dropped|>` is `special: false` → must be skipped,
        // and therefore survive skip_special_tokens=true. The latter is
        // the Ryan/Keiven concern: tool-call / reasoning markers are
        // universally declared `special: false` precisely so the parser
        // still sees them; a regression here would silently break
        // parsing without any of the parser unit tests turning red.
        const TOKENIZER_CONFIG_JSON: &str = r#"{
            "added_tokens_decoder": {
                "3": {"content": "<|special_kept|>",    "special": true,  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                "4": {"content": "<|special_dropped|>", "special": false, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            }
        }"#;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            TOKENIZER_CONFIG_JSON,
        )
        .unwrap();

        let mut tokenizer = HfTokenizer::from_file(dir.path().join("tokenizer.json")).unwrap();
        merge_special_tokens_from_config(&mut tokenizer, dir.path());

        // Registry assertion: only the special:true entry was promoted.
        let specials: Vec<String> = {
            let mut v: Vec<String> = tokenizer
                .get_added_tokens_decoder()
                .values()
                .filter(|t| t.special)
                .map(|t| t.content.clone())
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            specials,
            vec!["<unk>".to_string(), "<|special_kept|>".to_string()],
            "<|special_kept|> promoted; <|special_dropped|> stayed non-special"
        );

        // Decode round-trip assertion (the layer parser tests cannot
        // reach): the registry mutation actually changes downstream
        // decode behavior in the expected direction.
        let enc_kept = tokenizer.encode("<|special_kept|>", false).unwrap();
        let decoded_strip = tokenizer.decode(enc_kept.get_ids(), true).unwrap();
        assert!(
            !decoded_strip.contains("<|special_kept|>"),
            "promoted special:true token must be stripped under skip_special_tokens=true; got {decoded_strip:?}"
        );

        let enc_drop = tokenizer.encode("<|special_dropped|>", false).unwrap();
        let decoded_keep = tokenizer.decode(enc_drop.get_ids(), true).unwrap();
        assert!(
            decoded_keep.contains("<|special_dropped|>"),
            "non-promoted special:false token must survive skip_special_tokens=true; got {decoded_keep:?}"
        );
    }
}

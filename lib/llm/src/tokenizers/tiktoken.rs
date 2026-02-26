// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;

use base64::Engine as _;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

use super::{
    Encoding, Error, Result, TokenIdType,
    traits::{Decoder, Encoder, Tokenizer},
};

/// Number of reserved special-token slots to generate when filling gaps in the vocabulary.
/// Most tiktoken-based models reserve 256 IDs above the base vocabulary for special tokens.
const DEFAULT_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// Kimi BPE pattern from moonshotai/Kimi-K2-Instruct/tokenization_kimi.py
const KIMI_PATTERN: &str = r#"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

pub struct TikTokenTokenizer {
    bpe: CoreBPE,
    special_token_ids: HashSet<u32>,
}

impl TikTokenTokenizer {
    /// Create a TikTokenTokenizer from a tiktoken model file.
    ///
    /// # Arguments
    /// * `path` - Path to the `.model` or `.tiktoken` file (base64 rank-per-line format)
    /// * `pattern` - BPE regex pattern string
    /// * `special_tokens` - Map of special token strings to their IDs
    pub fn from_file(
        path: &str,
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self> {
        let encoder = parse_tiktoken_file(path)?;
        let special_token_ids: HashSet<u32> = special_tokens.values().copied().collect();

        let bpe = CoreBPE::new(encoder, special_tokens, pattern)
            .map_err(|err| Error::msg(format!("Error creating tiktoken BPE: {err}")))?;

        Ok(Self {
            bpe,
            special_token_ids,
        })
    }

    /// Create a TikTokenTokenizer from a tiktoken model file, auto-detecting
    /// the BPE pattern from `config.json` and special tokens from `tokenizer_config.json`.
    ///
    /// The tiktoken file and config files must be in the same directory.
    pub fn from_file_auto(path: &str) -> Result<Self> {
        let file_path = Path::new(path);
        let directory = file_path
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;

        let pattern = detect_bpe_pattern(directory)?;
        let encoder = parse_tiktoken_file(path)?;
        // Use max rank + 1 (not len) to avoid ID collisions with sparse/non-contiguous ranks
        let num_base_tokens = encoder.values().max().map_or(0, |&m| m + 1) as usize;
        let special_tokens = load_special_tokens(directory, num_base_tokens)?;
        let special_token_ids: HashSet<u32> = special_tokens.values().copied().collect();

        let bpe = CoreBPE::new(encoder, special_tokens, pattern)
            .map_err(|err| Error::msg(format!("Error creating tiktoken BPE: {err}")))?;

        Ok(Self {
            bpe,
            special_token_ids,
        })
    }
}

impl Encoder for TikTokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let token_ids: Vec<u32> = self.bpe.encode_with_special_tokens(input);
        Ok(Encoding::Sp(token_ids))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for TikTokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|&&id| !self.special_token_ids.contains(&id))
                .copied()
                .collect()
        } else {
            token_ids.to_vec()
        };

        self.bpe
            .decode(ids)
            .map_err(|err| Error::msg(format!("Error decoding tiktoken tokens: {err}")))
    }
}

impl Tokenizer for TikTokenTokenizer {}

/// Parse a tiktoken model file (base64-encoded token + rank per line).
fn parse_tiktoken_file(path: &str) -> Result<FxHashMap<Vec<u8>, u32>> {
    let contents = std::fs::read_to_string(path)
        .map_err(|err| Error::msg(format!("Failed to read tiktoken file '{path}': {err}")))?;

    let engine = base64::engine::general_purpose::STANDARD;
    let mut encoder = FxHashMap::default();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let token_b64 = parts
            .next()
            .ok_or_else(|| Error::msg(format!("Invalid tiktoken line (no token): {line}")))?;
        let rank_str = parts
            .next()
            .ok_or_else(|| Error::msg(format!("Invalid tiktoken line (no rank): {line}")))?;

        let token_bytes = engine
            .decode(token_b64)
            .map_err(|err| Error::msg(format!("Invalid base64 in tiktoken file: {err}")))?;
        let rank: u32 = rank_str
            .parse()
            .map_err(|err| Error::msg(format!("Invalid rank in tiktoken file: {err}")))?;

        encoder.insert(token_bytes, rank);
    }

    Ok(encoder)
}

/// Detect the BPE pattern for a model by reading `model_type` from `config.json`.
fn detect_bpe_pattern(directory: &Path) -> Result<&'static str> {
    let model_type: String = crate::file_json_field(&directory.join("config.json"), "model_type")
        .map_err(|err| {
        Error::msg(format!("Failed to read model_type from config.json: {err}"))
    })?;

    match model_type.as_str() {
        "kimi" | "kimi_k2" | "kimi_k25" | "deepseek_v3" => Ok(KIMI_PATTERN),
        _ => Err(Error::msg(format!(
            "Unsupported tiktoken model_type '{model_type}'. \
             Currently supported: kimi, kimi_k2, kimi_k25, deepseek_v3. \
             To add a new model type, extend detect_bpe_pattern() in tokenizers/tiktoken.rs \
             with the appropriate BPE regex pattern. \
             Alternatively, provide a tokenizer.json (HuggingFace format) instead."
        ))),
    }
}

/// Load special tokens from `tokenizer_config.json` in the model directory.
///
/// Reads the `added_tokens_decoder` field which maps string token IDs to token definitions.
/// Falls back to generating `<|reserved_token_{id}|>` names for unmapped IDs.
fn load_special_tokens(directory: &Path, num_base_tokens: usize) -> Result<FxHashMap<String, u32>> {
    let config_path = directory.join("tokenizer_config.json");
    let mut special_tokens = FxHashMap::default();

    if !config_path.exists() {
        // No tokenizer_config.json — generate default reserved tokens
        for i in 0..DEFAULT_NUM_RESERVED_SPECIAL_TOKENS {
            let id = num_base_tokens as u32 + i;
            special_tokens.insert(format!("<|reserved_token_{i}|>"), id);
        }
        return Ok(special_tokens);
    }

    let contents = std::fs::read_to_string(&config_path)
        .map_err(|err| Error::msg(format!("Failed to read tokenizer_config.json: {err}")))?;

    let config: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|err| Error::msg(format!("Failed to parse tokenizer_config.json: {err}")))?;

    if let Some(added_tokens) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
    {
        for (id_str, token_def) in added_tokens {
            let id: u32 = id_str.parse().map_err(|err| {
                Error::msg(format!(
                    "Invalid token ID '{id_str}' in added_tokens_decoder: {err}"
                ))
            })?;

            let content = token_def
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    // This shouldn't happen in well-formed configs, but handle gracefully
                    tracing::warn!("Missing 'content' field for token ID {id}");
                    ""
                });

            if !content.is_empty() {
                special_tokens.insert(content.to_string(), id);
            }
        }

        // Fill in any gaps with reserved tokens for the expected range
        let used_ids: HashSet<u32> = special_tokens.values().copied().collect();
        for i in 0..DEFAULT_NUM_RESERVED_SPECIAL_TOKENS {
            let id = num_base_tokens as u32 + i;
            if !used_ids.contains(&id) {
                special_tokens.insert(format!("<|reserved_token_{i}|>"), id);
            }
        }
    } else {
        // No added_tokens_decoder — generate default reserved tokens
        for i in 0..DEFAULT_NUM_RESERVED_SPECIAL_TOKENS {
            let id = num_base_tokens as u32 + i;
            special_tokens.insert(format!("<|reserved_token_{i}|>"), id);
        }
    }

    Ok(special_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_tiktoken_file(dir: &Path) -> String {
        let engine = base64::engine::general_purpose::STANDARD;
        let mut content = String::new();

        // Create some simple token entries: single bytes with sequential ranks
        let tokens: Vec<(&[u8], u32)> = vec![
            (b"h", 0),
            (b"e", 1),
            (b"l", 2),
            (b"o", 3),
            (b" ", 4),
            (b"w", 5),
            (b"r", 6),
            (b"d", 7),
            (b"he", 8),
            (b"ll", 9),
            (b"lo", 10),
            (b"wo", 11),
            (b"rl", 12),
            (b"hel", 13),
            (b"llo", 14),
            (b"wor", 15),
            (b"hell", 16),
            (b"ello", 17),
            (b"worl", 18),
            (b"hello", 19),
            (b"world", 20),
        ];

        for (token, rank) in tokens {
            let encoded = engine.encode(token);
            content.push_str(&format!("{encoded} {rank}\n"));
        }

        let file_path = dir.join("tiktoken.model");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_path.to_str().unwrap().to_string()
    }

    fn create_test_config(dir: &Path, model_type: &str) {
        let config = serde_json::json!({
            "model_type": model_type,
            "max_position_embeddings": 32768,
            "eos_token_id": [21]
        });
        let file_path = dir.join("config.json");
        std::fs::write(file_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    }

    fn create_test_tokenizer_config(dir: &Path, num_base_tokens: usize) {
        let mut added_tokens = serde_json::Map::new();
        let bos_id = num_base_tokens;
        let eos_id = num_base_tokens + 1;

        added_tokens.insert(
            bos_id.to_string(),
            serde_json::json!({"content": "[BOS]", "special": true}),
        );
        added_tokens.insert(
            eos_id.to_string(),
            serde_json::json!({"content": "[EOS]", "special": true}),
        );

        let config = serde_json::json!({
            "added_tokens_decoder": added_tokens
        });

        let file_path = dir.join("tokenizer_config.json");
        std::fs::write(file_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    }

    #[test]
    fn test_parse_tiktoken_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());
        let encoder = parse_tiktoken_file(&file_path).unwrap();
        assert_eq!(encoder.len(), 21);
        assert_eq!(encoder[b"hello".as_slice()], 19);
        assert_eq!(encoder[b"world".as_slice()], 20);
    }

    #[test]
    fn test_parse_tiktoken_file_missing() {
        let result = parse_tiktoken_file("/nonexistent/path/tiktoken.model");
        assert!(result.is_err());
    }

    #[test]
    fn test_tiktoken_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());

        let mut special_tokens = FxHashMap::default();
        special_tokens.insert("[BOS]".to_string(), 21_u32);
        special_tokens.insert("[EOS]".to_string(), 22_u32);

        // Use a simple pattern for testing
        let pattern = r"[\w]+|[^\w\s]+|\s+";

        let tokenizer = TikTokenTokenizer::from_file(&file_path, pattern, special_tokens).unwrap();

        // Test encode
        let encoding = tokenizer.encode("hello world").unwrap();
        let ids = encoding.token_ids();
        assert!(!ids.is_empty());

        // Test decode roundtrip
        let decoded = tokenizer.decode(ids, false).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_tiktoken_encoding_variant() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());

        let special_tokens = FxHashMap::default();
        let pattern = r"[\w]+|[^\w\s]+|\s+";

        let tokenizer = TikTokenTokenizer::from_file(&file_path, pattern, special_tokens).unwrap();
        let encoding = tokenizer.encode("hello").unwrap();

        // Verify it produces the Sp variant
        match &encoding {
            Encoding::Sp(_) => {}
            other => panic!("Expected Encoding::Sp, got {:?}", other),
        }
    }

    #[test]
    fn test_tiktoken_skip_special_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());

        let mut special_tokens = FxHashMap::default();
        special_tokens.insert("[BOS]".to_string(), 21_u32);
        special_tokens.insert("[EOS]".to_string(), 22_u32);

        let pattern = r"[\w]+|[^\w\s]+|\s+";

        let tokenizer = TikTokenTokenizer::from_file(&file_path, pattern, special_tokens).unwrap();

        // Encode hello and prepend/append special tokens
        let encoding = tokenizer.encode("hello").unwrap();
        let mut ids = vec![21u32]; // [BOS]
        ids.extend(encoding.token_ids());
        ids.push(22); // [EOS]

        // Decode with skip_special_tokens=true should strip special tokens
        let decoded_skip = tokenizer.decode(&ids, true).unwrap();
        assert_eq!(decoded_skip, "hello");

        // Decode with skip_special_tokens=false should include them
        let decoded_all = tokenizer.decode(&ids, false).unwrap();
        assert!(decoded_all.contains("hello"));
    }

    #[test]
    fn test_tiktoken_from_file_auto() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());

        create_test_config(dir.path(), "kimi");
        create_test_tokenizer_config(dir.path(), 21);

        let tokenizer = TikTokenTokenizer::from_file_auto(&file_path).unwrap();

        // Basic encode/decode roundtrip
        let encoding = tokenizer.encode("hello world").unwrap();
        let ids = encoding.token_ids();
        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(ids, false).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_detect_bpe_pattern_unknown() {
        let dir = tempfile::tempdir().unwrap();
        create_test_config(dir.path(), "unknown_model");
        let result = detect_bpe_pattern(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_special_tokens_no_config() {
        let dir = tempfile::tempdir().unwrap();
        let tokens = load_special_tokens(dir.path(), 100).unwrap();
        assert_eq!(tokens.len(), 256);
        assert_eq!(tokens["<|reserved_token_0|>"], 100);
        assert_eq!(tokens["<|reserved_token_255|>"], 355);
    }

    #[test]
    fn test_load_special_tokens_with_config() {
        let dir = tempfile::tempdir().unwrap();
        create_test_tokenizer_config(dir.path(), 100);
        let tokens = load_special_tokens(dir.path(), 100).unwrap();
        assert_eq!(tokens["[BOS]"], 100);
        assert_eq!(tokens["[EOS]"], 101);
        // Should also have reserved tokens filling gaps
        assert!(tokens.len() > 2);
    }

    #[test]
    fn test_tiktoken_encode_batch() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_test_tiktoken_file(dir.path());

        let special_tokens = FxHashMap::default();
        let pattern = r"[\w]+|[^\w\s]+|\s+";

        let tokenizer = TikTokenTokenizer::from_file(&file_path, pattern, special_tokens).unwrap();

        let inputs = &["hello", "world"];
        let encodings = tokenizer.encode_batch(inputs).unwrap();
        assert_eq!(encodings.len(), 2);

        for (encoding, input) in encodings.iter().zip(inputs.iter()) {
            let decoded = tokenizer.decode(encoding.token_ids(), false).unwrap();
            assert_eq!(decoded, *input);
        }
    }
}

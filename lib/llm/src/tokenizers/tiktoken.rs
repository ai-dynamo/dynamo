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
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Number of reserved special-token slots to generate when filling gaps in the vocabulary.
/// Most tiktoken-based models reserve 256 IDs above the base vocabulary for special tokens.
const DEFAULT_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// Kimi BPE pattern from moonshotai/Kimi-K2-Instruct/tokenization_kimi.py
const KIMI_PATTERN: &str = r#"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

pub struct TikTokenTokenizer {
    bpe: CoreBPE,
    special_token_ids: HashSet<u32>,
    decoder_tokens: FxHashMap<u32, Vec<u8>>,
    special_tokens_decoder: FxHashMap<u32, Vec<u8>>,
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
        let decoder_tokens: FxHashMap<u32, Vec<u8>> = encoder
            .iter()
            .map(|(bytes, &id)| (id, bytes.clone()))
            .collect();
        let special_tokens_decoder: FxHashMap<u32, Vec<u8>> = special_tokens
            .iter()
            .map(|(token, &id)| (id, token.as_bytes().to_vec()))
            .collect();
        let special_token_ids: HashSet<u32> = special_tokens.values().copied().collect();

        let bpe = CoreBPE::new(encoder, special_tokens, pattern)
            .map_err(|err| Error::msg(format!("Error creating tiktoken BPE: {err}")))?;

        Ok(Self {
            bpe,
            special_token_ids,
            decoder_tokens,
            special_tokens_decoder,
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
        let decoder_tokens: FxHashMap<u32, Vec<u8>> = encoder
            .iter()
            .map(|(bytes, &id)| (id, bytes.clone()))
            .collect();
        // Use max rank + 1 (not len) to avoid ID collisions with sparse/non-contiguous ranks
        let num_base_tokens = encoder.values().max().map_or(0, |&m| m + 1) as usize;
        let special_tokens = load_special_tokens(directory, num_base_tokens)?;
        let special_tokens_decoder: FxHashMap<u32, Vec<u8>> = special_tokens
            .iter()
            .map(|(token, &id)| (id, token.as_bytes().to_vec()))
            .collect();
        let special_token_ids: HashSet<u32> = special_tokens.values().copied().collect();

        let bpe = CoreBPE::new(encoder, special_tokens, pattern)
            .map_err(|err| Error::msg(format!("Error creating tiktoken BPE: {err}")))?;

        Ok(Self {
            bpe,
            special_token_ids,
            decoder_tokens,
            special_tokens_decoder,
        })
    }
}

impl Encoder for TikTokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.encode_with_special_tokens(input, true)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_with_special_tokens(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let token_ids: Vec<u32> = if add_special_tokens {
            self.bpe.encode_with_special_tokens(input)
        } else {
            self.bpe.encode_ordinary(input)
        };
        Ok(Encoding::Sp(token_ids))
    }
}

impl Decoder for TikTokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        let ids: Vec<u32> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|&&id| !self.special_token_ids.contains(&id))
                .copied()
                .collect()
        } else {
            token_ids.to_vec()
        };

        // Try strict UTF-8 first: valid bytes get `Complete` with zero extra allocation
        // (takes ownership of the Vec). This correctly handles vocabulary tokens whose
        // raw bytes are EF BF BD (legitimate U+FFFD) -- they are valid UTF-8 and must
        // not be confused with incomplete multi-byte sequences.
        //
        // On failure, fall back to lossy conversion so partial multi-byte sequences
        // become U+FFFD, then classify via the trailing-FFFD heuristic. This path is
        // only hit during incremental detokenization of byte-fallback tokens.
        let bytes: Vec<u8> = self.bpe._decode_native_and_split(ids).flatten().collect();
        match String::from_utf8(bytes) {
            Ok(text) => Ok(DecodeResult::Complete(text)),
            Err(e) => {
                let text = String::from_utf8_lossy(e.as_bytes()).into_owned();
                Ok(DecodeResult::from_decoded(text))
            }
        }
    }
}

impl Tokenizer for TikTokenTokenizer {
    fn convert_ids_to_tokens(&self, token_ids: &[TokenIdType]) -> Result<Vec<String>> {
        Ok(token_ids
            .iter()
            .map(|id| {
                self.decoder_tokens
                    .get(id)
                    .or_else(|| self.special_tokens_decoder.get(id))
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .unwrap_or_default()
            })
            .collect())
    }
}

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
        // baseten-admin/Kimi-2.5-text-nvfp4-v3 model has model_type: "deepseek_v3" in its config.json
        // because Kimi K2.5 is built on the DeepSeek V3 architecture.
        // it still ships the Kimi tiktoken tokenizer file, so the KIMI_PATTERN BPE regex is the
        // correct pattern to use.  No pure DeepSeek V3 model uses tiktoken.model files
        // (they use tokenizer.json instead) so this match is safe.
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
            special_tokens.insert(format!("<|reserved_token_{id}|>"), id);
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
                special_tokens.insert(format!("<|reserved_token_{id}|>"), id);
            }
        }
    } else {
        // No added_tokens_decoder — generate default reserved tokens
        for i in 0..DEFAULT_NUM_RESERVED_SPECIAL_TOKENS {
            let id = num_base_tokens as u32 + i;
            special_tokens.insert(format!("<|reserved_token_{id}|>"), id);
        }
    }

    Ok(special_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::DecodeStream;
    use std::io::Write;
    use std::sync::Arc;

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
        let decoded: String = tokenizer.decode(ids, false).unwrap().into();
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
        let decoded_skip: String = tokenizer.decode(&ids, true).unwrap().into();
        assert_eq!(decoded_skip, "hello");

        // Decode with skip_special_tokens=false should include them
        let decoded_all: String = tokenizer.decode(&ids, false).unwrap().into();
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

        let decoded: String = tokenizer.decode(ids, false).unwrap().into();
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
        assert_eq!(tokens["<|reserved_token_100|>"], 100);
        assert_eq!(tokens["<|reserved_token_355|>"], 355);
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

    /// Helper: create a tiktoken file that includes raw byte tokens (byte fallback tokens).
    fn create_test_tiktoken_file_with_byte_tokens(dir: &Path) -> String {
        let engine = base64::engine::general_purpose::STANDARD;
        let mut content = String::new();

        let tokens: Vec<(&[u8], u32)> = vec![
            (b"h", 0),
            (b"e", 1),
            (b"l", 2),
            (b"o", 3),
            (b" ", 4),
            (b"hello", 5),
        ];

        for (token, rank) in &tokens {
            let encoded = engine.encode(token);
            content.push_str(&format!("{encoded} {rank}\n"));
        }

        // Byte-fallback tokens: individual bytes that form CJK character "你" (U+4F60)
        // UTF-8 encoding: 0xE4 0xBD 0xA0
        let byte_tokens: Vec<(Vec<u8>, u32)> =
            vec![(vec![0xE4], 100), (vec![0xBD], 101), (vec![0xA0], 102)];

        for (token, rank) in &byte_tokens {
            let encoded = engine.encode(token);
            content.push_str(&format!("{encoded} {rank}\n"));
        }

        // Bytes for emoji "😀" (U+1F600) — 4-byte UTF-8: 0xF0 0x9F 0x98 0x80
        let emoji_tokens: Vec<(Vec<u8>, u32)> = vec![
            (vec![0xF0], 200),
            (vec![0x9F], 201),
            (vec![0x98], 202),
            (vec![0x80], 203),
        ];

        for (token, rank) in &emoji_tokens {
            let encoded = engine.encode(token);
            content.push_str(&format!("{encoded} {rank}\n"));
        }

        // Legitimate U+FFFD token: valid UTF-8 bytes EF BF BD (replacement character
        // as an actual vocabulary entry, not an artifact of lossy conversion)
        let fffd_token: Vec<(Vec<u8>, u32)> = vec![(vec![0xEF, 0xBF, 0xBD], 300)];

        for (token, rank) in &fffd_token {
            let encoded = engine.encode(token);
            content.push_str(&format!("{encoded} {rank}\n"));
        }

        let file_path = dir.join("tiktoken.model");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_path.to_str().unwrap().to_string()
    }

    fn create_byte_token_tokenizer(dir: &Path) -> TikTokenTokenizer {
        let file_path = create_test_tiktoken_file_with_byte_tokens(dir);
        let special_tokens = FxHashMap::default();
        let pattern = r"[\w]+|[^\w\s]+|\s+";
        TikTokenTokenizer::from_file(&file_path, pattern, special_tokens).unwrap()
    }

    /// Reproduces the original panic: decoding a single byte-fallback token that is
    /// part of a multi-byte UTF-8 character. Before the fix, CoreBPE::decode() would
    /// call String::from_utf8() on [0xE4] and error with "incomplete utf-8 byte sequence".
    #[test]
    fn test_decode_single_incomplete_utf8_byte_does_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[100], false);
        assert!(
            result.is_ok(),
            "decode() should not error on incomplete UTF-8 bytes"
        );
        let decode_result = result.unwrap();
        assert!(
            decode_result.is_partial(),
            "incomplete UTF-8 byte should produce DecodeResult::Partial, got: {:?}",
            decode_result
        );
    }

    /// Without the fix, fails with "incomplete utf-8 byte sequence" from CoreBPE::decode().
    #[test]
    fn test_decode_two_of_three_utf8_bytes_does_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[100, 101], false);
        assert!(result.is_ok());
        let decode_result = result.unwrap();
        assert!(
            decode_result.is_partial(),
            "incomplete 2-of-3 UTF-8 bytes should produce DecodeResult::Partial, got: {:?}",
            decode_result
        );
    }

    /// When all bytes of a multi-byte character are present, the concatenated bytes form
    /// valid UTF-8, so this test passes both before and after the fix. It serves as a
    /// correctness check that the lossy conversion doesn't corrupt complete characters.
    #[test]
    fn test_decode_complete_multibyte_utf8_produces_correct_char() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[100, 101, 102], false);
        assert!(result.is_ok());
        assert_eq!(String::from(result.unwrap()), "你");
    }

    /// All 4 emoji bytes together form valid UTF-8, so this passes both before and after
    /// the fix. Validates that lossy conversion doesn't alter complete multi-byte sequences.
    #[test]
    fn test_decode_complete_4byte_emoji_from_byte_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[200, 201, 202, 203], false);
        assert!(result.is_ok());
        assert_eq!(String::from(result.unwrap()), "😀");
    }

    /// Regression test: a vocabulary token whose raw bytes are EF BF BD (the valid
    /// UTF-8 encoding of U+FFFD) must decode as `Complete`, not `Partial`. Before the
    /// from_utf8 fast-path fix, from_utf8_lossy + the trailing-FFFD heuristic would
    /// misclassify this as Partial, causing the incremental decoder to suppress it.
    #[test]
    fn test_decode_legitimate_replacement_char_token_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[300], false);
        assert!(result.is_ok());
        let decode_result = result.unwrap();
        assert!(
            decode_result.is_complete(),
            "legitimate U+FFFD vocab token must be Complete, got: {:?}",
            decode_result
        );
        assert_eq!(decode_result.as_str(), "\u{FFFD}");
    }

    /// Without the fix, fails with "incomplete utf-8 byte sequence" from CoreBPE::decode().
    #[test]
    fn test_decode_partial_emoji_does_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[200], false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_partial());
    }

    /// Without the fix, fails with "incomplete utf-8 byte sequence" from CoreBPE::decode().
    #[test]
    fn test_decode_mixed_ascii_and_incomplete_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());

        let result = tokenizer.decode(&[5, 100], false);
        assert!(result.is_ok());
        let decode_result = result.unwrap();
        assert!(
            decode_result.is_partial(),
            "trailing incomplete byte should produce DecodeResult::Partial"
        );
        let text: String = decode_result.into();
        assert!(
            text.starts_with("hello"),
            "should start with 'hello', got: {:?}",
            text
        );
    }

    /// End-to-end incremental detokenization: DecodeStream buffers partial bytes,
    /// emits the complete character once all bytes arrive.
    /// Without the fix, fails with "incomplete utf-8 byte sequence" from CoreBPE::decode().
    #[test]
    fn test_decode_stream_incremental_multibyte_reassembly() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());
        let tokenizer_arc: Arc<dyn crate::tokenizers::traits::Tokenizer> = Arc::new(tokenizer);

        let mut stream = DecodeStream::new(tokenizer_arc, &[5], false);

        let r1 = stream.step(100).unwrap();
        assert_eq!(r1, None, "first byte of 3-byte char should be buffered");

        let r2 = stream.step(101).unwrap();
        assert_eq!(r2, None, "second byte of 3-byte char should be buffered");

        let r3 = stream.step(102).unwrap();
        assert!(r3.is_some(), "third byte should complete the character");
        assert_eq!(r3.unwrap(), "你");
    }

    /// Without the fix, fails with "incomplete utf-8 byte sequence" from CoreBPE::decode().
    #[test]
    fn test_decode_stream_incremental_emoji_reassembly() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = create_byte_token_tokenizer(dir.path());
        let tokenizer_arc: Arc<dyn crate::tokenizers::traits::Tokenizer> = Arc::new(tokenizer);

        let mut stream = DecodeStream::new(tokenizer_arc, &[5], false);

        let r1 = stream.step(200).unwrap();
        assert_eq!(r1, None, "byte 1/4 of emoji should be buffered");

        let r2 = stream.step(201).unwrap();
        assert_eq!(r2, None, "byte 2/4 of emoji should be buffered");

        let r3 = stream.step(202).unwrap();
        assert_eq!(r3, None, "byte 3/4 of emoji should be buffered");

        let r4 = stream.step(203).unwrap();
        assert!(r4.is_some(), "byte 4/4 should complete the emoji");
        assert_eq!(r4.unwrap(), "😀");
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
            let decoded: String = tokenizer
                .decode(encoding.token_ids(), false)
                .unwrap()
                .into();
            assert_eq!(decoded, *input);
        }
    }

    /// Helper: create a tiktoken file containing all 256 single-byte tokens (ranks 0..255).
    /// This gives a complete byte-level base vocabulary so any ASCII string can be encoded.
    fn create_byte_level_tiktoken_file(dir: &Path) -> String {
        let engine = base64::engine::general_purpose::STANDARD;
        let mut content = String::new();
        for byte_val in 0u16..256 {
            let encoded = engine.encode([byte_val as u8]);
            content.push_str(&format!("{encoded} {byte_val}\n"));
        }
        let file_path = dir.join("tiktoken.model");
        std::fs::write(&file_path, &content).unwrap();
        file_path.to_str().unwrap().to_string()
    }

    /// Regression test for Kimi K2.5 tokenizer inflation.
    ///
    /// Python's tokenization_kimi.py names unnamed reserved tokens by absolute ID:
    ///   `<|reserved_token_{absolute_id}|>`
    ///
    /// The Rust code previously used relative offsets (0..255) as the naming index,
    /// so when a prompt contained `<|reserved_token_258|>` the Rust tokenizer did NOT
    /// recognize it as a special token. Each occurrence was encoded as multiple BPE
    /// tokens instead of 1, inflating an 8192-token prompt to 9038 tokens and causing
    /// TRT-LLM to reject the request.
    #[test]
    fn test_reserved_token_absolute_id_naming_kimi_k25_regression() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_byte_level_tiktoken_file(dir.path());

        // config.json with kimi model type -> triggers KIMI_PATTERN
        create_test_config(dir.path(), "kimi");

        // tokenizer_config.json: BOS at 256, EOS at 257. Base vocab is IDs 0..255.
        create_test_tokenizer_config(dir.path(), 256);

        let tokenizer = TikTokenTokenizer::from_file_auto(&file_path).unwrap();

        // ID 256 = [BOS], ID 257 = [EOS].
        // ID 258 = first UNNAMED reserved token.
        //   With fix:   named <|reserved_token_258|>  (absolute ID)
        //   Before fix: named <|reserved_token_2|>    (relative offset)

        // Single unnamed reserved token should be recognized as 1 special token.
        let single = "<|reserved_token_258|>";
        let enc = tokenizer.encode(single).unwrap();
        assert_eq!(
            enc.token_ids().len(),
            1,
            "'{single}' should be 1 special token, got {} tokens: {:?}. \
             This means fallback naming still uses relative offsets instead of absolute IDs.",
            enc.token_ids().len(),
            enc.token_ids()
        );
        assert_eq!(enc.token_ids()[0], 258);

        // Multiple unnamed reserved tokens in sequence (mini version of the benchmark).
        // IDs 258..268 are all unnamed; with the fix they're <|reserved_token_258|>..267.
        let multi: String = (258u32..268)
            .map(|id| format!("<|reserved_token_{id}|>"))
            .collect();
        let enc_multi = tokenizer.encode(&multi).unwrap();
        assert_eq!(
            enc_multi.token_ids().len(),
            10,
            "10 reserved token strings should produce exactly 10 tokens, got {}: {:?}",
            enc_multi.token_ids().len(),
            enc_multi.token_ids()
        );
        let expected_ids: Vec<u32> = (258..268).collect();
        assert_eq!(enc_multi.token_ids(), &expected_ids);
    }

    /// Confirm that the old relative-offset naming would cause token inflation.
    /// Manually builds a tokenizer whose special-token map uses the WRONG names
    /// (relative offsets), then shows the same string encodes as many tokens.
    #[test]
    fn test_relative_offset_naming_causes_inflation() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = create_byte_level_tiktoken_file(dir.path());

        let _encoder = parse_tiktoken_file(&file_path).unwrap();
        let num_base_tokens = 256usize;

        // Build special tokens with the OLD (buggy) relative-offset naming
        let mut bad_special_tokens: FxHashMap<String, u32> = FxHashMap::default();
        bad_special_tokens.insert("[BOS]".to_string(), 256);
        bad_special_tokens.insert("[EOS]".to_string(), 257);
        for i in 0..DEFAULT_NUM_RESERVED_SPECIAL_TOKENS {
            let id = num_base_tokens as u32 + i;
            if id != 256 && id != 257 {
                // OLD naming: relative offset i, not absolute id
                bad_special_tokens.insert(format!("<|reserved_token_{i}|>"), id);
            }
        }

        let bad_tokenizer =
            TikTokenTokenizer::from_file(&file_path, KIMI_PATTERN, bad_special_tokens).unwrap();

        // With the wrong naming, <|reserved_token_258|> is NOT recognized as special.
        // It gets split into byte-level BPE tokens -> many more than 1.
        let input = "<|reserved_token_258|>";
        let enc = bad_tokenizer.encode(input).unwrap();
        assert!(
            enc.token_ids().len() > 1,
            "With buggy relative-offset naming, '{}' should NOT be recognized as a \
             single special token. Got {} token(s): {:?}",
            input,
            enc.token_ids().len(),
            enc.token_ids()
        );

        // Show the inflation: 10 reserved tokens produce far more than 10 BPE tokens.
        let multi: String = (258u32..268)
            .map(|id| format!("<|reserved_token_{id}|>"))
            .collect();
        let enc_multi = bad_tokenizer.encode(&multi).unwrap();
        assert!(
            enc_multi.token_ids().len() > 10,
            "With buggy naming, 10 reserved token strings should inflate beyond 10 tokens. \
             Got {}",
            enc_multi.token_ids().len(),
        );
    }
}

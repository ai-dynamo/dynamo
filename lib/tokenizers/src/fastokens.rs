// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fastokens backend using the `fastokens` crate for high-performance BPE encoding.
//!
//! `fastokens` only supports encoding, so this module provides a hybrid tokenizer that
//! uses `fastokens` for encoding and falls back to `HuggingFaceTokenizer` for decoding.
//! Both are loaded from the same `tokenizer.json` file.

use std::path::Path;

use rayon::prelude::*;

use super::{
    Encoding, Error, Result, TokenIdType,
    hf::HuggingFaceTokenizer,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Hybrid tokenizer: fast BPE encoding via `fastokens`, decoding via HuggingFace.
///
/// Both backends are loaded from the same `tokenizer.json` file.
pub struct FastTokenizer {
    fast_encoder: fastokens::Tokenizer,
    hf_decoder: HuggingFaceTokenizer,
}

impl FastTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let fast_encoder = fastokens::Tokenizer::from_file(Path::new(path))
            .map_err(|e| Error::msg(format!("Error loading fastokens tokenizer: {e}")))?;
        let hf_decoder = HuggingFaceTokenizer::from_file(path)?;
        Ok(Self {
            fast_encoder,
            hf_decoder,
        })
    }
}

impl Encoder for FastTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let ids = self
            .fast_encoder
            .encode(input)
            .map_err(|e| Error::msg(format!("Fastokens encode error: {e}")))?;
        Ok(Encoding::Sp(ids))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for FastTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.hf_decoder.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for FastTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HuggingFaceTokenizer;

    // Minimal synthetic BPE tokenizer with no normalizer or post-processor --
    // compatible with fastokens. Vocab covers: H,T,a,d,e,h,i,l,o,r,s,t,w + punctuation.
    const TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llm/tests/data/sample-models/minimal-bpe/tokenizer.json"
    );

    #[test]
    fn test_fast_encode_decode_roundtrip() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        // Encode then decode: verifies both paths execute without error.
        // With a null decoder, HF inserts spaces between tokens so exact equality
        // is not expected here -- we just verify the operations succeed and produce
        // non-empty results.
        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();
        assert!(!encoding.token_ids().is_empty());
        let decoded: String = tokenizer.decode(encoding.token_ids(), true).unwrap().into();
        assert!(!decoded.is_empty());
        // The decoded text should contain the same non-space characters
        let enc_chars: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let dec_chars: String = decoded.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            enc_chars, dec_chars,
            "non-space characters must be preserved"
        );
    }

    #[test]
    fn test_fast_matches_hf_encoding() {
        let fast = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();

        for text in &["Hello, world!", "Hello", " world", "He llo"] {
            let fast_ids = fast.encode(text).unwrap();
            let hf_ids = hf.encode(text).unwrap();
            assert_eq!(
                fast_ids.token_ids(),
                hf_ids.token_ids(),
                "fastokens and HuggingFace must produce identical token IDs for '{text}'"
            );
        }
    }

    #[test]
    fn test_fast_batch_encode() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let inputs = &["Hello", " world", "Hello, world!"];
        let encodings = tokenizer.encode_batch(inputs).unwrap();
        assert_eq!(encodings.len(), inputs.len());
        for (enc, input) in encodings.iter().zip(inputs.iter()) {
            assert!(
                !enc.token_ids().is_empty(),
                "encoding for '{input}' must be non-empty"
            );
        }
    }

    #[test]
    fn test_fast_with_decode_stream() {
        use crate::Tokenizer as TokenizerWrapper;
        use std::sync::Arc;

        let tokenizer = Arc::new(FastTokenizer::from_file(TOKENIZER_PATH).unwrap());
        let wrapper = TokenizerWrapper::from(tokenizer);

        // Encode a prompt and a continuation, then step through the decode stream
        let prompt_ids = wrapper.encode("Hello").unwrap().token_ids().to_vec();
        let continuation = ", world!";
        let cont_ids = wrapper.encode(continuation).unwrap().token_ids().to_vec();

        let mut stream = wrapper.decode_stream(&prompt_ids, true);
        // Accumulate incremental chunks from decode_stream
        let mut accumulated = String::new();
        for id in &cont_ids {
            if let Some(chunk) = stream.step(*id).unwrap() {
                accumulated.push_str(&chunk);
            }
        }

        // DecodeStream uses prompt tokens as context, so the expected text is
        // decode(prompt + continuation) minus decode(prompt) -- not a bare
        // decode(continuation) which lacks the surrounding context.
        let mut all_ids = prompt_ids.clone();
        all_ids.extend_from_slice(&cont_ids);
        let full_text: String = wrapper.decode(&all_ids, true).unwrap().into();
        let prompt_text: String = wrapper.decode(&prompt_ids, true).unwrap().into();
        let expected = &full_text[prompt_text.len()..];
        assert_eq!(
            accumulated, expected,
            "streamed chunks must equal context-aware decoded continuation"
        );
    }
}

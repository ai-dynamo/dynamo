// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tokenizers::tokenizer::Tokenizer as HfTokenizer;

use super::{
    Encoding, Error, Result, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

pub struct HuggingFaceTokenizer {
    tokenizer: HfTokenizer,
}

impl HuggingFaceTokenizer {
    pub fn from_file(model_name: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(model_name)
            .map_err(|err| Error::msg(format!("Error loading tokenizer: {}", err)))?;

        Ok(HuggingFaceTokenizer { tokenizer })
    }

    pub fn from_tokenizer(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // Use add_special_tokens=true to match TikTokenTokenizer::encode() behaviour.
        // Both backends must agree on whether BOS/EOS are included so that callers
        // (e.g. /v1/tokenize, rl_tokenize_prompt) get consistent token counts
        // regardless of which backend is active.  Callers that explicitly need no
        // special tokens should call encode_with_special_tokens(input, false) directly.
        self.encode_with_special_tokens(input, true)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let hf_encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), true) // true to match encode() above
            .map_err(|err| Error::msg(format!("Error batch tokenizing input: {err}")))?;

        let encodings = hf_encodings
            .into_iter()
            .map(|enc| Encoding::Hf(Box::new(enc)))
            .collect();

        Ok(encodings)
    }

    fn encode_with_special_tokens(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // This self.tokenizer is the library
        let encoding = self
            .tokenizer
            .encode(input, add_special_tokens)
            .map_err(|err| Error::msg(format!("Error tokenizing input: {err}")))?;

        Ok(Encoding::Hf(Box::new(encoding)))
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

impl Tokenizer for HuggingFaceTokenizer {
    fn convert_ids_to_tokens(&self, token_ids: &[TokenIdType]) -> Result<Vec<String>> {
        Ok(token_ids
            .iter()
            .map(|&id| self.tokenizer.id_to_token(id).unwrap_or_default())
            .collect())
    }
}

impl From<HfTokenizer> for HuggingFaceTokenizer {
    fn from(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }
}

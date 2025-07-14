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

use crate::tokenizers::{
    traits::{Decoder, Encoder, Tokenizer},
    Encoding, Error, Result, TokenIdType,
};

use sentencepiece::SentencePieceProcessor;

/// A tokenizer implementation using the SentencePiece tokenization algorithm.
/// This tokenizer can encode text into tokens and decode tokens back into text.
pub struct SentencePieceTokenizer {
    /// The underlying SentencePiece processor instance
    spp: SentencePieceProcessor,
}

impl SentencePieceTokenizer {
    /// Creates a new SentencePieceTokenizer from a model file.
    ///
    /// # Arguments
    /// * `tokenizer_name` - Path to the SentencePiece model file
    ///
    /// # Returns
    /// * `Result<Self>` - A new tokenizer instance or an error if loading fails
    pub fn from_file(tokenizer_name: &str) -> Result<Self> {
        let spp = SentencePieceProcessor::open(tokenizer_name)
            .map_err(|err| Error::msg(format!("Error loading tokenizer: {}", err)))?;

        Ok(Self { spp })
    }
}

impl Encoder for SentencePieceTokenizer {
    /// Encodes a string input into tokens using the SentencePiece model.
    ///
    /// # Arguments
    /// * `input` - The text to encode
    ///
    /// # Returns
    /// * `Result<Encoding>` - The encoded tokens, including IDs, text, and character spans
    fn encode(&self, input: &str) -> Result<Encoding> {
        let encoding = self
            .spp
            .encode(input)
            .map_err(|err| Error::msg(format!("Error encoding input: {}", err)))?;

        let token_ids = encoding.into_iter().map(|piece| piece.id).collect();
        Ok(Encoding::Sp(token_ids))
    }

    /// Encodes multiple string inputs into tokens using the SentencePiece model.
    ///
    /// # Arguments
    /// * `inputs` - The texts to encode
    ///
    /// # Returns
    /// * `Result<Vec<Encoding>>` - The encoded tokens for each input
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for SentencePieceTokenizer {
    /// Decodes a sequence of token IDs back into text.
    ///
    /// # Arguments
    /// * `token_ids` - The sequence of token IDs to decode
    /// * `skip_special_tokens` - Currently unsupported in SentencePieceTokenizer and
    ///   it will return an error if true
    ///
    /// # Returns
    /// * `Result<String>` - The decoded text
    ///
    /// # Errors
    /// * Returns an error if skip_special_tokens is true
    /// * Returns an error if the decoding process fails
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        if skip_special_tokens {
            return Err(Error::msg(
                "SentencePieceTokenizer does not support skip_special_tokens=true.",
            ));
        }

        let text = self
            .spp
            .decode_piece_ids(token_ids)
            .map_err(|err| Error::msg(format!("Error decoding input: {}", err)))?;

        Ok(text)
    }
}

/// Implement the Tokenizer trait for SentencePieceTokenizer
impl Tokenizer for SentencePieceTokenizer {}

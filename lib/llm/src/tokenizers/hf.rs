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

use tokenizers::tokenizer::Tokenizer as HfTokenizer;

use super::{
    traits::{Decoder, Encoder, Tokenizer},
    Encoding, Error, Result, TokenIdType,
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
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // This calls into the library
        let text = self
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|err| Error::msg(format!("Error de-tokenizing input: {err}")))?;

        Ok(text)
    }
}

impl Tokenizer for HuggingFaceTokenizer {}

impl From<HfTokenizer> for HuggingFaceTokenizer {
    fn from(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }
}

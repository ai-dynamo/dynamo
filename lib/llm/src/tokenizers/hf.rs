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

    pub async fn from_repo_id(repo_id: &str, revision: Option<&str>) -> Result<Self> {
        use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};

        // Build the API client
        let api = ApiBuilder::new().with_progress(false).build()?;

        // Create the repository reference
        let repo = match revision {
            Some(rev) => Repo::with_revision(repo_id.to_string(), RepoType::Model, rev.to_string()),
            None => Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string()),
        };

        // Download the tokenizer.json file
        let repo_builder = api.repo(repo);
        let file_path = repo_builder
            .get("tokenizer.json")
            .await
            .map_err(|err| Error::msg(format!("Failed to download tokenizer.json: {}", err)))?;

        // Load the tokenizer from the downloaded file
        Self::from_file(
            file_path
                .to_str()
                .ok_or_else(|| Error::msg("Invalid path".to_string()))?,
        )
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|err| Error::msg(format!("Error encoding input: {}", err)))?;

        let token_ids = encoding.get_ids().to_vec();
        let tokens = encoding.get_tokens().to_vec();
        let spans = encoding.get_offsets().to_vec();

        Ok(Encoding {
            token_ids,
            tokens,
            spans,
        })
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|err| Error::msg(format!("Error decoding input: {}", err)))?;

        Ok(text)
    }
}

impl Tokenizer for HuggingFaceTokenizer {}

impl From<HfTokenizer> for HuggingFaceTokenizer {
    fn from(tokenizer: HfTokenizer) -> Self {
        HuggingFaceTokenizer { tokenizer }
    }
}

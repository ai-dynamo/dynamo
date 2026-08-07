// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use hf_tokenizers::tokenizer::pipeline::{Inputs, PipelineTokenizer};
use hf_tokenizers::{AddedToken, Tokenizer as SourceTokenizer};
use tokenizers::Tokenizer as LegacyTokenizer;

use crate::tokenizers::{
    Encoding, Error, HuggingFaceTokenizer, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

pub(crate) struct HfRuntimeTokenizer {
    encoder: PipelineTokenizer,
    decoder: HuggingFaceTokenizer,
    options: TokenizerOptions,
}

impl HfRuntimeTokenizer {
    pub(crate) fn from_file(path: &Path, legacy: LegacyTokenizer) -> Result<Self> {
        let mut source = SourceTokenizer::from_file(path)
            .map_err(|error| Error::msg(format!("Error loading tokenizer: {error}")))?;

        sync_special_tokens(&legacy, &mut source)?;

        if source.get_truncation().is_some() {
            source
                .with_truncation(None)
                .map_err(|error| Error::msg(format!("Error disabling truncation: {error}")))?;
        }
        if source.get_padding().is_some() {
            source.with_padding(None);
        }

        let encoder = PipelineTokenizer::try_from(&source)
            .map_err(|error| Error::msg(format!("Error initializing tokenizer: {error}")))?;

        Ok(Self {
            encoder,
            decoder: HuggingFaceTokenizer::from_tokenizer(legacy),
            options: TokenizerOptions::default(),
        })
    }

    fn encode_inputs(&self, inputs: impl Into<Inputs>) -> Result<Vec<Vec<u32>>> {
        self.encoder
            .encode(inputs, self.options.add_special_tokens)
            .wait()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|tokens| tokens.into_iter().map(|token| token.id).collect())
                    .collect()
            })
            .map_err(|error| Error::msg(format!("Error tokenizing input: {error}")))
    }
}

fn sync_special_tokens(legacy: &LegacyTokenizer, source: &mut SourceTokenizer) -> Result<()> {
    let mut special_tokens: Vec<_> = legacy
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| token.special)
        .collect();
    special_tokens.sort_by_key(|(id, _)| *id);

    let special_tokens = special_tokens.into_iter().map(|(_, token)| {
        AddedToken::from(token.content, true)
            .single_word(token.single_word)
            .lstrip(token.lstrip)
            .rstrip(token.rstrip)
            .normalized(token.normalized)
    });

    source
        .add_special_tokens(special_tokens)
        .map(|_| ())
        .map_err(|error| Error::msg(format!("Error synchronizing special tokens: {error}")))
}

impl Encoder for HfRuntimeTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let mut batch = self.encode_inputs(input)?;
        if batch.len() != 1 {
            return Err(Error::msg(format!(
                "Tokenizer returned {} encodings for one input",
                batch.len()
            )));
        }
        Ok(Encoding::Sp(batch.pop().expect("length checked above")))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let batch = self.encode_inputs(inputs)?;
        if batch.len() != inputs.len() {
            return Err(Error::msg(format!(
                "Tokenizer returned {} encodings for {} inputs",
                batch.len(),
                inputs.len()
            )));
        }
        Ok(batch.into_iter().map(Encoding::Sp).collect())
    }
}

impl Decoder for HfRuntimeTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.decoder.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for HfRuntimeTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        if self.options.add_special_tokens {
            return Err(Error::msg(
                "HuggingFace tokenizers configured with add_special_tokens=true must remain uncached",
            ));
        }
        Ok(())
    }

    fn with_options(mut self, options: TokenizerOptions) -> Self {
        self.options = options;
        self
    }
}

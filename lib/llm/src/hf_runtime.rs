// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, ensure};
use hf_tokenizers::{
    DecoderRuntime,
    pipeline::{EncodeOptions, Inputs, Override, PipelineModel, PipelineTokenizer},
};

use crate::tokenizers::{
    Encoding, Error, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// The legacy tokenizer is used only to prepare configuration at startup.
/// Serving owns a single RC pipeline for both encoding and decoding.
pub(crate) struct HfRuntimeTokenizer {
    pipeline: PipelineTokenizer,
    options: EncodeOptions,
    source: String,
}

impl HfRuntimeTokenizer {
    pub(crate) fn from_tokenizer(prepared: tokenizers::Tokenizer, source: String) -> Result<Self> {
        let json = prepared
            .to_string(false)
            .map_err(Error::msg)
            .with_context(|| format!("failed to serialize prepared tokenizer for {source}"))?;
        let json = hf_tokenizers::canonicalize_str(&json)
            .with_context(|| format!("failed to convert tokenizer to HF rc.2 for {source}"))?;
        let pipeline = hf_tokenizers::from_json(&json)
            .map_err(Error::msg)
            .with_context(|| format!("failed to construct HF rc.2 tokenizer for {source}"))?;

        // rc.2 can reassign added IDs above a sparse vocabulary, and uses the
        // normalized matching form as decoded text. Reject either change rather
        // than silently changing the IDs sent to the model or its generated text.
        for (id, token) in prepared.get_added_tokens_decoder() {
            ensure!(
                pipeline
                    .get_added_vocabulary()
                    .simple_id_to_token(id)
                    .as_deref()
                    == Some(token.content.as_str()),
                "unsupported added token {id} for {source}: HF rc.2 changes its ID or decoded form"
            );
        }

        // rc.2 decodes byte-level BPE directly from bytes, bypassing the decoder.
        // Only accept configurations for which that shortcut preserves semantics.
        if matches!(pipeline.get_model(), PipelineModel::BPE(bpe) if bpe.is_byte_level()) {
            let supported = match pipeline.get_decoder() {
                Some(DecoderRuntime::ByteLevel(_)) => true,
                Some(DecoderRuntime::Sequence(decoders)) => {
                    matches!(decoders.as_slice(), [DecoderRuntime::ByteLevel(_)])
                }
                _ => false,
            };
            ensure!(
                supported,
                "unsupported byte-level BPE decoder for {source}: HF rc.2 requires a direct ByteLevel decoder or a sequence containing only that decoder"
            );
        }

        Ok(Self {
            pipeline,
            options: EncodeOptions {
                add_special_tokens: false,
                padding: Override::Off,
                truncation: Override::Off,
                ..Default::default()
            },
            source,
        })
    }

    fn encode_inputs(&self, inputs: impl Into<Inputs>, expected: usize) -> Result<Vec<Encoding>> {
        let batch = self
            .pipeline
            .encode(inputs, &self.options)
            .wait()
            .map_err(Error::msg)
            .with_context(|| format!("HF rc.2 encode failed for {}", self.source))?;
        ensure!(
            batch.len() == expected,
            "HF rc.2 returned {} encodings for {expected} inputs for {}",
            batch.len(),
            self.source
        );
        Ok(batch
            .into_iter()
            .map(|encoding| Encoding::Sp(encoding.ids().iter().map(|token| token.id()).collect()))
            .collect())
    }
}

impl Encoder for HfRuntimeTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let mut batch = self.encode_inputs(input, 1)?;
        Ok(batch.pop().expect("cardinality checked above"))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.encode_inputs(inputs, inputs.len())
    }
}

impl Decoder for HfRuntimeTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.pipeline
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from_decoded)
            .map_err(Error::msg)
            .with_context(|| format!("HF rc.2 decode failed for {}", self.source))
    }
}

impl Tokenizer for HfRuntimeTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        ensure!(
            !self.options.add_special_tokens,
            "HF tokenizers configured with add_special_tokens=true must remain uncached"
        );
        Ok(())
    }

    fn with_options(mut self, options: TokenizerOptions) -> Self {
        self.options.add_special_tokens = options.add_special_tokens;
        self
    }
}

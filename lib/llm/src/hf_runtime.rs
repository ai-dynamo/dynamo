// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::{Context, ensure};
use hf_tokenizers::{
    DecoderRuntime,
    pipeline::{
        EncodeOptions, Encoding as RcEncoding, Inputs, Override, PipelineModel, PipelineToken,
        PipelineTokenizer,
    },
};

use crate::tokenizers::{
    EncodeSegment, Encoding, Error, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// The release that serves encoding and decoding, named in every error.
const RUNTIME: &str = "HF tokenizers 1.0.0-rc.4";

/// The legacy tokenizer is used only to prepare configuration at startup.
/// Serving owns a single RC pipeline for both encoding and decoding.
pub(crate) struct HfRuntimeTokenizer {
    pipeline: PipelineTokenizer,
    options: EncodeOptions,
    /// `<0xNN>` ids under a ByteFallback decoder. A trailing run of them can
    /// still decode differently once the next byte arrives.
    byte_fallback_ids: HashSet<TokenIdType>,
    special_ids: HashSet<TokenIdType>,
    source: String,
}

impl HfRuntimeTokenizer {
    pub(crate) fn from_tokenizer(prepared: tokenizers::Tokenizer, source: String) -> Result<Self> {
        let json = prepared
            .to_string(false)
            .map_err(Error::msg)
            .with_context(|| format!("failed to serialize prepared tokenizer for {source}"))?;
        let json = hf_tokenizers::canonicalize_str(&json)
            .with_context(|| format!("failed to convert tokenizer to {RUNTIME} for {source}"))?;
        let pipeline = hf_tokenizers::from_json(&json)
            .map_err(Error::msg)
            .with_context(|| format!("failed to construct {RUNTIME} tokenizer for {source}"))?;

        // Every prepared added token must keep its ID and spelling. The RC keys
        // a normalized added token by its normalized form: the original
        // spelling has no ID, and a special one is skipped where legacy keeps
        // its text. Reject that rather than silently changing the IDs sent to
        // the model or its text.
        for (id, token) in prepared.get_added_tokens_decoder() {
            ensure!(
                pipeline
                    .get_added_vocabulary()
                    .id_to_token_bytes(id)
                    .map(|(bytes, _)| bytes)
                    == Some(token.content.as_bytes()),
                "unsupported added token {id} for {source}: {RUNTIME} changes its ID or decoded form"
            );
        }

        // The RC decodes byte-level BPE directly from bytes, bypassing the decoder.
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
                "unsupported byte-level BPE decoder for {source}: {RUNTIME} requires a direct ByteLevel decoder or a sequence containing only that decoder"
            );
        }

        fn has_byte_fallback(decoder: &DecoderRuntime) -> bool {
            match decoder {
                DecoderRuntime::ByteFallback(_) => true,
                DecoderRuntime::Sequence(decoders) => decoders.iter().any(has_byte_fallback),
                _ => false,
            }
        }
        let byte_fallback_ids = if pipeline.get_decoder().is_some_and(has_byte_fallback) {
            pipeline
                .get_vocab(true)
                .into_iter()
                .filter_map(|(token, id)| {
                    (token.len() == 6
                        && token.starts_with("<0x")
                        && token.ends_with('>')
                        && u8::from_str_radix(&token[3..5], 16).is_ok())
                    .then_some(id)
                })
                .collect()
        } else {
            HashSet::new()
        };
        let special_ids = pipeline
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect();

        Ok(Self {
            pipeline,
            options: EncodeOptions {
                add_special_tokens: false,
                padding: Override::Off,
                truncation: Override::Off,
                ..Default::default()
            },
            byte_fallback_ids,
            special_ids,
            source,
        })
    }

    fn encode_with(
        &self,
        inputs: impl Into<Inputs>,
        expected: usize,
        options: &EncodeOptions,
    ) -> Result<Vec<RcEncoding>> {
        let batch = self
            .pipeline
            .encode(inputs, options)
            .wait()
            .map_err(Error::msg)
            .with_context(|| format!("{RUNTIME} encode failed for {}", self.source))?;
        ensure!(
            batch.len() == expected,
            "{RUNTIME} returned {} encodings for {expected} inputs for {}",
            batch.len(),
            self.source
        );
        Ok(batch)
    }

    fn encode_inputs(&self, inputs: impl Into<Inputs>, expected: usize) -> Result<Vec<Encoding>> {
        Ok(self
            .encode_with(inputs, expected, &self.options)?
            .iter()
            .map(|encoding| Encoding::Sp(PipelineToken::cast_slice(encoding.ids()).to_vec()))
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

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        // Post-processing applies once to the joined sequence, not at every
        // trust boundary. Untrusted segments encode special-token spellings as
        // ordinary text.
        let mut options = EncodeOptions {
            add_special_tokens: false,
            ..self.options.clone()
        };
        let mut ids = Vec::new();
        for segment in segments {
            options.encode_special_tokens = !segment.allow_special;
            let batch = self.encode_with(segment.text, 1, &options)?;
            ids.extend_from_slice(batch[0].ids());
        }
        if self.options.add_special_tokens {
            let processed = self
                .pipeline
                .get_post_processor()
                .single
                .post_process::<true>(ids, None);
            ids = processed.ids().to_vec();
        }
        Ok(Encoding::Sp(PipelineToken::cast_slice(&ids).to_vec()))
    }
}

impl Decoder for HfRuntimeTokenizer {
    fn has_unstable_suffix(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> bool {
        token_ids
            .iter()
            .rev()
            .find(|id| !skip_special_tokens || !self.special_ids.contains(id))
            .is_some_and(|id| self.byte_fallback_ids.contains(id))
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.pipeline
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from_decoded)
            .map_err(Error::msg)
            .with_context(|| format!("{RUNTIME} decode failed for {}", self.source))
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

    fn vocab_size(&self) -> Option<usize> {
        Some(self.pipeline.get_vocab_size(true))
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        Ok(self.pipeline.token_to_id(token))
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        let mut ids: Vec<_> = self.special_ids.iter().copied().collect();
        ids.sort_unstable();
        Ok(ids)
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(self.pipeline.get_post_processor().single.n_special())
    }
}

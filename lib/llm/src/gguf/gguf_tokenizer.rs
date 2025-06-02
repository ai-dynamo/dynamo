// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adapted from mistral.rs
//
// MIT License
//
// Copyright (c) 2025 Eric Buehler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// https://github.com/huggingface/transformers/blob/8685b3c5d2dd2550527773d2a02499495a759e31/src/transformers/convert_slow_tokenizer.py

use std::collections::HashMap;

use anyhow::Result;
use itertools::Itertools;
use tokenizers::{
    decoders::{
        self, byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, strip::Strip,
    },
    models::{bpe::BpeBuilder, unigram::Unigram},
    normalizers::{self, Prepend, Replace},
    pre_tokenizers,
    processors::{
        self,
        template::{self, TemplateProcessing},
    },
    AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

use crate::gguf::gguf_metadata::ContentMetadata;

use super::Content;

pub(crate) struct GgufTokenizerConversion {
    pub tokenizer: Tokenizer,
    // we most likely need those later
    pub _bos: Option<String>,
    pub _eos: Option<String>,
    pub _unk: Option<String>,
}

struct PropsGGUF {
    model: String,
    tokens: Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    eos: u32,
    bos: u32,
    add_bos_token: Option<bool>,
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self, Self::Error> {
        let required = ["model", "tokens", "eos_token_id", "bos_token_id"];
        c.has_required_keys(&required)?;

        let props = Self {
            model: c.get_value("model")?,
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id")?,
            bos: c.get_value("bos_token_id")?,
            add_bos_token: c.get_value("add_bos_token").ok(),
        };

        Ok(props)
    }
}

struct AddedTokensCollection {
    bos: String,
    eos: String,
    unk: Option<String>,
}

pub fn convert_gguf_to_hf_tokenizer(content: &Content) -> Result<GgufTokenizerConversion> {
    let metadata = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata: content.get_metadata(),
    };
    let props = PropsGGUF::try_from(metadata)?;

    let (tokenizer, kind, special_tokens) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        other => {
            anyhow::bail!("Tokenizer model `{other}` not supported.");
        }
    };

    info!(
        "GGUF tokenizer model is `{model}`, kind: `{kind:?}`, num tokens: {}, num added tokens: {}, num merges: {}, num scores: {}",
        tokenizer.get_vocab_size(true),
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
        model = props.model,
    );
    tracing::trace!("GGUF Tokenizer: {tokenizer:?}");

    let AddedTokensCollection { bos, eos, unk } = special_tokens;

    Ok(GgufTokenizerConversion {
        tokenizer,
        _bos: Some(bos),
        _eos: Some(eos),
        _unk: unk,
    })
}

// TODO: Add support for additional tokenizer models: WordPiece, WordLevel
// https://docs.rs/tokenizers/latest/tokenizers/models/enum.ModelWrapper.html
#[derive(Debug)]
enum TokenizerKind {
    Unigram,
    Bpe,
}

/// Add the special tokens and return their string representations
fn add_special_tokens(
    p: &PropsGGUF,
    tokenizer: &mut Tokenizer,
    bos: u32,
    eos: u32,
    unk: Option<u32>,
) -> AddedTokensCollection {
    // Add special tokens (bos, eos, unk):
    let mut special_tokens: [Option<String>; 3] = Default::default();

    // A little bit awkward here since eos/bos are assumed not options so we need to handle an Option
    for (i, token_id) in [Some(bos), Some(eos), unk].into_iter().enumerate() {
        if let Some(token_id) = token_id {
            let token = p.tokens[token_id as usize].as_str();
            special_tokens[i] = Some(token.to_string());
            tokenizer.add_special_tokens(&[AddedToken::from(token.to_string(), true)]);
        }
    }

    // Destructure array of options:
    let [bos_str, eos_str, unk_str] = special_tokens;
    // Would need to unwrap bos/eos here, or change the struct types
    AddedTokensCollection {
        bos: bos_str.unwrap(),
        eos: eos_str.unwrap(),
        unk: unk_str,
    }
}

fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind, AddedTokensCollection)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    // Unigram (SentencePiece) default UNK is 0
    let unk = unk.unwrap_or(0);

    // Create the Tokenizer model:
    let model = {
        let vocab: Vec<(String, f64)> = {
            let Some(s) = p.scores.as_ref() else {
                anyhow::bail!(
                    "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
                );
            };
            let scores = s.iter().cloned().map(|f_32| f_32 as f64);

            p.tokens.iter().cloned().zip(scores).collect()
        };

        Unigram::from(vocab, Some(unk as usize), true).map_err(anyhow::Error::msg)?
    };

    // Decoder + Normalizer config reference:
    // https://github.com/EricLBuehler/mistral.rs/pull/389#discussion_r1630620763
    let decoder = Decoder::Sequence(vec![
        Decoder::Replace("▁", " "),
        Decoder::ByteFallback,
        Decoder::Fuse,
        Decoder::Strip(' ', 1, 0),
    ]);

    let normalizer = Normalizer::Sequence(vec![
        Normalizer::Prepend("▁"),
        Normalizer::Replace(" ", "▁"),
    ]);

    let mut tokenizer: Tokenizer = TokenizerX::new(
        ModelWrapper::Unigram(model),
        Some(decoder),
        Some(normalizer),
    )?;

    // Add special tokens (bos, eos, unk):
    let special_tokens = add_special_tokens(p, &mut tokenizer, bos, eos, Some(unk));

    Ok((tokenizer, TokenizerKind::Unigram, special_tokens))
}

fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind, AddedTokensCollection)> {
    // BPE merges have each string item as a space-delimited pair:
    // https://github.com/EricLBuehler/mistral.rs/pull/397#discussion_r1631988370
    let merges = p
        .merges
        .as_ref()
        .ok_or(anyhow::Error::msg("BPE tokenizer must include merges"))?
        .iter()
        .map(|merge| {
            let split: (&str, &str) = merge
                .splitn(2, ' ')
                .collect_tuple()
                .expect("Failed to convert split into 2-tuple");
            (split.0.to_string(), split.1.to_string())
        })
        .collect::<Vec<_>>();

    let mut vocab = HashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(token.clone(), i as u32);
    }

    let PropsGGUF {
        eos,
        bos,
        unk,
        add_bos_token,
        ..
    } = *p;

    let mut bpe = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = unk {
        bpe = bpe.unk_token(p.tokens[unk as usize].to_string());
    };

    let bpe = bpe.build().map_err(anyhow::Error::msg)?;

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::BPE(bpe),
        Some(Decoder::ByteLevel(true, true, true)),
        None,
    )?;
    tokenizer.with_pre_tokenizer(Some(pre_tokenizers::byte_level::ByteLevel::new(
        false, true, true,
    )));
    if add_bos_token.is_some_and(|x| x) {
        let mut special_toks = HashMap::new();
        special_toks.insert(
            p.tokens[bos as usize].clone(),
            template::SpecialToken::new(
                p.tokens[bos as usize].clone(),
                vec![bos],
                vec![p.tokens[bos as usize].clone()],
            )
            .unwrap(),
        );
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single(format!("{}:0 $A:0", p.tokens[bos as usize]))
                .unwrap()
                .try_pair(format!("{}:0 $A:0 $B:1", p.tokens[bos as usize]))
                .unwrap()
                .special_tokens(special_toks)
                .build()
                .unwrap(),
        ));
    } else {
        tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
            true, false, true,
        )));
    }

    let special_tokens = add_special_tokens(p, &mut tokenizer, bos, eos, unk);

    Ok((tokenizer, TokenizerKind::Bpe, special_tokens))
}

// This is a workaround to have a better builder API.
// Upstream `TokenizerBuilder` is difficult to work with:
// https://github.com/huggingface/tokenizers/issues/1549
struct TokenizerX;

impl TokenizerX {
    #[allow(clippy::new_ret_no_self)]
    fn new<'a>(
        model: ModelWrapper,
        decoder: Option<Decoder<'a>>,
        normalizer: Option<Normalizer<'a>>,
    ) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::new(model);

        // Handle local enum to remote enum type:
        if let Some(decoder) = decoder {
            let d = DecoderWrapper::try_from(decoder)?;
            tokenizer.with_decoder(Some(d));
        }
        if let Some(normalizer) = normalizer {
            let n: NormalizerWrapper = NormalizerWrapper::try_from(normalizer)?;
            tokenizer.with_normalizer(Some(n));
        }

        Ok(tokenizer)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/decoders/enum.DecoderWrapper.html
enum Decoder<'a> {
    ByteFallback,
    Fuse,
    Replace(&'a str, &'a str),
    Strip(char, usize, usize),
    Sequence(Vec<Self>),
    ByteLevel(bool, bool, bool),
}

// Convert into upstream type wrapped enum variants:
impl TryFrom<Decoder<'_>> for DecoderWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Decoder) -> Result<Self, Self::Error> {
        let value: DecoderWrapper = match variant {
            Decoder::ByteFallback => ByteFallback::default().into(),
            Decoder::Fuse => Fuse::default().into(),
            Decoder::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Decoder::Strip(content, start, stop) => Strip::new(content, start, stop).into(),
            Decoder::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(DecoderWrapper::try_from)
                    .collect::<Result<Vec<DecoderWrapper>>>()?;

                decoders::sequence::Sequence::new(seq).into()
            }
            Decoder::ByteLevel(add_prefix_space, trim_offsets, use_regex) => {
                ByteLevel::new(add_prefix_space, trim_offsets, use_regex).into()
            }
        };

        Ok(value)
    }
}

// Convenient alternative to upstream:
// https://docs.rs/tokenizers/latest/tokenizers/normalizers/enum.NormalizerWrapper.html
enum Normalizer<'a> {
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> Result<Self, Self::Error> {
        let value: NormalizerWrapper = match variant {
            Normalizer::Prepend(prepend) => Prepend::new(prepend.to_owned()).into(),
            Normalizer::Replace(pattern, content) => Replace::new(pattern, content)
                .map_err(anyhow::Error::msg)?
                .into(),
            Normalizer::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(NormalizerWrapper::try_from)
                    .collect::<Result<Vec<NormalizerWrapper>>>()?;

                normalizers::Sequence::new(seq).into()
            }
        };

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use tokenizers::Tokenizer;

    #[allow(dead_code)]
    #[derive(Debug)]
    enum TokenizerType {
        /// Mistral v0.1 tokenizer
        Llama,
        Replit,
        Gpt2,
        Rwkv,
    }

    fn get_gguf_tokenizer(tokenizer: TokenizerType) -> Result<Tokenizer> {
        match tokenizer {
            TokenizerType::Llama => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let filename = api.get("llama_gguf_tokenizer.json").unwrap();
                let tokenizer = Tokenizer::from_file(filename).expect("Valid tokenizer");
                Ok(tokenizer)
            }
            TokenizerType::Gpt2 => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let filename = api.get("gpt2_gguf_tokenizer.json").unwrap();
                let tokenizer = Tokenizer::from_file(filename).expect("Valid tokenizer");
                Ok(tokenizer)
            }
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    fn get_hf_tokenizer(tokenizer: TokenizerType) -> Result<Tokenizer> {
        match tokenizer {
            TokenizerType::Llama => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let tokenizer_filename = api.get("tokenizer.json").unwrap();
                Ok(Tokenizer::from_file(tokenizer_filename).unwrap())
            }
            TokenizerType::Gpt2 => {
                let api = ApiBuilder::new().with_progress(true).build().unwrap();
                let api = api.repo(Repo::with_revision(
                    "EricB/mistralrs_tests".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let tokenizer_filename = api.get("tokenizer_gpt2.json").unwrap();
                Ok(Tokenizer::from_file(tokenizer_filename).unwrap())
            }
            other => anyhow::bail!("Cannot get testing HF tokenizer for type {other:?}"),
        }
    }

    // Content based upon https://github.com/ggerganov/llama.cpp/blob/master/tests/test-tokenizer-random.py#L99-L161
    fn get_test_passage() -> String {
        let passage = "Hello, world! \n🚀 (normal) 😶‍🌫️ (compound emoji, zwj sequence) ✅ (emoji as single token)\n你好世界！\nNǐ hǎo shìjiè!";

        passage.to_owned()
    }

    // The provided passage should encode and decode back into the same passage string:
    fn codec_roundtrip(
        tokenizer: &Tokenizer,
        passage: &str,
        add_special_tokens: bool,
    ) -> Result<String> {
        let tokenized = tokenizer
            .encode_fast(passage, add_special_tokens)
            .map_err(anyhow::Error::msg)?;

        // NOTE: The special tokens bool param meaning differs between encode() / decode():
        decode(tokenizer, tokenized.get_ids(), !add_special_tokens)
    }

    fn decode(
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(anyhow::Error::msg)
    }

    #[test]
    fn test_encode_decode_llama() -> Result<()> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Llama)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Llama)?;

        // Without adding special tokens
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), false)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), false)?;
        assert_eq!(hf_decoded, gguf_decoded);
        assert_eq!(passage, gguf_decoded);

        // With special tokens added
        // SKIPPED:
        // - Bugged the GGUF tokenizer does not prepend `<s> `
        // - Due to HF tokenizer using BPE (tokenizer.json) while GGUF tokenizer uses Unigram (metadata)?
        /*
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), true)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), true)?;
        assert_eq!(hf_decoded, gguf_decoded);
        */

        #[allow(clippy::cast_possible_truncation)]
        let mut tokens = (0..hf_tokenizer.get_vocab_size(false) as u32).collect::<Vec<_>>();
        tokens.shuffle(&mut rng());

        // Without skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, false)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, false)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, true)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, true)?;
        assert_eq!(hf_decoded, gguf_decoded);

        Ok(())
    }

    #[test]
    fn test_encode_decode_gpt2() -> Result<()> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let passage = get_test_passage();
        let hf_tokenizer = get_hf_tokenizer(TokenizerType::Gpt2)?;
        let gguf_tokenizer = get_gguf_tokenizer(TokenizerType::Gpt2)?;

        // Without adding special tokens
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), false)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), false)?;
        assert_eq!(hf_decoded, gguf_decoded);
        assert_eq!(passage, gguf_decoded);

        // With special tokens added
        // SKIPPED:
        // - Bugged the GGUF tokenizer does not prepend `<s> `
        // - Due to HF tokenizer using BPE (tokenizer.json) while GGUF tokenizer uses Unigram (metadata)?
        /*
        let hf_decoded = codec_roundtrip(&hf_tokenizer, passage.as_str(), true)?;
        let gguf_decoded = codec_roundtrip(&gguf_tokenizer, passage.as_str(), true)?;
        assert_eq!(hf_decoded, gguf_decoded);
        */

        #[allow(clippy::cast_possible_truncation)]
        let mut tokens = (0..hf_tokenizer.get_vocab_size(false) as u32).collect::<Vec<_>>();
        tokens.shuffle(&mut rng());

        // Without skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, false)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, false)?;
        assert_eq!(hf_decoded, gguf_decoded);

        // With skipping special tokens
        let hf_decoded = decode(&hf_tokenizer, &tokens, true)?;
        let gguf_decoded = decode(&gguf_tokenizer, &tokens, true)?;
        assert_eq!(hf_decoded, gguf_decoded);

        Ok(())
    }
}

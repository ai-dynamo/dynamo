// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod hf;

// TODO: Add tokenizer benchmarks
// TODO: Enable README.md as a module doc
// #[doc = include_str!("../README.md")]

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::{ops::Deref, path::Path};

use crate::protocols::TokenIdType;
pub use anyhow::{Error, Result};

pub use hf::HuggingFaceTokenizer;

/// Represents the type of tokenizer being used
#[derive(Debug)]
pub enum TokenizerType {
    HuggingFace(String),
}

/// character offsets in the original text
pub type Offsets = (usize, usize);

/// Contains the results of tokenizing text: token IDs, string tokens, and their spans
#[derive(Debug, Clone)]
pub enum Encoding {
    /// Hugging Face
    Hf(Box<tokenizers::tokenizer::Encoding>),
    /// Sentence Piece
    Sp(Vec<TokenIdType>),
}

impl Encoding {
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Encoding::Hf(inner) => inner.get_ids(),
            Encoding::Sp(inner) => inner,
        }
    }
}

impl Hash for Encoding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.token_ids().hash(state);
    }
}

pub mod traits {
    use super::*;

    pub trait Encoder: Send + Sync {
        fn encode(&self, input: &str) -> Result<Encoding>;
        fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>>;
    }

    pub trait Decoder: Send + Sync {
        fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String>;
    }

    pub trait Tokenizer: Encoder + Decoder {
        // fn get_vocab_size(&self) -> usize;
        // fn make_unique_clone(&self) -> Box<dyn Tokenizer>;
    }
}

impl Encoding {
    pub fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Main tokenizer wrapper that provides a unified interface for different tokenizer implementations
#[derive(Clone)]
pub struct Tokenizer(Arc<dyn traits::Tokenizer>);

impl Tokenizer {
    pub fn from_file(file_path: &str) -> Result<Tokenizer> {
        Ok(Tokenizer(create_tokenizer_from_file(file_path)?))
    }

    /// Create a stateful sequence object for decoding token_ids into text
    pub fn decode_stream(
        &self,
        prompt_token_ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> DecodeStream {
        DecodeStream::new(self.0.clone(), prompt_token_ids, skip_special_tokens)
    }
}

impl Deref for Tokenizer {
    type Target = Arc<dyn traits::Tokenizer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Arc<dyn traits::Tokenizer>> for Tokenizer {
    fn from(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        Tokenizer(tokenizer)
    }
}

impl<T> From<Arc<T>> for Tokenizer
where
    T: traits::Tokenizer + 'static, // 'static is required to ensure T can be safely put into an Arc
{
    fn from(tokenizer: Arc<T>) -> Self {
        Tokenizer(tokenizer)
    }
}

/// Create a tokenizer from a file path to a tokenizer file.
/// The file extension is used to determine the tokenizer type.
/// Supported file types are:
/// - json: HuggingFace tokenizer
pub fn create_tokenizer_from_file(file_path: &str) -> Result<Arc<dyn traits::Tokenizer>> {
    let path = Path::new(file_path);
    let extension = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| Error::msg("Failed to read file extension".to_string()))?;

    match extension {
        "json" => {
            let tokenizer = HuggingFaceTokenizer::from_file(file_path)?;
            Ok(Arc::new(tokenizer))
        }
        _ => Err(Error::msg("Unsupported file type".to_string())),
    }
}

// With incremental detokenization, we need to consider the final context tokens when handling the initial decode tokens.
// This is the initial offset from the end of the context that we start decoding from.
// Both Huggingface TGI and vLLM use this same value.
// See: https://github.com/huggingface/text-generation-inference/blob/24c2bff65924801ddf90fa24fcc72752d4f45538/server/text_generation_server/models/mamba.py#L169
// and https://github.com/vllm-project/vllm/blob/da2705198fa19030a25d0bea437f7be6547d47d4/vllm/transformers_utils/detokenizer_utils.py#L51
const INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 5;

/// DecodeStream will keep the state necessary to produce individual chunks of
/// strings given an input stream of token_ids.
///
/// This is necessary because decoding in general cannot achieve that since strings
/// depend on surrounding ids to provide a valid string. Typically stripping extra spaces.
pub struct DecodeStream {
    /// The tokenizer used to decode token_ids
    tokenizer: Arc<dyn traits::Tokenizer>,

    skip_special_tokens: bool,
    /// A temporary buffer of the necessary token_ids needed
    /// to produce valid string chunks.
    /// This typically contains 3 parts:
    ///  - read
    ///  - prefix
    ///  - rest
    ///
    /// Read is the bit necessary to surround the prefix
    /// so decoding the whole ids produces a valid prefix.
    /// Prefix is the previously produced string, kept around to trim off of
    /// the next valid chunk
    all_token_ids: Vec<u32>,

    prefix_offset: usize,

    read_offset: usize,
}

impl DecodeStream {
    pub fn new(
        tokenizer: Arc<dyn traits::Tokenizer>,
        prompt_token_ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> Self {
        let num_input_tokens = prompt_token_ids.len();
        let prompt_token_ids = prompt_token_ids.to_vec();
        Self {
            tokenizer,
            skip_special_tokens,
            all_token_ids: prompt_token_ids,
            prefix_offset: num_input_tokens
                .saturating_sub(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET),
            read_offset: num_input_tokens,
        }
    }

    /// Step appends a token_id to the internal state and tries to produce a text chunk.
    ///
    /// Implementation directly copied from Huggingface's TGI:
    /// https://github.com/huggingface/text-generation-inference/blob/24c2bff65924801ddf90fa24fcc72752d4f45538/server/text_generation_server/models/model.py#L144
    ///
    /// Returning `None` means the given id is not enough to produce a chunk.
    /// This typically happens with `byte_fallback` options where some tokens do not
    /// represent valid UTF-8, and only follow-up token_ids will help produce
    /// a valid chunk.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.all_token_ids.push(id);

        let prefix_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..self.read_offset],
            self.skip_special_tokens,
        )?;

        let new_text = self.tokenizer.decode(
            &self.all_token_ids[self.prefix_offset..],
            self.skip_special_tokens,
        )?;

        if new_text.len() > prefix_text.len() && !new_text.ends_with("�") {
            let new_text = new_text[prefix_text.len()..].to_string();

            self.prefix_offset = self.read_offset;
            self.read_offset = self.all_token_ids.len();

            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

/// Maintains state for an ongoing sequence of tokens and their decoded text
pub struct Sequence {
    /// Encodes text -> token_ids
    tokenizer: Tokenizer,

    /// The current sequence of token ids
    token_ids: Vec<TokenIdType>,

    /// The position in the current sequence the last decoded token completed
    prefix_offset: usize,

    /// Current position in the sequence
    read_offset: usize,
}

impl std::fmt::Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("tokenizer", &"Arc<dyn Tokenizer>")
            .field(
                "token_ids",
                &format_args!("{}", {
                    let token_ids = self.token_ids();
                    if token_ids.len() <= 20 {
                        format!("{:?}", token_ids)
                    } else {
                        let first_ten = &token_ids[..10];
                        let last_ten = &token_ids[token_ids.len() - 10..];
                        format!("{:?} ... {:?}", first_ten, last_ten)
                    }
                }),
            )
            .field("prefix_offset", &self.prefix_offset)
            .field("read_offset", &self.read_offset)
            .field("token count", &self.token_ids.len())
            .finish()
    }
}

impl Sequence {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.prefix_offset = 0;
        self.read_offset = 0;
    }

    pub fn append_text(&mut self, input: &str) -> Result<()> {
        // let tokenizer = self.tokenizer.read().map_err(|err| {
        //     Error::msg(format!("Failed to acquire read lock on tokenizer: {}", err))
        // })?;

        let encoding = self.tokenizer.encode(input)?;
        self.token_ids.extend(encoding.token_ids());
        Ok(())
    }

    // Based on
    // https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
    // under Apache 2.0 license
    pub fn append_token_id(&mut self, token_id: TokenIdType) -> Result<String> {
        self.token_ids.push(token_id);
        // log::trace!("pushed token_id: {}", token_id);

        let prefix_text = self
            .tokenizer
            .decode(&self.token_ids[self.prefix_offset..self.read_offset], false)?;

        let new_text = self
            .tokenizer
            .decode(&self.token_ids[self.prefix_offset..], false)?;

        // if the end character of the previous returned sequence is a multi-byte character
        // then we can not split the text on that byte offset, so we roll back to the byte offset
        // of the start of that character
        let mut prefix_text_len = prefix_text.len();
        while !new_text.is_char_boundary(prefix_text_len) && prefix_text_len > 0 {
            prefix_text_len -= 1;
        }
        let prefix_text_len = prefix_text_len;

        if new_text.len() > prefix_text.len() {
            if new_text.ends_with("�") {
                return Ok("".to_string());
            } else {
                // shift and update the state
                let new_text = new_text[prefix_text_len..].to_string().replace("�", "");
                self.prefix_offset = self.read_offset;
                self.read_offset = self.token_ids.len();
                return Ok(new_text);
            }
        }

        Ok("".to_string())
    }

    pub fn tokenizer(&self) -> Tokenizer {
        self.tokenizer.clone()
    }

    pub fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }

    pub fn text(&self) -> Result<String> {
        // let tokenizer = self.tokenizer.read().map_err(|err| {
        //     Error::msg(format!("Failed to acquire read lock on tokenizer: {}", err))
        // })?;
        self.tokenizer.decode(&self.token_ids, false)
    }
}

/// The output conditions/values of a SequenceDecoder::add_token_id operation.
/// Result of decoding a token, indicating whether text was produced or a stop condition was met
pub enum SequenceDecoderOutput {
    /// The text for the appended token_id
    Text(String),

    /// A sequence of token_ids has been partially matched a stop sequence, so the text is held
    /// until either a match or a divergence
    Held,

    /// Indicates that a stop sequence has been matched and the decoder is stopped.
    /// Subsequent calls to append_token_id will return an error
    Stopped,

    /// Indicates that a stop token_id has been matched and the decoder is stopped.
    /// Subsequent calls to append_token_id will return an error
    /// The text for the stop token_id is returned
    StoppedWithText(String),
}

/// A Sequence for decoding a stream of token ids into text and detecting stop sequences.
/// A stop sequence is either a matching token_id or a sequence of texts/strings which match.
/// Matches happen first at the token-level, then at the sequence-level. Hidden takes precedence
/// over visible. For example, if you put the same token_id in both `stop_token_ids_visible` and
/// `stop_token_ids_hidden`, the token_id will be treated as hidden.
#[derive(Debug)]
pub struct StopSequenceDecoder {
    // The current sequence of token ids
    sequence: Sequence,

    // Stop Tokens - the presence of any one of these should trigger a stop
    // If found, the text for the matched token will be returned
    stop_token_ids_visible: Vec<TokenIdType>,

    // Stop Tokens - the presence of any one of these should trigger a stop
    // If found, the text for the matched token will NOT be returned
    stop_token_ids_hidden: Vec<TokenIdType>,

    // Stop Words - the presence of any one of these should trigger a stop
    // If found, the text for the matched token will be returned
    #[allow(dead_code)]
    stop_sequences_visible: Vec<String>,

    // Stop Words - the presence of any one of these should trigger a stop
    // If found, the text for the matched token will NOT be returned
    stop_sequences_hidden: Vec<String>,

    // If the decoder has observed and returned a stop SequenceDecoderOutput,
    // futhur calls to append_token_id will return an error
    stopped: bool,

    // text jail - if a partial stop sequence is being observed, we hold/jail the text
    // until either the stop sequence is matched or the sequence is reset by a divergence
    state: String,
}

impl StopSequenceDecoder {
    /// Builder object for configurating a StopSequenceDecoder
    pub fn builder(tokenizer: Tokenizer) -> StopSequenceDecoderBuilder {
        StopSequenceDecoderBuilder::new(tokenizer)
    }

    /// Add a token_id to the sequence and return the SequenceDecoderOutput
    pub fn append_token_id(&mut self, token_id: TokenIdType) -> Result<SequenceDecoderOutput> {
        if self.stopped {
            return Err(Error::msg("Decoder is stopped"));
        }

        // update the sequence
        let text = self.sequence.append_token_id(token_id)?;

        // append the text to the state
        self.state.push_str(text.as_str());

        let mut stop: bool = false;
        let mut visible: bool = false;

        if self.stop_token_ids_visible.contains(&token_id) {
            stop = true;
            visible = true;
        }

        if self.stop_token_ids_hidden.contains(&token_id) {
            stop = true;
            visible = false;
        }

        if stop {
            self.stopped = true;
            let state = std::mem::take(&mut self.state);
            if visible {
                return Ok(SequenceDecoderOutput::StoppedWithText(state));
            }
            return Ok(SequenceDecoderOutput::Stopped);
        }

        // determine if state matches any of the stop sequences
        for stop_sequence in self.stop_sequences_hidden.iter() {
            if stop_sequence.starts_with(&self.state) {
                if stop_sequence == &self.state {
                    // on matched stop sequence, we do NOT return the jailed stop sequence
                    self.stopped = true;
                    return Ok(SequenceDecoderOutput::Stopped);
                } else {
                    return Ok(SequenceDecoderOutput::Held);
                }
            }
        }

        let state = std::mem::take(&mut self.state);
        Ok(SequenceDecoderOutput::Text(state))
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.token_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sequence.token_ids.len()
    }

    pub fn is_complete(&self) -> bool {
        self.stopped
    }

    pub fn close(&mut self) {
        self.stopped = true;
    }
}

pub struct StopSequenceDecoderBuilder {
    tokenizer: Tokenizer,
    stop_token_ids_visible: Vec<TokenIdType>,
    stop_token_ids_hidden: Vec<TokenIdType>,
    stop_sequences_visible: Vec<String>,
    stop_sequences_hidden: Vec<String>,
}

impl StopSequenceDecoderBuilder {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            stop_token_ids_visible: Vec::new(),
            stop_token_ids_hidden: Vec::new(),
            stop_sequences_visible: Vec::new(),
            stop_sequences_hidden: Vec::new(),
        }
    }

    /// Adds a visible stop token id to the StopSequenceDecoder
    pub fn add_stop_token_id_visible(mut self, token_id: TokenIdType) -> Self {
        self.stop_token_ids_visible.push(token_id);
        self
    }

    /// Adds a list of visible stop token ids to the StopSequenceDecoder
    /// Each token_id is added as for an individual match
    pub fn add_stop_token_ids_visible(mut self, token_ids: &[TokenIdType]) -> Self {
        self.stop_token_ids_visible.extend(token_ids);
        self
    }

    /// Adds a hidden stop token id to the StopSequenceDecoder
    pub fn add_stop_token_id_hidden(mut self, token_id: TokenIdType) -> Self {
        self.stop_token_ids_hidden.push(token_id);
        self
    }

    /// Adds a list of hidden stop token ids to the StopSequenceDecoder
    /// Each token_id is added as for an individual match
    pub fn add_stop_token_ids_hidden(mut self, token_ids: &[TokenIdType]) -> Self {
        self.stop_token_ids_hidden.extend(token_ids);
        self
    }

    pub fn add_stop_sequence_visible(mut self, text: &str) -> Self {
        self.stop_sequences_visible.push(text.to_string());
        self
    }

    pub fn add_stop_sequences_visible(mut self, strings: &[&str]) -> Self {
        self.stop_sequences_visible
            .extend(strings.iter().map(|text| text.to_string()));
        self
    }

    pub fn add_stop_sequence_hidden(mut self, text: &str) -> Self {
        self.stop_sequences_hidden.push(text.to_string());
        self
    }

    pub fn add_stop_sequences_hidden(mut self, strings: &[&str]) -> Self {
        self.stop_sequences_hidden
            .extend(strings.iter().map(|text| text.to_string()));
        self
    }

    pub fn build(self) -> Result<StopSequenceDecoder> {
        Ok(StopSequenceDecoder {
            sequence: Sequence::new(self.tokenizer.clone()),
            stop_token_ids_visible: self.stop_token_ids_visible,
            stop_token_ids_hidden: self.stop_token_ids_hidden,
            stop_sequences_visible: self.stop_sequences_visible,
            stop_sequences_hidden: self.stop_sequences_hidden,
            stopped: false,
            state: String::new(),
        })
    }
}

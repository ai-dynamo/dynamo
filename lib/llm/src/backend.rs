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

//! Backend
//!
//! An [`Backend`] is the final stage of the pipeline. It represents the execution of the LLM
//! on some processing hardware.
//!
//! At minimum, the Backend is split into two components, the [`Backend`] itself and a downstream [`ExecutionContext`].
//!
//! The [`ExecutionContext`] can be thought of as the core driver of the forward pass, whereas the [`Backend`] is the
//! manager of all resources and concurrent tasks surrounding the LLM execution context / forward pass.
//!
//! For almost every known scenario, detokenization and initial post processing must happen in the Backend.
//! Further post-processing can happen in the response stream. One example is the jailing mechanism for partial
//! hidden stop condition matches, which can be handled in the response stream rather than the backend.

use std::{collections::HashSet, sync::Arc};

use anyhow::{Error, Result};
use futures::stream::{self, StreamExt};
use tracing as log;

use crate::model_card::model::{ModelDeploymentCard, TokenizerKind};
use dynamo_runtime::{
    pipeline::{
        async_trait, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream,
        ServerStreamingEngine, SingleIn,
    },
    protocols::annotated::Annotated,
};

use crate::protocols::{
    common::{
        llm_backend::{BackendInput, BackendOutput, FinishReason, LLMEngineOutput},
        StopConditions,
    },
    TokenIdType,
};
use crate::tokenizers::{DecodeStream, HuggingFaceTokenizer, Tokenizer};
use tokenizers::Tokenizer as HfTokenizer;

/// Represents the output stream from the execution engine
pub type ExecutionOutputStream = Annotated<LLMEngineOutput>;

/// Context for executing LLM inference, engine consumes backend input and produces execution output stream
pub type ExecutionContext = ServerStreamingEngine<BackendInput, ExecutionOutputStream>;

/// Backend handles resource management and orchestrates LLM execution
#[allow(dead_code)]
pub struct Backend {
    pub tokenizer: Option<Tokenizer>, // Handles token encoding/decoding
    validate_engine_decode: bool,     // Enable validation of engine decoding
}

/// Internal state for managing token decoding and stream processing
#[allow(dead_code)]
struct DecoderUnfoldState {
    stream: ManyOut<ExecutionOutputStream>,
    decoder: Decoder,
    validate_engine_decode: bool,
}

impl Backend {
    pub async fn from_tokenizer(tokenizer: HfTokenizer) -> Result<Arc<Self>> {
        let tokenizer = HuggingFaceTokenizer::from_tokenizer(tokenizer);
        let tokenizer = Tokenizer::from(Arc::new(tokenizer));

        Ok(Arc::new(Self {
            tokenizer: Some(tokenizer),
            validate_engine_decode: false,
        }))
    }

    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let tokenizer = match &mdc.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => {
                HfTokenizer::from_file(file).map_err(Error::msg)?
            }
            Some(TokenizerKind::GGUF(t)) => *t.clone(),
            None => {
                return Ok(Arc::new(Self {
                    tokenizer: None,
                    validate_engine_decode: false,
                }));
            }
        };
        Self::from_tokenizer(tokenizer).await
    }

    fn decoder(
        &self,
        stream: ManyOut<ExecutionOutputStream>,
        stop_conditions: StopConditions,
    ) -> anyhow::Result<DecoderUnfoldState> {
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            anyhow::bail!("Backend built from blank ModelDeploymentCard, no tokenizer");
        };
        let decoder = Decoder::new(tokenizer.decode_stream(false), stop_conditions);

        Ok(DecoderUnfoldState {
            stream,
            decoder,
            validate_engine_decode: self.validate_engine_decode,
        })
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<BackendInput>,
        ManyOut<Annotated<BackendOutput>>,
        SingleIn<BackendInput>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Backend
{
    async fn generate(
        &self,
        request: SingleIn<BackendInput>,
        next: ServerStreamingEngine<BackendInput, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>> {
        let stop_conditions = request.stop_conditions.clone();
        let next_stream = next.generate(request).await?;

        let context = next_stream.context();
        let state = self.decoder(next_stream, stop_conditions)?;

        let processed_stream = stream::unfold(state, |mut state| async move {
            match state.stream.next().await {
                Some(output) => {
                    // move to state.process_output
                    // handle any error conditions / unwraps here

                    // events are pass thru
                    if output.is_event() || output.data.is_none() {
                        return Some((output, state));
                    }

                    // if we have a data field without an event, then we might need to update the data
                    if let Some(data) = &output.data {
                        if data.text.is_some() && !state.validate_engine_decode {
                            return Some((output, state));
                        }
                    }

                    let data = output.data.as_ref().unwrap();

                    let result = state.decoder.process_token_ids(&data.token_ids).unwrap();

                    // todo - propagate finish reason details - possibly an annotation
                    let finish_reason = match &result.stop_trigger {
                        Some(StopTrigger::MaxTokensLimit) => Some(FinishReason::Length),
                        Some(StopTrigger::HiddenStopTokenDetected(_)) => Some(FinishReason::Stop),
                        Some(StopTrigger::HiddenStopSequenceDetected(_)) => {
                            Some(FinishReason::Stop)
                        }
                        None => None,
                    };

                    if data.finish_reason.is_none() && finish_reason.is_some() {
                        tracing::debug!(
                            ?result.stop_trigger,
                            "upstream did not provide a finish reason; issuing a stop_generation request to free resources",
                        );
                        state.stream.context().stop_generating();
                    }

                    let text = result.text;
                    let tokens = result.tokens;

                    if state.validate_engine_decode {
                        if data.finish_reason != finish_reason {
                            log::warn!(
                                "finish reason mismatch: expected {:?}, got {:?}",
                                data.finish_reason,
                                finish_reason
                            );
                        }

                        if data.text.is_some() && data.text != text {
                            log::warn!("text mismatch: expected {:?}, got {:?}", data.text, text);
                        }
                    }

                    // update output in-place
                    let mut output = output;
                    let mut data = output.data.take().unwrap();

                    data.finish_reason = finish_reason;
                    data.text = text;
                    data.tokens = Some(tokens);

                    output.data = Some(data);

                    Some((output, state))
                }

                None => None,
            }
        });

        // convert stream of processed Annotated<LLMEngineOutput> to Annotated<BackendOutput>
        //let mdcsum = self.mdcsum.clone();
        let stream = processed_stream.map(move |output| {
            output.map_data(|data| {
                Ok(BackendOutput {
                    token_ids: data.token_ids,
                    tokens: data.tokens.unwrap_or_default(),
                    text: data.text,
                    cum_log_probs: data.cum_log_probs,
                    log_probs: data.log_probs,
                    finish_reason: data.finish_reason,
                    //mdcsum: mdcsum.clone(),
                })
            })
        });

        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

// todo - add visible stop conditions
// visible_stop_ids: HashSet<TokenIdType>,
// visible_stop_sequences: Vec<String>,

/// The [`Decoder`] object could be a member of either the internal LLM engine or part of the
/// postprocessor. If in the postprocessor, should be minimally in the same process or at very minimum
/// on the same physical machine connected by an IPC.
#[allow(dead_code)]
pub struct Decoder {
    decode_stream: DecodeStream,

    // do not trigger stop conditions until at least this many tokens have been generated
    min_tokens: u32,

    // single tokens that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated
    hidden_stop_ids: HashSet<TokenIdType>,

    // text sequences that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated
    hidden_stop_sequences: Vec<String>,

    // number of generated tokens
    generated_tokens: u32,

    // content jailed by partial hidden stop matches
    jail: String,

    // maximum number of bytes for the largest stop sequence
    jail_max_bytes: usize,

    // the number of bytes currently jailed
    jailed_bytes: usize,
    // mdcsum
    //mdcsum: String,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum StopTrigger {
    MaxTokensLimit,
    HiddenStopTokenDetected(TokenIdType),
    HiddenStopSequenceDetected(String),
}

impl StopTrigger {
    pub fn should_hide_text(&self) -> bool {
        match self {
            StopTrigger::MaxTokensLimit => false,
            StopTrigger::HiddenStopTokenDetected(_) => true,
            StopTrigger::HiddenStopSequenceDetected(_) => true,
        }
    }
}

pub struct StepResult {
    pub token: Option<String>,
    pub stop_trigger: Option<StopTrigger>,
}

impl StepResult {
    fn ok(token: Option<String>) -> Self {
        Self {
            token,
            stop_trigger: None,
        }
    }

    fn with_stop_trigger(token: Option<String>, stop_trigger: StopTrigger) -> Self {
        Self {
            token,
            stop_trigger: Some(stop_trigger),
        }
    }
}

/// Result of processing a sequence of tokens
pub struct SeqResult {
    pub tokens: Vec<Option<String>>,       // Individual decoded tokens
    pub text: Option<String>,              // Combined decoded text
    pub stop_trigger: Option<StopTrigger>, // Reason for stopping generation, if any
}

#[allow(dead_code)]
impl Decoder {
    pub fn new(
        decode_stream: DecodeStream,
        stop_condition: StopConditions,
        //mdcsum: String,
    ) -> Self {
        let hidden_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_hidden
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();

        let hidden_stop_sequences: Vec<String> = stop_condition
            .stop
            .unwrap_or_default()
            .iter()
            .map(|x| x.to_string())
            .collect();

        let jail_max_bytes = hidden_stop_sequences
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap_or(0);

        Self {
            decode_stream,
            hidden_stop_ids,
            hidden_stop_sequences,
            //visible_stop_ids: HashSet::new(),
            //visible_stop_sequences: Vec::new(),
            min_tokens: stop_condition.min_tokens.unwrap_or(0),
            generated_tokens: 0,
            jail: String::new(),
            jail_max_bytes,
            jailed_bytes: 0,
        }
    }

    /// Minimum amount of work to determine if a given generated/decoded sequence should be stopped
    /// This method can be called by the inner most loop of the LLM engine or minimally in the same
    /// process as the LLM engine.
    ///
    /// In the future, this method may kick off async cpu/tokio tasks and or async cuda tasks to
    /// handle logits post-processing and/or other tasks.
    pub fn step(&mut self, token_id: TokenIdType) -> Result<StepResult> {
        // increment the generated tokens
        self.generated_tokens += 1;

        // decode the token
        let token = self.decode_stream.step(token_id)?;

        // stop conditions to not apply until the minimum number of tokens have been generated
        if self.generated_tokens < self.min_tokens {
            return Ok(StepResult::ok(token));
        }

        // check for hidden stop tokens - eos takes precedence
        if self.hidden_stop_ids.contains(&token_id) {
            return Ok(StepResult::with_stop_trigger(
                token,
                StopTrigger::HiddenStopTokenDetected(token_id),
            ));
        }

        // check stop sequences - the jail will always hold at least the largest stop sequence
        // if jail_max_bytes is 0, then there are no stop sequences
        if self.jail_max_bytes > 0 {
            if let Some(token) = &token {
                let pre_append = self.jail.len();
                log::debug!("pre_append: {}", pre_append);
                log::debug!("jail: {}", self.jail);
                self.jail.push_str(token);
                log::debug!("post_append: {}", self.jail.len());
                log::debug!("jail: {}", self.jail);

                for seq in &self.hidden_stop_sequences {
                    log::debug!("stop seq: {}", seq);
                    if let Some(offset) =
                        galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                    {
                        log::debug!("offset: {}", offset);
                        // return only new bytes after pre_append .. offset+seq.len()
                        // example: seq = "ox", token = "boxes", return "b"
                        // note: this changes when we start jailing tokens for partial matches
                        // on the suffix of the jail with prefixes of the stop sequences
                        //
                        // we might have returned a partial match, if so, then offset < pre_append
                        // in that case, we return the empty string
                        let partial_token = if offset >= pre_append {
                            self.jail[pre_append..offset].to_string()
                        } else {
                            "".to_string()
                        };
                        return Ok(StepResult::with_stop_trigger(
                            Some(partial_token),
                            StopTrigger::HiddenStopSequenceDetected(seq.to_string()),
                        ));
                    }
                }

                if self.jail.len() > self.jail_max_bytes {
                    // truncate the jail
                    let drain_len = self.jail.len() - self.jail_max_bytes;
                    self.jail.drain(0..drain_len);
                }
            }
        }

        Ok(StepResult::ok(token))
    }

    pub fn process_token_ids(&mut self, token_ids: &[TokenIdType]) -> Result<SeqResult> {
        let mut text: Option<String> = None;
        let mut tokens = Vec::new();

        for token_id in token_ids {
            let StepResult {
                token,
                stop_trigger,
            } = self.step(*token_id)?;

            let hide_text = stop_trigger
                .as_ref()
                .map(|x| x.should_hide_text())
                .unwrap_or(false);

            if !hide_text {
                if let Some(token) = &token {
                    text.get_or_insert_with(String::new).push_str(token);
                }
            }
            tokens.push(token);

            if let Some(stop_trigger) = stop_trigger {
                return Ok(SeqResult {
                    tokens,
                    text,
                    stop_trigger: Some(stop_trigger),
                });
            }
        }

        Ok(SeqResult {
            tokens,
            text,
            stop_trigger: None,
        })
    }

    fn return_token(&self, token: Option<String>) -> StepResult {
        StepResult {
            token,
            stop_trigger: None,
        }
    }

    fn return_with_stop_trigger(
        &self,
        token: Option<String>,
        stop_trigger: StopTrigger,
    ) -> StepResult {
        StepResult {
            token,
            stop_trigger: Some(stop_trigger),
        }
    }

    fn jailed_string(&self) -> Option<String> {
        if self.jailed_bytes > 0 {
            // get the last jailed_bytes from the jail
            Some(self.jail[self.jail.len() - self.jailed_bytes..].to_string())
        } else {
            None
        }
    }
}

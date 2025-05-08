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

use std::{
    num::NonZeroU32,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use async_stream::stream;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::error as pipeline_error;
use dynamo_runtime::pipeline::{async_trait, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{CancellationToken, ErrorContext, Result};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::protocols::common::llm_backend::{BackendInput, LLMEngineOutput};
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;

/// If user does not provide a max_tokens limit prompt+output to this many
const DEFAULT_MAX_TOKENS: u32 = 8192;

// I'm not entirely sure what this is. The model context size surely comes from the GGUF??
const CONTEXT_SIZE: u32 = 8192;

static LLAMA_BACKEND: tokio::sync::OnceCell<LlamaBackend> = tokio::sync::OnceCell::const_new();
pub(crate) static LLAMA_MODEL: tokio::sync::OnceCell<LlamaModel> =
    tokio::sync::OnceCell::const_new();
const NUM_CONTEXTS: usize = 3;
static LLAMA_CONTEXTS: [OnceLock<Mutex<ContextWrapper>>; NUM_CONTEXTS] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new()];

// Newtype to simplify LlamaContext lifetime
#[derive(Debug)]
struct ContextWrapper(LlamaContext<'static>);
unsafe impl Send for ContextWrapper {} // LlamaContext has a NonNull which is !Send
unsafe impl Sync for ContextWrapper {} // LlamaContext has a NonNull which is !Sync

pub async fn make_engine(
    cancel_token: CancellationToken,
    model_path: &Path,
) -> pipeline_error::Result<ExecutionContext> {
    let engine = LlamacppEngine::new(cancel_token, model_path).await?;
    let engine: ExecutionContext = Arc::new(engine);
    Ok(engine)
}

struct WorkRequest {
    request: PreprocessedRequest,
    response_channel: tokio::sync::mpsc::Sender<Annotated<LLMEngineOutput>>,
}

struct LlamacppEngine {
    cancel_token: CancellationToken,
    req_tx: tokio::sync::mpsc::Sender<WorkRequest>,
}

impl LlamacppEngine {
    async fn new(
        cancel_token: CancellationToken,
        model_path: &Path,
    ) -> pipeline_error::Result<Self> {
        let backend = LlamaBackend::init()?;
        let model = load_model(&backend, model_path)?;
        LLAMA_MODEL.set(model)?;

        let (ctx_set, ctx_get) = tokio::sync::mpsc::channel(NUM_CONTEXTS);
        // Safety: NonZeroU32::new only errors if we give it a zero
        let context_size = NonZeroU32::new(CONTEXT_SIZE).unwrap();
        let llama_ctx_params = LlamaContextParams::default().with_n_ctx(Some(context_size));
        for (i, ctx_holder) in LLAMA_CONTEXTS.iter().enumerate().take(NUM_CONTEXTS) {
            let llama_ctx = LLAMA_MODEL
                .get()
                .unwrap() // Safety: We put it in a few lines up
                .new_context(&backend, llama_ctx_params.clone())
                .with_context(|| "unable to create the llama_context")?;
            let _ = ctx_holder.set(Mutex::new(ContextWrapper(llama_ctx)));
            let _ = ctx_set.send(i).await;
        }
        LLAMA_BACKEND.set(backend)?;

        let (req_tx, req_rx) = tokio::sync::mpsc::channel(2);
        let ct = cancel_token.clone();
        tokio::task::spawn(worker(ct, req_rx, ctx_get, ctx_set));

        Ok(LlamacppEngine {
            cancel_token,
            req_tx,
        })
    }
}

fn load_model(backend: &LlamaBackend, model_path: &Path) -> Result<LlamaModel> {
    let model_params = {
        if cfg!(any(feature = "cuda", feature = "vulkan")) {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
    };
    LlamaModel::load_from_file(backend, model_path, &model_params)
        .with_context(|| "unable to load model")
}

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for LlamacppEngine
{
    async fn generate(
        &self,
        request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = request.into_parts();
        let ctx = context.context();
        let request_id = ctx.id().to_string();

        let (tx, mut rx) = tokio::sync::mpsc::channel(128);
        let work_request = WorkRequest {
            request,
            response_channel: tx,
        };

        self.req_tx.send(work_request).await?;

        let cancel_token = self.cancel_token.clone();
        let output = stream! {
            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::trace!(request_id, "LlamacppEngine.generate stopped by cancel token");
                        break;
                    }
                    from_llamacpp = rx.recv() => {
                        match from_llamacpp {
                            Some(out) => {
                                yield out;
                            },
                            None => {
                                tracing::trace!(request_id, "generate: response channel closed");
                                break;
                            }
                        }
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

// Run this in a thread
async fn worker(
    cancel_token: CancellationToken,
    mut req_rx: tokio::sync::mpsc::Receiver<WorkRequest>,
    mut ctx_get: tokio::sync::mpsc::Receiver<usize>,
    ctx_set: tokio::sync::mpsc::Sender<usize>,
) {
    loop {
        let maybe_work_request = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_work_request = req_rx.recv() => {
                maybe_work_request
            }
        };
        let Some(work_request) = maybe_work_request else {
            tracing::error!("llamacpp work request sender channel closed. Worker exit");
            break;
        };
        // will block if there are already NUM_CONTEXTS requests in flight
        let Some(ctx_pos) = ctx_get.recv().await else {
            unreachable!("We don't close ctx_set");
        };
        let ct = cancel_token.clone();
        let inner_ctx_set = ctx_set.clone();

        tokio::task::spawn_blocking(move || {
            let mut ctx = LLAMA_CONTEXTS[ctx_pos].get().unwrap().lock().unwrap();
            if let Err(err) = run_request(ct, work_request, &mut ctx) {
                tracing::error!("run_request error: {err:#}");
            }
            let _ = inner_ctx_set.blocking_send(ctx_pos);
        });
    }
}

fn run_request(
    cancel_token: CancellationToken,
    work_request: WorkRequest,
    llama_context: &mut ContextWrapper,
) -> Result<()> {
    let tokens_list: Vec<LlamaToken> = work_request
        .request
        .token_ids
        .into_iter()
        .map(|u| LlamaToken::new(u as i32))
        .collect();

    let limit = DEFAULT_MAX_TOKENS; // - prompt_tokens;
    let max_output_tokens = std::cmp::min(
        work_request
            .request
            .stop_conditions
            .max_tokens
            .unwrap_or(limit),
        limit,
    );

    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(std::cmp::max(512, max_output_tokens as usize), 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .with_context(|| format!("Failed adding token pos {i} to batch"))?;
    }

    // "decode" means "run forward pass"
    llama_context
        .0
        .decode(&mut batch)
        .with_context(|| "llama_decode failed on first pass")?;

    let mut sampler = LlamaSampler::greedy();
    let mut n_cur = batch.n_tokens() as u32;

    let mut used_output_tokens = 0;
    while !cancel_token.is_cancelled() {
        // sample the next token
        let n_tokens = batch.n_tokens();
        let token = sampler.sample(&llama_context.0, n_tokens - 1);
        sampler.accept(token);

        // is it an end of stream?
        // This is probably safe for concurrent access
        if LLAMA_MODEL.get().unwrap().is_eog_token(token) {
            work_request
                .response_channel
                .blocking_send(Annotated::from_data(LLMEngineOutput::stop()))
                .with_context(|| "Failed sending stop to response_channel")?;
            break;
        }

        let engine_out = LLMEngineOutput {
            // todo - propagate mdcsum
            token_ids: vec![token.0 as u32],
            tokens: None,
            text: None,
            //text: if output.text.is_empty() { None } else { Some(output.text) },
            cum_log_probs: None, // TODO output.cumulative_logprob.map(|v| v as f64),
            log_probs: None,     // TODO  output.logprobs
            finish_reason: None,
        };
        work_request
            .response_channel
            .blocking_send(Annotated::from_data(engine_out))
            .with_context(|| "Failed forwarding engine output to response_channel")?;

        batch.clear();
        if let Err(err) = batch.add(token, n_cur as i32, &[0], true) {
            let err_msg = format!(
                "batch add error, probably insufficient space in buffer, aborting request. {err}."
            );
            tracing::error!(err_msg);
            let _ = work_request
                .response_channel
                .blocking_send(Annotated::from_data(LLMEngineOutput::error(err_msg)));
            break;
        }
        n_cur += 1;

        used_output_tokens += 1;
        if used_output_tokens > max_output_tokens {
            let _ = work_request
                .response_channel
                .blocking_send(Annotated::from_data(LLMEngineOutput::length()));
            break;
        }

        llama_context
            .0
            .decode(&mut batch)
            .with_context(|| "llama_decode failed during loop")?;
    }
    if cancel_token.is_cancelled() {
        let _ = work_request
            .response_channel
            .blocking_send(Annotated::from_data(LLMEngineOutput::stop()));
    }

    // Clean context for next use
    llama_context.0.clear_kv_cache();
    llama_context.0.reset_timings();

    Ok(())
}

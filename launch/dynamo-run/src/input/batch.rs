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

use anyhow::Context as _;
use async_openai::types::FinishReason;
use dynamo_llm::model_card::model::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::request_template::RequestTemplate;
use dynamo_llm::types::openai::chat_completions::{
    NvCreateChatCompletionRequest, OpenAIChatCompletionsStreamingEngine,
};
use dynamo_runtime::{pipeline::Context, runtime::CancellationToken, Runtime};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use crate::input::common;
use crate::{EngineConfig, Flags};

/// Max tokens in each response.
/// TODO: For batch mode this should be the full context size of the model
const MAX_TOKENS: u32 = 8192;

const OUTPUT_FILENAME: &str = "output.jsonl";

#[derive(Serialize, Deserialize, Default, Debug)]
struct Entry {
    // The input files only have this
    text: String,

    response: Option<String>,

    #[serde(default)]
    tokens_in: usize,

    #[serde(default)]
    tokens_out: usize,

    #[serde(default)]
    elapsed_ms: usize,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    finish_reason: Option<FinishReason>,

    #[serde(skip, default)]
    request_id: usize,
}

pub async fn run(
    runtime: Runtime,
    flags: Flags,
    card: ModelDeploymentCard,
    input_jsonl: PathBuf,
    engine_config: EngineConfig,
    template: Option<RequestTemplate>,
) -> anyhow::Result<()> {
    let cancel_token = runtime.primary_token();
    // Check if the path exists and is a directory
    if !input_jsonl.exists() || !input_jsonl.is_file() {
        anyhow::bail!(
            "Missing or not a file: {}. Should be a JSON Lines file.",
            input_jsonl.display()
        );
    }

    let prepared_engine = common::prepare_engine(runtime, flags, engine_config).await?;
    let service_name_ref = Arc::new(prepared_engine.service_name);

    let pre_processor = if card.has_tokenizer() {
        Some(OpenAIPreprocessor::new(card).await?)
    } else {
        None
    };
    let (done_entries_tx, done_entries_rx) = tokio::sync::mpsc::channel(64);
    let dw_cancel_token = cancel_token.clone();
    let mut output_file = input_jsonl.clone();
    output_file.set_file_name(OUTPUT_FILENAME);
    tokio::spawn(async move {
        if let Err(err) = output_writer(dw_cancel_token, done_entries_rx, &output_file).await {
            tracing::error!(%err, "Failed writing output to {}", output_file.display());
        }
    });

    let tokens_in = Arc::new(AtomicU64::new(0));
    let tokens_out = Arc::new(AtomicU64::new(0));
    let mut handles = vec![];
    let mut num_entries = 0;
    let input_file = tokio::fs::File::open(&input_jsonl)
        .await
        .with_context(|| input_jsonl.display().to_string())?;
    let buffered_input = tokio::io::BufReader::new(input_file);

    tracing::info!("Timer start.");
    let start = Instant::now();
    let mut lines = buffered_input.lines();
    let template: Option<Arc<RequestTemplate>> = template.map(Arc::new);
    while let Ok(Some(line)) = lines.next_line().await {
        if cancel_token.is_cancelled() {
            break;
        }
        if line.is_empty() {
            continue;
        }
        let request_id = num_entries;
        num_entries += 1;
        let mut entry: Entry = match serde_json::from_str(&line) {
            Ok(entry) => entry,
            Err(err) => {
                anyhow::bail!("Error parsing entry: '{line}'. {err}");
            }
        };
        entry.request_id = request_id;

        let engine = prepared_engine.engine.clone();
        let pre_processor = pre_processor.clone();
        let tokens_in = tokens_in.clone();
        let tokens_out = tokens_out.clone();
        let done_entries_tx = done_entries_tx.clone();
        let service_name_ref = service_name_ref.clone();
        let template_clone = template.clone();
        let handle = tokio::spawn(async move {
            let local_start = Instant::now();
            let response = match evaluate(
                request_id,
                service_name_ref.as_str(),
                engine,
                &mut entry,
                template_clone,
            )
            .await
            {
                Ok(r) => r,
                Err(err) => {
                    tracing::error!(%err, entry.text, "Failed evaluating prompt");
                    return;
                }
            };
            let local_elapsed = Instant::now() - local_start;
            entry.elapsed_ms = local_elapsed.as_millis() as usize;

            if let Some(pre) = pre_processor {
                // Note this does not include the prompt template. Probably TODO
                entry.tokens_in = match pre.tokenize(&entry.text) {
                    Ok(encoding) => encoding.token_ids.len(),
                    Err(err) => {
                        tracing::warn!(%err, entry.text, "Failed tokenizing prompt");
                        0
                    }
                };
                entry.tokens_out = match pre.tokenize(&response) {
                    Ok(encoding) => encoding.token_ids.len(),
                    Err(err) => {
                        tracing::warn!(%err, response, "Failed tokenizing response");
                        0
                    }
                };
                tokens_in.fetch_add(entry.tokens_in as u64, Ordering::Relaxed);
                tokens_out.fetch_add(entry.tokens_out as u64, Ordering::Relaxed);
            }
            entry.response = Some(response);

            let _ = done_entries_tx.send(entry).await;
        });
        handles.push(handle);
    }
    tokio::select! {
        _ = cancel_token.cancelled() => {
            // Don't print stats
            return Ok(());
        }
        _ = futures::future::join_all(handles) => {
        }
    }
    let elapsed = Instant::now() - start;
    let elapsed_clean = Duration::from_millis(elapsed.as_millis() as u64);
    let tokens_in = Arc::into_inner(tokens_in).unwrap().into_inner();
    let tokens_out = Arc::into_inner(tokens_out).unwrap().into_inner();
    tokio::time::sleep(Duration::from_millis(1)).await; // Let output_writer finish stdout write
    tracing::info!(
        "Ran {} files in {}. Tokens in: {} ({}/s). Tokens out: {} ({}/s)",
        num_entries,
        humantime::format_duration(elapsed_clean),
        tokens_in,
        tokens_in / cmp::max(elapsed.as_secs(), 1),
        tokens_out,
        tokens_out / cmp::max(elapsed.as_secs(), 1),
    );
    cancel_token.cancel(); // stop everything else

    Ok(())
}

// Run a single prompt through the engine
async fn evaluate(
    request_id: usize,
    service_name: &str,
    engine: OpenAIChatCompletionsStreamingEngine,
    entry: &mut Entry,
    template: Option<Arc<RequestTemplate>>,
) -> anyhow::Result<String> {
    let user_message = async_openai::types::ChatCompletionRequestMessage::User(
        async_openai::types::ChatCompletionRequestUserMessage {
            content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                entry.text.clone(),
            ),
            name: None,
        },
    );
    let inner = async_openai::types::CreateChatCompletionRequestArgs::default()
        .messages(vec![user_message])
        .model(
            template
                .as_ref()
                .map_or_else(|| service_name.to_string(), |t| t.model.clone()),
        )
        .stream(true)
        .max_completion_tokens(
            template
                .as_ref()
                .map_or(MAX_TOKENS, |t| t.max_completion_tokens),
        )
        .temperature(template.as_ref().map_or(0.7, |t| t.temperature))
        .build()?;
    let req = NvCreateChatCompletionRequest { inner, nvext: None };
    let mut stream = engine.generate(Context::new(req)).await?;
    let mut output = String::new();
    while let Some(item) = stream.next().await {
        match (item.data.as_ref(), item.event.as_deref()) {
            (Some(data), _) => {
                // Normal case
                let choice = data.inner.choices.first();
                let chat_comp = choice.as_ref().unwrap();
                if let Some(c) = &chat_comp.delta.content {
                    output += c;
                }
                entry.finish_reason = chat_comp.finish_reason;
                if chat_comp.finish_reason.is_some() {
                    tracing::trace!(
                        request_id,
                        "finish reason: {:?}",
                        chat_comp.finish_reason.unwrap()
                    );
                    break;
                }
            }
            (None, Some("error")) => {
                tracing::error!(request_id, "the error case");
                // There's only one error but we loop in case that changes
                for err in item.comment.unwrap_or_default() {
                    tracing::error!(request_id, "Engine error: {err}");
                }
            }
            (None, Some(annotation)) => {
                tracing::debug!(request_id, "Annotation. {annotation}: {:?}", item.comment);
            }
            _ => {
                unreachable!("Event from engine with no data, no error, no annotation.");
            }
        }
    }
    Ok(output)
}

async fn output_writer(
    cancel_token: CancellationToken,
    mut entries_rx: tokio::sync::mpsc::Receiver<Entry>,
    output_file: &Path,
) -> anyhow::Result<()> {
    let mut num_completed = 0;
    let mut f = tokio::fs::File::create(output_file).await?;
    loop {
        let entry = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_entry = entries_rx.recv() => {
                match maybe_entry {
                    Some(entry) => entry,
                    None => {break;}
                }
            }
        };
        let mut s = serde_json::to_string(&entry)?;
        s.push('\n');
        f.write_all(s.as_bytes()).await?;

        num_completed += 1;
        // TODO: Progress bar. We'd have to count the lines in the input first,
        // and the input maybe be large
        tracing::info!(entry.request_id, entry.tokens_out, "Saved {num_completed}");
    }
    Ok(())
}

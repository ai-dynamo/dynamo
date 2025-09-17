// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::request_template::RequestTemplate;
use crate::types::openai::chat_completions::{
    NvCreateChatCompletionRequest, OpenAIChatCompletionsStreamingEngine,
};
use dynamo_runtime::{Runtime, pipeline::Context, runtime::CancellationToken};
use futures::StreamExt;
use std::io::{ErrorKind, Write};

use crate::entrypoint::EngineConfig;
use crate::entrypoint::input::common;

/// Max response tokens for each single query. Must be less than model context size.
/// TODO: Cmd line flag to overwrite this
const MAX_TOKENS: u32 = 8192;

pub async fn run(
    runtime: Runtime,
    single_prompt: Option<String>,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let cancel_token = runtime.primary_token();
    let prepared_engine = common::prepare_engine(runtime, engine_config).await?;
    // TODO: Pass prepared_engine directly
    main_loop(
        cancel_token,
        &prepared_engine.service_name,
        prepared_engine.engine,
        single_prompt,
        prepared_engine.inspect_template,
        prepared_engine.request_template,
    )
    .await
}

async fn main_loop(
    cancel_token: CancellationToken,
    service_name: &str,
    engine: OpenAIChatCompletionsStreamingEngine,
    mut initial_prompt: Option<String>,
    _inspect_template: bool,
    template: Option<RequestTemplate>,
) -> anyhow::Result<()> {
    if initial_prompt.is_none() {
        tracing::info!("Ctrl-c to exit");
    }
    let theme = dialoguer::theme::ColorfulTheme::default();

    // Initial prompt is the pipe case: `echo "Hello" | dynamo-run ..`
    // We run that single prompt and exit
    let single = initial_prompt.is_some();
    let mut history = dialoguer::BasicHistory::default();
    let mut messages = vec![];
    while !cancel_token.is_cancelled() {
        // User input
        let prompt = match initial_prompt.take() {
            Some(p) => p,
            None => {
                let input_ui = dialoguer::Input::<String>::with_theme(&theme)
                    .history_with(&mut history)
                    .with_prompt("User");
                match input_ui.interact_text() {
                    Ok(prompt) => prompt,
                    Err(dialoguer::Error::IO(err)) => {
                        match err.kind() {
                            ErrorKind::Interrupted => {
                                // Ctrl-C
                                // Unfortunately I could not make dialoguer handle Ctrl-d
                            }
                            k => {
                                tracing::info!("IO error: {k}");
                            }
                        }
                        break;
                    }
                }
            }
        };

        // Construct messages
        let user_message = dynamo_async_openai::types::ChatCompletionRequestMessage::User(
            dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    prompt,
                ),
                name: None,
            },
        );
        messages.push(user_message);
        // Request
        let inner = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
            .messages(messages.clone())
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
            .n(1) // only generate one response
            .build()?;

        let req = NvCreateChatCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
        };

        // Call the model
        let mut stream = match engine.generate(Context::new(req)).await {
            Ok(stream) => stream,
            Err(err) => {
                tracing::error!(%err, "Request failed.");
                continue;
            }
        };

        // Stream the output to stdout
        let mut stdout = std::io::stdout();
        let mut assistant_message = String::new();
        while let Some(item) = stream.next().await {
            if cancel_token.is_cancelled() {
                break;
            }
            match (item.data.as_ref(), item.event.as_deref()) {
                (Some(data), _) => {
                    // Normal case
                    let entry = data.choices.first();
                    let chat_comp = entry.as_ref().unwrap();
                    if let Some(c) = &chat_comp.delta.content {
                        let _ = stdout.write(c.as_bytes());
                        let _ = stdout.flush();
                        assistant_message += c;
                    }
                    if let Some(reason) = chat_comp.finish_reason {
                        tracing::trace!("finish reason: {reason:?}");
                        break;
                    }
                }
                (None, Some("error")) => {
                    // There's only one error but we loop in case that changes
                    for err in item.comment.unwrap_or_default() {
                        tracing::error!("Engine error: {err}");
                    }
                }
                (None, Some(annotation)) => {
                    tracing::debug!("Annotation. {annotation}: {:?}", item.comment);
                }
                _ => {
                    unreachable!("Event from engine with no data, no error, no annotation.");
                }
            }
        }
        println!();

        let assistant_content =
            dynamo_async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                assistant_message,
            );

        let assistant_message = dynamo_async_openai::types::ChatCompletionRequestMessage::Assistant(
            dynamo_async_openai::types::ChatCompletionRequestAssistantMessage {
                content: Some(assistant_content),
                ..Default::default()
            },
        );
        messages.push(assistant_message);

        if single {
            break;
        }
    }
    cancel_token.cancel(); // stop everything else
    println!();
    Ok(())
}

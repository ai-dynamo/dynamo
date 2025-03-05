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

use dynemo_llm::{
    backend::Backend,
    preprocessor::OpenAIPreprocessor,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};
use dynemo_runtime::{
    pipeline::{Context, ManyOut, Operator, ServiceBackend, ServiceFrontend, SingleIn, Source},
    runtime::CancellationToken,
};
use futures::StreamExt;
use std::{
    io::{ErrorKind, Read, Write},
    sync::Arc,
};

use crate::EngineConfig;

/// Max response tokens for each single query. Must be less than model context size.
const MAX_TOKENS: u32 = 8192;

/// Output of `isatty` if the fd is indeed a TTY
const IS_A_TTY: i32 = 1;

pub async fn run(
    cancel_token: CancellationToken,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let (service_name, engine, inspect_template): (
        String,
        OpenAIChatCompletionsStreamingEngine,
        bool,
    ) = match engine_config {
        EngineConfig::Dynamic(client) => {
            // The service_name isn't used for text chat outside of logs,
            // so use the path. That avoids having to listen on etcd for model registration.
            let service_name = client.path();
            tracing::info!("Model: {service_name}");
            (service_name, Arc::new(client), false)
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
        } => {
            tracing::info!("Model: {service_name}");
            (service_name, engine, false)
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let frontend = ServiceFrontend::<
                SingleIn<NvCreateChatCompletionRequest>,
                ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(*card.clone())
                .await?
                .into_operator();
            let backend = Backend::from_mdc(*card.clone()).await?.into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);

            let pipeline = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            tracing::info!("Model: {service_name} with pre-processing");
            (service_name, pipeline, true)
        }
        EngineConfig::None => unreachable!(),
    };
    main_loop(cancel_token, &service_name, engine, inspect_template).await
}

#[allow(deprecated)]
async fn main_loop(
    cancel_token: CancellationToken,
    service_name: &str,
    engine: OpenAIChatCompletionsStreamingEngine,
    _inspect_template: bool,
) -> anyhow::Result<()> {
    tracing::info!("Ctrl-c to exit");
    let theme = dialoguer::theme::ColorfulTheme::default();

    let mut initial_prompt = if unsafe { libc::isatty(libc::STDIN_FILENO) == IS_A_TTY } {
        None
    } else {
        // Something piped in, use that as initial prompt
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input).unwrap();
        Some(input)
    };

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
        let user_message = async_openai::types::ChatCompletionRequestMessage::User(
            async_openai::types::ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt),
                name: None,
            },
        );
        messages.push(user_message);

        // Request
        let inner = async_openai::types::CreateChatCompletionRequestArgs::default()
            .messages(messages.clone())
            .model(service_name)
            .stream(true)
            .max_tokens(MAX_TOKENS)
            .build()?;

        // TODO We cannot set min_tokens with async-openai
        // if inspect_template {
        //     // This makes the pre-processor ignore stop tokens
        //     req_builder.min_tokens(8192);
        // }

        let req = NvCreateChatCompletionRequest { inner, nvext: None };

        // Call the model
        let mut stream = engine.generate(Context::new(req)).await?;

        // Stream the output to stdout
        let mut stdout = std::io::stdout();
        let mut assistant_message = String::new();
        while let Some(item) = stream.next().await {
            let data = item.data.as_ref().unwrap();
            let entry = data.inner.choices.first();
            let chat_comp = entry.as_ref().unwrap();
            if let Some(c) = &chat_comp.delta.content {
                let _ = stdout.write(c.as_bytes());
                let _ = stdout.flush();
                assistant_message += c;
            }
            if chat_comp.finish_reason.is_some() {
                tracing::trace!("finish reason: {:?}", chat_comp.finish_reason.unwrap());
                break;
            }
        }
        println!();

        let assistant_content =
            async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                assistant_message,
            );

        // ALLOW: function_call is deprecated
        let assistant_message = async_openai::types::ChatCompletionRequestMessage::Assistant(
            async_openai::types::ChatCompletionRequestAssistantMessage {
                content: Some(assistant_content),
                refusal: None,
                name: None,
                audio: None,
                tool_calls: None,
                function_call: None,
            },
        );
        messages.push(assistant_message);
    }
    println!();
    Ok(())
}

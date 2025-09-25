// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker selection pipeline for querying routing decisions.
//!
//! This module provides a simplified pipeline that gets worker selection information
//! without performing full inference. When a request contains the "query_instance_id"
//! annotation, the router returns:
//!
//! 1. A "worker_instance_id" annotation with the selected worker ID
//! 2. A "token_data" annotation with the tokenized prompt
//!
//! Usage:
//! ```rust,ignore
//! let engine = build_worker_selection_pipeline_chat(...).await?;
//! let request = create_request_with_annotation("query_instance_id");
//! let response_stream = engine.generate(request).await?;
//!
//! // Option 1: Use the helper function (recommended)
//! let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await?;
//!
//! // Option 2: Manual extraction
//! let mut stream = engine.generate(request).await?;
//! while let Some(response) = stream.next().await {
//!     if let Some(event) = &response.event {
//!         match event.as_str() {
//!             "worker_instance_id" => { /* extract worker ID from comment field */ }
//!             "token_data" => { /* extract tokens from comment field */ }
//!             _ => {}
//!         }
//!     }
//! }
//! ```

use std::{pin::Pin, sync::Arc};

use serde_json;

// Constants
const GENERATE_ENDPOINT: &str = "generate";

use crate::{
    kv_router::{KvPushRouter, KvRouter, KvRouterConfig},
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            nvext::NvExt,
        },
    },
    types::Annotated,
};

use dynamo_runtime::{
    component::Client,
    engine::AsyncEngineStream,
    pipeline::{
        ManyOut, Operator, PushRouter, RouterMode, SegmentSource, ServiceBackend, ServiceEngine,
        SingleIn, Source,
    },
};

/// Helper function to extract worker selection information from the annotation stream
/// This demonstrates how to process the annotations returned by the router
pub async fn extract_worker_selection_from_stream(
    mut stream: Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
) -> anyhow::Result<(i64, Vec<u32>)> {
    use futures::StreamExt;

    println!("!![DEBUG] extract_worker_selection_from_stream: Starting stream processing");
    let mut worker_id = 0i64;
    let mut tokens = Vec::<u32>::new();
    let mut response_count = 0;

    while let Some(response) = stream.next().await {
        response_count += 1;
        println!(
            "!![DEBUG] extract_worker_selection_from_stream: Processing response #{}",
            response_count
        );
        println!(
            "!![DEBUG] extract_worker_selection_from_stream: Response event: {:?}",
            response.event
        );
        println!(
            "!![DEBUG] extract_worker_selection_from_stream: Response comment: {:?}",
            response.comment
        );
        println!(
            "!![DEBUG] extract_worker_selection_from_stream: Response data: {:?}",
            response.data.is_some()
        );

        if let Some(event) = &response.event {
            match event.as_str() {
                "worker_instance_id" => {
                    println!(
                        "!![DEBUG] extract_worker_selection_from_stream: Found worker_instance_id event"
                    );
                    if let Some(comments) = &response.comment {
                        println!(
                            "!![DEBUG] extract_worker_selection_from_stream: Comments array: {:?}",
                            comments
                        );
                        if let Some(first_comment) = comments.first() {
                            println!(
                                "!![DEBUG] extract_worker_selection_from_stream: First comment: '{}'",
                                first_comment
                            );
                            // Try JSON deserialization as string first (handles quoted strings like "1732646935200805498")
                            match serde_json::from_str::<String>(first_comment) {
                                Ok(id_string) => {
                                    println!(
                                        "!![DEBUG] extract_worker_selection_from_stream: JSON-deserialized string: '{}'",
                                        id_string
                                    );
                                    if let Ok(parsed_id) = id_string.parse::<i64>() {
                                        worker_id = parsed_id;
                                        println!(
                                            "!![DEBUG] extract_worker_selection_from_stream: Successfully parsed worker_id from JSON string: {}",
                                            worker_id
                                        );
                                    } else {
                                        println!(
                                            "!![DEBUG] extract_worker_selection_from_stream: Failed to parse number from JSON string: '{}'",
                                            id_string
                                        );
                                    }
                                }
                                Err(_) => {
                                    // Fallback to direct parsing (handles unquoted numbers)
                                    if let Ok(parsed_id) = first_comment.parse::<i64>() {
                                        worker_id = parsed_id;
                                        println!(
                                            "!![DEBUG] extract_worker_selection_from_stream: Successfully direct-parsed worker_id: {}",
                                            worker_id
                                        );
                                    } else {
                                        println!(
                                            "!![DEBUG] extract_worker_selection_from_stream: Failed to parse worker_id from: '{}'",
                                            first_comment
                                        );
                                    }
                                }
                            }
                        } else {
                            println!(
                                "!![DEBUG] extract_worker_selection_from_stream: Comments array is empty"
                            );
                        }
                    } else {
                        println!(
                            "!![DEBUG] extract_worker_selection_from_stream: No comments field in response"
                        );
                    }
                }
                "token_data" => {
                    println!(
                        "!![DEBUG] extract_worker_selection_from_stream: Found token_data event"
                    );
                    if let Some(comments) = &response.comment {
                        println!(
                            "!![DEBUG] extract_worker_selection_from_stream: Token comments array: {:?}",
                            comments
                        );
                        if let Some(first_comment) = comments.first() {
                            println!(
                                "!![DEBUG] extract_worker_selection_from_stream: Token comment: '{}'",
                                first_comment
                            );
                            match serde_json::from_str::<Vec<u32>>(first_comment) {
                                Ok(parsed_tokens) => {
                                    tokens = parsed_tokens;
                                    println!(
                                        "!![DEBUG] extract_worker_selection_from_stream: Successfully parsed {} tokens",
                                        tokens.len()
                                    );
                                }
                                Err(e) => {
                                    println!(
                                        "!![DEBUG] extract_worker_selection_from_stream: Failed to parse tokens from '{}': {}",
                                        first_comment, e
                                    );
                                }
                            }
                        } else {
                            println!(
                                "!![DEBUG] extract_worker_selection_from_stream: Token comments array is empty"
                            );
                        }
                    } else {
                        println!(
                            "!![DEBUG] extract_worker_selection_from_stream: No comments field in token_data response"
                        );
                    }
                }
                _ => {
                    println!(
                        "!![DEBUG] extract_worker_selection_from_stream: Unknown event type: '{}'",
                        event
                    );
                }
            }
        } else {
            println!("!![DEBUG] extract_worker_selection_from_stream: Response has no event field");
        }
    }

    println!(
        "!![DEBUG] extract_worker_selection_from_stream: Finished processing {} responses",
        response_count
    );
    println!(
        "!![DEBUG] extract_worker_selection_from_stream: Final worker_id={}, tokens.len()={}",
        worker_id,
        tokens.len()
    );
    Ok((worker_id, tokens))
}

/// Utility function to add the "query_instance_id" annotation to an OpenAI request
///
/// This function modifies the request to include the annotation that signals the KV router
/// to return worker selection information (worker_instance_id and token_data) instead of
/// performing actual inference.
///
/// # Parameters
/// - `request`: Mutable reference to the OpenAI chat completion request
///
/// # Returns
/// The same request with the "query_instance_id" annotation added
pub fn add_query_instance_id(
    request: &mut NvCreateChatCompletionRequest,
) -> &mut NvCreateChatCompletionRequest {
    // Create or modify the nvext field to include the query_instance_id annotation
    match request.nvext.as_mut() {
        Some(nvext) => {
            // NvExt already exists, add annotation to it
            match nvext.annotations.as_mut() {
                Some(annotations) => {
                    // Annotations vector exists, add if not already present
                    if !annotations.contains(&"query_instance_id".to_string()) {
                        annotations.push("query_instance_id".to_string());
                    }
                }
                None => {
                    // No annotations vector, create one with our annotation
                    nvext.annotations = Some(vec!["query_instance_id".to_string()]);
                }
            }
        }
        None => {
            // No nvext field, create one with our annotation
            request.nvext = Some(
                NvExt::builder()
                    .add_annotation("query_instance_id")
                    .build()
                    .expect("NvExt builder should not fail"),
            );
        }
    }

    request
}

/// Utility function to add worker_instance_id annotation to an OpenAI request
pub fn add_worker_instance_id_annotation(
    request: &mut NvCreateChatCompletionRequest,
    worker_id: i64,
) -> &mut NvCreateChatCompletionRequest {
    let worker_id_str = worker_id.to_string();

    match request.nvext.as_mut() {
        Some(nvext) => {
            match nvext.annotations.as_mut() {
                Some(annotations) => {
                    // Remove existing worker_instance_id if present
                    annotations.retain(|ann| !ann.starts_with("worker_instance_id:"));
                    annotations.push(format!("worker_instance_id:{}", worker_id_str));
                }
                None => {
                    nvext.annotations = Some(vec![format!("worker_instance_id:{}", worker_id_str)]);
                }
            }
        }
        None => {
            request.nvext = Some(
                NvExt::builder()
                    .add_annotation(format!("worker_instance_id:{}", worker_id_str))
                    .build()
                    .expect("NvExt builder should not fail"),
            );
        }
    }

    request
}

/// Utility function to add token_data annotation to an OpenAI request
pub fn add_token_data_annotation<'a>(
    request: &'a mut NvCreateChatCompletionRequest,
    tokens: &[u32],
) -> &'a mut NvCreateChatCompletionRequest {
    let tokens_json = serde_json::to_string(tokens).unwrap_or_default();

    match request.nvext.as_mut() {
        Some(nvext) => {
            match nvext.annotations.as_mut() {
                Some(annotations) => {
                    // Remove existing token_data if present
                    annotations.retain(|ann| !ann.starts_with("token_data:"));
                    annotations.push(format!("token_data:{}", tokens_json));
                }
                None => {
                    nvext.annotations = Some(vec![format!("token_data:{}", tokens_json)]);
                }
            }
        }
        None => {
            request.nvext = Some(
                NvExt::builder()
                    .add_annotation(format!("token_data:{}", tokens_json))
                    .build()
                    .expect("NvExt builder should not fail"),
            );
        }
    }

    request
}

/// Wrapper function that queries worker selection and annotates the original request
///
/// This function performs the complete flow:
/// 1. Clones the original request and adds "query_instance_id" annotation
/// 2. Calls engine.generate() with the modified request
/// 3. Extracts worker_instance_id and tokens from the response stream
/// 4. Adds worker_instance_id and token_data annotations to the original request
/// 5. Returns (worker_id, tokens, annotated_original_request)
///
/// # Parameters
/// - `engine`: The worker selection pipeline engine
/// - `original_request`: The original OpenAI request to process
///
/// # Returns
/// A tuple containing (worker_instance_id, tokens, modified_original_request)
/// where the modified_original_request has worker_instance_id and token_data annotations added
///
/// # Example
/// ```rust,ignore
/// let (worker_id, tokens, annotated_request) =
///     query_worker_selection_and_annotate(&engine, original_request).await?;
/// ```
pub async fn query_worker_selection_and_annotate(
    engine: &ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
    mut original_request: NvCreateChatCompletionRequest,
) -> anyhow::Result<(i64, Vec<u32>, NvCreateChatCompletionRequest)> {
    // Clone the request and add query_instance_id annotation
    let mut query_request = original_request.clone();
    add_query_instance_id(&mut query_request);

    // Create SingleIn and generate
    let single_in = SingleIn::new(query_request);
    let response_stream = engine.generate(single_in).await?;

    // Extract worker selection from stream
    println!(
        "!![DEBUG] query_worker_selection_and_annotate: About to extract worker selection from stream"
    );
    let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await?;
    println!(
        "!![DEBUG] query_worker_selection_and_annotate: Extracted worker_id={}, tokens.len()={}",
        worker_id,
        tokens.len()
    );

    // Add worker_instance_id and tokens to original request's nvext
    add_worker_instance_id_annotation(&mut original_request, worker_id);
    add_token_data_annotation(&mut original_request, &tokens);

    Ok((worker_id, tokens, original_request))
}

/// Build a worker selection pipeline specifically for Chat Completion requests
///
/// This pipeline: frontend -> preprocessor -> backend -> migration -> router
/// The router handles query_instance_id annotations and returns worker_instance_id and token_data annotations.
pub async fn build_worker_selection_pipeline_chat(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
    hf_tokenizer: tokenizers::Tokenizer,
) -> anyhow::Result<
    ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
> {
    use crate::backend::Backend;
    use crate::migration::Migration;
    use crate::preprocessor::prompt::PromptFormatter;

    let PromptFormatter::OAI(formatter) = PromptFormatter::from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, hf_tokenizer.clone())?;

    let frontend = SegmentSource::<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >::new();
    let preprocessor_op = preprocessor.into_operator();
    let backend = Backend::from_tokenizer(hf_tokenizer).into_operator();
    let migration = Migration::from_mdc(card).into_operator();

    let router =
        PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
            client.clone(),
            router_mode,
            busy_threshold,
        )
        .await?;

    let service_backend = match router_mode {
        RouterMode::Random | RouterMode::RoundRobin | RouterMode::Direct(_) => {
            ServiceBackend::from_engine(Arc::new(router))
        }
        RouterMode::KV => {
            let Some(chooser) = chooser else {
                anyhow::bail!("RouterMode::KV requires KvRouter to not be null");
            };
            let kv_push_router = KvPushRouter::new(router, chooser);
            ServiceBackend::from_engine(Arc::new(kv_push_router))
        }
    };

    // Build bidirectional pipeline (router processes query_instance_id and returns annotations via backward path)
    let engine = frontend
        .link(preprocessor_op.forward_edge())?
        .link(backend.forward_edge())?
        .link(migration.forward_edge())?
        .link(service_backend)?
        .link(migration.backward_edge())?
        .link(backend.backward_edge())?
        .link(preprocessor_op.backward_edge())?
        .link(frontend)?;

    Ok(engine)
}

/// Helper function to create worker selection pipeline for OpenAI Chat Completion requests
///
/// This is a concrete implementation that works specifically with NvCreateChatCompletionRequest
/// and is designed for use with C bindings. Uses the "generate" endpoint by default.
///
/// # Parameters
/// - `namespace`: namespace name
/// - `component_name`: component name
/// - `model_name`: Name/slug of the model to load
/// - `router_mode`: How to route requests (KV, RoundRobin, etc.)
/// - `busy_threshold`: Optional threshold for busy worker detection
/// - `kv_router_config`: Optional KV router configuration (only used when router_mode is KV)
///
/// # Returns
/// A configured worker selection pipeline ready to use
pub async fn create_worker_selection_pipeline_chat(
    namespace: &str,
    component_name: &str,
    model_name: &str,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    kv_router_config: Option<KvRouterConfig>,
) -> anyhow::Result<
    ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
> {
    use crate::{discovery::ModelManager, model_card::ModelDeploymentCard};
    use anyhow::Context;
    use dynamo_runtime::{
        DistributedRuntime, Runtime, distributed::DistributedConfig, slug::Slug,
        traits::DistributedRuntimeProvider,
    };

    // --- IMPORTANT CHANGE ---
    // Create a fresh Runtime + DistributedRuntime, then *leak* the DistributedRuntime
    // so it won't be dropped inside an async context (which triggers Tokio's panic).
    //
    // This is acceptable here because this function is typically called once at startup;
    // if it's called multiple times, you'll leak once per call. If that's a concern,
    // switch to a global OnceCell/OnceLock to cache one instance instead.
    let runtime = Runtime::from_settings()?;
    let dst_config = DistributedConfig::from_settings(false);
    let drt_owned = DistributedRuntime::new(runtime, dst_config).await?;
    let distributed_runtime: &'static DistributedRuntime = Box::leak(Box::new(drt_owned));

    // Create Component and Client
    let ns = distributed_runtime.namespace(namespace)?;
    let component = ns.component(component_name)?;
    let endpoint = component.endpoint(GENERATE_ENDPOINT);
    let client = endpoint.client().await?;

    // Load ModelDeploymentCard
    let model_slug = Slug::from_string(model_name);
    let card = match ModelDeploymentCard::load_from_store(&model_slug, component.drt()).await {
        Ok(Some(card)) => card,
        Ok(None) => anyhow::bail!("ModelDeploymentCard not found for model: {}", model_name),
        Err(err) => anyhow::bail!(
            "Error fetching ModelDeploymentCard from storage under key {model_slug}. {err}"
        ),
    };

    // Get tokenizer from the model card
    let hf_tokenizer = card
        .tokenizer_hf()
        .with_context(|| format!("Failed to load tokenizer for model: {}", model_name))?;

    // Create KV chooser if using KV routing mode
    let chooser = if router_mode == RouterMode::KV {
        let model_manager = std::sync::Arc::new(ModelManager::new());
        Some(
            model_manager
                .kv_chooser_for(
                    &card.display_name,
                    &component,
                    card.kv_cache_block_size,
                    kv_router_config,
                )
                .await?,
        )
    } else {
        None
    };

    // Build and return the worker selection pipeline
    build_worker_selection_pipeline_chat(
        &card,
        &client,
        router_mode,
        busy_threshold,
        chooser,
        hf_tokenizer,
    )
    .await
}

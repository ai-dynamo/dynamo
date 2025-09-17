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
//! let engine = build_worker_selection_pipeline(...).await?;
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

use crate::{
    backend::Backend,
    kv_router::{KvPushRouter, KvRouter},
    migration::Migration,
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    types::Annotated,
};

use dynamo_runtime::{
    component::Client,
    engine::AsyncEngineStream,
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, SegmentSource, ServiceBackend,
        ServiceEngine, SingleIn, Source,
    },
};

/// Build a worker selection pipeline that gets routing decisions from the router
///
/// This pipeline: frontend -> preprocessor -> backend -> migration -> router
/// The router handles query_instance_id annotations and returns worker_instance_id and token_data annotations.
pub async fn build_worker_selection_pipeline_with_preprocessor<Req>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
    preprocessor: Arc<OpenAIPreprocessor>,
    hf_tokenizer: tokenizers::Tokenizer,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<LLMEngineOutput>>>>
where
    Req: dynamo_runtime::engine::Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
        >,
{
    let frontend = SegmentSource::<SingleIn<Req>, ManyOut<Annotated<LLMEngineOutput>>>::new();
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

    // Build pipeline - forward path only (router handles query_instance_id and returns annotations)
    frontend
        .link(preprocessor_op.forward_edge())?
        .link(backend.forward_edge())?
        .link(migration.forward_edge())?
        .link(service_backend)?;

    Ok(frontend)
}

/// Convenience function that creates a preprocessor and calls build_worker_selection_pipeline_with_preprocessor
pub async fn build_worker_selection_pipeline<Req>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
    hf_tokenizer: tokenizers::Tokenizer,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<LLMEngineOutput>>>>
where
    Req: dynamo_runtime::engine::Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
        >,
{
    use crate::preprocessor::prompt::PromptFormatter;

    let PromptFormatter::OAI(formatter) = PromptFormatter::from_mdc(card)?;
    let preprocessor =
        OpenAIPreprocessor::new_with_parts(card.clone(), formatter, hf_tokenizer.clone())?;

    build_worker_selection_pipeline_with_preprocessor(
        card,
        client,
        router_mode,
        busy_threshold,
        chooser,
        preprocessor,
        hf_tokenizer,
    )
    .await
}

/// Helper function to extract worker selection information from the annotation stream
/// This demonstrates how to process the annotations returned by the router
pub async fn extract_worker_selection_from_stream(
    mut stream: Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
) -> anyhow::Result<(i64, Vec<u32>)> {
    use futures::StreamExt;

    let mut worker_id = 0i64;
    let mut tokens = Vec::<u32>::new();

    while let Some(response) = stream.next().await {
        if let Some(event) = &response.event {
            match event.as_str() {
                "worker_instance_id" => {
                    worker_id = response
                        .comment
                        .as_ref()
                        .and_then(|comments| comments.first())
                        .and_then(|v| v.parse::<i64>().ok())
                        .unwrap_or(0);
                }
                "token_data" => {
                    tokens = response
                        .comment
                        .as_ref()
                        .and_then(|comments| comments.first())
                        .and_then(|v| serde_json::from_str::<Vec<u32>>(v).ok())
                        .unwrap_or_default();
                }
                _ => {}
            }
        }
    }

    Ok((worker_id, tokens))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[tokio::test]
    #[ignore] // Requires full distributed setup
    async fn test_worker_selection_pipeline() {
        // This test would require:
        // - A real ModelDeploymentCard
        // - A Component client connected to workers
        // - A KvRouter with actual worker state

        // Example test structure:
        // let engine = build_worker_selection_pipeline(...).await.unwrap();
        //
        // // Create a request with query_instance_id annotation
        // let request = create_test_request_with_annotation("query_instance_id");
        // let response_stream = engine.generate(request).await.unwrap();
        //
        // // Use the helper function to extract worker selection information
        // let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await.unwrap();
        //
        // assert!(worker_id > 0);
        // assert!(!tokens.is_empty());
    }
}

/*
/// Helper function to create worker selection pipeline from C string parameters
///
/// This function demonstrates how to create all the necessary parameters for
/// `build_worker_selection_pipeline` when you only have namespace, component,
/// and model information from C FFI.
///
/// # Parameters
/// - `namespace_c_str`: C string pointer to namespace name
/// - `component_c_str`: C string pointer to component name
/// - `endpoint_name`: Name of the endpoint to connect to (e.g., "inference")
/// - `model_name`: Name/slug of the model to load
/// - `router_mode`: How to route requests (KV, RoundRobin, etc.)
/// - `busy_threshold`: Optional threshold for busy worker detection
///
/// # Returns
/// A configured worker selection pipeline ready to use
///
/// # Example Usage in C FFI context:
/// ```rust,ignore
/// let pipeline = create_worker_selection_pipeline_from_c_params(
///     namespace_c_str,
///     component_c_str,
///     "inference",
///     "llama3-8b-instruct",
///     RouterMode::KV,
///     Some(0.8)
/// ).await?;
///
/// // Use pipeline to get worker selection
/// let request = create_request_with_annotation("query_instance_id");
/// let response_stream = pipeline.generate(request).await?;
/// let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await?;
/// ```
pub async fn create_worker_selection_pipeline_from_c_params<Req>(
    namespace_c_str: *const std::os::raw::c_char,
    component_c_str: *const std::os::raw::c_char,
    endpoint_name: &str,
    model_name: &str,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<LLMEngineOutput>>>>
where
    Req: dynamo_runtime::engine::Data,
    OpenAIPreprocessor: Operator<
            Context<Req>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
            Context<PreprocessedRequest>,
            Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
        >,
{
    use std::ffi::CStr;
    use dynamo_runtime::{
        Runtime, DistributedRuntime,
        storage::{EtcdStorage, KeyValueStoreManager},
        slug::Slug,
    };
    use crate::{
        model_card::{ModelDeploymentCard, ROOT_PATH as MODEL_ROOT_PATH},
        discovery::ModelManager,
    };

    // 1. Convert C strings to Rust strings
    let namespace_str = unsafe {
        CStr::from_ptr(namespace_c_str)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid namespace string: {}", e))?
    };
    let component_str = unsafe {
        CStr::from_ptr(component_c_str)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid component string: {}", e))?
    };

    // 2. Create Runtime and DistributedRuntime
    let runtime = Runtime::new().await?;
    let distributed_runtime = DistributedRuntime::from_settings(runtime).await?;

    // 3. Create Component and Client
    let namespace = distributed_runtime.namespace(namespace_str)?;
    let component = namespace.component(component_str)?;
    let endpoint = component.endpoint(endpoint_name);
    let client = endpoint.client().await?;

    // 4. Load ModelDeploymentCard
    let model_slug = Slug::from_string(model_name);
    let card = match ModelDeploymentCard::load_from_store(&model_slug, &runtime)

    // 5. Get HuggingFace tokenizer from the model card
    let hf_tokenizer = card.tokenizer_hf()
        .with_context(|| format!("Failed to load tokenizer for model: {}", model_name))?;

    // 6. Create KV chooser if using KV routing mode
    let chooser = if router_mode == RouterMode::KV {
        let model_manager = std::sync::Arc::new(ModelManager::new());
        Some(
            model_manager.kv_chooser_for(
                &card.display_name,
                &component,
                card.kv_cache_block_size,
                None, // Use default KV router config
            ).await?
        )
    } else {
        None
    };

    // 7. Build and return the worker selection pipeline
    build_worker_selection_pipeline(
        &card,
        &client,
        router_mode,
        busy_threshold,
        chooser,
        hf_tokenizer,
    ).await
}

*/

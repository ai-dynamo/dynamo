// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A WorkerSet represents a group of workers deployed from the same configuration,
//! identified by their shared namespace. Each WorkerSet owns a complete pipeline
//! (engines, KV router, prefill router) built from its specific ModelDeploymentCard.

use std::sync::Arc;

use tokio::sync::watch;

use dynamo_renderer::OAIPromptFormatter;

use crate::{
    discovery::KvWorkerMonitor,
    kv_router::{KvRouter, PrefillRouter},
    model_card::{ModelDeploymentCard, ModelInfo},
    tokenizers::Tokenizer,
    types::{
        RealtimeBidirectionalEngine,
        generic::tensor::TensorStreamingEngine,
        openai::{
            audios::OpenAIAudiosStreamingEngine,
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, images::OpenAIImagesStreamingEngine,
            videos::OpenAIVideosStreamingEngine,
        },
    },
};

/// Captured-at-construction inputs for the `/tokenize` and `/detokenize` HTTP
/// endpoints.
///
/// Mirrors the per-WorkerSet caching that the chat/completions engines get
/// implicitly through `OpenAIPreprocessor` (`preprocessor.rs:310-353`): the
/// tokenizer, default prompt formatter, and model_info are loaded once at
/// WorkerSet construction and held in memory so subsequent requests don't
/// touch the MDC store or disk. This makes /tokenize survive the file-backed
/// worker's WorkerSet being torn down, the same way chat completions
/// already do — `Model::get_tokenize_handle` selects any live WorkerSet.
pub struct TokenizeHandle {
    pub display_name: String,
    pub tokenizer: Tokenizer,
    /// Default chat formatter built from the MDC. `None` when the MDC has no
    /// `prompt_formatter` artifact (completions-only models) — chat-tokenize
    /// then surfaces a clean error instead of silently falling back.
    pub formatter: Option<Arc<dyn OAIPromptFormatter>>,
    /// Cached model_info (eos_token_ids etc.). `None` if absent on the MDC.
    pub model_info: Option<Arc<dyn ModelInfo>>,
    pub context_length: u32,
}

impl TokenizeHandle {
    /// Direct constructor from pre-loaded Arcs. The watcher loads the
    /// formatter and ModelInfo once and shares the same Arcs with the chat /
    /// completions preprocessors, so we don't re-parse `tokenizer_config.json`
    /// and `config.json` per consumer.
    pub fn new(
        display_name: String,
        tokenizer: Tokenizer,
        formatter: Option<Arc<dyn OAIPromptFormatter>>,
        model_info: Option<Arc<dyn ModelInfo>>,
        context_length: u32,
    ) -> Self {
        Self {
            display_name,
            tokenizer,
            formatter,
            model_info,
            context_length,
        }
    }
}

/// A set of workers from the same namespace/configuration with their own pipeline.
pub struct WorkerSet {
    /// Full namespace (e.g., "ns-abc12345")
    namespace: String,

    /// MDC checksum for this set's configuration
    mdcsum: String,

    /// The model deployment card used to build this set's pipeline
    card: ModelDeploymentCard,

    // Engines — each WorkerSet owns its own pipelines
    pub(crate) chat_engine: Option<OpenAIChatCompletionsStreamingEngine>,
    pub(crate) completions_engine: Option<OpenAICompletionsStreamingEngine>,
    pub(crate) embeddings_engine: Option<OpenAIEmbeddingsStreamingEngine>,
    pub(crate) images_engine: Option<OpenAIImagesStreamingEngine>,
    pub(crate) videos_engine: Option<OpenAIVideosStreamingEngine>,
    pub(crate) audios_engine: Option<OpenAIAudiosStreamingEngine>,
    pub(crate) tensor_engine: Option<TensorStreamingEngine>,
    pub(crate) realtime_engine: Option<RealtimeBidirectionalEngine>,

    /// KV router for this set's workers (if KV mode)
    pub(crate) kv_router: Option<Arc<KvRouter>>,

    /// Worker monitor for load-based rejection
    pub(crate) worker_monitor: Option<KvWorkerMonitor>,

    /// Prefill router for disaggregated serving. Stored here so the watcher can
    /// deactivate it when all prefill workers die, and reactivate when they rejoin.
    pub(crate) prefill_router: Option<Arc<PrefillRouter>>,

    /// In-memory state needed by the `/tokenize` and `/detokenize` HTTP
    /// endpoints. Populated by the watcher at WorkerSet construction so the
    /// endpoints don't re-resolve through `ModelManager.cards` (which churns
    /// when individual workers come and go).
    pub(crate) tokenize_handle: Option<Arc<TokenizeHandle>>,

    /// Watcher for available instance IDs (from the Client's discovery watch).
    /// None for in-process models (http/grpc) which don't have a discovery client.
    instance_count_rx: Option<watch::Receiver<Vec<u64>>>,
}

impl WorkerSet {
    pub fn new(namespace: String, mdcsum: String, card: ModelDeploymentCard) -> Self {
        Self {
            namespace,
            mdcsum,
            card,
            chat_engine: None,
            completions_engine: None,
            embeddings_engine: None,
            images_engine: None,
            videos_engine: None,
            audios_engine: None,
            tensor_engine: None,
            realtime_engine: None,
            kv_router: None,
            worker_monitor: None,
            prefill_router: None,
            tokenize_handle: None,
            instance_count_rx: None,
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    pub fn card(&self) -> &ModelDeploymentCard {
        &self.card
    }

    pub fn has_chat_engine(&self) -> bool {
        self.chat_engine.is_some()
    }

    pub fn has_completions_engine(&self) -> bool {
        self.completions_engine.is_some()
    }

    pub fn has_embeddings_engine(&self) -> bool {
        self.embeddings_engine.is_some()
    }

    pub fn has_images_engine(&self) -> bool {
        self.images_engine.is_some()
    }

    pub fn has_videos_engine(&self) -> bool {
        self.videos_engine.is_some()
    }

    pub fn has_audios_engine(&self) -> bool {
        self.audios_engine.is_some()
    }

    pub fn has_tensor_engine(&self) -> bool {
        self.tensor_engine.is_some()
    }

    pub fn has_realtime_engine(&self) -> bool {
        self.realtime_engine.is_some()
    }

    /// Whether this set has any decode engine (chat or completions)
    pub fn has_decode_engine(&self) -> bool {
        self.has_chat_engine() || self.has_completions_engine()
    }

    /// Whether this set has any engine capable of producing output for an
    /// inference request. Single source of truth for the "is something attached
    /// that can serve a request?" question — keep the engine-kind list here so
    /// new modalities don't need to be added in multiple readiness predicates.
    pub fn has_any_serving_engine(&self) -> bool {
        self.has_chat_engine()
            || self.has_completions_engine()
            || self.has_embeddings_engine()
            || self.has_images_engine()
            || self.has_tensor_engine()
            || self.has_videos_engine()
            || self.has_audios_engine()
            || self.has_realtime_engine()
    }

    /// Whether this set tracks a prefill model (no engine, just lifecycle)
    pub fn is_prefill_set(&self) -> bool {
        !self.has_any_serving_engine()
    }

    /// In-memory inputs for `/tokenize` and `/detokenize`. `None` for
    /// WorkerSets whose model has no Rust-loadable tokenizer (e.g. models
    /// served via a Python chat_engine_factory).
    pub fn tokenize_handle(&self) -> Option<Arc<TokenizeHandle>> {
        self.tokenize_handle.clone()
    }

    pub fn has_tokenize_handle(&self) -> bool {
        self.tokenize_handle.is_some()
    }

    /// Build ParsingOptions from this WorkerSet's card configuration.
    pub fn parsing_options(&self) -> crate::protocols::openai::ParsingOptions {
        crate::protocols::openai::ParsingOptions::new(
            self.card.runtime_config.tool_call_parser.clone(),
            self.card.runtime_config.reasoning_parser.clone(),
        )
    }

    /// Number of active workers in this set, derived from the Client's discovery watcher.
    /// Returns 1 for in-process models (no watcher) since they always have one local worker.
    pub fn worker_count(&self) -> usize {
        match &self.instance_count_rx {
            Some(rx) => rx.borrow().len(),
            None => 1,
        }
    }

    /// Store the instance watcher from the Client's discovery system.
    /// Must be called before the WorkerSet is wrapped in Arc.
    pub fn set_instance_watcher(&mut self, rx: watch::Receiver<Vec<u64>>) {
        self.instance_count_rx = Some(rx);
    }

    /// Whether this WorkerSet can serve requests. Delegates to the prefill router
    /// if one exists; otherwise always returns true.
    /// When the prefill router is deactivated and enforce_disagg is set, this returns
    /// false, causing the model to be hidden from /v1/models and requests to be rejected.
    pub fn can_serve_requests(&self) -> bool {
        self.prefill_router
            .as_ref()
            .is_none_or(|pr| pr.can_serve_requests())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;
    use crate::types::Annotated;
    use crate::types::generic::tensor::{NvCreateTensorRequest, NvCreateTensorResponse};
    use crate::types::openai::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};
    use crate::types::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    };
    use crate::types::openai::completions::{
        NvCreateCompletionRequest, NvCreateCompletionResponse,
    };
    use crate::types::openai::embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse};
    use crate::types::openai::images::{NvCreateImageRequest, NvImagesResponse};
    use crate::types::openai::videos::{NvCreateVideoRequest, NvVideosResponse};
    use async_trait::async_trait;
    use dynamo_runtime::engine::AsyncEngine;
    use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
    use std::marker::PhantomData;

    fn make_worker_set(namespace: &str, mdcsum: &str) -> WorkerSet {
        WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        )
    }

    /// Generic stub satisfying any `ServerStreamingEngine<Req, Annotated<Resp>>` trait
    /// object. `generate` is unreachable: the stub exists only to populate typed engine
    /// slots on `WorkerSet` so `is_prefill_set`'s exclusion logic can be exercised per
    /// field. `Req` / `Resp` are inferred from the assignment-site engine alias.
    struct StubEngine<Req, Resp>(PhantomData<fn() -> (Req, Resp)>);

    impl<Req, Resp> StubEngine<Req, Resp> {
        fn new() -> Arc<Self> {
            Arc::new(Self(PhantomData))
        }
    }

    #[async_trait]
    impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>
        for StubEngine<Req, Resp>
    where
        Req: dynamo_runtime::engine::Data,
        Resp: dynamo_runtime::engine::Data,
    {
        async fn generate(&self, _req: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
            unimplemented!("stub for is_prefill_set classification tests only")
        }
    }

    #[test]
    fn test_worker_set_basics() {
        let ws = make_worker_set("ns1", "abc123");
        assert_eq!(ws.namespace(), "ns1");
        assert_eq!(ws.mdcsum(), "abc123");
    }

    #[test]
    fn test_no_engines_by_default() {
        let ws = make_worker_set("ns1", "abc123");
        assert!(!ws.has_chat_engine());
        assert!(!ws.has_completions_engine());
        assert!(!ws.has_embeddings_engine());
        assert!(!ws.has_images_engine());
        assert!(!ws.has_videos_engine());
        assert!(!ws.has_audios_engine());
        assert!(!ws.has_tensor_engine());
        assert!(!ws.has_realtime_engine());
        assert!(!ws.has_decode_engine());
        assert!(ws.is_prefill_set());
    }

    /// `is_prefill_set` must exclude every serving-engine field on `WorkerSet`. If a new
    /// engine variant is added without updating `is_prefill_set`, a worker that registers
    /// only that engine would be misclassified as prefill — silent and easy to miss in
    /// integration tests. This walks each engine in isolation so the failing arm names
    /// itself.
    #[test]
    fn test_any_serving_engine_excludes_prefill() {
        macro_rules! check {
            ($field:ident, $has:ident, $engine:expr, $label:literal) => {{
                let mut ws = make_worker_set("ns1", "abc123");
                ws.$field = Some($engine);
                assert!(ws.$has());
                assert!(
                    !ws.is_prefill_set(),
                    concat!($label, "-only WorkerSet must not be classified as prefill")
                );
            }};
        }

        check!(
            chat_engine,
            has_chat_engine,
            StubEngine::<NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse>::new(
            ),
            "chat"
        );
        check!(
            completions_engine,
            has_completions_engine,
            StubEngine::<NvCreateCompletionRequest, NvCreateCompletionResponse>::new(),
            "completions"
        );
        check!(
            embeddings_engine,
            has_embeddings_engine,
            StubEngine::<NvCreateEmbeddingRequest, NvCreateEmbeddingResponse>::new(),
            "embeddings"
        );
        check!(
            images_engine,
            has_images_engine,
            StubEngine::<NvCreateImageRequest, NvImagesResponse>::new(),
            "images"
        );
        check!(
            videos_engine,
            has_videos_engine,
            StubEngine::<NvCreateVideoRequest, NvVideosResponse>::new(),
            "videos"
        );
        check!(
            audios_engine,
            has_audios_engine,
            StubEngine::<NvCreateAudioSpeechRequest, NvAudioSpeechResponse>::new(),
            "audios"
        );
        check!(
            tensor_engine,
            has_tensor_engine,
            StubEngine::<NvCreateTensorRequest, NvCreateTensorResponse>::new(),
            "tensor"
        );
        check!(
            realtime_engine,
            has_realtime_engine,
            Arc::new(crate::engines::EchoBidirectionalEngine),
            "realtime"
        );
    }

    #[test]
    fn test_worker_count_without_watcher() {
        // In-process models have no discovery watcher; worker_count defaults to 1
        let ws = make_worker_set("ns1", "abc");
        assert_eq!(ws.worker_count(), 1);
    }

    #[test]
    fn test_worker_count_with_watcher() {
        let mut ws = make_worker_set("ns1", "abc");

        // Simulate a discovery watcher with 3 workers
        let (tx, rx) = watch::channel(vec![1, 2, 3]);
        ws.set_instance_watcher(rx);
        assert_eq!(ws.worker_count(), 3);

        // Workers leave → count drops
        tx.send(vec![1]).unwrap();
        assert_eq!(ws.worker_count(), 1);

        // All workers gone → count is 0
        tx.send(vec![]).unwrap();
        assert_eq!(ws.worker_count(), 0);
    }

    #[test]
    fn test_worker_count_with_empty_watcher() {
        // Discovery watcher starts empty (no workers have joined yet)
        let mut ws = make_worker_set("ns1", "abc");
        let (_tx, rx) = watch::channel::<Vec<u64>>(vec![]);
        ws.set_instance_watcher(rx);
        assert_eq!(ws.worker_count(), 0);
    }

    #[test]
    fn test_worker_count_updates_on_join() {
        let mut ws = make_worker_set("ns1", "abc");
        let (tx, rx) = watch::channel::<Vec<u64>>(vec![]);
        ws.set_instance_watcher(rx);
        assert_eq!(ws.worker_count(), 0);

        // Workers join one by one
        tx.send(vec![100]).unwrap();
        assert_eq!(ws.worker_count(), 1);

        tx.send(vec![100, 200]).unwrap();
        assert_eq!(ws.worker_count(), 2);

        tx.send(vec![100, 200, 300]).unwrap();
        assert_eq!(ws.worker_count(), 3);
    }
}

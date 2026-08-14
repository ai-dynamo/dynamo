// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A WorkerSet represents a group of workers behind one serving endpoint. Each
//! WorkerSet owns a complete pipeline (engines, KV router, prefill router) built
//! from its specific ModelDeploymentCard.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, Data};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::EndpointId;
use tokio::sync::watch;

use crate::{
    discovery::{KvWorkerMonitor, allocator::AllocatorTrimOnDrop},
    kv_router::{EncoderRouter, KvRouter, PrefillRouter},
    model_card::ModelDeploymentCard,
    types::{
        RealtimeBidirectionalEngine,
        generic::tensor::TensorStreamingEngine,
        openai::{
            audios::OpenAIAudiosStreamingEngine,
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, generate::GenerateStreamingEngine,
            images::OpenAIImagesStreamingEngine, videos::OpenAIVideosStreamingEngine,
        },
    },
};

type StreamingEngine<Req, Resp> = Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Resp>, Error>>;

struct RequestLifetimeEngine<Req, Resp>
where
    Req: AsyncEngineContextProvider + Send + 'static,
    Resp: AsyncEngineContextProvider + 'static,
{
    inner: Arc<dyn AsyncEngine<Req, Resp, Error>>,
    teardown: Arc<AllocatorTrimOnDrop>,
}

#[async_trait]
impl<Req, Resp> AsyncEngine<Req, Resp, Error> for RequestLifetimeEngine<Req, Resp>
where
    Req: AsyncEngineContextProvider + Send + 'static,
    Resp: AsyncEngineContextProvider + 'static,
{
    async fn generate(&self, request: Req) -> Result<Resp, Error> {
        request.context().retain(self.teardown.clone());
        let response = self.inner.generate(request).await?;
        response.context().retain(self.teardown.clone());
        Ok(response)
    }
}

fn retain_teardown_until_requests_finish<Req, Resp>(
    engine: Option<Arc<dyn AsyncEngine<Req, Resp, Error>>>,
    teardown: &Arc<AllocatorTrimOnDrop>,
) -> Option<Arc<dyn AsyncEngine<Req, Resp, Error>>>
where
    Req: AsyncEngineContextProvider + Send + 'static,
    Resp: AsyncEngineContextProvider + 'static,
{
    engine.map(|inner| {
        Arc::new(RequestLifetimeEngine {
            inner,
            teardown: teardown.clone(),
        }) as Arc<dyn AsyncEngine<Req, Resp, Error>>
    })
}

struct LoraContextEngine<Req: Data, Resp: Data> {
    inner: StreamingEngine<Req, Resp>,
    lora_name: String,
}

#[async_trait]
impl<Req: Data, Resp: Data> AsyncEngine<SingleIn<Req>, ManyOut<Resp>, Error>
    for LoraContextEngine<Req, Resp>
{
    async fn generate(&self, mut request: SingleIn<Req>) -> Result<ManyOut<Resp>, Error> {
        request.insert(
            crate::preprocessor::LORA_NAME_CONTEXT_KEY,
            self.lora_name.clone(),
        );
        self.inner.generate(request).await
    }
}

struct LoraGenerateEngine {
    inner: GenerateStreamingEngine,
    lora_name: String,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<crate::protocols::common::preprocessor::PreprocessedRequest>,
        ManyOut<crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>>,
        Error,
    > for LoraGenerateEngine
{
    async fn generate(
        &self,
        mut request: SingleIn<crate::protocols::common::preprocessor::PreprocessedRequest>,
    ) -> Result<
        ManyOut<crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>>,
        Error,
    > {
        request.routing.get_or_insert_default().lora_name = Some(self.lora_name.clone());
        self.inner.generate(request).await
    }
}

fn lora_context_engine<Req: Data, Resp: Data>(
    engine: &Option<StreamingEngine<Req, Resp>>,
    lora_name: &str,
) -> Option<StreamingEngine<Req, Resp>> {
    engine.as_ref().map(|inner| {
        Arc::new(LoraContextEngine {
            inner: inner.clone(),
            lora_name: lora_name.to_string(),
        }) as Arc<dyn AsyncEngine<SingleIn<Req>, ManyOut<Resp>, Error>>
    })
}

/// A set of workers from the same namespace/configuration with their own pipeline.
pub struct WorkerSet {
    /// Full namespace (e.g., "ns-abc12345")
    namespace: String,

    /// Exact serving pool identity. Discovery-backed WorkerSets always set
    /// this; in-process models have no distributed endpoint.
    endpoint_id: Option<EndpointId>,

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
    pub(crate) generate_engine: Option<GenerateStreamingEngine>,

    /// KV router for this set's workers (if KV mode)
    pub(crate) kv_router: Option<Arc<KvRouter>>,

    /// Worker monitor for load-based rejection
    pub(crate) worker_monitor: Option<KvWorkerMonitor>,

    /// Prefill router for disaggregated serving. Stored here so the watcher can
    /// deactivate it when all prefill workers die, and reactivate when they rejoin.
    pub(crate) prefill_router: Option<Arc<PrefillRouter>>,

    /// Optional multimodal encoder hop. Stored for discovery-driven
    /// deactivation/reactivation when Encode workers leave or rejoin.
    pub(crate) encoder_router: Option<Arc<EncoderRouter>>,

    /// Watcher for available instance IDs (from the Client's discovery watch).
    /// None for in-process models (http/grpc) which don't have a discovery client.
    instance_count_rx: Option<watch::Receiver<Vec<u64>>>,

    /// Drops after engine fields and after every active request context releases it.
    allocator_trim: Option<Arc<AllocatorTrimOnDrop>>,
    allocator_trim_wrapped: bool,
}

impl WorkerSet {
    pub fn new(namespace: String, mdcsum: String, card: ModelDeploymentCard) -> Self {
        Self {
            namespace,
            endpoint_id: None,
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
            generate_engine: None,
            kv_router: None,
            worker_monitor: None,
            prefill_router: None,
            encoder_router: None,
            instance_count_rx: None,
            allocator_trim: None,
            allocator_trim_wrapped: false,
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn endpoint_id(&self) -> Option<&EndpointId> {
        self.endpoint_id.as_ref()
    }

    pub(crate) fn set_endpoint_id(&mut self, endpoint_id: EndpointId) {
        self.endpoint_id = Some(endpoint_id);
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

    pub fn has_generate_engine(&self) -> bool {
        self.generate_engine.is_some()
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
            || self.has_generate_engine()
    }

    /// Whether this set tracks an Encode worker. Encode WorkerSets carry
    /// no serving engines (the watcher's Encode role gate skips
    /// pipeline construction) -- if we let `is_prefill_set` classify
    /// them, model-displayability logic would gate /v1/models on a
    /// PrefillRouter that doesn't exist for Encode. Keep the two
    /// mutually exclusive.
    ///
    /// **Role-based, not engine-field-based.** Unlike `has_chat_engine()`
    /// / `has_completions_engine()` / etc. (which inspect typed engine
    /// slots on the WorkerSet), `is_encode_set` reads `card.worker_type`
    /// directly. The Encode role intentionally has no `encode_engine`
    /// field -- Encode workers don't expose a public OpenAI-shaped
    /// endpoint, so there is nothing to slot. The role itself is the
    /// contract.
    pub fn is_encode_set(&self) -> bool {
        matches!(
            self.card.worker_type,
            Some(crate::worker_type::WorkerType::Encode),
        )
    }

    /// Whether this set tracks a prefill model (no engine, just
    /// lifecycle). Excludes Encode sets, which also lack engines but
    /// are not gated through PrefillRouter.
    pub fn is_prefill_set(&self) -> bool {
        !self.is_encode_set() && !self.has_any_serving_engine()
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

    pub(crate) fn initialize_allocator_trim_on_teardown(&mut self) -> Arc<AllocatorTrimOnDrop> {
        self.allocator_trim
            .get_or_insert_with(|| Arc::new(AllocatorTrimOnDrop::new()))
            .clone()
    }

    pub(crate) fn enable_allocator_trim_on_teardown(&mut self) {
        if self.allocator_trim_wrapped {
            return;
        }
        let teardown = self.initialize_allocator_trim_on_teardown();
        macro_rules! retain_for_requests {
            ($field:ident) => {
                self.$field = retain_teardown_until_requests_finish(self.$field.take(), &teardown);
            };
        }
        retain_for_requests!(chat_engine);
        retain_for_requests!(completions_engine);
        retain_for_requests!(embeddings_engine);
        retain_for_requests!(images_engine);
        retain_for_requests!(videos_engine);
        retain_for_requests!(audios_engine);
        retain_for_requests!(tensor_engine);
        retain_for_requests!(realtime_engine);
        retain_for_requests!(generate_engine);
        self.allocator_trim_wrapped = true;
    }

    pub(crate) fn adapter_view(&self, card: ModelDeploymentCard) -> Self {
        let lora_name = card
            .lora
            .as_ref()
            .expect("adapter views require LoRA metadata")
            .name
            .clone();
        let mdcsum = card.mdcsum().to_string();
        let generate_engine = self.generate_engine.as_ref().map(|inner| {
            Arc::new(LoraGenerateEngine {
                inner: inner.clone(),
                lora_name: lora_name.clone(),
            }) as GenerateStreamingEngine
        });
        let mut view = Self {
            namespace: self.namespace.clone(),
            endpoint_id: self.endpoint_id.clone(),
            mdcsum,
            card,
            chat_engine: lora_context_engine(&self.chat_engine, &lora_name),
            completions_engine: lora_context_engine(&self.completions_engine, &lora_name),
            embeddings_engine: lora_context_engine(&self.embeddings_engine, &lora_name),
            images_engine: lora_context_engine(&self.images_engine, &lora_name),
            videos_engine: lora_context_engine(&self.videos_engine, &lora_name),
            audios_engine: lora_context_engine(&self.audios_engine, &lora_name),
            tensor_engine: lora_context_engine(&self.tensor_engine, &lora_name),
            // The bidirectional realtime engine cannot carry the server-streaming context
            // wrapper. Do not expose the base weights through an adapter model name.
            realtime_engine: None,
            generate_engine,
            kv_router: self.kv_router.clone(),
            worker_monitor: self.worker_monitor.clone(),
            prefill_router: self.prefill_router.clone(),
            encoder_router: self.encoder_router.clone(),
            instance_count_rx: self.instance_count_rx.clone(),
            allocator_trim: None,
            allocator_trim_wrapped: false,
        };
        if self.allocator_trim.is_some() {
            view.enable_allocator_trim_on_teardown();
        }
        view
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::llm_backend::LLMEngineOutput;
    use crate::protocols::common::preprocessor::PreprocessedRequest;
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
    use std::{marker::PhantomData, sync::Mutex};

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

    struct CaptureGenerateEngine {
        observed_lora: Arc<Mutex<Option<String>>>,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CaptureGenerateEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            *self.observed_lora.lock().unwrap() = request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.clone());
            Err(anyhow::anyhow!("captured request"))
        }
    }

    #[test]
    fn test_worker_set_basics() {
        let ws = make_worker_set("ns1", "abc123");
        assert_eq!(ws.namespace(), "ns1");
        assert_eq!(ws.mdcsum(), "abc123");
    }

    #[tokio::test]
    async fn adapter_view_routes_generate_requests_with_adapter_identity() {
        let observed_lora = Arc::new(Mutex::new(None));
        let mut base = make_worker_set("ns1", "abc123");
        base.generate_engine = Some(Arc::new(CaptureGenerateEngine {
            observed_lora: observed_lora.clone(),
        }));
        let mut adapter_card = ModelDeploymentCard::with_name_only("adapter-model");
        adapter_card.lora = Some(crate::model_card::LoraInfo {
            name: "adapter-model".to_string(),
            max_gpu_lora_count: Some(4),
        });
        let adapter = base.adapter_view(adapter_card);
        let request = PreprocessedRequest::builder()
            .model("adapter-model".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();

        let result = adapter
            .generate_engine
            .as_ref()
            .unwrap()
            .generate(SingleIn::new(request))
            .await;

        assert!(result.is_err());
        assert_eq!(
            observed_lora.lock().unwrap().as_deref(),
            Some("adapter-model")
        );
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
        assert!(!ws.has_generate_engine());
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
        check!(
            generate_engine,
            has_generate_engine,
            StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
            "generate"
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

    // -------------------------------------------------------------------
    // Encode-set classification
    //
    // Encode WorkerSets carry no serving engines (the watcher's role
    // gate skips pipeline construction), so the legacy "no engines =
    // prefill" rule would misclassify them. is_encode_set distinguishes
    // them via card.worker_type and is_prefill_set excludes them so the
    // two predicates stay mutually exclusive.
    // -------------------------------------------------------------------

    fn make_encode_worker_set() -> WorkerSet {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Encode);
        WorkerSet::new("ns1".to_string(), "abc".to_string(), card)
    }

    #[test]
    fn encode_set_is_classified_as_encode_not_prefill() {
        let ws = make_encode_worker_set();
        assert!(ws.is_encode_set());
        // The two predicates must be mutually exclusive: an Encode set
        // has no engines but must NOT be classified as prefill, since
        // model-displayability logic gates /v1/models on PrefillRouter
        // for prefill sets and Encode workers have no such router.
        assert!(!ws.is_prefill_set());
    }

    #[test]
    fn non_encode_engineless_set_stays_classified_as_prefill() {
        // Regression guard: the existing "engineless = prefill" rule
        // must still hold for worker_type = None / Prefill / Decode /
        // Aggregated. Only Encode is carved out.
        let mut card_none = ModelDeploymentCard::default();
        card_none.worker_type = None;
        let ws = WorkerSet::new("ns1".to_string(), "abc".to_string(), card_none);
        assert!(!ws.is_encode_set());
        assert!(ws.is_prefill_set());

        for role in [
            crate::worker_type::WorkerType::Prefill,
            crate::worker_type::WorkerType::Decode,
            crate::worker_type::WorkerType::Aggregated,
        ] {
            let mut card = ModelDeploymentCard::default();
            card.worker_type = Some(role);
            let ws = WorkerSet::new("ns1".to_string(), "abc".to_string(), card);
            assert!(!ws.is_encode_set(), "{:?} should not be Encode", role);
            assert!(
                ws.is_prefill_set(),
                "{:?} should remain prefill-classified",
                role
            );
        }
    }
}

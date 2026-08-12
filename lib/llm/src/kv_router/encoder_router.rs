// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional multimodal encoder hop for token-serving pipelines.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Context as _, Result};
use arc_swap::ArcSwapOption;
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    engine::AsyncEngine,
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, ServerStreamingEngine, SingleIn,
        async_trait,
    },
    protocols::{EndpointId, annotated::Annotated, maybe_error::MaybeError},
};

use crate::protocols::common::{
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::TraceLink,
};

type EncodePushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

struct EncoderBinding {
    endpoint_id: EndpointId,
    router: Arc<EncodePushRouter>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum EncoderLifecycleState {
    Pending = 0,
    Active = 1,
    Unavailable = 2,
}

impl EncoderLifecycleState {
    fn load(value: u8) -> Self {
        match value {
            0 => Self::Pending,
            1 => Self::Active,
            2 => Self::Unavailable,
            value => panic!("invalid encoder lifecycle state: {value}"),
        }
    }
}

/// Forward-only operator that optionally runs a multimodal Encode worker.
///
/// The router is present on every token pipeline but remains a passthrough
/// until discovery supplies an Encode endpoint for the same model namespace.
/// Encode workers are selected round-robin independently of the downstream
/// token router mode; they do not participate in KV-aware routing.
pub struct EncoderRouter {
    binding: ArcSwapOption<EncoderBinding>,
    target: Mutex<Option<EndpointId>>,
    target_tx: Option<watch::Sender<Option<Endpoint>>>,
    cancel_token: CancellationToken,
    lifecycle: AtomicU8,
    require_handoff: bool,
    model_name: String,
    namespace: String,
}

impl Drop for EncoderRouter {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}

impl EncoderRouter {
    /// Create a permanently-disabled passthrough router.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            binding: ArcSwapOption::empty(),
            target: Mutex::new(None),
            target_tx: None,
            cancel_token: CancellationToken::new(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            require_handoff: false,
            model_name: String::new(),
            namespace: String::new(),
        })
    }

    /// Create a router whose endpoint is driven by committed discovery topology.
    pub fn new(model_name: String, namespace: String, require_handoff: bool) -> Arc<Self> {
        let cancel_token = CancellationToken::new();
        let (target_tx, target_rx) = watch::channel(None);
        let router = Arc::new(Self {
            binding: ArcSwapOption::empty(),
            target: Mutex::new(None),
            target_tx: Some(target_tx),
            cancel_token: cancel_token.clone(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            require_handoff,
            model_name,
            namespace,
        });

        tokio::spawn(Self::drive_target(
            Arc::downgrade(&router),
            target_rx,
            cancel_token,
        ));

        router
    }

    async fn build(endpoint: Endpoint) -> Result<EncoderBinding> {
        let endpoint_id = endpoint.id();
        let client = endpoint.client().await?;
        let router =
            EncodePushRouter::from_client_with_monitor(client, RouterMode::RoundRobin, None)
                .await?;
        Ok(EncoderBinding {
            endpoint_id,
            router: Arc::new(router),
        })
    }

    async fn drive_target(
        router: std::sync::Weak<Self>,
        mut target_rx: watch::Receiver<Option<Endpoint>>,
        cancel_token: CancellationToken,
    ) {
        loop {
            let target = target_rx.borrow_and_update().clone();
            let Some(endpoint) = target else {
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => return,
                    changed = target_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                continue;
            };
            let endpoint_id = endpoint.id();
            let reuses_binding = router.upgrade().is_some_and(|router| {
                router
                    .binding
                    .load_full()
                    .is_some_and(|binding| binding.endpoint_id == endpoint_id)
                    && router.lifecycle_state() == EncoderLifecycleState::Active
            });
            if reuses_binding {
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => return,
                    changed = target_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                continue;
            }
            let build = Self::build(endpoint);
            tokio::pin!(build);
            let result = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => return,
                changed = target_rx.changed() => {
                    if changed.is_err() {
                        return;
                    }
                    continue;
                }
                result = &mut build => result,
            };

            let Some(router) = router.upgrade() else {
                return;
            };
            match result {
                Ok(binding) => {
                    let current_target = router.target.lock();
                    if current_target.as_ref() != Some(&endpoint_id) {
                        continue;
                    }
                    router.binding.store(Some(Arc::new(binding)));
                    router
                        .lifecycle
                        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
                    drop(current_target);
                    tracing::info!(
                        model = %router.model_name,
                        namespace = %router.namespace,
                        %endpoint_id,
                        "Encoder router target activated"
                    );
                }
                Err(error) => {
                    if router.target.lock().as_ref() != Some(&endpoint_id) {
                        continue;
                    }
                    tracing::error!(
                        %error,
                        model = %router.model_name,
                        namespace = %router.namespace,
                        %endpoint_id,
                        "Failed to activate encoder router target"
                    );
                    drop(router);
                    tokio::select! {
                        biased;
                        _ = cancel_token.cancelled() => return,
                        changed = target_rx.changed() => {
                            if changed.is_err() {
                                return;
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                    }
                }
            }
        }
    }

    fn lifecycle_state(&self) -> EncoderLifecycleState {
        EncoderLifecycleState::load(self.lifecycle.load(Ordering::Acquire))
    }

    /// Update the desired Encode endpoint. Clearing is synchronous so requests
    /// holding an older catalog snapshot stop using a removed endpoint before
    /// the new catalog is published.
    pub(crate) fn set_target(&self, target: Option<Endpoint>) {
        let target_id = target.as_ref().map(Endpoint::id);
        let mut current = self.target.lock();
        if *current == target_id {
            return;
        }
        *current = target_id.clone();
        let reuses_binding = target_id.is_some()
            && self
                .binding
                .load_full()
                .is_some_and(|binding| Some(&binding.endpoint_id) == target_id.as_ref());
        let lifecycle = if target.is_none() {
            EncoderLifecycleState::Unavailable
        } else if reuses_binding {
            EncoderLifecycleState::Active
        } else {
            self.binding.store(None);
            EncoderLifecycleState::Pending
        };
        self.lifecycle.store(lifecycle as u8, Ordering::Release);
        if let Some(target_tx) = &self.target_tx {
            target_tx.send_replace(target);
        }
    }

    #[cfg(test)]
    pub(crate) fn target_endpoint_id(&self) -> Option<EndpointId> {
        self.target.lock().clone()
    }

    fn should_encode(request: &PreprocessedRequest) -> bool {
        !request.is_probe
            && request.encoder_result.is_none()
            && request
                .multi_modal_data
                .as_ref()
                .is_some_and(|media| media.values().any(|items| !items.is_empty()))
    }

    async fn consume_encode_stream(
        mut response: ManyOut<Annotated<LLMEngineOutput>>,
    ) -> Result<(serde_json::Value, Option<TraceLink>)> {
        let mut terminal = None;
        while let Some(item) = response.next().await {
            if let Some(error) = item.err() {
                return Err(anyhow::anyhow!(error)).context("Encode worker returned an error");
            }
            let Some(output) = item.data else {
                continue;
            };
            if output.finish_reason.is_some() {
                terminal = Some(output);
            }
        }

        let terminal = terminal.context("Encode worker stream ended without a terminal chunk")?;
        let result = terminal
            .encoder_result
            .filter(serde_json::Value::is_object)
            .context("Encode worker terminal is missing an object-shaped encoder_result")?;
        Ok((result, terminal.worker_trace_link))
    }

    fn install_encode_result(
        request: &mut PreprocessedRequest,
        encoder_result: serde_json::Value,
        worker_link: Option<TraceLink>,
    ) {
        request.encoder_result = Some(encoder_result);
        // The descriptor replaces the raw image payload. Keeping both would
        // waste request-plane bandwidth and risks downstream double decoding.
        request.multi_modal_data = None;
        request.migration_link = worker_link;
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for EncoderRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (mut request, context) = request.into_parts();
        if !Self::should_encode(&request) {
            return next.generate(context.map(|_| request)).await;
        }
        if self.lifecycle_state() != EncoderLifecycleState::Active {
            if self.require_handoff {
                anyhow::bail!("custom encoder handoff is required but no Encode worker is active");
            }
            return next.generate(context.map(|_| request)).await;
        }

        let encode_context = Context::with_id_and_metadata(
            request.clone(),
            context.id().to_string(),
            context.metadata().clone(),
        );
        let encode_result = async {
            let binding = self
                .binding
                .load_full()
                .context("Encoder router is active but not initialized")?;
            let response = binding.router.generate(encode_context).await?;
            Self::consume_encode_stream(response).await
        }
        .await;

        match encode_result {
            Ok((encoder_result, worker_link)) => {
                // Once the Encode worker has emitted a transfer handle, always hand it
                // to the downstream worker even if the caller disconnected. The
                // receiver owns transfer completion and buffer release.
                Self::install_encode_result(&mut request, encoder_result, worker_link);
            }
            Err(error) => {
                if self.require_handoff {
                    return Err(error).context("Required custom encoder hop failed");
                }
                tracing::error!(
                    %error,
                    model = %self.model_name,
                    namespace = %self.namespace,
                    "Encoder hop failed; falling back to downstream inline encoding"
                );
            }
        }
        next.generate(context.map(|_| request)).await
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Mutex};

    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::{
        engine::AsyncEngineContextProvider,
        pipeline::{Error, ResponseStream, context::Controller},
    };

    use crate::protocols::common::preprocessor::MultimodalData;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

    use super::*;

    fn stream_of(items: Vec<Annotated<LLMEngineOutput>>) -> ManyOut<Annotated<LLMEngineOutput>> {
        ResponseStream::new(
            Box::pin(stream::iter(items)),
            Arc::new(Controller::default()),
        )
    }

    #[derive(Default)]
    struct CaptureEngine {
        request: Mutex<Option<PreprocessedRequest>>,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CaptureEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> std::result::Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            self.request
                .lock()
                .unwrap()
                .replace(request.content().clone());
            Ok(ResponseStream::new(
                Box::pin(stream::empty()),
                request.context(),
            ))
        }
    }

    fn multimodal_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .multi_modal_data(Some(HashMap::from([(
                "image".to_string(),
                vec![MultimodalData::RawUrl(
                    "data:image/png;base64,cGF5bG9hZA==".to_string(),
                )],
            )])))
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn pending_activation_does_not_keep_router_alive() {
        let router = EncoderRouter::new("model".into(), "namespace".into(), false);
        let weak = Arc::downgrade(&router);

        drop(router);

        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn encoder_failure_falls_back_to_downstream() {
        let router = EncoderRouter::disabled();
        router
            .lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        let downstream = Arc::new(CaptureEngine::default());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            downstream.clone();

        let _response = router
            .generate(SingleIn::new(multimodal_request()), next)
            .await
            .expect("encoder failure should fall through to downstream");

        let request = downstream
            .request
            .lock()
            .unwrap()
            .take()
            .expect("downstream must receive the original request");
        assert!(request.encoder_result.is_none());
        assert!(request.multi_modal_data.is_some());
    }

    #[tokio::test]
    async fn required_encoder_is_fail_closed_while_inactive() {
        let router = EncoderRouter::new("model".into(), "namespace".into(), true);
        let downstream = Arc::new(CaptureEngine::default());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            downstream.clone();

        let result = router
            .generate(SingleIn::new(multimodal_request()), next)
            .await;
        let error = result
            .err()
            .expect("required handoff must reject an inactive Encode route");

        assert!(error.to_string().contains("no Encode worker is active"));
        assert!(downstream.request.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn required_encoder_failure_does_not_call_downstream() {
        let router = EncoderRouter::new("model".into(), "namespace".into(), true);
        router
            .lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        let downstream = Arc::new(CaptureEngine::default());
        let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            downstream.clone();

        let result = router
            .generate(SingleIn::new(multimodal_request()), next)
            .await;
        let error = result
            .err()
            .expect("required handoff must propagate Encode failure");

        assert!(
            error
                .to_string()
                .contains("Required custom encoder hop failed")
        );
        assert!(downstream.request.lock().unwrap().is_none());
    }

    #[test]
    fn successful_encoder_result_replaces_raw_media() {
        let mut request = multimodal_request();
        let descriptor = json!({"schema_version": 1});

        EncoderRouter::install_encode_result(&mut request, descriptor.clone(), None);

        assert_eq!(request.encoder_result, Some(descriptor));
        assert!(request.multi_modal_data.is_none());
    }

    #[tokio::test]
    async fn consumes_object_shaped_encode_terminal() {
        let output = LLMEngineOutput::encode_terminal(
            json!({"schema_version": 1}).as_object().unwrap().clone(),
        );
        let (result, _) =
            EncoderRouter::consume_encode_stream(stream_of(vec![Annotated::from_data(output)]))
                .await
                .unwrap();
        assert_eq!(result, json!({"schema_version": 1}));
    }

    #[tokio::test]
    async fn rejects_terminal_without_encoder_result() {
        let result = EncoderRouter::consume_encode_stream(stream_of(vec![Annotated::from_data(
            LLMEngineOutput::stop(),
        )]))
        .await;
        assert!(result.is_err());
    }
}

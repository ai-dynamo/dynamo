// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multimodal encoder hop for token-serving pipelines.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Context as _, Result};
use arc_swap::ArcSwapOption;
use futures::{StreamExt, stream};
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{
        Context, ManyOut, Operator, PipelineError, PushRouter, ResponseStream, RouterMode,
        ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated},
};

use crate::discovery::{WorkerSetTarget, WorkerSetTargetId};
use crate::protocols::common::{
    FinishReason,
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::{MultimodalData, TraceLink},
};

type EncodePushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

/// Encode workers that can release an unclaimed result through their own endpoint
/// set this flag on the terminal. A cleanup call carries the original request ID
/// in context metadata under the same key and the opaque result in the request.
pub const ENCODER_RELEASE: &str = "dynamo.encoder.release";

struct EncoderLease {
    router: Arc<EncodePushRouter>,
    worker_id: u64,
    caller: Arc<dyn AsyncEngineContext>,
    model: String,
    result: Option<serde_json::Value>,
}

impl EncoderLease {
    fn release(&mut self) {
        let Some(result) = self.result.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let model = self.model.clone();
        let request_id = self.caller.id().to_owned();
        let worker_id = self.worker_id;
        let router = self.router.clone();
        runtime.spawn(async move {
            let release = async {
                let request = PreprocessedRequest::builder()
                    .model(model)
                    .token_ids(Vec::new())
                    .stop_conditions(Default::default())
                    .sampling_options(Default::default())
                    .output_options(Default::default())
                    .encoder_result(Some(result))
                    .build()?;
                let mut call = Context::new(request);
                call.insert_metadata(ENCODER_RELEASE, &request_id);
                let mut response = router.direct(call, worker_id).await?;
                while let Some(item) = response.next().await {
                    if item.into_data()?.is_some_and(|out| out.finish_reason == Some(FinishReason::Stop)) {
                        return Ok(());
                    }
                }
                anyhow::bail!("encoder release ended without an acknowledgement")
            };
            let timeout = std::time::Duration::from_secs(2);
            if !matches!(tokio::time::timeout(timeout, release).await, Ok(Ok(()))) {
                tracing::debug!(%request_id, worker_id, "producer expiry will reclaim the unconfirmed encoder release");
            }
        });
    }
}

impl Drop for EncoderLease {
    fn drop(&mut self) {
        if self.caller.is_stopped() {
            self.release();
        }
    }
}

// A connection timeout or interrupted stream may leave a live consumer. Only
// revoke on a known rejection or cancellation. Lost receipts expire at E.
fn definitively_rejected(error: &(dyn std::error::Error + 'static)) -> bool {
    use super::prefill_router::PrefillError;
    let mut current = Some(error);
    while let Some(cause) = current {
        if matches!(
            cause.downcast_ref::<PipelineError>(),
            Some(PipelineError::NoRoutableWorkers)
        ) || matches!(
            cause.downcast_ref::<PrefillError>(),
            Some(PrefillError::NotActivated | PrefillError::Cancelled)
        ) || cause.downcast_ref::<DynamoError>().is_some_and(|e| {
            // These semantic reasons survive the request-plane wire, including
            // an admission refusal wrapped in a transport prologue error.
            matches!(
                e.reason().as_str(),
                "capacity.pool_exhausted"
                    | "capacity.worker_overloaded"
                    | "backend.worker_unavailable"
                    | "request.cancelled"
                    | "backend.cancelled"
            ) || e.class() == ErrorType::InvalidRequest
        }) {
            return true;
        }
        current = cause.source();
    }
    false
}

/// A required encoder failed before the language worker received the request.
#[derive(Debug, thiserror::Error)]
#[error("Required encoder stage failed")]
pub struct RequiredEncoderStage(#[from] anyhow::Error);

struct EncoderBinding {
    target_id: WorkerSetTargetId,
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

/// Forward-only operator that routes media to a multimodal Encode worker.
///
/// Consumers that require an encoder result propagate staging failures. Other
/// token pipelines can encode inline while the external encoder is unavailable.
/// Encode workers are selected round-robin independently of the downstream
/// token router mode; they do not participate in KV-aware routing.
pub struct EncoderRouter {
    binding: ArcSwapOption<EncoderBinding>,
    target: Mutex<Option<WorkerSetTargetId>>,
    target_tx: Option<watch::Sender<Option<WorkerSetTarget>>>,
    cancel_token: CancellationToken,
    lifecycle: AtomicU8,
    requires_encoder_result: bool,
    model_name: String,
    namespace: String,
}

impl Drop for EncoderRouter {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}

impl EncoderRouter {
    /// Returns `None` when this router does not target the group, and an empty
    /// set when it targets the endpoint but its binding is not yet available.
    pub(crate) fn available_worker_ids_for(
        &self,
        group_id: &str,
        endpoint: &EndpointId,
    ) -> Option<std::collections::HashSet<u64>> {
        // Target changes and activation publish binding/lifecycle under this same lock.
        let target_id = self.target.lock();
        let binding = self.binding.load();
        let binding_matches = binding.as_ref().is_some_and(|binding| {
            &binding.router.client.endpoint.id() == endpoint
                && binding.target_id.matches_group(group_id, endpoint)
        });
        let target_matches = self.target_tx.as_ref().map(|target_tx| {
            target_tx.borrow().as_ref().is_some_and(|target| {
                &target.endpoint().id() == endpoint
                    && target_id
                        .as_ref()
                        .is_some_and(|id| id.matches_group(group_id, endpoint))
            })
        });
        if !target_matches.unwrap_or(binding_matches) {
            return None;
        }
        Some(
            if self.cancel_token.is_cancelled()
                || self.lifecycle_state() != EncoderLifecycleState::Active
            {
                std::collections::HashSet::new()
            } else {
                binding
                    .as_ref()
                    .filter(|binding| {
                        &binding.router.client.endpoint.id() == endpoint
                            && binding.target_id.matches_group(group_id, endpoint)
                    })
                    .map(|binding| {
                        binding
                            .router
                            .client
                            .available_instance_ids()
                            .map(|ids| ids.as_ref().clone())
                            .unwrap_or_default()
                    })
                    .unwrap_or_default()
            },
        )
    }

    /// Create a permanently-disabled passthrough router.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            binding: ArcSwapOption::empty(),
            target: Mutex::new(None),
            target_tx: None,
            cancel_token: CancellationToken::new(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            requires_encoder_result: false,
            model_name: String::new(),
            namespace: String::new(),
        })
    }

    /// Create a router whose endpoint is driven by committed discovery topology.
    pub fn new(model_name: String, namespace: String) -> Arc<Self> {
        Self::new_inner(model_name, namespace, false, None)
    }

    pub(crate) fn new_with_task_guard(
        model_name: String,
        namespace: String,
        requires_encoder_result: bool,
        task_guard: dynamo_runtime::engine::EngineContextGuard,
    ) -> Arc<Self> {
        Self::new_inner(
            model_name,
            namespace,
            requires_encoder_result,
            Some(task_guard),
        )
    }

    fn new_inner(
        model_name: String,
        namespace: String,
        requires_encoder_result: bool,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
    ) -> Arc<Self> {
        let cancel_token = CancellationToken::new();
        let (target_tx, target_rx) = watch::channel(None);
        let router = Arc::new(Self {
            binding: ArcSwapOption::empty(),
            target: Mutex::new(None),
            target_tx: Some(target_tx),
            cancel_token: cancel_token.clone(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            model_name,
            namespace,
            requires_encoder_result,
        });

        let router_weak = Arc::downgrade(&router);
        tokio::spawn(async move {
            let _task_guard = task_guard;
            Self::drive_target(router_weak, target_rx, cancel_token).await;
        });

        router
    }

    async fn build(
        target: WorkerSetTarget,
        cancel_token: CancellationToken,
    ) -> Result<EncoderBinding> {
        let target_id = target.id();
        let client = target.client(cancel_token).await?;
        let router = EncodePushRouter::from_client_with_options(
            client,
            RouterMode::RoundRobin,
            crate::kv_router::plugins::push_router_options(),
        )
        .await?;
        Ok(EncoderBinding {
            target_id,
            router: Arc::new(router),
        })
    }

    async fn drive_target(
        router: std::sync::Weak<Self>,
        mut target_rx: watch::Receiver<Option<WorkerSetTarget>>,
        cancel_token: CancellationToken,
    ) {
        loop {
            let target = target_rx.borrow_and_update().clone();
            let Some(target) = target else {
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
            let target_id = target.id();
            let endpoint_id = target.endpoint().id();
            let reuses_binding = router.upgrade().is_some_and(|router| {
                router
                    .binding
                    .load_full()
                    .is_some_and(|binding| binding.target_id == target_id)
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
            let build = Self::build(target, cancel_token.child_token());
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
                    if current_target.as_ref() != Some(&target_id) {
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
                    if router.target.lock().as_ref() != Some(&target_id) {
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
    pub(crate) fn set_target(&self, target: Option<WorkerSetTarget>) {
        let target_id = target.as_ref().map(WorkerSetTarget::id);
        let mut current = self.target.lock();
        if *current == target_id {
            return;
        }
        *current = target_id.clone();
        let reuses_binding = target_id.is_some()
            && self
                .binding
                .load_full()
                .is_some_and(|binding| Some(&binding.target_id) == target_id.as_ref());
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
    pub(crate) fn target_endpoint_id(&self) -> Option<dynamo_runtime::protocols::EndpointId> {
        self.target_tx
            .as_ref()
            .and_then(|tx| tx.borrow().as_ref().map(|target| target.endpoint().id()))
    }

    fn should_encode(request: &PreprocessedRequest) -> bool {
        !request.is_probe
            && request.encoder_result.is_none()
            && request.multi_modal_data.as_ref().is_some_and(|media| {
                media.values().any(|items| !items.is_empty())
                    && !media
                        .values()
                        .flatten()
                        .any(|item| matches!(item, MultimodalData::Json(_)))
            })
    }

    async fn consume_encode_stream(
        mut response: ManyOut<Annotated<LLMEngineOutput>>,
    ) -> Result<(serde_json::Value, Option<TraceLink>, bool)> {
        let mut terminal = None;
        while let Some(item) = response.next().await {
            let Some(output) = item
                .into_data()
                .context("Encode worker returned an error")?
            else {
                continue;
            };
            if output.finish_reason.is_some() {
                terminal = Some(output);
            }
        }

        let terminal = terminal.context("Encode worker stream ended without a terminal chunk")?;
        let result = terminal
            .encoder_result
            .map(|result| *result)
            .filter(serde_json::Value::is_object)
            .context("Encode worker terminal is missing an object-shaped encoder_result")?;
        let releasable = terminal
            .extra_args
            .as_deref()
            .and_then(|v| v.get(ENCODER_RELEASE))
            .and_then(|v| v.as_bool())
            == Some(true);
        Ok((
            result,
            terminal.worker_trace_link.map(|link| *link),
            releasable,
        ))
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
        let active = self.lifecycle_state() == EncoderLifecycleState::Active;
        if (!active && !self.requires_encoder_result) || !Self::should_encode(&request) {
            return next.generate(context.map(|_| request)).await;
        }
        crate::routing_policy::check_stage(
            &request,
            crate::routing_policy::ExecutionStage::Encoder,
        )?;

        let encode_result = async {
            let binding = self.binding.load_full().filter(|_| active).ok_or_else(|| {
                DynamoError::builder()
                    .error_type(ErrorType::WorkerUnavailable)
                    .message("No encoder is available for this model")
                    .build()
            })?;
            let mut encode_context = Context::with_id_and_metadata(
                request.clone(),
                context.id().to_string(),
                context.metadata().clone(),
            );
            encode_context.metadata_mut().remove(ENCODER_RELEASE);
            let (worker_id, response) = binding
                .router
                .select_and_dispatch(encode_context, |_, worker_id| Ok(worker_id))
                .await?;
            let (result, link, releasable) = Self::consume_encode_stream(response).await?;
            let lease = releasable.then(|| EncoderLease {
                router: binding.router.clone(),
                worker_id,
                caller: context.context(),
                model: request.model.clone(),
                result: Some(result.clone()),
            });
            anyhow::Ok((result, link, lease))
        }
        .await;

        let lease = match encode_result {
            Ok((encoder_result, worker_link, lease)) => {
                request.encoder_result = Some(encoder_result);
                request.migration_link = worker_link;
                lease
            }
            Err(error) if self.requires_encoder_result => {
                return Err(RequiredEncoderStage(error).into());
            }
            Err(error) => {
                tracing::error!(
                    %error,
                    model = %self.model_name,
                    namespace = %self.namespace,
                    "Encoder hop failed; falling back to downstream inline encoding"
                );
                None
            }
        };
        if lease.is_some() && context.controller().is_stopped() {
            return Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated::from_data(
                    LLMEngineOutput::cancelled(),
                )])),
                context.context(),
            ));
        }
        let response = next.generate(context.map(|_| request)).await;
        let Some(mut lease) = lease else {
            return response;
        };
        let response = response.inspect_err(|error| {
            if definitively_rejected(error.as_ref()) {
                lease.release();
            }
        })?;
        let ctx = response.context();
        let response = response.inspect(move |item| {
            if item
                .error
                .as_ref()
                .is_some_and(|e| definitively_rejected(e))
                || item
                    .data
                    .as_ref()
                    .is_some_and(|out| out.finish_reason == Some(FinishReason::Cancelled))
            {
                lease.release();
            } else if item.data.as_ref().is_some_and(|out| {
                !out.token_ids.is_empty()
                    || matches!(
                        out.finish_reason,
                        Some(FinishReason::Stop | FinishReason::Length | FinishReason::EoS)
                    )
            }) {
                lease.result = None;
            }
        });
        Ok(ResponseStream::new(Box::pin(response), ctx))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::{Mutex, atomic::AtomicUsize},
        time::Duration,
    };

    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::EventTransportKind,
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        engine::AsyncEngine,
        pipeline::{Error, context::Controller, network::Ingress},
        storage::kv,
    };

    use crate::discovery::CommittedWorkerSetTarget;
    use crate::model_card::ModelDeploymentCard;
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

    #[test]
    fn opaque_json_skips_the_encoder_even_when_mixed_with_media() {
        let mut request = multimodal_request();
        assert!(EncoderRouter::should_encode(&request));
        request.multi_modal_data.as_mut().unwrap().insert(
            "chemistry".into(),
            vec![MultimodalData::Json(serde_json::json!({"atoms":["C"]}))],
        );
        assert!(!EncoderRouter::should_encode(&request));
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

    struct EncodeWorker(
        u64,
        Arc<AtomicUsize>,
        tokio::sync::mpsc::UnboundedSender<(u64, String)>,
    );

    struct RouteOutcome(Option<ErrorType>);

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for RouteOutcome
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            if let Some(kind) = &self.0 {
                return Err(DynamoError::builder()
                    .error_type(*kind)
                    .message("route failed")
                    .build()
                    .into());
            }
            Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated::from_data(
                    LLMEngineOutput::cancelled(),
                )])),
                request.context(),
            ))
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for EncodeWorker
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            if let Some(original_id) = request.metadata().get(ENCODER_RELEASE) {
                let receipt = request.encoder_result.as_ref().unwrap();
                assert_eq!(receipt["worker_id"], self.0);
                assert_eq!(receipt["stage_request"], *original_id);
                assert_ne!(request.id(), original_id);
                assert!(request.multi_modal_data.is_none() && request.token_ids.is_empty());
                self.2.send((self.0, original_id.clone())).unwrap();
                return Ok(stream_of(vec![Annotated::from_data(
                    LLMEngineOutput::encode_terminal(Default::default()),
                )]));
            }
            self.1.fetch_add(1, Ordering::SeqCst);
            let mut output = LLMEngineOutput::encode_terminal(
                json!({"worker_id": self.0, "stage_request": request.id()})
                    .as_object()
                    .unwrap()
                    .clone(),
            );
            output.extra_args = Some(Box::new(json!({(ENCODER_RELEASE): true})));
            Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated::from_data(output)])),
                request.context(),
            ))
        }
    }

    async fn encoded_worker(router: &EncoderRouter) -> Option<u64> {
        let downstream = Arc::new(CaptureEngine::default());
        router
            .generate(SingleIn::new(multimodal_request()), downstream.clone())
            .await
            .unwrap();
        downstream
            .request
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .encoder_result
            .and_then(|result| result["worker_id"].as_u64())
    }

    #[tokio::test]
    async fn committed_encoder_admission_survives_membership_changes_and_endpoint_reuse() {
        // Each request-plane server needs a live runtime for its process-wide
        // accept loop, independent of the surrounding test harness's runtimes.
        const TEST: &str = concat!(
            module_path!(),
            "::committed_encoder_admission_survives_membership_changes_and_endpoint_reuse"
        );
        let test_name = TEST.split_once("::").unwrap().1;
        if std::env::var("DYNAMO_ROUTING_HOP_TEST").as_deref() != Ok(test_name) {
            let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", test_name, "--nocapture"])
                .env("DYNAMO_ROUTING_HOP_TEST", test_name)
                .env("DYN_TCP_RPC_HOST", "127.0.0.1")
                .env("DYN_TCP_RPC_PORT", "0")
                .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
                .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
                .kill_on_drop(true);
            let output = tokio::time::timeout(Duration::from_secs(30), child.output())
                .await
                .expect("encoder subprocess must finish within its deadline")
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let runtime = Runtime::from_current().unwrap();
        let store = tempfile::tempdir().unwrap();
        let config = || DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(store.path().into())),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            response_plane: None,
            event_transport_kind: EventTransportKind::Zmq,
        };
        let namespace = format!("encoder-admission-{}", uuid::Uuid::new_v4());
        let mut workers = Vec::new();
        let mut runtimes = Vec::new();
        let encode_calls = Arc::new(AtomicUsize::new(0));
        let (release_tx, mut releases) = tokio::sync::mpsc::unbounded_channel();
        for _ in 0..3 {
            let drt = DistributedRuntime::new(runtime.clone(), config())
                .await
                .unwrap();
            let endpoint = drt
                .namespace(namespace.clone())
                .unwrap()
                .component("workers")
                .unwrap()
                .endpoint("encode");
            let worker = endpoint
                .endpoint_builder()
                .handler(
                    Ingress::for_engine(Arc::new(EncodeWorker(
                        drt.connection_id(),
                        encode_calls.clone(),
                        release_tx.clone(),
                    )))
                    .unwrap(),
                )
                .start_with_registration()
                .await
                .unwrap();
            workers.push(worker);
            runtimes.push(drt);
        }
        let drt = DistributedRuntime::new(runtime.clone(), config())
            .await
            .unwrap();
        let endpoint = drt
            .namespace(namespace.clone())
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("encode");
        let raw_client = endpoint.client().await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut instances = raw_client.instance_source.as_ref().clone();
            while instances.borrow_and_update().len() != 3 {
                instances.changed().await.unwrap();
            }
        })
        .await
        .expect("all three workers must be visible in raw discovery");
        let ids: Vec<_> = workers
            .iter()
            .map(|worker| worker.instance().id())
            .collect();
        let (admissions, admitted_ids) = watch::channel(vec![ids[0]]);
        let card = Arc::new(ModelDeploymentCard::with_name_only("model"));
        let group_id = endpoint.id().to_string();
        let target = |generation, admitted_ids| {
            WorkerSetTarget::Committed(CommittedWorkerSetTarget {
                endpoint: endpoint.clone(),
                group: endpoint.id().to_string(),
                generation,
                card: card.clone(),
                admitted_ids,
            })
        };
        let router = EncoderRouter::new("model".into(), namespace);
        router.set_target(Some(target(1, admitted_ids)));
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(worker) = encoded_worker(&router).await {
                    assert_eq!(worker, ids[0]);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("committed encoder must activate");
        assert_eq!(
            router.available_worker_ids_for(&group_id, &endpoint.id()),
            Some(HashSet::from([ids[0]]))
        );
        assert!(
            router
                .available_worker_ids_for("other-group", &endpoint.id())
                .is_none()
        );
        let calls_before = encode_calls.load(Ordering::SeqCst);
        let downstream = Arc::new(CaptureEngine::default());
        let mut runtime_request = multimodal_request();
        runtime_request.routing_mut().lora_resolution_version =
            Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION);
        let result = router
            .generate(SingleIn::new(runtime_request.clone()), downstream.clone())
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("runtime LoRA must not dispatch to an encoder"),
        };
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::InvalidArgument],
            &[]
        ));
        assert!(
            error
                .to_string()
                .contains("runtime_lora_encoder_unsupported")
        );
        assert_eq!(encode_calls.load(Ordering::SeqCst), calls_before);
        assert!(downstream.request.lock().unwrap().is_none());

        runtime_request.multi_modal_data = None;
        router
            .generate(SingleIn::new(runtime_request), downstream.clone())
            .await
            .unwrap();
        assert_eq!(encode_calls.load(Ordering::SeqCst), calls_before);
        let text_request = downstream.request.lock().unwrap().take().unwrap();
        assert_eq!(
            text_request.routing.unwrap().lora_resolution_version,
            Some(crate::lora::runtime::RUNTIME_LORA_PROTOCOL_VERSION)
        );
        assert!(text_request.encoder_result.is_none());
        assert_eq!(encoded_worker(&router).await, Some(ids[0]));
        assert_eq!(encode_calls.load(Ordering::SeqCst), calls_before + 1);
        for _ in 0..6 {
            assert_eq!(encoded_worker(&router).await, Some(ids[0]));
        }

        admissions.send_replace(vec![ids[0], ids[1]]);
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut seen = HashSet::new();
            while seen.len() < 2 {
                let worker = encoded_worker(&router).await.unwrap();
                assert!(
                    ids[..2].contains(&worker),
                    "rejected encoder received a request"
                );
                seen.insert(worker);
            }
        })
        .await
        .expect("compatible members must become reachable through the admission channel");

        // A transport timeout can leave a live consumer; a refusal and a
        // terminal cancellation must each revoke the exact producer receipt.
        for (kind, revoke) in [
            (Some(ErrorType::ConnectionTimeout), false),
            (Some(ErrorType::InvalidRequest), true),
            (None, true),
        ] {
            let request = SingleIn::new(multimodal_request());
            let id = request.id().to_owned();
            let output = router.generate(request, Arc::new(RouteOutcome(kind))).await;
            if let Ok(stream) = output {
                stream.collect::<Vec<_>>().await;
            }
            if revoke {
                let (worker, released_id) =
                    tokio::time::timeout(Duration::from_secs(2), releases.recv())
                        .await
                        .unwrap()
                        .unwrap();
                assert_eq!(released_id, id);
                assert!(ids[..2].contains(&worker));
            } else {
                assert!(
                    tokio::time::timeout(Duration::from_millis(100), releases.recv())
                        .await
                        .is_err()
                );
            }
        }
        let request = SingleIn::new(multimodal_request());
        let caller = request.context();
        let response = router
            .generate(request, Arc::new(CaptureEngine::default()))
            .await
            .unwrap();
        caller.stop();
        drop(response);
        let (_, released_id) = tokio::time::timeout(Duration::from_secs(2), releases.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(released_id, caller.id());
        assert!(releases.try_recv().is_err());

        let retired = router.binding.load_full().unwrap();
        admissions.send_replace(Vec::new());
        drop(admissions);
        router.set_target(None);
        assert!(
            router
                .available_worker_ids_for(&group_id, &endpoint.id())
                .is_none()
        );
        let (_successor_admissions, successor_ids) = watch::channel(vec![ids[2]]);
        router.set_target(Some(target(2, successor_ids)));
        assert_eq!(
            router.available_worker_ids_for(&group_id, &endpoint.id()),
            Some(HashSet::new())
        );
        assert!(
            router
                .available_worker_ids_for("other-group", &endpoint.id())
                .is_none()
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(worker) = encoded_worker(&router).await {
                    assert_eq!(worker, ids[2]);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("same-endpoint successor must activate with its own admission");
        assert_eq!(
            router.available_worker_ids_for(&group_id, &endpoint.id()),
            Some(HashSet::from([ids[2]]))
        );
        for _ in 0..6 {
            assert_eq!(encoded_worker(&router).await, Some(ids[2]));
        }
        assert!(
            tokio::time::timeout(
                Duration::from_secs(5),
                retired
                    .router
                    .generate(SingleIn::new(multimodal_request()),)
            )
            .await
            .expect("retired admission must fail promptly")
            .is_err()
        );

        drop(router);
        for worker in workers {
            worker.shutdown().await.unwrap();
        }
        runtime.shutdown();
    }

    #[tokio::test]
    async fn binding_availability_tracks_local_inhibition_and_cancellation() {
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::collections::HashSet;

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("encoder-availability")
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate");
        let router = EncoderRouter::disabled();
        assert!(
            router
                .available_worker_ids_for("legacy", &endpoint.id())
                .is_none()
        );
        let binding = Arc::new(
            EncoderRouter::build(
                WorkerSetTarget::Legacy(endpoint.clone()),
                CancellationToken::new(),
            )
            .await
            .unwrap(),
        );
        binding
            .router
            .client
            .override_discovered_instances(vec![1, 2]);
        router.binding.store(Some(binding.clone()));
        router
            .lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        assert_eq!(
            router.available_worker_ids_for("legacy", &endpoint.id()),
            Some(HashSet::from([1, 2]))
        );
        binding.router.client.report_instance_down(1);
        assert_eq!(
            router.available_worker_ids_for("legacy", &endpoint.id()),
            Some(HashSet::from([2]))
        );
        let mut other = endpoint.id();
        other.name = "other".into();
        assert!(router.available_worker_ids_for("legacy", &other).is_none());
        router.cancel_token.cancel();
        assert_eq!(
            router.available_worker_ids_for("legacy", &endpoint.id()),
            Some(HashSet::new())
        );
        runtime.shutdown();
    }

    #[tokio::test]
    async fn binding_availability_waits_for_rebind_publication() {
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::{sync::mpsc, time::Duration};

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let component = distributed
            .namespace("encoder-rebind")
            .unwrap()
            .component("workers")
            .unwrap();
        let original = component.endpoint("original");
        let replacement = Arc::new(
            EncoderRouter::build(
                WorkerSetTarget::Legacy(component.endpoint("replacement")),
                CancellationToken::new(),
            )
            .await
            .unwrap(),
        );
        let router = EncoderRouter::disabled();
        let binding = Arc::new(
            EncoderRouter::build(
                WorkerSetTarget::Legacy(original.clone()),
                CancellationToken::new(),
            )
            .await
            .unwrap(),
        );
        binding.router.client.override_discovered_instances(vec![1]);
        router.binding.store(Some(binding));
        router
            .lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        let mut target = router.target.lock();
        *target = Some(WorkerSetTargetId::Legacy(original.id()));
        let (started_tx, started_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let reader_router = router.clone();
        let reader = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            result_tx
                .send(reader_router.available_worker_ids_for("legacy", &original.id()))
                .unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            result_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        *target = Some(replacement.target_id.clone());
        router.binding.store(Some(replacement));
        router
            .lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        drop(target);
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            None
        );
        reader.join().unwrap();
        runtime.shutdown();
    }

    #[tokio::test]
    async fn pending_activation_does_not_keep_router_alive() {
        let router = EncoderRouter::new("model".into(), "namespace".into());
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
    async fn required_encoder_absence_is_unavailable_without_dispatching_media() {
        let router = EncoderRouter::new_inner("model".into(), "namespace".into(), true, None);
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(multimodal_request()), downstream.clone())
            .await
            .expect_err("required encoder is still pending");
        assert!(error.is::<RequiredEncoderStage>());
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[ErrorType::WorkerUnavailable],
            &[],
        ));
        assert!(downstream.request.lock().unwrap().is_none());

        let mut text = multimodal_request();
        text.multi_modal_data = None;
        router
            .generate(SingleIn::new(text), downstream.clone())
            .await
            .unwrap();
        assert!(downstream.request.lock().unwrap().is_some());
    }

    #[test]
    fn only_local_empty_pool_proves_no_consumer_was_dispatched() {
        let unavailable = || {
            DynamoError::builder()
                .error_type(ErrorType::Unavailable)
                .message("no workers")
        };
        let local = unavailable()
            .local_cause(PipelineError::NoRoutableWorkers)
            .build();
        let remote = unavailable().build();
        assert!(definitively_rejected(&local));
        assert!(!definitively_rejected(&remote));
        assert_eq!(
            serde_json::to_value(local).unwrap(),
            serde_json::to_value(remote).unwrap()
        );
    }

    #[tokio::test]
    async fn consumes_object_shaped_encode_terminal() {
        let output = LLMEngineOutput::encode_terminal(
            json!({"schema_version": 1}).as_object().unwrap().clone(),
        );
        let (result, _, _) =
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

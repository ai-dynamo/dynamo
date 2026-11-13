// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional multimodal encoder hop for token-serving pipelines.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use anyhow::{Context as _, Result};
use arc_swap::ArcSwapOption;
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    error::{DynamoError, ErrorClass},
    pipeline::{
        Context, ManyOut, Operator, PushRouter, ResponseStream, RouterMode, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated, maybe_error::MaybeError},
};

use crate::discovery::{WorkerSetTarget, WorkerSetTargetId};
use crate::protocols::common::{
    FinishReason,
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::TraceLink,
};

mod cleanup;
use cleanup::EncoderResultCleanup;

type EncodePushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

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

/// Forward-only operator that optionally runs a multimodal Encode worker.
///
/// The router is present on every token pipeline but remains a passthrough
/// until discovery supplies an Encode endpoint for the same model namespace.
/// Encode workers are selected round-robin independently of the downstream
/// token router mode; they do not participate in KV-aware routing.
pub struct EncoderRouter {
    binding: ArcSwapOption<EncoderBinding>,
    target: Mutex<Option<WorkerSetTargetId>>,
    target_tx: Option<watch::Sender<Option<WorkerSetTarget>>>,
    cancel_token: CancellationToken,
    lifecycle: AtomicU8,
    required: AtomicBool,
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
            required: AtomicBool::new(false),
            model_name: String::new(),
            namespace: String::new(),
        })
    }

    /// Create a router whose endpoint is driven by committed discovery topology.
    pub fn new(model_name: String, namespace: String) -> Arc<Self> {
        Self::new_inner(model_name, namespace, None)
    }

    pub(crate) fn new_with_task_guard(
        model_name: String,
        namespace: String,
        task_guard: dynamo_runtime::engine::EngineContextGuard,
    ) -> Arc<Self> {
        Self::new_inner(model_name, namespace, Some(task_guard))
    }

    fn new_inner(
        model_name: String,
        namespace: String,
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
            required: AtomicBool::new(false),
            model_name,
            namespace,
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

    pub(crate) fn set_required(&self, required: bool) {
        self.required.store(required, Ordering::Release);
    }

    pub(crate) fn is_required(&self) -> bool {
        self.required.load(Ordering::Acquire)
    }

    fn unavailable(&self, message: &str) -> anyhow::Error {
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message(format!(
                "Required encoder for model {} in namespace {}: {message}",
                self.model_name, self.namespace
            ))
            .build()
            .into()
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
            && request.get_annotation_value("query_instance_id").is_none()
            && request.encoder_result.is_none()
            && request
                .multi_modal_data
                .as_ref()
                .is_some_and(|media| media.values().any(|items| !items.is_empty()))
    }

    async fn consume_encode_stream(
        mut response: ManyOut<Annotated<LLMEngineOutput>>,
        mut cleanup: Option<(&mut EncoderResultCleanup, &str)>,
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
                if let Some((cleanup, model)) = cleanup.as_mut() {
                    cleanup.observe(model, &output)?;
                }
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
        let required = self.is_required();
        if self.lifecycle_state() != EncoderLifecycleState::Active {
            if required {
                return Err(self.unavailable("no active encoder is available"));
            }
            return next.generate(context.map(|_| request)).await;
        }

        let mut encode_request = request.clone();
        if let Some(routing) = &mut encode_request.routing {
            routing.backend_instance_id = None;
            routing.prefill_worker_id = None;
            routing.decode_worker_id = None;
            routing.dp_rank = None;
            routing.prefill_dp_rank = None;
        }
        let encode_context = Context::with_id_and_metadata(
            encode_request,
            context.id().to_string(),
            context.metadata().clone(),
        );
        let encode_result = async {
            let binding = self
                .binding
                .load_full()
                .ok_or_else(|| self.unavailable("encoder binding changed during activation"))?;
            let (worker_id, response) = binding
                .router
                .select_and_dispatch(encode_context, |_, worker_id| Ok(worker_id))
                .await?;
            let mut cleanup = EncoderResultCleanup {
                router: binding.router.clone(),
                worker_id,
                request: None,
            };
            let (result, link) =
                Self::consume_encode_stream(response, Some((&mut cleanup, &request.model))).await?;
            Ok::<_, anyhow::Error>((result, link, cleanup))
        }
        .await;

        let cleanup = match encode_result {
            Ok((encoder_result, worker_link, cleanup)) => {
                request.encoder_result = Some(encoder_result);
                request.migration_link = worker_link;
                cleanup
            }
            Err(error) => {
                if required {
                    return Err(error.context("Required encoder hop failed"));
                }
                tracing::error!(
                    %error,
                    model = %self.model_name,
                    namespace = %self.namespace,
                    "Encoder hop failed; falling back to downstream inline encoding"
                );
                return next.generate(context.map(|_| request)).await;
            }
        };
        let mut response = next.generate(context.map(|_| request)).await?;
        let response_context = response.context();
        let stream = async_stream::stream! {
            let mut cleanup = cleanup;
            while let Some(item) = response.next().await {
                if item.data.as_ref().is_some_and(|output| {
                    !output.token_ids.is_empty()
                        || matches!(output.finish_reason, Some(FinishReason::EoS | FinishReason::Length | FinishReason::Stop))
                }) {
                    // Tokens or successful completion prove the consumer claimed the handoff.
                    cleanup.request = None;
                }
                yield item;
            }
        };
        Ok(ResponseStream::new(Box::pin(stream), response_context))
    }
}

#[cfg(test)]
mod tests;

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional multimodal encoder hop for token-serving pipelines.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};
use futures::StreamExt;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    engine::AsyncEngine,
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, ServerStreamingEngine, SingleIn,
        async_trait,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::protocols::common::{
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::TraceLink,
};

type EncodePushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

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
    router: OnceLock<Arc<EncodePushRouter>>,
    cancel_token: CancellationToken,
    lifecycle: AtomicU8,
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
            router: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            model_name: String::new(),
            namespace: String::new(),
        })
    }

    /// Create a router that activates when discovery observes an Encode peer.
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_name: String,
        namespace: String,
    ) -> Arc<Self> {
        let cancel_token = CancellationToken::new();
        let router = Arc::new(Self {
            router: OnceLock::new(),
            cancel_token: cancel_token.clone(),
            lifecycle: AtomicU8::new(EncoderLifecycleState::Pending as u8),
            model_name,
            namespace,
        });

        let router_weak = Arc::downgrade(&router);
        tokio::spawn(async move {
            tokio::select! {
                result = activation_rx => {
                    let Ok(endpoint) = result else {
                        tracing::debug!("Encoder router activation channel closed");
                        return;
                    };
                    let Some(router) = router_weak.upgrade() else {
                        tracing::debug!("Encoder router dropped before activation");
                        return;
                    };
                    if let Err(error) = router.activate(endpoint).await {
                        tracing::error!(%error, "Failed to activate encoder router");
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Encoder router activation cancelled");
                }
            }
        });

        router
    }

    async fn activate(&self, endpoint: Endpoint) -> Result<()> {
        let client = endpoint.client().await?;
        let router =
            EncodePushRouter::from_client_with_monitor(client, RouterMode::RoundRobin, None)
                .await?;
        let _ = self.router.set(Arc::new(router));
        self.lifecycle
            .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        tracing::info!(
            model = %self.model_name,
            namespace = %self.namespace,
            "Encoder router activated"
        );
        Ok(())
    }

    fn lifecycle_state(&self) -> EncoderLifecycleState {
        EncoderLifecycleState::load(self.lifecycle.load(Ordering::Acquire))
    }

    pub fn deactivate(&self) {
        if self.router.get().is_some() {
            self.lifecycle
                .store(EncoderLifecycleState::Unavailable as u8, Ordering::Release);
        }
    }

    pub fn reactivate(&self) {
        if self.router.get().is_some() {
            self.lifecycle
                .store(EncoderLifecycleState::Active as u8, Ordering::Release);
        }
    }

    pub fn is_deactivated(&self) -> bool {
        self.lifecycle_state() == EncoderLifecycleState::Unavailable
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
        if self.lifecycle_state() != EncoderLifecycleState::Active || !Self::should_encode(&request)
        {
            return next.generate(context.map(|_| request)).await;
        }

        let router = self
            .router
            .get()
            .context("Encoder router is active but not initialized")?;
        let encode_context = Context::with_id_and_metadata(
            request.clone(),
            context.id().to_string(),
            context.metadata().clone(),
        );
        let response = router.generate(encode_context).await?;
        let (encoder_result, worker_link) = Self::consume_encode_stream(response).await?;

        // Once the Encode worker has emitted a transfer handle, always hand it
        // to the downstream worker even if the caller disconnected. The
        // receiver owns transfer completion and buffer release.
        request.encoder_result = Some(encoder_result);
        request.migration_link = worker_link;
        next.generate(context.map(|_| request)).await
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::pipeline::{ResponseStream, context::Controller};

    use super::*;

    fn stream_of(items: Vec<Annotated<LLMEngineOutput>>) -> ManyOut<Annotated<LLMEngineOutput>> {
        ResponseStream::new(
            Box::pin(stream::iter(items)),
            Arc::new(Controller::default()),
        )
    }

    #[tokio::test]
    async fn pending_activation_does_not_keep_router_alive() {
        let (_activation_tx, activation_rx) = oneshot::channel();
        let router = EncoderRouter::new(activation_rx, "model".into(), "namespace".into());
        let weak = Arc::downgrade(&router);

        drop(router);

        assert!(weak.upgrade().is_none());
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

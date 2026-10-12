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
    engine::AsyncEngine,
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, ServerStreamingEngine, SingleIn,
        async_trait,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::discovery::{WorkerSetTarget, WorkerSetTargetId};
use crate::protocols::common::{
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::TraceLink,
};

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
        let router =
            EncodePushRouter::from_client_with_monitor(client, RouterMode::RoundRobin, None)
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
                request.encoder_result = Some(encoder_result);
                request.migration_link = worker_link;
            }
            Err(error) => {
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
    use std::{
        collections::{HashMap, HashSet},
        sync::Mutex,
        time::Duration,
    };

    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::EventTransportKind,
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        engine::AsyncEngineContextProvider,
        pipeline::{Error, ResponseStream, context::Controller, network::Ingress},
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

    struct EncodeWorker(u64);

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for EncodeWorker
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let output = LLMEngineOutput::encode_terminal(
                json!({"worker_id": self.0}).as_object().unwrap().clone(),
            );
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
                .handler(Ingress::for_engine(Arc::new(EncodeWorker(drt.connection_id()))).unwrap())
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

        let retired = router.binding.load_full().unwrap();
        admissions.send_replace(Vec::new());
        drop(admissions);
        router.set_target(None);
        let (_successor_admissions, successor_ids) = watch::channel(vec![ids[2]]);
        router.set_target(Some(target(2, successor_ids)));
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

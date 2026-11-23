// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Namespace admission exercised through real discovery, TCP workers, and HTTP requests.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::discovery::{ModelManager, ModelWatcher};
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::namespace::{NamespaceFilter, NamespacePrefixMode};
use dynamo_llm::protocols::Annotated;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::component::StartedEndpoint;
use dynamo_runtime::discovery::{
    DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventTransportKind,
};
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::storage::kv;
use dynamo_runtime::{CancellationToken, DistributedRuntime, Runtime};
use reqwest::StatusCode;
use serde_json::{Value, json};

#[path = "common/ports.rs"]
mod ports;

const WINDOW: Duration = Duration::from_secs(10);

// The shared TCP listener is process-global; each test needs its own lifetime.
async fn in_fresh_process(name: &str) -> bool {
    if std::env::var("DYNAMO_NAMESPACE_SCOPE_TEST").as_deref() == Ok(name) {
        return false;
    }
    let output = tokio::time::timeout(
        Duration::from_secs(45),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("DYNAMO_NAMESPACE_SCOPE_TEST", name)
            .env("DYN_TCP_RPC_HOST", "127.0.0.1")
            .env("DYN_TCP_RPC_PORT", "0")
            .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
            .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("namespace test subprocess must finish")
    .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
        "subprocess must execute the selected test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    true
}

struct EchoWorker(&'static str);

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        anyhow::Error,
    > for EchoWorker
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>> {
        let (request, context) = request.transfer(());
        let response = serde_json::from_value(json!({
            "id": "namespace-echo",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": request.inner.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": self.0}, "finish_reason": "stop"}],
        }))?;
        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(async move {
                Annotated::from_data(response)
            })),
            context.context(),
        ))
    }
}

fn config() -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::Memory),
        nats_config: None,
        request_plane: RequestPlaneMode::Tcp,
        response_plane: None,
        event_transport_kind: EventTransportKind::Zmq,
    }
}

struct Worker {
    drt: DistributedRuntime,
    serving: StartedEndpoint,
    registration: DiscoveryInstance,
}

impl Worker {
    async fn start(
        runtime: &DistributedRuntime,
        namespace: &str,
        model: &str,
        origin: &'static str,
    ) -> Self {
        let drt = runtime.clone();
        let endpoint = drt
            .namespace(namespace)
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate");
        let serving = endpoint
            .endpoint_builder()
            .handler(Ingress::for_engine(Arc::new(EchoWorker(origin))).unwrap())
            .start_with_registration()
            .await
            .unwrap();
        let mut card = ModelDeploymentCard::with_name_only(model);
        card.model_input = ModelInput::Text;
        card.model_type = ModelType::Chat;
        card.worker_type = Some(WorkerType::Aggregated);
        card.source_path = Some(format!("namespace-fixture/{model}"));
        let registration = drt
            .discovery()
            .register(
                DiscoverySpec::from_model(
                    namespace.into(),
                    "workers".into(),
                    "generate".into(),
                    &card,
                )
                .unwrap(),
            )
            .await
            .unwrap();
        Self {
            drt,
            serving,
            registration,
        }
    }

    async fn shutdown(self) {
        self.drt
            .discovery()
            .unregister(self.registration)
            .await
            .unwrap();
        self.serving.shutdown().await.unwrap();
    }
}

struct Frontend {
    base_url: String,
    client: reqwest::Client,
    manager: Arc<ModelManager>,
    cancel: CancellationToken,
    watch: tokio::task::JoinHandle<()>,
    http: tokio::task::JoinHandle<Result<()>>,
}

impl Frontend {
    async fn start(
        runtime: &DistributedRuntime,
        scope: NamespaceFilter,
        prefix_mode: NamespacePrefixMode,
    ) -> Self {
        let drt = runtime.clone();
        let (listener, port) = ports::bind_random_port().await;
        let service = HttpService::builder()
            .host("127.0.0.1")
            .port(port)
            .enable_chat_endpoints(true)
            .build()
            .unwrap();
        let manager = service.state().manager_clone();
        let cancel = CancellationToken::new();
        let stream = drt
            .discovery()
            .list_and_watch(DiscoveryQuery::AllModels, Some(cancel.clone()))
            .await
            .unwrap();
        let mut watcher = ModelWatcher::new(
            drt,
            manager.clone(),
            RouterConfig {
                router_mode: dynamo_runtime::pipeline::RouterMode::RoundRobin,
                ..Default::default()
            },
            0,
            None,
            None,
            None,
            service.state().metrics_clone(),
        );
        watcher.set_namespace_prefix_mode(prefix_mode);
        let watcher = Arc::new(watcher);
        let watch = tokio::spawn(watcher.watch(stream, scope));
        let http = service.spawn_with_listener(cancel.clone(), listener).await;
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(WINDOW)
                .build()
                .unwrap(),
            manager,
            cancel,
            watch,
            http,
        }
    }

    async fn wait_namespaces(&self, name: &str, namespaces: &[&str]) {
        tokio::time::timeout(WINDOW, async {
            loop {
                if let Some(model) = self.manager.get_model(name)
                    && self.manager.is_model_ready_to_serve(name)
                    && model.distinct_namespaces_sorted() == namespaces
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("expected worker namespaces must become ready");
    }

    async fn chat(&self, model: &str) -> (StatusCode, Value) {
        let response = self.client.post(format!("{}/v1/chat/completions", self.base_url)).json(&json!({
            "model": model, "messages": [{"role": "user", "content": "identify worker"}], "stream": false, "max_tokens": 1,
        })).send().await.unwrap();
        let status = response.status();
        (status, response.json().await.unwrap())
    }

    async fn assert_origin(&self, model: &str, origin: &str) {
        let (status, response) = self.chat(model).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(
            response["choices"][0]["message"]["content"], origin,
            "{response}"
        );
    }

    async fn shutdown(self) {
        self.cancel.cancel();
        tokio::time::timeout(WINDOW, self.watch)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(WINDOW, self.http)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn namespace_scope_controls_http_routing_and_worker_rollover() {
    if in_fresh_process("namespace_scope_controls_http_routing_and_worker_rollover").await {
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    let base = Worker::start(&drt, "default-foo", "foo-model", "base").await;
    let sibling = Worker::start(&drt, "default-foo-bar", "bar-model", "sibling").await;
    let legacy = Worker::start(&drt, "default-foo-legacy", "legacy-model", "legacy").await;
    let literal = NamespaceFilter::from_namespace_and_prefix(None, Some("default-foo"));
    let strict =
        Frontend::start(&drt, literal.clone(), NamespacePrefixMode::WorkerGeneration).await;
    let manual = Frontend::start(&drt, literal, NamespacePrefixMode::Literal).await;
    let exact = Frontend::start(
        &drt,
        NamespaceFilter::Exact("default-foo".into()),
        NamespacePrefixMode::WorkerGeneration,
    )
    .await;

    strict.wait_namespaces("foo-model", &["default-foo"]).await;
    strict
        .wait_namespaces("legacy-model", &["default-foo-legacy"])
        .await;
    manual
        .wait_namespaces("bar-model", &["default-foo-bar"])
        .await;
    exact.wait_namespaces("foo-model", &["default-foo"]).await;
    strict.assert_origin("foo-model", "base").await;
    strict.assert_origin("legacy-model", "legacy").await;
    manual.assert_origin("bar-model", "sibling").await;
    exact.assert_origin("foo-model", "base").await;
    assert_eq!(strict.chat("bar-model").await.0, StatusCode::NOT_FOUND);
    assert_eq!(exact.chat("legacy-model").await.0, StatusCode::NOT_FOUND);

    let generation = Worker::start(&drt, "default-foo-1a2b3c4d", "foo-model", "generation").await;
    strict
        .wait_namespaces("foo-model", &["default-foo", "default-foo-1a2b3c4d"])
        .await;
    exact.wait_namespaces("foo-model", &["default-foo"]).await;
    exact.assert_origin("foo-model", "base").await;
    base.shutdown().await;
    strict
        .wait_namespaces("foo-model", &["default-foo-1a2b3c4d"])
        .await;
    for _ in 0..3 {
        strict.assert_origin("foo-model", "generation").await;
        assert_eq!(strict.chat("bar-model").await.0, StatusCode::NOT_FOUND);
    }

    generation.shutdown().await;
    legacy.shutdown().await;
    sibling.shutdown().await;
    strict.shutdown().await;
    manual.shutdown().await;
    exact.shutdown().await;
    runtime.shutdown();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a client gets from the real HTTP service when the frontend's discovery
//! scope, rather than the card itself, decides whether a model exists.
//!
//! The status code is decided above `Model::get_chat_engine`: the chat handler
//! runs the per-model readiness gate first, so only a model name that is absent
//! from the catalog reaches the engine lookup and answers 404. A case that keeps
//! a name out of the catalog therefore has to read the status off the socket,
//! not off an engine accessor.
//!
//! The shape driven here is the GlobalRouter's own registration
//! (`components/src/dynamo/global_router/__main__.py`, `_serve_disagg`): in one
//! namespace, a `Tokens` + `Prefill` card that carries no OpenAI surface, and a
//! `Tokens` + `Chat|Completions` card that is the only one that can carry the
//! chat engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::discovery::{ModelManager, ModelWatcher};
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::namespace::NamespaceFilter;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryInstance};
use dynamo_runtime::{
    CancellationToken, DistributedRuntime, Runtime, distributed::DistributedConfig,
};
use futures::StreamExt;
use reqwest::StatusCode;

#[path = "common/ports.rs"]
mod ports;
use ports::bind_random_port;

const MODEL: &str = "global-router-model";
/// `config.json` plus a loadable `tokenizer.json` and no weight files: what a
/// registration with `ignore_weights=true` leaves in the cache.
const SNAPSHOT_WITH_TOKENIZER: &str = "mock-llama-3.1-8b-instruct";

const COMMIT_WINDOW: Duration = Duration::from_secs(5);

/// A frontend scoped to a sibling deployment's prefix must answer 404, not serve
/// that deployment's workers under its model name. See [`NamespaceFilter::matches`]
/// for why a bare `starts_with` crosses the boundary.
#[tokio::test]
async fn chat_completions_answers_404_for_a_sibling_prefix_namespace() {
    let frontend = Frontend::start(
        NamespaceFilter::Prefix("myns-dgd".to_string()),
        "myns-dgd2",
        SNAPSHOT_WITH_TOKENIZER,
    )
    .await;
    assert!(
        !frontend.model_committed_within(COMMIT_WINDOW).await,
        "a sibling deployment's namespace must not reach the catalog"
    );
    assert_eq!(frontend.chat_status().await, StatusCode::NOT_FOUND);
    frontend.shutdown().await;
}

/// The real HTTP service, fed by the real model watcher over an in-process
/// discovery stream. No message bus, no GPU.
struct Frontend {
    base_url: String,
    client: reqwest::Client,
    manager: Arc<ModelManager>,
    runtime: Runtime,
    cancel: CancellationToken,
    events: Option<DiscoverySender>,
    watch: tokio::task::JoinHandle<()>,
    http: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl Frontend {
    /// Serve `namespace`'s GlobalRouter card pair through a frontend scoped by
    /// `namespace_filter`. Returns once the watcher has applied both discovery
    /// events; whether either card commits is what the caller measures.
    async fn start(namespace_filter: NamespaceFilter, namespace: &str, snapshot: &str) -> Self {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();

        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .enable_chat_endpoints(true)
            .build()
            .unwrap();
        let manager = service.state().manager_clone();

        let watcher = Arc::new(ModelWatcher::new(
            drt.clone(),
            manager.clone(),
            RouterConfig::default(),
            0,
            None,
            None,
            None,
            service.state().metrics_clone(),
        ));
        let (events, stream) = discovery_stream();
        let watch = tokio::spawn(watcher.watch(stream, namespace_filter));

        let cancel = CancellationToken::new();
        let http = service.spawn_with_listener(cancel.clone(), listener).await;

        let source = global_router_card(snapshot);
        for (endpoint, model_type, worker_type, needs) in global_router_card_shapes() {
            drt.namespace(namespace)
                .unwrap()
                .component("workers")
                .unwrap()
                .endpoint(endpoint)
                .register_endpoint_instance()
                .await
                .unwrap();
            let mut card = source.clone();
            card.model_type = model_type;
            card.worker_type = Some(worker_type);
            card.needs = needs;
            apply_discovery_event(
                &events,
                DiscoveryEvent::Added(DiscoveryInstance::Model {
                    namespace: namespace.to_string(),
                    component: "workers".to_string(),
                    endpoint: endpoint.to_string(),
                    instance_id: drt.discovery().instance_id(),
                    card_json: serde_json::to_value(&card).unwrap(),
                    model_suffix: None,
                }),
            )
            .await;
        }

        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            manager,
            runtime,
            cancel,
            events: Some(events),
            watch,
            http,
        }
    }

    /// Whether any WorkerSet of the model commits within `window`. Worker sets
    /// are built off the discovery thread, so absence has to be waited out
    /// rather than read once.
    async fn model_committed_within(&self, window: Duration) -> bool {
        tokio::time::timeout(window, async {
            while !self.manager.has_registered_model(MODEL) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .is_ok()
    }

    async fn chat_status(&self) -> StatusCode {
        self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": false,
            }))
            .send()
            .await
            .unwrap()
            .status()
    }

    async fn shutdown(mut self) {
        self.cancel.cancel();
        drop(self.events.take());
        self.watch.await.unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.http).await;
        self.runtime.shutdown();
    }
}

/// The GlobalRouter forwards already-tokenized requests, so both of its cards
/// take the `Tokens` branch of worker-set preparation.
fn global_router_card(snapshot: &str) -> ModelDeploymentCard {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models")
        .join(snapshot);
    let mut card = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    card.set_name(MODEL);
    card.model_input = ModelInput::Tokens;
    card
}

#[allow(clippy::type_complexity)]
fn global_router_card_shapes() -> [(&'static str, ModelType, WorkerType, Vec<Vec<WorkerType>>); 2] {
    [
        (
            "prefill_generate",
            ModelType::Prefill,
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
        ),
        (
            "decode_generate",
            ModelType::Chat | ModelType::Completions,
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
        ),
    ]
}

type DiscoverySender =
    tokio::sync::mpsc::UnboundedSender<(DiscoveryEvent, tokio::sync::oneshot::Sender<()>)>;

/// A discovery stream the test drives by hand. Each event carries an
/// acknowledgement that fires when the watcher comes back for the next one, so
/// a case knows the scope decision for an event has already been made.
fn discovery_stream() -> (DiscoverySender, dynamo_runtime::discovery::DiscoveryStream) {
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let stream = futures::stream::unfold(
        (event_rx, None::<tokio::sync::oneshot::Sender<()>>),
        |(mut receiver, applied)| async {
            if let Some(applied) = applied {
                let _ = applied.send(());
            }
            receiver
                .recv()
                .await
                .map(|(event, applied)| (Ok(event), (receiver, Some(applied))))
        },
    )
    .boxed();
    (event_tx, stream)
}

async fn apply_discovery_event(events: &DiscoverySender, event: DiscoveryEvent) {
    let (applied, acknowledged) = tokio::sync::oneshot::channel();
    events.send((event, applied)).unwrap();
    tokio::time::timeout(Duration::from_secs(5), acknowledged)
        .await
        .expect("watcher stopped consuming discovery events")
        .unwrap();
}

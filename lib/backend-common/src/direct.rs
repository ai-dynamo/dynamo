// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host engines this process does not run, for a frontend that dials them directly.
//!
//! In sidecar mode a worker process beside the engine runs an [`LLMEngine`]: it
//! registers the engine with discovery and serves the request plane through
//! [`EngineAdapter`]. In direct mode the frontend runs that same engine
//! implementation and adapter itself. It registers each engine in its own
//! discovery under the engine's own instance id, so model watching, routing and
//! prefill activation run unchanged, and its routers hand each request to
//! [`DirectDispatch`], which calls the engine's adapter in-process.
//!
//! An engine that stops serving ([`LLMEngine::engine_lost`]) leaves discovery and
//! is probed again until it serves, then registered again.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_llm::local_model::model_card_instance;
use dynamo_llm::local_model::runtime_config::DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::{Instance, TransportType};
use dynamo_runtime::discovery::DiscoveryInstance;
use dynamo_runtime::pipeline::{
    AddressedRequest, AsyncEngine, Context, Error, ManyIn, ManyOut, SingleIn, StreamingDispatch,
};
use dynamo_runtime::protocols::annotated::Annotated;
use tokio_util::sync::CancellationToken;

use crate::adapter::EngineAdapter;
use crate::disagg::DisaggregationMode;
use crate::engine::{EngineConfig, KvEventSource, LLMEngine};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::worker::{
    WorkerConfig, build_local_model, resolve_model_type, resolve_worker_type_and_needs,
};

/// Delay before the first probe of an engine that stopped serving.
const REPROBE_INITIAL_DELAY: Duration = Duration::from_secs(1);
/// Upper bound on the delay between probes of an engine that is not serving.
const REPROBE_MAX_DELAY: Duration = Duration::from_secs(30);

/// One engine the frontend dials, as the operator listed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectEngineSpec {
    /// Address of the engine's native API, which also identifies the engine.
    pub url: String,
    pub mode: DisaggregationMode,
}

/// Builds the engine implementation that talks to one listed engine; the same
/// implementation a sidecar for that engine runs.
pub type DirectEngineFactory =
    Arc<dyn Fn(&DirectEngineSpec) -> Result<Arc<dyn LLMEngine>, DynamoError> + Send + Sync>;

/// The routers' final hop in direct mode: the adapter of the engine each
/// request was routed to.
#[derive(Default)]
pub struct DirectDispatch {
    adapters: parking_lot::RwLock<HashMap<u64, Arc<EngineAdapter>>>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for DirectDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (request, _, instance) = addressed.into_parts();
        let instance_id = instance
            .ok_or_else(|| {
                error(
                    ErrorType::CannotConnect,
                    "direct dispatch needs the routed instance",
                )
            })?
            .id();
        let adapter = self
            .adapters
            .read()
            .get(&instance_id)
            .cloned()
            .ok_or_else(|| {
                error(
                    ErrorType::CannotConnect,
                    format!("engine {instance_id} is not serving"),
                )
            })?;
        adapter.generate(Context::rejoin(request, context)).await
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(error(
            ErrorType::InvalidArgument,
            "engines dialed directly take unary requests only",
        )
        .into())
    }
}

/// Probe and register every listed engine, then keep each registered while it
/// serves. Returns once every engine serves; fails if any cannot be started.
///
/// `config` is the worker configuration a sidecar for these engines would run
/// with. Each engine's role comes from its spec, not from `config`.
pub async fn serve_direct_engines(
    drt: &DistributedRuntime,
    config: WorkerConfig,
    specs: Vec<DirectEngineSpec>,
    factory: DirectEngineFactory,
) -> Result<Arc<DirectDispatch>, DynamoError> {
    if specs.is_empty() {
        return Err(error(
            ErrorType::InvalidArgument,
            "direct mode needs at least one engine",
        ));
    }
    let mut urls = HashSet::new();
    for spec in &specs {
        if !urls.insert(normalized_url(&spec.url)) {
            return Err(error(
                ErrorType::InvalidArgument,
                format!("engine {} is listed twice", spec.url),
            ));
        }
    }

    let dispatch = Arc::new(DirectDispatch::default());
    let mut hosts: Vec<(DirectEngineHost, HostedEngine)> = Vec::with_capacity(specs.len());
    for spec in specs {
        let host = DirectEngineHost {
            drt: drt.clone(),
            config: config.clone(),
            instance_id: instance_id_for_url(&spec.url),
            spec,
            factory: factory.clone(),
            dispatch: dispatch.clone(),
        };
        // Release the engines started so far if a later one fails.
        let hosted = match host.start().await {
            Ok(hosted) => hosted,
            Err(start_error) => {
                for (host, hosted) in hosts {
                    host.stop(hosted).await;
                }
                return Err(start_error);
            }
        };
        hosts.push((host, hosted));
    }
    let shutdown = drt.primary_token();
    for (host, hosted) in hosts {
        tokio::spawn(host.supervise(hosted, shutdown.clone()));
    }
    Ok(dispatch)
}

/// Stable instance id for an engine, shared by every frontend that lists it.
pub fn instance_id_for_url(url: &str) -> u64 {
    dynamo_runtime::discovery::hash_pod_name(&normalized_url(url))
}

fn normalized_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

/// One listed engine and everything needed to (re)register it.
struct DirectEngineHost {
    drt: DistributedRuntime,
    config: WorkerConfig,
    spec: DirectEngineSpec,
    instance_id: u64,
    factory: DirectEngineFactory,
    dispatch: Arc<DirectDispatch>,
}

/// An engine that serves and the discovery records that publish it.
struct HostedEngine {
    engine: Arc<dyn LLMEngine>,
    registered: Vec<DiscoveryInstance>,
}

impl DirectEngineHost {
    /// Start the engine, make it dispatchable, then publish it.
    async fn start(&self) -> Result<HostedEngine, DynamoError> {
        let engine = (self.factory)(&self.spec)?;
        let started = async {
            let engine_config = engine.start(self.instance_id).await?;
            let (endpoint, model) = self.discovery_records(&engine, &engine_config).await?;
            Ok::<_, DynamoError>((endpoint, model))
        }
        .await;
        let (endpoint, model) = match started {
            Ok(records) => records,
            Err(start_error) => {
                self.cleanup(&engine).await;
                return Err(start_error);
            }
        };

        self.dispatch.adapters.write().insert(
            self.instance_id,
            Arc::new(EngineAdapter::new(engine.clone(), self.spec.mode)),
        );
        let mut hosted = HostedEngine {
            engine,
            registered: Vec::with_capacity(2),
        };
        // The endpoint first, so routers built when the model card appears
        // already see the instance.
        for record in [endpoint, model] {
            match self.drt.discovery().register_instance(record).await {
                Ok(registered) => hosted.registered.push(registered),
                Err(register_error) => {
                    self.stop(hosted).await;
                    return Err(error(
                        ErrorType::Backend(BackendError::Unknown),
                        format!("register engine {}: {register_error}", self.spec.url),
                    ));
                }
            }
        }
        tracing::info!(
            url = %self.spec.url,
            instance_id = self.instance_id,
            mode = ?self.spec.mode,
            "serving engine directly"
        );
        Ok(hosted)
    }

    /// Withdraw the engine from discovery and dispatch, then release it.
    async fn stop(&self, hosted: HostedEngine) {
        for record in hosted.registered.into_iter().rev() {
            if let Err(unregister_error) = self.drt.discovery().unregister(record).await {
                tracing::warn!(url = %self.spec.url, %unregister_error, "engine unregistration failed");
            }
        }
        self.dispatch.adapters.write().remove(&self.instance_id);
        self.cleanup(&hosted.engine).await;
    }

    async fn cleanup(&self, engine: &Arc<dyn LLMEngine>) {
        if let Err(cleanup_error) = engine.cleanup().await {
            tracing::warn!(url = %self.spec.url, %cleanup_error, "engine cleanup failed");
        }
    }

    /// Keep the engine published while it serves; when it stops, withdraw it
    /// and probe until it serves again.
    async fn supervise(self, mut hosted: HostedEngine, shutdown: CancellationToken) {
        loop {
            let engine = hosted.engine.clone();
            let lost = tokio::select! {
                _ = shutdown.cancelled() => None,
                lost = engine.engine_lost() => Some(lost),
            };
            let Some(lost) = lost else {
                self.stop(hosted).await;
                return;
            };
            tracing::warn!(
                url = %self.spec.url,
                reason = %lost,
                "engine stopped serving; removed it from routing"
            );
            self.stop(hosted).await;

            let mut delay = REPROBE_INITIAL_DELAY;
            hosted = loop {
                tokio::select! {
                    _ = shutdown.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }
                match self.start().await {
                    Ok(hosted) => break hosted,
                    Err(probe_error) => {
                        tracing::debug!(url = %self.spec.url, %probe_error, "engine is not serving yet");
                        delay = (delay * 2).min(REPROBE_MAX_DELAY);
                    }
                }
            };
        }
    }

    /// The endpoint instance and model card a sidecar for this engine would
    /// publish, under the engine's own instance id.
    async fn discovery_records(
        &self,
        engine: &Arc<dyn LLMEngine>,
        engine_config: &EngineConfig,
    ) -> Result<(DiscoveryInstance, DiscoveryInstance), DynamoError> {
        let mut config = self.config.clone();
        config.disaggregation_mode = self.spec.mode;
        if config.model_name.is_empty() {
            config.model_name = engine_config.model.clone();
        }
        let model_type = resolve_model_type(&config)?;
        let (worker_type, needs) = resolve_worker_type_and_needs(&config);

        let mut engine_config = engine_config.clone();
        record_kv_event_sources(
            &mut engine_config.runtime_data,
            engine.kv_event_sources().await?,
        )?;
        let local_model = build_local_model(&config, &engine_config, false).await?;
        let mut card = local_model.card().clone();
        card.model_type = model_type;
        card.model_input = config.model_input;
        card.worker_type = Some(worker_type);
        card.needs = needs;

        let component = match self.spec.mode {
            DisaggregationMode::Aggregated => config.component.clone(),
            mode => mode.discovery_component().to_string(),
        };
        let endpoint = DiscoveryInstance::Endpoint(Instance {
            namespace: config.namespace.clone(),
            component: component.clone(),
            endpoint: config.endpoint.clone(),
            instance_id: self.instance_id,
            transport: TransportType::Grpc(self.spec.url.clone()),
            device_type: None,
            request_plane_codec: None,
        });
        let model = model_card_instance(
            config.namespace,
            component,
            config.endpoint,
            &card,
            self.instance_id,
        )
        .map_err(|card_error| {
            error(
                ErrorType::Backend(BackendError::Unknown),
                format!("model card: {card_error}"),
            )
        })?;
        Ok((endpoint, model))
    }
}

/// Name the engine's ZMQ KV-event publishers in its runtime data, by rank, so
/// the frontend subscribes to them itself.
fn record_kv_event_sources(
    runtime_data: &mut HashMap<String, serde_json::Value>,
    sources: Vec<KvEventSource>,
) -> Result<(), DynamoError> {
    let mut endpoints = serde_json::Map::new();
    for source in sources {
        match source {
            KvEventSource::Zmq {
                endpoint, dp_rank, ..
            } => {
                endpoints.insert(dp_rank.to_string(), endpoint.into());
            }
            KvEventSource::Push { dp_rank, .. } => {
                return Err(error(
                    ErrorType::InvalidArgument,
                    format!(
                        "DP rank {dp_rank} pushes KV events in-process; a frontend can only subscribe to ZMQ publishers"
                    ),
                ));
            }
        }
    }
    if endpoints.is_empty() {
        return Ok(());
    }
    runtime_data.insert(
        DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY.to_string(),
        endpoints.into(),
    );
    Ok(())
}

fn error(error_type: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(error_type)
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use dynamo_llm::discovery::{ModelManager, ModelWatcher};
    use dynamo_llm::entrypoint::RouterConfig;
    use dynamo_llm::http::service::metrics::Metrics;
    use dynamo_llm::types::openai::chat_completions::{
        NvCreateChatCompletionRequest, OpenAIChatCompletionsStreamingEngine,
    };
    use dynamo_runtime::Runtime;
    use dynamo_runtime::discovery::DiscoveryQuery;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::namespace::NamespaceFilter;
    use dynamo_runtime::protocols::maybe_error::MaybeError;
    use futures::StreamExt;
    use futures::stream::BoxStream;
    use tokio::sync::Notify;

    use super::*;
    use crate::engine::{GenerateContext, LlmRegistration, chunk};

    const MODEL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llm/tests/data/sample-models/mock-llama-3.1-8b-instruct"
    );
    const A: &str = "http://engine-a:30001";
    const B: &str = "http://engine-b:30001";

    /// What the test controls about one listed engine, across incarnations.
    #[derive(Default)]
    struct EngineState {
        up: AtomicBool,
        served: AtomicUsize,
        lost: Notify,
    }

    /// One incarnation of a listed engine: answers every request with one token.
    /// `fail_next`, shared by every engine, makes the next request any of them
    /// receives fail mid-stream as if its engine went away.
    struct FakeEngine {
        state: Arc<EngineState>,
        fail_next: Arc<AtomicBool>,
    }

    #[async_trait]
    impl LLMEngine for FakeEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            if !self.state.up.load(Ordering::SeqCst) {
                return Err(error(
                    ErrorType::Backend(BackendError::CannotConnect),
                    "engine is down",
                ));
            }
            Ok(EngineConfig {
                model: MODEL.to_string(),
                served_model_name: Some("direct-test-model".to_string()),
                llm: Some(LlmRegistration {
                    kv_cache_block_size: Some(16),
                    context_length: Some(4096),
                    ..LlmRegistration::default()
                }),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Ok(futures::stream::iter([Err(error(
                    ErrorType::Backend(BackendError::EngineShutdown),
                    "engine went away mid-stream",
                ))])
                .boxed());
            }
            self.state.served.fetch_add(1, Ordering::SeqCst);
            Ok(futures::stream::iter([Ok(chunk::token(1)), Ok(LLMEngineOutput::stop())]).boxed())
        }

        async fn engine_lost(&self) -> DynamoError {
            self.state.lost.notified().await;
            error(
                ErrorType::Backend(BackendError::EngineShutdown),
                "engine stopped",
            )
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    /// A frontend serving engines `A` and `B` directly, built the way the HTTP
    /// frontend builds a dynamic one.
    struct Frontend {
        drt: DistributedRuntime,
        namespace: String,
        states: HashMap<&'static str, Arc<EngineState>>,
        fail_next: Arc<AtomicBool>,
        chat: OpenAIChatCompletionsStreamingEngine,
        model: String,
    }

    impl Frontend {
        async fn start(namespace: &str, migration_limit: u32) -> Self {
            let runtime = Runtime::from_current().unwrap();
            let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
                .await
                .unwrap();
            let states: HashMap<&'static str, Arc<EngineState>> = [A, B]
                .into_iter()
                .map(|url| {
                    let state = Arc::new(EngineState::default());
                    state.up.store(true, Ordering::SeqCst);
                    (url, state)
                })
                .collect();
            let fail_next = Arc::new(AtomicBool::new(false));
            let factory_states = states.clone();
            let factory_fail_next = fail_next.clone();
            let factory: DirectEngineFactory = Arc::new(move |spec: &DirectEngineSpec| {
                Ok(Arc::new(FakeEngine {
                    state: factory_states[spec.url.as_str()].clone(),
                    fail_next: factory_fail_next.clone(),
                }) as Arc<dyn LLMEngine>)
            });
            let specs = [A, B]
                .map(|url| DirectEngineSpec {
                    url: url.to_string(),
                    mode: DisaggregationMode::Aggregated,
                })
                .to_vec();
            let config = WorkerConfig {
                namespace: namespace.to_string(),
                ..WorkerConfig::default()
            };
            let dispatch = serve_direct_engines(&drt, config, specs, factory)
                .await
                .unwrap();

            let manager = Arc::new(ModelManager::new());
            manager.set_direct_dispatch(dispatch).unwrap();
            let watcher = Arc::new(ModelWatcher::new(
                drt.clone(),
                manager.clone(),
                RouterConfig::default(),
                migration_limit,
                None,
                None,
                None,
                Arc::new(Metrics::new()),
            ));
            let filter = NamespaceFilter::Exact(namespace.to_string());
            let models = drt
                .discovery()
                .list_and_watch(filter.model_query(), Some(drt.primary_token()))
                .await
                .unwrap();
            tokio::spawn(watcher.clone().watch(models, filter));
            let model =
                tokio::time::timeout(Duration::from_secs(30), watcher.wait_for_chat_model())
                    .await
                    .expect("the hosted engines' model must be served");
            let chat = manager.get_chat_completions_engine(&model).unwrap();
            Self {
                drt,
                namespace: namespace.to_string(),
                states,
                fail_next,
                chat,
                model,
            }
        }

        fn served(&self, url: &str) -> usize {
            self.states[url].served.load(Ordering::SeqCst)
        }

        /// One chat request; `true` when it completed without an error.
        async fn chat(&self) -> bool {
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                    "stream": true,
                }))
                .unwrap();
            match self.chat.generate(Context::new(request)).await {
                Ok(stream) => {
                    let items: Vec<_> = stream.collect().await;
                    !items.is_empty() && items.iter().all(|item| item.err().is_none())
                }
                Err(_) => false,
            }
        }

        /// Send requests until `done` holds, tolerating requests that race a
        /// membership change.
        async fn chat_until(&self, done: impl Fn(&Self) -> bool) {
            tokio::time::timeout(Duration::from_secs(30), async {
                while !done(self) {
                    self.chat().await;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("routing did not reach the expected state");
        }

        async fn listed(&self, url: &str) -> bool {
            let config = WorkerConfig::default();
            let query = DiscoveryQuery::Endpoint {
                namespace: self.namespace.clone(),
                component: config.component,
                endpoint: config.endpoint,
            };
            self.drt
                .discovery()
                .list(query)
                .await
                .unwrap()
                .iter()
                .any(|instance| instance.instance_id() == instance_id_for_url(url))
        }
    }

    #[tokio::test]
    async fn routes_to_hosted_engines_and_follows_their_loss_and_return() {
        let frontend = Frontend::start("direct-membership", 0).await;
        assert_eq!(frontend.model, "direct-test-model");

        // Round robin reaches both engines through the in-process final hop.
        frontend
            .chat_until(|f| f.served(A) > 0 && f.served(B) > 0)
            .await;
        assert!(frontend.chat().await);

        // A lost engine leaves discovery and stops receiving requests.
        let a = frontend.states[A].clone();
        a.up.store(false, Ordering::SeqCst);
        a.lost.notify_one();
        tokio::time::timeout(Duration::from_secs(10), async {
            while frontend.listed(A).await {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the lost engine must leave discovery");
        let (a_before, b_before) = (frontend.served(A), frontend.served(B));
        frontend.chat_until(|f| f.served(B) >= b_before + 4).await;
        assert_eq!(frontend.served(A), a_before);

        // Once it serves again it is probed, registered and routed to again.
        a.up.store(true, Ordering::SeqCst);
        frontend.chat_until(|f| f.served(A) > a_before).await;
        assert!(frontend.listed(A).await);
    }

    #[tokio::test]
    async fn a_request_whose_engine_goes_away_migrates_to_another_engine() {
        let frontend = Frontend::start("direct-migration", 1).await;
        frontend
            .chat_until(|f| f.served(A) > 0 && f.served(B) > 0)
            .await;

        // Whichever engine the next request reaches fails it mid-stream, the
        // way a sidecar reports an engine that went away.
        frontend.fail_next.store(true, Ordering::SeqCst);
        let before = frontend.served(A) + frontend.served(B);
        assert!(frontend.chat().await, "the request must complete elsewhere");
        assert_eq!(frontend.served(A) + frontend.served(B), before + 1);
    }

    #[test]
    fn kv_event_publishers_are_recorded_by_rank() {
        let mut runtime_data = HashMap::new();
        record_kv_event_sources(
            &mut runtime_data,
            (0..2)
                .map(|dp_rank| KvEventSource::Zmq {
                    endpoint: format!("tcp://engine:{}", 5557 + dp_rank),
                    topic: String::new(),
                    dp_rank,
                    image_token_id: None,
                    heartbeat_timeout: None,
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            runtime_data[DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY],
            serde_json::json!({"0": "tcp://engine:5557", "1": "tcp://engine:5558"})
        );

        let push = KvEventSource::Push {
            on_ready: Box::new(|_| Ok(())),
            dp_rank: 0,
        };
        assert!(record_kv_event_sources(&mut HashMap::new(), vec![push]).is_err());
    }

    #[test]
    fn instance_ids_ignore_surrounding_whitespace_and_trailing_slashes() {
        assert_eq!(
            instance_id_for_url("http://engine:30001"),
            instance_id_for_url(" http://engine:30001/ ")
        );
        assert_ne!(
            instance_id_for_url("http://engine:30001"),
            instance_id_for_url("http://engine:30002")
        );
    }
}

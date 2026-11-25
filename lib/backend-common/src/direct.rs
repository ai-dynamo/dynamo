// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host engines this process does not run, for a frontend that dials them directly.
//!
//! In sidecar mode a worker process beside the engine runs an [`LLMEngine`]: it
//! registers the engine with discovery and serves the request plane through
//! [`EngineAdapter`]. In direct mode the frontend runs that same engine
//! implementation and adapter itself. It registers each engine in its own
//! discovery under the engine's own instance id as a `TransportType::Direct`
//! worker, so model watching, routing and prefill activation run unchanged.
//! [`DirectDispatch`] is registered as the runtime's direct dispatch, so routers
//! hand each request for those workers to it, and it calls the engine's
//! adapter in-process.
//!
//! An engine that stops serving ([`LLMEngine::engine_lost`]) leaves discovery and
//! is probed again until it serves, then registered again. An engine whose
//! configuration this frontend rejects is logged as an error and not probed
//! again.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_llm::local_model::model_card_instance;
use dynamo_llm::local_model::runtime_config::{
    DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY, DIRECT_KV_EVENTS_IDEMPOTENT_STORES_RUNTIME_KEY,
};
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::{Instance, TransportType};
use dynamo_runtime::discovery::DiscoveryInstance;
use dynamo_runtime::pipeline::network::StreamPrologueError;
use dynamo_runtime::pipeline::network::egress::addressed_router::pre_stream_failure_error;
use dynamo_runtime::pipeline::{
    AddressedRequest, AsyncEngine, AsyncEngineContextProvider, Context, Error, ManyIn, ManyOut,
    ResponseStream, SingleIn, StreamingDispatch,
};
use dynamo_runtime::protocols::annotated::Annotated;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::Instrument;

use crate::adapter::EngineAdapter;
use crate::disagg::DisaggregationMode;
use crate::engine::{EngineConfig, KvEventSource, LLMEngine};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::worker::{
    WorkerConfig, build_local_model, resolve_model_type, resolve_worker_type_and_needs,
    validate_llm_model_input,
};

/// Delay after the first failed probe, and before the first probe of an engine
/// that stopped serving. It doubles after each failed probe and each time the
/// engine is lost, until the engine has served for `REPROBE_MAX_DELAY`.
const REPROBE_INITIAL_DELAY: Duration = Duration::from_secs(1);
/// Upper bound on the delay between probes of an engine that is not serving,
/// and how long an engine must serve before that delay starts over.
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
///
/// The host never cancels or times out `LLMEngine::start`, as `Worker` does
/// not: the engine must bound it itself (the sidecar transports return
/// `startup_timeout` errors). A start that never returns leaves its engine
/// unregistered and never probed again.
pub type DirectEngineFactory =
    Arc<dyn Fn(&DirectEngineSpec) -> Result<Arc<dyn LLMEngine>, DynamoError> + Send + Sync>;

/// The routers' final hop in direct mode: the adapter of the engine each
/// request was routed to.
#[derive(Default)]
pub struct DirectDispatch {
    adapters: parking_lot::RwLock<HashMap<u64, Arc<EngineAdapter>>>,
    /// One supervisor per engine. Nothing waits for them on shutdown: an
    /// engine's cleanup releases only this process's client, not the engine,
    /// which runs elsewhere.
    supervisors: TaskTracker,
}

impl DirectDispatch {
    /// Resolves once runtime shutdown has drained, withdrawn and released
    /// every engine.
    #[cfg(test)]
    async fn stopped(&self) {
        self.supervisors.wait().await;
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for DirectDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let establish_timeout = addressed.establish_timeout();
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
        // As on the request plane: the engine has this long to open its
        // response stream, and a failure before the stream reports the engine
        // down, so a retry does not pick it again.
        let engine_context = context.context();
        let setup = adapter.generate(Context::rejoin(request, context));
        let stream = match establish_timeout {
            Some(timeout) => tokio::time::timeout(timeout, setup).await.map_err(|_| {
                tracing::warn!(
                    request_id = engine_context.id(),
                    instance_id,
                    timeout_secs = timeout.as_secs_f64(),
                    "engine did not establish its response stream before the timeout"
                );
                error(
                    ErrorType::ResponseTimeout,
                    format!(
                        "engine {instance_id} did not establish its response stream within {timeout:?}"
                    ),
                )
            })?,
            None => setup.await,
        };
        match stream {
            Ok(stream) => Ok(stream),
            // The client went away, not the engine: end the stream, as the
            // request plane does, instead of reporting the engine down.
            Err(_) if engine_context.is_stopped() || engine_context.is_killed() => Ok(
                ResponseStream::new(Box::pin(futures::stream::empty()), engine_context),
            ),
            Err(setup_error) => {
                let engine_error: &(dyn std::error::Error + 'static) = setup_error.as_ref();
                Err(pre_stream_failure_error(StreamPrologueError::new(
                    setup_error.to_string(),
                    DynamoError::from(engine_error),
                ))
                .into())
            }
        }
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(error(
            ErrorType::Backend(BackendError::InvalidArgument),
            "engines dialed directly take unary requests only",
        )
        .into())
    }
}

/// Host every listed engine, each on its own as a sidecar would: probe it until
/// it serves, register it, and keep it registered while it serves. Returns once
/// every engine's supervisor runs, without waiting for any engine to serve; an
/// engine that is down is probed with backoff until it serves, and the others
/// serve meanwhile. Fails only on configuration errors, including an engine the
/// factory cannot build. An engine that starts but whose configuration this
/// frontend rejects (an `InvalidArgument` error, such as KV events it cannot
/// subscribe to) is logged as an error and not probed again.
///
/// `config` is the worker configuration a sidecar for these engines would run
/// with. Each engine's role comes from its spec, not from `config`.
///
/// `drt` must use process-local discovery: the engines are listed by the
/// operator, not discovered, and their records name addresses only this
/// frontend can dial.
///
/// On runtime shutdown the engines keep serving for the HTTP service's drain
/// (`DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`), so the requests it has admitted
/// finish, and are released after it.
pub async fn serve_direct_engines(
    drt: &DistributedRuntime,
    config: WorkerConfig,
    specs: Vec<DirectEngineSpec>,
    factory: DirectEngineFactory,
) -> Result<Arc<DirectDispatch>, DynamoError> {
    let shutdown_drain = dynamo_runtime::config::http_graceful_shutdown_timeout();
    serve_engines(drt, config, specs, factory, shutdown_drain).await
}

/// [`serve_direct_engines`] with the shutdown drain window given.
async fn serve_engines(
    drt: &DistributedRuntime,
    config: WorkerConfig,
    specs: Vec<DirectEngineSpec>,
    factory: DirectEngineFactory,
    shutdown_drain: Duration,
) -> Result<Arc<DirectDispatch>, DynamoError> {
    if !drt.discovery_is_process_local() {
        return Err(error(
            ErrorType::Backend(BackendError::InvalidArgument),
            "direct mode needs process-local discovery (--discovery-backend mem); \
             engines are listed, not discovered",
        ));
    }
    if specs.is_empty() {
        return Err(error(
            ErrorType::Backend(BackendError::InvalidArgument),
            "direct mode needs at least one engine",
        ));
    }
    // As the sidecar `Worker` checks: the engines are `LLMEngine`s, and only
    // the token pipeline's requests have a direct dispatch to reach them.
    validate_llm_model_input(config.model_input)?;
    let mut urls = HashSet::new();
    for spec in &specs {
        if !urls.insert(normalized_url(&spec.url)) {
            return Err(error(
                ErrorType::Backend(BackendError::InvalidArgument),
                format!("engine {} is listed twice", spec.url),
            ));
        }
    }

    // Build every engine before any probe: a factory error is a configuration
    // error, so it fails startup instead of being probed forever. Nothing is
    // registered with the runtime yet, so the caller can correct its list and
    // call again.
    let mut engines = Vec::with_capacity(specs.len());
    for spec in specs {
        let engine = factory(&spec)?;
        engines.push((spec, engine));
    }

    // Before any engine is published: routers check for this dispatch once,
    // when the first model card builds them.
    let dispatch = Arc::new(DirectDispatch::default());
    drt.register_direct_dispatch::<PreprocessedRequest, Annotated<LLMEngineOutput>>(
        dispatch.clone(),
    )
    .map_err(|register_error| {
        error(
            ErrorType::Backend(BackendError::InvalidArgument),
            format!("register the direct dispatch: {register_error}"),
        )
    })?;
    let shutdown = drt.primary_token();
    for (spec, engine) in engines {
        let host = DirectEngineHost {
            drt: drt.clone(),
            config: config.clone(),
            instance_id: instance_id_for_url(&spec.url),
            spec,
            factory: factory.clone(),
            dispatch: dispatch.clone(),
            shutdown_drain,
        };
        dispatch
            .supervisors
            .spawn(host.supervise(engine, shutdown.clone()).in_current_span());
    }
    dispatch.supervisors.close();
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
    /// How long the engine keeps serving once runtime shutdown starts.
    shutdown_drain: Duration,
}

/// An engine that serves and the discovery records that publish it.
struct HostedEngine {
    engine: Arc<dyn LLMEngine>,
    registered: Vec<DiscoveryInstance>,
    /// Cancelled as `engine.cleanup()` starts, so the adapter reports requests
    /// the engine ends during cleanup as `EngineShutdown`, which the frontend
    /// migrates; see `EngineAdapter::with_engine_cleanup`.
    engine_cleanup: CancellationToken,
}

impl DirectEngineHost {
    /// Start the engine, make it dispatchable, then publish it. Nothing cancels
    /// `engine.start()`: the `LLMEngine` contract makes `cleanup` safe only once
    /// `start` has returned, and `Worker` never cancels it either.
    async fn start(&self, engine: Arc<dyn LLMEngine>) -> Result<HostedEngine, ProbeError> {
        let records = match engine.start(self.instance_id).await {
            Ok(engine_config) => self
                .discovery_records(&engine, &engine_config)
                .await
                .map_err(ProbeError::NotRegistered),
            Err(start_error) => Err(ProbeError::NotServing(start_error)),
        };
        let (endpoint, model) = match records {
            Ok(records) => records,
            Err(probe_error) => {
                self.cleanup(&engine).await;
                return Err(probe_error);
            }
        };

        let engine_cleanup = CancellationToken::new();
        self.dispatch.adapters.write().insert(
            self.instance_id,
            Arc::new(
                EngineAdapter::new(engine.clone(), self.spec.mode)
                    .with_engine_cleanup(engine_cleanup.clone()),
            ),
        );
        let mut hosted = HostedEngine {
            engine,
            registered: Vec::with_capacity(2),
            engine_cleanup,
        };
        // The endpoint first, so routers built when the model card appears
        // already see the instance.
        for record in [endpoint, model] {
            match self.drt.discovery().register_instance(record).await {
                Ok(registered) => hosted.registered.push(registered),
                Err(register_error) => {
                    self.stop(hosted).await;
                    return Err(ProbeError::NotRegistered(error(
                        ErrorType::Backend(BackendError::Unknown),
                        format!("register engine {}: {register_error}", self.spec.url),
                    )));
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
        hosted.engine_cleanup.cancel();
        self.cleanup(&hosted.engine).await;
    }

    async fn cleanup(&self, engine: &Arc<dyn LLMEngine>) {
        if let Err(cleanup_error) = engine.cleanup().await {
            tracing::warn!(url = %self.spec.url, %cleanup_error, "engine cleanup failed");
        }
    }

    /// Probe the engine until it serves, keep it published while it serves, and
    /// when it stops, withdraw it and probe again. Probes back off from
    /// `REPROBE_INITIAL_DELAY` to `REPROBE_MAX_DELAY`, and so do the returns of
    /// an engine that is lost again soon after it registers. A probe's start runs
    /// to completion even on shutdown. Stops probing,
    /// with an error, when the engine cannot be built or its configuration is
    /// rejected, since every later probe would fail the same way.
    async fn supervise(self, engine: Arc<dyn LLMEngine>, shutdown: CancellationToken) {
        let mut engine = Some(engine);
        let mut delay = REPROBE_INITIAL_DELAY;
        loop {
            let hosted = loop {
                let engine = match engine.take().map_or_else(|| (self.factory)(&self.spec), Ok) {
                    Ok(engine) => engine,
                    Err(build_error) => {
                        tracing::error!(
                            url = %self.spec.url,
                            %build_error,
                            "engine cannot be built; no longer probing it"
                        );
                        return;
                    }
                };
                match self.start(engine).await {
                    Ok(hosted) => break hosted,
                    Err(ProbeError::NotServing(probe_error)) => {
                        tracing::warn!(
                            url = %self.spec.url,
                            %probe_error,
                            retry_in = ?delay,
                            "engine is not serving yet"
                        );
                    }
                    Err(ProbeError::NotRegistered(rejected))
                        if is_configuration_error(&rejected) =>
                    {
                        tracing::error!(
                            url = %self.spec.url,
                            %rejected,
                            "engine configuration is rejected; no longer probing it"
                        );
                        return;
                    }
                    Err(ProbeError::NotRegistered(register_error)) => {
                        tracing::warn!(
                            url = %self.spec.url,
                            %register_error,
                            retry_in = ?delay,
                            "engine serves but could not be registered"
                        );
                    }
                }
                tokio::select! {
                    _ = shutdown.cancelled() => return,
                    _ = tokio::time::sleep(delay) => {}
                }
                delay = (delay * 2).min(REPROBE_MAX_DELAY);
            };

            let serving_since = tokio::time::Instant::now();
            let serving = hosted.engine.clone();
            // One `engine_lost` future per hosted engine, as the trait describes.
            let mut engine_lost = serving.engine_lost();
            let lost = tokio::select! {
                _ = shutdown.cancelled() => None,
                lost = &mut engine_lost => Some(lost),
            };
            let Some(lost) = lost else {
                // The same token starts the HTTP service's drain. The requests
                // it has admitted, in flight here or still to be routed, need
                // the engine until that drain ends; cleanup would fail them
                // with no engine left to migrate to.
                tokio::select! {
                    _ = tokio::time::sleep(self.shutdown_drain) => {}
                    _ = &mut engine_lost => {}
                }
                self.stop(hosted).await;
                return;
            };
            tracing::warn!(
                url = %self.spec.url,
                reason = %lost,
                "engine stopped serving; removed it from routing"
            );
            self.stop(hosted).await;
            // An engine that registers and is lost straight away would otherwise
            // take traffic and be withdrawn once a second, indefinitely.
            if serving_since.elapsed() >= REPROBE_MAX_DELAY {
                delay = REPROBE_INITIAL_DELAY;
            }
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = (delay * 2).min(REPROBE_MAX_DELAY);
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
        // As `Worker::setup_publishing`: with KV routing off, the engine is
        // not asked for its KV event sources, so one this frontend could not
        // subscribe to does not keep the engine from serving.
        if config.enable_kv_routing {
            record_kv_event_sources(
                &mut engine_config.runtime_data,
                engine.kv_event_sources().await?,
            )?;
        }
        let local_model = build_local_model(&config, &engine_config, false, None).await?;
        let mut card = local_model.into_card();
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
            transport: TransportType::Direct(self.spec.url.clone()),
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

/// Why a probe did not leave its engine serving. The engine is released either way.
enum ProbeError {
    /// `engine.start()` failed: the engine does not serve yet.
    NotServing(DynamoError),
    /// The engine started, but this frontend could not register it.
    NotRegistered(DynamoError),
}

/// An error every later probe would repeat: the engine's configuration, not
/// its state, is wrong.
fn is_configuration_error(error: &DynamoError) -> bool {
    matches!(
        error.error_type(),
        ErrorType::InvalidArgument | ErrorType::Backend(BackendError::InvalidArgument)
    )
}

/// Name the engine's ZMQ KV-event publishers in its runtime data, by rank, so
/// the frontend subscribes to them itself.
///
/// The endpoint, the rank and `idempotent_stores` reach the router's ZMQ
/// ingress; stores are idempotent there only when every publisher of the
/// engine says so. Unlike the sidecar's subscriber, that ingress:
/// - subscribes to every topic on the socket, not only `topic`;
/// - does not rewrite multimodal events with `image_token_id`, so KV routing
///   of requests with images is not exact, as when the engine sets none;
/// - does not watch `heartbeat_timeout`: a publisher that goes quiet keeps
///   its blocks in the index until the engine is lost.
fn record_kv_event_sources(
    runtime_data: &mut HashMap<String, serde_json::Value>,
    sources: Vec<KvEventSource>,
) -> Result<(), DynamoError> {
    let mut endpoints = serde_json::Map::new();
    let mut idempotent_stores = true;
    for source in sources {
        match source {
            KvEventSource::Zmq {
                endpoint,
                dp_rank,
                idempotent_stores: idempotent,
                topic: _,
                image_token_id: _,
                heartbeat_timeout: _,
            } => {
                endpoints.insert(dp_rank.to_string(), endpoint.into());
                idempotent_stores &= idempotent;
            }
            KvEventSource::Push { dp_rank, .. } => {
                return Err(error(
                    ErrorType::Backend(BackendError::InvalidArgument),
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
    if idempotent_stores {
        runtime_data.insert(
            DIRECT_KV_EVENTS_IDEMPOTENT_STORES_RUNTIME_KEY.to_string(),
            true.into(),
        );
    }
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

    use dynamo_http_server::metrics::Metrics;
    use dynamo_llm::discovery::{ModelManager, ModelWatcher};
    use dynamo_llm::entrypoint::RouterConfig;
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
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
    use crate::engine::{GenerateContext, LLMEngineOutputExt, LlmRegistration, chunk};

    const MODEL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../llm/tests/data/sample-models/mock-llama-3.1-8b-instruct"
    );
    const A: &str = "http://engine-a:30001";
    const B: &str = "http://engine-b:30001";
    /// How long the hosted engines keep serving after shutdown, unless a test
    /// asks for longer.
    const SHUTDOWN_DRAIN: Duration = Duration::from_millis(200);
    /// Tokens per KV block, as the engines report it.
    const BLOCK_SIZE: usize = 16;

    /// What the test controls about one listed engine, across incarnations.
    #[derive(Default)]
    struct EngineState {
        up: AtomicBool,
        served: AtomicUsize,
        lost: Notify,
        /// Signalled by `cleanup`, which ends a held request as cancelled.
        cleaned: Notify,
        /// Signalled by the test to complete a held request.
        released: Notify,
        /// The ZMQ KV-event publisher the engine advertises, if any.
        kv_events: Option<String>,
        cleanups: AtomicUsize,
        /// Calls to `start`. While `start_hangs` is set, a start returns only
        /// once `start_released` is notified.
        starts: AtomicUsize,
        start_hangs: AtomicBool,
        start_released: Notify,
        /// Advertise an in-process KV-event source, which a frontend rejects.
        kv_push: AtomicBool,
        /// When each `start` was called.
        started_at: parking_lot::Mutex<Vec<tokio::time::Instant>>,
        /// Report the engine lost as soon as it registers.
        flaps: AtomicBool,
        /// Fail every request before its response stream, as an engine that
        /// cannot be reached does.
        refuses: AtomicBool,
        /// Never open a response stream.
        stalls: AtomicBool,
        /// The prompt tokens of the last request served.
        prompt: parking_lot::Mutex<Vec<u32>>,
        /// The most prompt tokens the router took as cached here for a request.
        cached_tokens: AtomicUsize,
    }

    /// One incarnation of a listed engine: answers every request with one token.
    /// `hold_next`, shared by every engine, makes
    /// the next request wait until its engine is cleaned up, which ends it as
    /// cancelled, or until the test releases it, which completes it; the engine
    /// that holds it is posted to `held`.
    struct FakeEngine {
        state: Arc<EngineState>,
        hold_next: Arc<AtomicBool>,
        held: Arc<parking_lot::Mutex<Option<Arc<EngineState>>>>,
    }

    #[async_trait]
    impl LLMEngine for FakeEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            self.state.starts.fetch_add(1, Ordering::SeqCst);
            self.state
                .started_at
                .lock()
                .push(tokio::time::Instant::now());
            if self.state.start_hangs.load(Ordering::SeqCst) {
                self.state.start_released.notified().await;
            }
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
                    kv_cache_block_size: Some(BLOCK_SIZE as u32),
                    context_length: Some(4096),
                    ..LlmRegistration::default()
                }),
                ..EngineConfig::default()
            })
        }

        async fn generate(
            &self,
            request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            if self.state.refuses.load(Ordering::SeqCst) {
                return Err(error(
                    ErrorType::Backend(BackendError::CannotConnect),
                    "engine is unreachable",
                ));
            }
            if self.state.stalls.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            if self.hold_next.swap(false, Ordering::SeqCst) {
                *self.held.lock() = Some(self.state.clone());
                let state = self.state.clone();
                return Ok(futures::stream::once(async move {
                    tokio::select! {
                        _ = state.cleaned.notified() => Ok(LLMEngineOutput::cancelled()),
                        _ = state.released.notified() => {
                            Ok(LLMEngineOutput::stop().with_tokens(vec![1]))
                        }
                    }
                })
                .boxed());
            }
            self.state.served.fetch_add(1, Ordering::SeqCst);
            if let Some(cached) = request.tracker.as_ref().and_then(|t| t.cached_tokens()) {
                self.state.cached_tokens.fetch_max(cached, Ordering::SeqCst);
            }
            *self.state.prompt.lock() = request.token_ids.to_vec();
            Ok(futures::stream::iter([Ok(chunk::token(1)), Ok(LLMEngineOutput::stop())]).boxed())
        }

        async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
            if self.state.kv_push.load(Ordering::SeqCst) {
                return Ok(vec![KvEventSource::Push {
                    on_ready: Box::new(|_| Ok(())),
                    dp_rank: 0,
                }]);
            }
            Ok(self
                .state
                .kv_events
                .iter()
                .map(|endpoint| KvEventSource::Zmq {
                    endpoint: endpoint.clone(),
                    topic: String::new(),
                    dp_rank: 0,
                    image_token_id: None,
                    heartbeat_timeout: None,
                    idempotent_stores: false,
                })
                .collect())
        }

        async fn engine_lost(&self) -> DynamoError {
            if !self.state.flaps.load(Ordering::SeqCst) {
                self.state.lost.notified().await;
            }
            error(
                ErrorType::Backend(BackendError::EngineShutdown),
                "engine stopped",
            )
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            self.state.cleanups.fetch_add(1, Ordering::SeqCst);
            self.state.cleaned.notify_one();
            Ok(())
        }
    }

    /// A frontend serving engines `A` and `B` directly, built the way the HTTP
    /// frontend builds a dynamic one.
    struct Frontend {
        drt: DistributedRuntime,
        namespace: String,
        states: HashMap<&'static str, Arc<EngineState>>,
        hold_next: Arc<AtomicBool>,
        held: Arc<parking_lot::Mutex<Option<Arc<EngineState>>>>,
        dispatch: Arc<DirectDispatch>,
        chat: OpenAIChatCompletionsStreamingEngine,
        model: String,
    }

    impl Frontend {
        async fn start(namespace: &str, migration_limit: u32) -> Self {
            Self::start_with(
                namespace,
                migration_limit,
                RouterConfig::default(),
                None,
                true,
                SHUTDOWN_DRAIN,
            )
            .await
        }

        /// `a_kv_events` is the KV-event publisher engine `A` advertises; `B`
        /// advertises none. `a_up` is whether `A` serves at startup; `B` does.
        /// `shutdown_drain` is how long the engines keep serving after shutdown.
        async fn start_with(
            namespace: &str,
            migration_limit: u32,
            router_config: RouterConfig,
            a_kv_events: Option<String>,
            a_up: bool,
            shutdown_drain: Duration,
        ) -> Self {
            let runtime = Runtime::from_current().unwrap();
            let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
                .await
                .unwrap();
            let states: HashMap<&'static str, Arc<EngineState>> = [A, B]
                .into_iter()
                .map(|url| {
                    let state = Arc::new(EngineState {
                        kv_events: (url == A).then(|| a_kv_events.clone()).flatten(),
                        ..EngineState::default()
                    });
                    state.up.store(url == B || a_up, Ordering::SeqCst);
                    (url, state)
                })
                .collect();
            let hold_next = Arc::new(AtomicBool::new(false));
            let held = Arc::new(parking_lot::Mutex::new(None));
            let factory_states = states.clone();
            let factory_hold_next = hold_next.clone();
            let factory_held = held.clone();
            let factory: DirectEngineFactory = Arc::new(move |spec: &DirectEngineSpec| {
                Ok(Arc::new(FakeEngine {
                    state: factory_states[spec.url.as_str()].clone(),
                    hold_next: factory_hold_next.clone(),
                    held: factory_held.clone(),
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
            let dispatch = serve_engines(&drt, config, specs, factory, shutdown_drain)
                .await
                .unwrap();

            let manager = Arc::new(ModelManager::new());
            let watcher = Arc::new(ModelWatcher::new(
                drt.clone(),
                manager.clone(),
                router_config,
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
                hold_next,
                held,
                dispatch,
                chat,
                model,
            }
        }

        fn served(&self, url: &str) -> usize {
            self.states[url].served.load(Ordering::SeqCst)
        }

        /// One chat request; `true` when it completed without an error.
        async fn chat(&self) -> bool {
            self.chat_with("hello").await
        }

        async fn chat_with(&self, prompt: &str) -> bool {
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
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

        /// The engine that holds the request `hold_next` asked for.
        async fn holding(&self) -> Arc<EngineState> {
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    if let Some(state) = self.held.lock().take() {
                        return state;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("an engine must hold the request")
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
        assert!(
            frontend
                .drt
                .direct_dispatch::<PreprocessedRequest, Annotated<LLMEngineOutput>>()
                .is_some()
        );

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

        // Shutdown releases both engines, once they have served through the
        // drain, before the dispatch reports stopped.
        let cleanups = |url: &str| frontend.states[url].cleanups.load(Ordering::SeqCst);
        let (a_cleanups, b_cleanups) = (cleanups(A), cleanups(B));
        frontend.drt.primary_token().cancel();
        tokio::time::timeout(Duration::from_secs(10), frontend.dispatch.stopped())
            .await
            .expect("shutdown must release every engine");
        assert_eq!((cleanups(A), cleanups(B)), (a_cleanups + 1, b_cleanups + 1));
    }

    #[tokio::test]
    async fn shutdown_waits_for_a_reprobe_start_then_releases_the_engine() {
        let frontend = Frontend::start("direct-reprobe-shutdown", 0).await;
        let a = frontend.states[A].clone();
        let starts = a.starts.load(Ordering::SeqCst);
        a.start_hangs.store(true, Ordering::SeqCst);
        a.lost.notify_one();
        tokio::time::timeout(Duration::from_secs(10), async {
            while a.starts.load(Ordering::SeqCst) == starts {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the lost engine must be probed again");

        // The probe's start is never cancelled: shutdown waits for it, and the
        // engine is released only once it returns.
        let cleanups = a.cleanups.load(Ordering::SeqCst);
        frontend.drt.primary_token().cancel();
        assert!(
            tokio::time::timeout(Duration::from_millis(200), frontend.dispatch.stopped())
                .await
                .is_err(),
            "shutdown must not cancel a start still running"
        );
        assert_eq!(a.cleanups.load(Ordering::SeqCst), cleanups);
        a.start_released.notify_one();
        tokio::time::timeout(Duration::from_secs(10), frontend.dispatch.stopped())
            .await
            .expect("shutdown must release the engine once its start returns");
        assert_eq!(a.cleanups.load(Ordering::SeqCst), cleanups + 1);
    }

    #[tokio::test]
    async fn an_engine_down_at_startup_is_probed_until_it_serves() {
        // Startup returns while `A` is down; `B` serves meanwhile.
        let frontend = Frontend::start_with(
            "direct-startup-down",
            0,
            RouterConfig::default(),
            None,
            false,
            SHUTDOWN_DRAIN,
        )
        .await;
        let a = frontend.states[A].clone();
        frontend.chat_until(|f| f.served(B) >= 4).await;
        assert_eq!(frontend.served(A), 0);
        assert!(!frontend.listed(A).await);
        assert!(a.starts.load(Ordering::SeqCst) >= 1);

        // Once `A` serves, the next probe registers it and it is routed to.
        a.up.store(true, Ordering::SeqCst);
        frontend.chat_until(|f| f.served(A) > 0).await;
        assert!(frontend.listed(A).await);
    }

    #[tokio::test(start_paused = true)]
    async fn an_engine_lost_soon_after_it_registers_keeps_backing_off() {
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        // `A` starts and registers, then reports itself lost straight away.
        let state = Arc::new(EngineState {
            up: AtomicBool::new(true),
            flaps: AtomicBool::new(true),
            ..EngineState::default()
        });
        let factory_state = state.clone();
        let factory: DirectEngineFactory = Arc::new(move |_: &DirectEngineSpec| {
            Ok(Arc::new(FakeEngine {
                state: factory_state.clone(),
                hold_next: Arc::default(),
                held: Arc::default(),
            }) as Arc<dyn LLMEngine>)
        });
        let spec = DirectEngineSpec {
            url: A.to_string(),
            mode: DisaggregationMode::Aggregated,
        };
        serve_direct_engines(&drt, WorkerConfig::default(), vec![spec], factory)
            .await
            .unwrap();
        let started = |count: usize| {
            let state = state.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(600), async {
                    while state.started_at.lock().len() < count {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("the engine must be probed again");
                state.started_at.lock().clone()
            }
        };

        // Each return waits twice as long as the one before.
        let starts = started(4).await;
        let waits: Vec<Duration> = starts.windows(2).map(|pair| pair[1] - pair[0]).collect();
        for (wait, at_least) in waits.iter().zip([1, 2, 4]) {
            assert!(
                *wait >= Duration::from_secs(at_least),
                "returns must back off: {waits:?}"
            );
        }

        // Once the engine has served for a while, the delay starts over.
        state.flaps.store(false, Ordering::SeqCst);
        let serving = started(5).await.len();
        tokio::time::sleep(REPROBE_MAX_DELAY).await;
        let lost_at = tokio::time::Instant::now();
        state.lost.notify_one();
        let returned_at = started(serving + 1).await[serving];
        assert!(
            returned_at - lost_at < REPROBE_INITIAL_DELAY * 2,
            "an engine that served for {REPROBE_MAX_DELAY:?} must be probed after the initial delay"
        );
        drt.shutdown();
    }

    #[tokio::test]
    async fn an_engine_whose_configuration_is_rejected_is_not_probed_again() {
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        // `A` starts, but advertises KV events a frontend cannot subscribe to.
        let state = Arc::new(EngineState {
            up: AtomicBool::new(true),
            kv_push: AtomicBool::new(true),
            ..EngineState::default()
        });
        let factory_state = state.clone();
        let factory: DirectEngineFactory = Arc::new(move |_: &DirectEngineSpec| {
            Ok(Arc::new(FakeEngine {
                state: factory_state.clone(),
                hold_next: Arc::default(),
                held: Arc::default(),
            }) as Arc<dyn LLMEngine>)
        });
        let spec = DirectEngineSpec {
            url: A.to_string(),
            mode: DisaggregationMode::Aggregated,
        };
        let dispatch = serve_direct_engines(&drt, WorkerConfig::default(), vec![spec], factory)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), dispatch.stopped())
            .await
            .expect("a rejected engine must not be probed again");
        assert_eq!(state.starts.load(Ordering::SeqCst), 1);
        assert_eq!(state.cleanups.load(Ordering::SeqCst), 1);
        drt.shutdown();
    }

    #[tokio::test]
    async fn kv_event_sources_are_not_asked_for_with_kv_routing_off() {
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        // `A` advertises KV events a frontend cannot subscribe to, which a
        // frontend that routes by KV events rejects.
        let state = Arc::new(EngineState {
            up: AtomicBool::new(true),
            kv_push: AtomicBool::new(true),
            ..EngineState::default()
        });
        let factory_state = state.clone();
        let factory: DirectEngineFactory = Arc::new(move |_: &DirectEngineSpec| {
            Ok(Arc::new(FakeEngine {
                state: factory_state.clone(),
                hold_next: Arc::default(),
                held: Arc::default(),
            }) as Arc<dyn LLMEngine>)
        });
        let spec = DirectEngineSpec {
            url: A.to_string(),
            mode: DisaggregationMode::Aggregated,
        };
        let config = WorkerConfig {
            enable_kv_routing: false,
            ..WorkerConfig::default()
        };
        let endpoint = DiscoveryQuery::Endpoint {
            namespace: config.namespace.clone(),
            component: config.component.clone(),
            endpoint: config.endpoint.clone(),
        };
        serve_direct_engines(&drt, config, vec![spec], factory)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            while drt
                .discovery()
                .list(endpoint.clone())
                .await
                .unwrap()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the engine must serve without KV events");
        drt.shutdown();
    }

    #[tokio::test]
    async fn an_engine_the_factory_cannot_build_fails_startup() {
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let state = Arc::new(EngineState::default());
        state.up.store(true, Ordering::SeqCst);
        let factory_state = state.clone();
        let factory: DirectEngineFactory = Arc::new(move |spec: &DirectEngineSpec| {
            if spec.url == B {
                return Err(error(ErrorType::InvalidArgument, "bad engine address"));
            }
            Ok(Arc::new(FakeEngine {
                state: factory_state.clone(),
                hold_next: Arc::default(),
                held: Arc::default(),
            }) as Arc<dyn LLMEngine>)
        });
        let specs = [A, B]
            .map(|url| DirectEngineSpec {
                url: url.to_string(),
                mode: DisaggregationMode::Aggregated,
            })
            .to_vec();
        let result = serve_direct_engines(
            &drt,
            WorkerConfig::default(),
            specs.clone(),
            factory.clone(),
        )
        .await;
        assert!(result.is_err());
        // No engine is started before every engine is built.
        assert_eq!(state.starts.load(Ordering::SeqCst), 0);

        // The failed call left nothing registered with the runtime: the list
        // can be corrected and served.
        let corrected = specs.into_iter().filter(|spec| spec.url == A).collect();
        serve_direct_engines(&drt, WorkerConfig::default(), corrected, factory)
            .await
            .expect("the corrected list must be served on the same runtime");
        drt.shutdown();
    }

    #[tokio::test]
    async fn a_request_its_lost_engine_cancels_during_cleanup_migrates() {
        let frontend = Frontend::start("direct-cleanup", 3).await;
        frontend
            .chat_until(|f| f.served(A) > 0 && f.served(B) > 0)
            .await;

        // The next request waits on its engine; that engine is then lost, and
        // its cleanup ends the request as cancelled while the client waits.
        frontend.hold_next.store(true, Ordering::SeqCst);
        let before = frontend.served(A) + frontend.served(B);
        let (completed, ()) = tokio::join!(frontend.chat(), async {
            let state = frontend.holding().await;
            state.up.store(false, Ordering::SeqCst);
            state.lost.notify_one();
        });
        assert!(completed, "the request must complete on the other engine");
        assert_eq!(frontend.served(A) + frontend.served(B), before + 1);
    }

    #[tokio::test]
    async fn shutdown_lets_a_request_in_flight_finish_before_releasing_its_engine() {
        // The drain outlasts the test, which waits only for the request.
        let frontend = Frontend::start_with(
            "direct-drain",
            0,
            RouterConfig::default(),
            None,
            true,
            Duration::from_secs(60),
        )
        .await;
        frontend
            .chat_until(|f| f.served(A) > 0 && f.served(B) > 0)
            .await;

        // Shutdown starts while a request is in flight. Its engine is not
        // cleaned up, which would end the request, and the request finishes.
        frontend.hold_next.store(true, Ordering::SeqCst);
        let (completed, ()) = tokio::join!(frontend.chat(), async {
            let state = frontend.holding().await;
            let cleanups = state.cleanups.load(Ordering::SeqCst);
            frontend.drt.primary_token().cancel();
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert_eq!(
                state.cleanups.load(Ordering::SeqCst),
                cleanups,
                "the engine must keep serving while the frontend drains"
            );
            state.released.notify_one();
        });
        assert!(completed, "a request in flight at shutdown must finish");
    }

    /// A dispatch that hosts one engine, and a request routed to that engine.
    fn dispatch_to(state: EngineState) -> (DirectDispatch, AddressedRequest<PreprocessedRequest>) {
        let engine = Arc::new(FakeEngine {
            state: Arc::new(state),
            hold_next: Arc::default(),
            held: Arc::default(),
        });
        let dispatch = DirectDispatch::default();
        dispatch.adapters.write().insert(
            instance_id_for_url(A),
            Arc::new(EngineAdapter::new(engine, DisaggregationMode::Aggregated)),
        );
        let request = PreprocessedRequest::builder()
            .model("direct-test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();
        let instance = Instance {
            namespace: "direct-dispatch".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            instance_id: instance_id_for_url(A),
            transport: TransportType::Direct(A.to_string()),
            device_type: None,
            request_plane_codec: None,
        };
        (dispatch, AddressedRequest::for_instance(request, instance))
    }

    #[tokio::test]
    async fn a_failure_before_the_response_stream_reports_the_engine_down() {
        let (dispatch, request) = dispatch_to(EngineState {
            refuses: AtomicBool::new(true),
            ..EngineState::default()
        });
        let Err(setup_error) = dispatch.generate(Context::new(request)).await else {
            panic!("the engine refuses the request");
        };
        // The outer `CannotConnect` is what the router reports an instance down
        // for; the engine's own error is its cause.
        assert_eq!(
            setup_error
                .downcast_ref::<DynamoError>()
                .map(DynamoError::error_type),
            Some(ErrorType::CannotConnect),
            "{setup_error}"
        );
        assert!(
            dynamo_runtime::error::match_error_chain(
                setup_error.as_ref(),
                &[ErrorType::Backend(BackendError::CannotConnect)],
                &[],
            ),
            "{setup_error}"
        );
    }

    #[tokio::test]
    async fn a_cancelled_request_that_fails_before_its_stream_ends_without_an_error() {
        let (dispatch, request) = dispatch_to(EngineState {
            refuses: AtomicBool::new(true),
            ..EngineState::default()
        });
        // The client goes away, and the engine fails the request for it. That
        // is not the engine's fault: an error here would report it down.
        let request = Context::new(request);
        request.context().stop_generating();
        let Ok(stream) = dispatch.generate(request).await else {
            panic!("a cancelled request must not fail as a pre-stream failure");
        };
        assert_eq!(stream.count().await, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn an_engine_that_does_not_open_its_stream_times_out() {
        let (dispatch, request) = dispatch_to(EngineState {
            stalls: AtomicBool::new(true),
            ..EngineState::default()
        });
        let request = request.with_establish_timeout(Some(Duration::from_secs(5)));
        let result = tokio::time::timeout(
            Duration::from_secs(60),
            dispatch.generate(Context::new(request)),
        )
        .await
        .expect("the establish timeout must end the request");
        let Err(timeout_error) = result else {
            panic!("the engine never opens a stream");
        };
        assert_eq!(
            timeout_error
                .downcast_ref::<DynamoError>()
                .map(DynamoError::error_type),
            Some(ErrorType::ResponseTimeout),
            "{timeout_error}"
        );
    }

    #[tokio::test]
    async fn kv_routing_scores_published_blocks_and_reaches_engines_without_publishers() {
        // `A` publishes its KV events on this socket; `B` advertises none.
        let context = zmq::Context::new();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let kv_events = publisher.get_last_endpoint().unwrap().unwrap();
        let router_config = RouterConfig {
            router_mode: dynamo_runtime::pipeline::RouterMode::KV,
            ..RouterConfig::default()
        };
        let frontend = Frontend::start_with(
            "direct-kv",
            0,
            router_config,
            Some(kv_events),
            true,
            SHUTDOWN_DRAIN,
        )
        .await;
        frontend
            .chat_until(|f| f.served(A) > 0 && f.served(B) > 0)
            .await;

        // The sample tokenizer knows only its special tokens: a prompt of
        // them fills two KV blocks.
        let prompt = "<|reserved_special_token_0|>".repeat(2 * BLOCK_SIZE);
        let a = frontend.states[A].clone();
        tokio::time::timeout(Duration::from_secs(30), async {
            while a.prompt.lock().len() < 2 * BLOCK_SIZE {
                frontend.chat_with(&prompt).await;
            }
        })
        .await
        .expect("the prompt must reach `A`");
        let tokens = a.prompt.lock().clone();
        let blocks = tokens.len() / BLOCK_SIZE;

        // `A` reports the prompt's blocks as stored, in vLLM's event format.
        // The router has subscribed to `A`'s publisher, so it indexes them and
        // routes the prompt to `A` with those blocks as its overlap.
        let stored = (
            "BlockStored",
            (0..blocks as u64).collect::<Vec<_>>(),
            Option::<u64>::None,
            tokens[..blocks * BLOCK_SIZE].to_vec(),
            BLOCK_SIZE,
            Option::<u64>::None,
            "GPU",
        );
        let batch = rmp_serde::to_vec(&(0.0_f64, vec![stored], Some(0_i32))).unwrap();
        tokio::time::timeout(Duration::from_secs(30), async {
            while a.cached_tokens.load(Ordering::SeqCst) == 0 {
                // A subscriber misses what is published before it connects;
                // it applies sequence 0 once.
                publisher
                    .send_multipart([Vec::new(), 0_u64.to_be_bytes().to_vec(), batch.clone()], 0)
                    .unwrap();
                frontend.chat_with(&prompt).await;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the router must score the blocks `A` published");
        assert_eq!(a.cached_tokens.load(Ordering::SeqCst), blocks * BLOCK_SIZE);
        assert_eq!(frontend.states[B].cached_tokens.load(Ordering::SeqCst), 0);
    }

    /// Setups direct mode cannot serve fail before any engine is built.
    #[tokio::test]
    async fn setups_direct_mode_cannot_serve_are_rejected() {
        async fn rejection(
            distributed: DistributedConfig,
            config: WorkerConfig,
        ) -> Option<ErrorType> {
            let drt = DistributedRuntime::new(Runtime::from_current().unwrap(), distributed)
                .await
                .unwrap();
            let factory: DirectEngineFactory = Arc::new(
                |_: &DirectEngineSpec| -> Result<Arc<dyn LLMEngine>, DynamoError> {
                    panic!("no engine is built for a rejected setup")
                },
            );
            let spec = DirectEngineSpec {
                url: A.to_string(),
                mode: DisaggregationMode::Aggregated,
            };
            let result = serve_direct_engines(&drt, config, vec![spec], factory).await;
            drt.shutdown();
            result.err().map(|rejected| rejected.error_type())
        }
        let invalid = Some(ErrorType::Backend(BackendError::InvalidArgument));

        // Discovery other processes share: direct engines are listed, not discovered.
        let store = tempfile::tempdir().unwrap();
        let shared = DistributedConfig {
            discovery_backend: dynamo_runtime::distributed::DiscoveryBackend::KvStore(
                dynamo_runtime::storage::kv::Selector::File(store.path().to_path_buf()),
            ),
            ..DistributedConfig::process_local()
        };
        assert_eq!(rejection(shared, WorkerConfig::default()).await, invalid);

        // A text model would be advertised as ready, and its requests would
        // find no direct dispatch.
        let text = WorkerConfig {
            model_input: dynamo_llm::model_type::ModelInput::Text,
            ..WorkerConfig::default()
        };
        assert_eq!(
            rejection(DistributedConfig::process_local(), text).await,
            invalid
        );
    }

    #[test]
    fn kv_event_publishers_are_recorded_by_rank() {
        // `idempotent` is whether each rank's publisher sends one remove per
        // block however many times it stored it.
        let recorded = |idempotent: [bool; 2]| {
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
                        idempotent_stores: idempotent[dp_rank as usize],
                    })
                    .collect(),
            )
            .unwrap();
            runtime_data
        };
        let runtime_data = recorded([false, false]);
        assert_eq!(
            runtime_data[DIRECT_KV_EVENTS_ENDPOINTS_RUNTIME_KEY],
            serde_json::json!({"0": "tcp://engine:5557", "1": "tcp://engine:5558"})
        );
        // Stores are idempotent only when every publisher says so.
        let idempotent = |runtime_data: &HashMap<String, serde_json::Value>| {
            runtime_data
                .get(DIRECT_KV_EVENTS_IDEMPOTENT_STORES_RUNTIME_KEY)
                .cloned()
        };
        assert_eq!(idempotent(&runtime_data), None);
        assert_eq!(idempotent(&recorded([true, false])), None);
        assert_eq!(idempotent(&recorded([true, true])), Some(true.into()));

        let push = KvEventSource::Push {
            on_ready: Box::new(|_| Ok(())),
            dp_rank: 0,
        };
        assert_eq!(
            record_kv_event_sources(&mut HashMap::new(), vec![push])
                .err()
                .map(|rejected| rejected.error_type()),
            Some(ErrorType::Backend(BackendError::InvalidArgument))
        );
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

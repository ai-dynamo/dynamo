// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, CompletionUsage, ComponentSnapshot, DisaggregationMode, DynamoError,
    EngineConfig, GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    LlmRegistration, MetricsBindings, MetricsCtx, ModelInput, PreprocessedRequest,
    PromptLogprobEntry, PromptLogprobs, SnapshotPublisher, WorkerConfig, usage,
};
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::lora::{LoRACache, LoRADownloader, LoRASource, LocalLoRASource, S3LoRASource};
use dynamo_llm::model_card::LoraInfo;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::protocols::common::preprocessor::MultimodalData;
use dynamo_llm::utils::lora_name_to_id;
use dynamo_llm::worker_type::WorkerType;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use tokio::sync::{OnceCell, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Discovery, Pool};
use crate::convert;
use crate::kv;
use crate::proto as pb;

fn request_error_stream(
    error: DynamoError,
) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
    Box::pin(futures::stream::once(async move { Err(error) }))
}

pub struct OpenEngineSidecar {
    endpoint: String,
    expected_engine: Option<String>,
    expected_schema_release: Option<String>,
    requested_model: Option<String>,
    transport: TransportConfig,
    bootstrap: Discovery,
    disaggregation_mode: DisaggregationMode,
    pool: OnceCell<Pool>,
    discovery: OnceCell<Discovery>,
    cancel: CancellationToken,
    fatal: watch::Sender<Option<String>>,
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
    lora_downloader: Option<LoRADownloader>,
    lora_discovery: OnceCell<LoraDiscovery>,
    loaded_loras: tokio::sync::Mutex<HashMap<String, pb::LoraAdapter>>,
    active_requests: Arc<Mutex<HashSet<String>>>,
    shutting_down: Arc<AtomicBool>,
}

struct LoraDiscovery {
    endpoint: dynamo_runtime::component::Endpoint,
    base_model: LocalModel,
    model_type: ModelType,
    worker_type: WorkerType,
    needs: Vec<Vec<WorkerType>>,
    models: tokio::sync::Mutex<HashMap<String, LocalModel>>,
}

impl LoraDiscovery {
    async fn attach(&self, adapter: &pb::LoraAdapter) -> Result<(), DynamoError> {
        let mut model = self.base_model.clone();
        model
            .attach(
                &self.endpoint,
                self.model_type,
                ModelInput::Tokens,
                Some(LoraInfo {
                    name: adapter.lora_name.clone(),
                    max_gpu_lora_count: None,
                }),
                Some(self.worker_type),
                self.needs.clone(),
            )
            .await
            .map_err(|error| {
                client::engine_shutdown(format!("register LoRA model card: {error}"))
            })?;
        self.models
            .lock()
            .await
            .insert(adapter.lora_name.clone(), model);
        Ok(())
    }

    async fn detach(&self, name: &str) -> Result<(), DynamoError> {
        LocalModel::detach_from_endpoint(&self.endpoint, Some(name))
            .await
            .map_err(|error| {
                client::engine_shutdown(format!("unregister LoRA model card: {error}"))
            })?;
        self.models.lock().await.remove(name);
        Ok(())
    }
}

impl OpenEngineSidecar {
    pub fn from_cli() -> Result<(Self, WorkerConfig), DynamoError> {
        let args = <Args as clap::Parser>::try_parse().unwrap_or_else(|error| error.exit());
        Self::from_parsed_args(args)
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(argv) => <Args as clap::Parser>::try_parse_from(argv),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|error| client::invalid_arg(error.to_string()))?;
        Self::from_parsed_args(args)
    }

    fn from_parsed_args(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        let endpoint = normalize_endpoint(&args.openengine_endpoint);
        let transport = args.transport();
        let bootstrap = bootstrap_discover(
            &endpoint,
            &transport,
            args.model.as_deref(),
            args.expected_engine.as_deref(),
            args.expected_schema_release.as_deref(),
        )?;
        validate_model(&bootstrap)?;
        let role = engine_role(&bootstrap)?;
        let disaggregation_mode = role_to_mode(role);
        let model_name = bootstrap
            .model
            .tokenizer
            .as_ref()
            .and_then(|tokenizer| nonempty(&tokenizer.source))
            .unwrap_or_else(|| bootstrap.model.model_id.clone());
        let served_model_name = (!bootstrap.model.served_model_name.is_empty())
            .then(|| bootstrap.model.served_model_name.clone());
        let (fatal, _) = watch::channel(None);
        let worker = WorkerConfig {
            namespace: args.namespace,
            component: component_for_role(role).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name,
            served_model_name,
            model_input: ModelInput::Tokens,
            reasoning_parser: nonempty(&bootstrap.model.reasoning_parser),
            tool_call_parser: nonempty(&bootstrap.model.tool_call_parser),
            // OpenEngine owns fetch/decode/preprocessing; the CPU sidecar must
            // receive dereferenceable URL/data inputs.
            media_decoder: None,
            ..Default::default()
        };
        tracing::info!(
            %endpoint,
            engine = %bootstrap.engine.engine_name,
            model = %bootstrap.model.model_id,
            ?role,
            "OpenEngine sidecar discovery complete"
        );
        Ok((
            Self {
                endpoint,
                expected_engine: args.expected_engine,
                expected_schema_release: args.expected_schema_release,
                requested_model: Some(bootstrap.selected_model.clone()),
                transport,
                bootstrap,
                disaggregation_mode,
                pool: OnceCell::new(),
                discovery: OnceCell::new(),
                cancel: CancellationToken::new(),
                fatal,
                background_tasks: Arc::new(Mutex::new(Vec::new())),
                lora_downloader: build_lora_downloader(),
                lora_discovery: OnceCell::new(),
                loaded_loras: tokio::sync::Mutex::new(HashMap::new()),
                active_requests: Arc::new(Mutex::new(HashSet::new())),
                shutting_down: Arc::new(AtomicBool::new(false)),
            },
            worker,
        ))
    }

    async fn await_ready(&self, client: &mut client::Control) -> Result<(), DynamoError> {
        let deadline = Instant::now() + self.transport.deadline;
        loop {
            match tokio::time::timeout(
                self.transport.connect_timeout,
                client.health(health_request()),
            )
            .await
            {
                Ok(Ok(response)) => {
                    let state = pb::HealthState::try_from(response.into_inner().state)
                        .unwrap_or(pb::HealthState::Unspecified);
                    match state {
                        pb::HealthState::Ready => return Ok(()),
                        pb::HealthState::Draining => {
                            return Err(client::engine_shutdown("OpenEngine is draining"));
                        }
                        _ => {}
                    }
                }
                Ok(Err(error)) => tracing::debug!(%error, "OpenEngine health poll failed"),
                Err(_) => tracing::debug!("OpenEngine health poll timed out"),
            }
            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "OpenEngine did not become READY within {:?}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep(self.transport.poll_interval).await;
        }
    }

    #[cfg(test)]
    pub(crate) fn enable_local_lora_for_test(&mut self) {
        self.lora_downloader = Some(LoRADownloader::new(
            vec![Arc::new(LocalLoRASource::new())],
            LoRACache::new(std::env::temp_dir().join("dynamo-openengine-lora-tests")),
        ));
    }

    #[cfg(test)]
    pub(crate) async fn lora_card_count(&self) -> usize {
        match self.lora_discovery.get() {
            Some(discovery) => discovery.models.lock().await.len(),
            None => 0,
        }
    }

    #[cfg(test)]
    pub(crate) async fn lora_card_display_name(&self, name: &str) -> Option<String> {
        let discovery = self.lora_discovery.get()?;
        discovery
            .models
            .lock()
            .await
            .get(name)
            .map(|model| model.display_name().to_string())
    }
}

#[async_trait]
impl LLMEngine for OpenEngineSidecar {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown(
                "OpenEngine sidecar already started",
            ));
        }
        let pool =
            Pool::connect(&self.endpoint, &self.transport, self.transport.connections).await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control).await?;
        let discovery = tokio::time::timeout(
            self.transport.deadline,
            client::discover(
                &mut control,
                self.requested_model.as_deref(),
                self.expected_engine.as_deref(),
                self.expected_schema_release.as_deref(),
                self.transport.deadline,
            ),
        )
        .await
        .map_err(|_| client::engine_shutdown("OpenEngine discovery timed out"))??;
        validate_model(&discovery)?;
        if role_to_mode(engine_role(&discovery)?) != self.disaggregation_mode {
            return Err(client::invalid_arg(
                "OpenEngine role changed between bootstrap discovery and startup",
            ));
        }
        if discovery.model.model_id != self.bootstrap.model.model_id {
            return Err(client::invalid_arg(
                "OpenEngine selected model changed between bootstrap discovery and startup",
            ));
        }
        let config = build_engine_config(&discovery);
        let connections = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("OpenEngine sidecar already started"))?;
        self.discovery
            .set(discovery)
            .map_err(|_| client::engine_shutdown("OpenEngine discovery already initialized"))?;
        tracing::info!(connections, model = %config.model, "OpenEngine sidecar started");
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let pool = self
            .pool
            .get()
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;
        let discovery = self
            .discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("generate called before discovery"))?;
        if request.is_probe {
            let prompt_tokens = request.token_ids.len() as u32;
            let response = tokio::time::timeout(
                self.transport.connect_timeout,
                pool.control_client().health(health_request()),
            )
            .await
            .map_err(|_| client::engine_shutdown("OpenEngine Health probe timed out"))?
            .map_err(|status| client::status_to_dynamo("Health probe", status))?;
            let state = pb::HealthState::try_from(response.into_inner().state)
                .unwrap_or(pb::HealthState::Unspecified);
            if state != pb::HealthState::Ready {
                return Err(client::engine_shutdown(format!(
                    "OpenEngine Health probe returned {state:?}"
                )));
            }
            return Ok(Box::pin(futures::stream::once(async move {
                Ok(LLMEngineOutput::stop().with_usage(usage(prompt_tokens, 0)))
            })));
        }
        let connector_uses_client_bootstrap = discovery
            .engine
            .kv_connector
            .as_ref()
            .and_then(|connector| connector.supports_client_bootstrap)
            == Some(true);
        if self.disaggregation_mode.is_decode()
            && request.prefill_result.is_none()
            && !(connector_uses_client_bootstrap && request.bootstrap_info.is_some())
        {
            return Ok(request_error_stream(client::invalid_arg(
                "decode worker requires a context-first prefill_result or advertised client bootstrap",
            )));
        }
        if let Err(error) =
            validate_request_capabilities(discovery, &request, self.disaggregation_mode)
        {
            return Ok(request_error_stream(error));
        }

        let request_id = ctx.id().to_string();
        let prompt_tokens = request.token_ids.len() as u32;
        let requested_outputs = request.sampling_options.n.unwrap_or(1).max(1) as usize;
        let is_prefill = self.disaggregation_mode.is_prefill();
        let remote_model = if discovery.model.served_model_name.is_empty() {
            discovery.selected_model.as_str()
        } else {
            discovery.model.served_model_name.as_str()
        };
        let mut grpc_message = match convert::build_generate_request(
            &request,
            &request_id,
            remote_model,
            is_prefill,
            discovery.model.supports_text_input == Some(true),
        ) {
            Ok(message) => message,
            Err(error) => return Ok(request_error_stream(error)),
        };
        if let Err(error) = apply_client_bootstrap(
            discovery,
            &request,
            &request_id,
            self.disaggregation_mode,
            &mut grpc_message,
        ) {
            return Ok(request_error_stream(error));
        }
        if let Err(error) =
            validate_decode_handoff(discovery, &grpc_message, self.disaggregation_mode)
        {
            return Ok(request_error_stream(error));
        }
        let requested_bootstrap = grpc_message
            .kv
            .as_ref()
            .and_then(|kv| kv.session.as_ref())
            .and_then(|session| session.bootstrap.clone());
        let metadata = match convert::generate_metadata(&request, ctx.metadata(), is_prefill) {
            Ok(metadata) => metadata,
            Err(error) => return Ok(request_error_stream(error)),
        };
        let mut grpc_request = tonic::Request::new(grpc_message);
        *grpc_request.metadata_mut() = metadata;
        let mut grpc_client = pool.stream_client();
        let abort_client = pool.control_client();
        let active_requests = self.active_requests.clone();
        let shutting_down = self.shutting_down.clone();
        let cancel = self.cancel.clone();
        let fatal = self.fatal.clone();
        let current_failure = self.fatal.borrow().clone();
        let response_discovery = discovery.clone();

        Ok(Box::pin(async_stream::stream! {
            let Some(mut abort_guard) = AbortOnDrop::new(
                request_id.clone(),
                abort_client,
                active_requests,
                shutting_down,
            ) else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            };
            if let Some(reason) = current_failure {
                yield Err(client::engine_shutdown(reason));
                return;
            }
            if ctx.is_stopped() || cancel.is_cancelled() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            }
            let opened = tokio::select! {
                biased;
                _ = ctx.stopped() => None,
                _ = cancel.cancelled() => None,
                response = grpc_client.generate(grpc_request) => Some(response),
            };
            let Some(opened) = opened else {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            };
            let mut stream = match opened {
                Ok(response) => response.into_inner(),
                Err(status) => {
                    let error = client::status_to_dynamo("Generate", status);
                    yield Err(error);
                    return;
                }
            };
            let mut generated = 0u32;
            let mut effective_prompt_tokens = prompt_tokens;
            let mut finished_outputs = HashSet::new();
            let mut prompt_logprobs: Option<PromptLogprobs> = None;
            let mut prompt_seen = false;
            loop {
                let message = tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(effective_prompt_tokens, generated)));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(LLMEngineOutput::cancelled().with_usage(usage(effective_prompt_tokens, generated)));
                        break;
                    }
                    message = stream.message() => message,
                };
                match message {
                    Ok(Some(response)) => {
                        if response.request_id != request_id {
                            let error = client::engine_shutdown(format!(
                                "OpenEngine returned request_id `{}` on stream `{request_id}`",
                                response.request_id
                            ));
                            let _ = fatal.send(Some(error.message().to_string()));
                            yield Err(error);
                            break;
                        }
                        let terminal_usage = response.usage.as_ref().map(openengine_usage);
                        if let Some(value) = response.usage.as_ref() {
                            effective_prompt_tokens = value.prompt_tokens;
                        }
                        match response.event {
                            Some(pb::generate_response::Event::Prompt(prompt)) => {
                                if response.usage.is_some() || prompt_seen {
                                    let error = client::engine_shutdown("OpenEngine returned duplicate/misplaced PromptOutput");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                prompt_seen = true;
                                prompt_logprobs = Some(prompt_logprobs_from_openengine(prompt.tokens));
                            }
                            Some(pb::generate_response::Event::Token(token)) => {
                                if is_prefill || response.usage.is_some() {
                                    let error = client::engine_shutdown("OpenEngine returned TokenOutput for a prefill role or with terminal usage");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let Some(output_index) = token.output_index else {
                                    let error = client::engine_shutdown("OpenEngine TokenOutput omitted required output_index");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                };
                                if output_index as usize >= requested_outputs || finished_outputs.contains(&output_index) {
                                    let error = client::engine_shutdown(format!("OpenEngine returned invalid TokenOutput index {output_index}"));
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                generated = generated.saturating_add(token.tokens.len() as u32);
                                yield Ok(convert::token_output(token));
                            }
                            Some(pb::generate_response::Event::PrefillReady(prefill)) => {
                                if !is_prefill {
                                    let error = client::engine_shutdown("OpenEngine returned PrefillReady for a non-prefill role");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let Some(session) = prefill.kv_session else {
                                    let error = client::engine_shutdown("OpenEngine PrefillReady omitted required kv_session");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                };
                                let Some(usage) = terminal_usage else {
                                    let error = client::engine_shutdown("OpenEngine PrefillReady omitted required terminal usage");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                };
                                let params = convert::kv_session_to_disagg_json(session);
                                let validated_session = match convert::disagg_json_to_kv_session(&params) {
                                    Ok(session) => session,
                                    Err(error) => {
                                        let error = client::engine_shutdown(format!("OpenEngine returned an invalid PrefillReady handoff: {error}"));
                                        let _ = fatal.send(Some(error.message().to_string()));
                                        yield Err(error);
                                        break;
                                    }
                                };
                                if let Err(error) = validate_prefill_handoff(
                                    &response_discovery,
                                    &validated_session,
                                    requested_bootstrap.as_ref(),
                                ) {
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let mut output = LLMEngineOutput::stop();
                                output.completion_usage = Some(usage);
                                output.disaggregated_params = Some(params);
                                attach_prompt_logprobs(&mut output, prompt_logprobs.take());
                                abort_guard.complete();
                                yield Ok(output);
                                break;
                            }
                            Some(pb::generate_response::Event::Finished(finished)) => {
                                if is_prefill {
                                    let error = client::engine_shutdown("OpenEngine returned GenerationFinished for a prefill role");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let Some(output_index) = finished.output_index else {
                                    let error = client::engine_shutdown("OpenEngine GenerationFinished omitted required output_index");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                };
                                if output_index as usize >= requested_outputs || !finished_outputs.insert(output_index) {
                                    let error = client::engine_shutdown(format!("OpenEngine returned invalid/duplicate terminal index {output_index}"));
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let reason = pb::FinishReason::try_from(finished.reason).unwrap_or(pb::FinishReason::Unspecified);
                                if reason == pb::FinishReason::Unspecified {
                                    let error = client::engine_shutdown("OpenEngine GenerationFinished used unspecified finish reason");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let all_outputs_finished = finished_outputs.len() >= requested_outputs;
                                if response.usage.is_some() != all_outputs_finished {
                                    let error = client::engine_shutdown("OpenEngine usage must appear only on the final output terminal");
                                    let _ = fatal.send(Some(error.message().to_string()));
                                    yield Err(error);
                                    break;
                                }
                                let mut output = finish_output(reason, finished.stop_match);
                                output.completion_usage = terminal_usage;
                                output.index = finished.output_index;
                                if all_outputs_finished {
                                    attach_prompt_logprobs(&mut output, prompt_logprobs.take());
                                    abort_guard.complete();
                                }
                                yield Ok(output);
                                if all_outputs_finished { break; }
                            }
                            Some(pb::generate_response::Event::Error(error)) => {
                                abort_guard.complete();
                                yield Err(client::engine_error_to_dynamo(&error));
                                break;
                            }
                            None => {}
                        }
                    }
                    Ok(None) => {
                        let error = client::engine_shutdown("OpenEngine closed Generate before a terminal event");
                        let _ = fatal.send(Some(error.message().to_string()));
                        yield Err(error);
                        break;
                    }
                    Err(status) => {
                        let error = client::status_to_dynamo("Generate stream", status);
                        yield Err(error);
                        break;
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        if let Some(pool) = self.pool.get() {
            abort_request(pool.control_client(), ctx.id().to_string()).await;
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.shutting_down.store(true, Ordering::SeqCst);
        let request_ids = self.active_requests.lock().drain().collect::<Vec<_>>();
        self.cancel.cancel();
        if let Some(pool) = self.pool.get() {
            let request_count = request_ids.len();
            let aborts = request_ids
                .into_iter()
                .map(|request_id| abort_request(pool.control_client(), request_id));
            if tokio::time::timeout(
                self.transport.connect_timeout,
                futures::future::join_all(aborts),
            )
            .await
            .is_err()
            {
                tracing::warn!(
                    request_count,
                    "OpenEngine Abort cleanup exceeded the shared timeout"
                );
            }
        }
        if let Some(discovery) = self.lora_discovery.get() {
            let names = self
                .loaded_loras
                .lock()
                .await
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for name in names {
                if let Err(error) = discovery.detach(&name).await {
                    tracing::warn!(%error, %name, "failed to detach LoRA card during cleanup");
                }
            }
        }
        let tasks = std::mem::take(&mut *self.background_tasks.lock());
        for task in tasks {
            task.abort();
        }
        Ok(())
    }

    async fn watch(&self) -> Result<(), DynamoError> {
        let pool = self
            .pool
            .get()
            .ok_or_else(|| client::engine_shutdown("watch called before start"))?;
        let mut grpc_client = pool.control_client();
        let mut fatal = self.fatal.subscribe();
        let mut interval = tokio::time::interval(self.transport.poll_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                changed = fatal.changed() => {
                    if changed.is_err() {
                        return Err(client::engine_shutdown("fatal watcher closed"));
                    }
                    if let Some(reason) = fatal.borrow().clone() {
                        return Err(client::engine_shutdown(reason));
                    }
                }
                _ = interval.tick() => {
                    let response = tokio::time::timeout(
                        self.transport.connect_timeout,
                        grpc_client.health(health_request()),
                    ).await.map_err(|_| client::engine_shutdown("OpenEngine Health timed out"))?
                        .map_err(|status| client::status_to_dynamo("Health", status))?;
                    let state = pb::HealthState::try_from(response.into_inner().state)
                        .unwrap_or(pb::HealthState::Unspecified);
                    if state != pb::HealthState::Ready {
                        return Err(client::engine_shutdown(format!("OpenEngine health changed to {state:?}")));
                    }
                }
            }
        }
    }

    async fn is_quiescent(&self) -> Result<Option<bool>, DynamoError> {
        let Some(pool) = self.pool.get() else {
            return Ok(None);
        };
        let load = tokio::time::timeout(
            self.transport.connect_timeout,
            pool.control_client().get_load(pb::GetLoadRequest {
                include_per_rank: false,
            }),
        )
        .await
        .map_err(|_| client::engine_shutdown("OpenEngine GetLoad timed out"))?
        .map_err(|status| client::status_to_dynamo("GetLoad", status))?
        .into_inner();
        Ok(Some(load_is_quiescent(&load)))
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let pool = self
            .pool
            .get()
            .ok_or_else(|| client::engine_shutdown("kv_event_sources called before start"))?;
        let discovery = self
            .discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("kv_event_sources called before discovery"))?;
        kv::discover_sources(
            pool.channel(),
            pool.control_client(),
            kv::SourceDiscovery {
                expected_ranks: advertised_dp_ranks(discovery)?,
                routing_image_token_id: discovery
                    .model
                    .multimodal_capabilities
                    .as_ref()
                    .and_then(|capabilities| capabilities.routing_image_token_id),
                deadline: self.transport.connect_timeout,
                cancel: self.cancel.clone(),
                tasks: self.background_tasks.clone(),
                fatal: self.fatal.clone(),
            },
        )
        .await
    }

    async fn setup_metrics(&self, _ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        let pool = self
            .pool
            .get()
            .ok_or_else(|| client::engine_shutdown("setup_metrics called before start"))?;
        let discovery = self
            .discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("setup_metrics called before discovery"))?;
        let parallelism = discovery.engine.parallelism.unwrap_or_default();
        let start = parallelism.data_parallel_start_rank.unwrap_or(0);
        let count = parallelism.data_parallel_size.unwrap_or(1).max(1);
        let ranks = (start..start + count).collect::<Vec<_>>();
        let channel = pool.channel();
        let cancel = self.cancel.clone();
        let interval = self.transport.load_poll_interval;
        let tasks = self.background_tasks.clone();
        Ok(MetricsBindings {
            dp_ranks: ranks.clone(),
            on_publisher_ready: Some(Box::new(move |publisher| {
                let task = tokio::spawn(load_loop(channel, publisher, ranks, interval, cancel));
                tasks.lock().push(task);
                Ok(())
            })),
        })
    }

    async fn drain(&self) -> Result<(), DynamoError> {
        let Some(pool) = self.pool.get() else {
            return Ok(());
        };
        let client_timeout = self.transport.drain_timeout;
        let deadline = Instant::now() + client_timeout;
        let deadline_ms = client_timeout.as_millis().min(u32::MAX as u128) as u32;
        let mut stream = tokio::time::timeout(
            self.transport.drain_timeout,
            pool.control_client().drain(pb::DrainRequest {
                stop_accepting_new_requests: true,
                deadline_ms: Some(deadline_ms),
                // The sidecar owns only its tracked request IDs. Let cleanup
                // abort those stragglers at the local deadline rather than
                // asking the shared engine server to abort unrelated HTTP
                // requests process-wide.
                abort_after_deadline: false,
            }),
        )
        .await
        .map_err(|_| client::engine_shutdown("OpenEngine Drain startup timed out"))?
        .map_err(|status| client::status_to_dynamo("Drain", status))?
        .into_inner();
        let mut drain_complete = false;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(client::engine_shutdown("OpenEngine Drain timed out"));
            }
            let response = tokio::time::timeout(remaining, stream.message())
                .await
                .map_err(|_| client::engine_shutdown("OpenEngine Drain stream timed out"))?
                .map_err(|status| client::status_to_dynamo("Drain stream", status))?;
            let Some(response) = response else {
                break;
            };
            match response.event {
                Some(pb::drain_response::Event::State(value))
                    if pb::DrainState::try_from(value).ok() == Some(pb::DrainState::Complete) =>
                {
                    drain_complete = true;
                    break;
                }
                Some(pb::drain_response::Event::Error(error)) => {
                    return Err(client::engine_error_to_dynamo(&error));
                }
                _ => {}
            }
        }
        if !drain_complete {
            return Err(client::engine_shutdown(
                "OpenEngine Drain stream ended without COMPLETE",
            ));
        }

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(client::engine_shutdown(
                    "OpenEngine remained non-quiescent after terminal Drain",
                ));
            }
            let load_timeout = self.transport.connect_timeout.min(remaining);
            match tokio::time::timeout(
                load_timeout,
                pool.control_client().get_load(pb::GetLoadRequest {
                    include_per_rank: false,
                }),
            )
            .await
            {
                Ok(Ok(response)) if load_is_quiescent(response.get_ref()) => return Ok(()),
                Ok(Ok(response)) => {
                    let load = response.into_inner();
                    tracing::debug!(
                        running_requests = ?load.running_requests,
                        queued_requests = ?load.queued_requests,
                        active_kv_sessions = ?load.active_kv_sessions,
                        "waiting for OpenEngine to quiesce after terminal Drain"
                    );
                }
                Ok(Err(status)) => {
                    tracing::debug!(
                        %status,
                        "OpenEngine GetLoad failed while verifying drain quiescence"
                    );
                }
                Err(_) => {
                    tracing::debug!(
                        "OpenEngine GetLoad timed out while verifying drain quiescence"
                    );
                }
            }
            tokio::time::sleep(
                self.transport
                    .load_poll_interval
                    .min(deadline.saturating_duration_since(Instant::now())),
            )
            .await;
        }
    }

    fn drain_before_discovery_unregister(&self) -> bool {
        false
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        if self.bootstrap.model.supports_lora == Some(true) {
            Ok(vec!["load_lora", "unload_lora", "list_loras"]
                .into_iter()
                .map(str::to_string)
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        let pool = self
            .pool
            .get()
            .ok_or_else(|| client::engine_shutdown("engine_update called before start"))?;
        let mut grpc_client = pool.control_client();
        match update.as_str() {
            "load_lora" => {
                let name = required_string(&body, "lora_name")?;
                let uri = body
                    .get("source")
                    .and_then(|source| source.get("uri"))
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| body.get("source_path").and_then(serde_json::Value::as_str))
                    .ok_or_else(|| client::invalid_arg("load_lora requires source.uri"))?;
                let downloader = self.lora_downloader.as_ref().ok_or_else(|| {
                    client::invalid_arg(
                        "LoRA downloading is disabled; set DYN_LORA_ENABLED=true and mount the shared cache",
                    )
                })?;
                let path = downloader.download_if_needed(uri).await.map_err(|error| {
                    client::invalid_arg(format!("download LoRA `{name}`: {error}"))
                })?;
                validate_lora_directory(&path)?;
                let path = std::path::absolute(&path)
                    .map_err(|error| client::invalid_arg(format!("resolve LoRA path: {error}")))?;
                let id = i64::from(lora_name_to_id(name));
                let requested = pb::LoraAdapter {
                    lora_id: id,
                    lora_name: name.to_string(),
                    source_path: path.to_string_lossy().into_owned(),
                };
                let mut loaded = self.loaded_loras.lock().await;
                if let Some(existing) = loaded.get(name) {
                    if existing == &requested {
                        return Ok(lora_response(Some(existing.clone()), Some(true)));
                    }
                    return Err(client::invalid_arg(format!(
                        "LoRA adapter `{name}` conflicts with the loaded ID/path"
                    )));
                }
                let response = grpc_client
                    .load_lora(pb::LoadLoraRequest {
                        adapter: Some(requested.clone()),
                    })
                    .await
                    .map_err(|status| client::status_to_dynamo("LoadLora", status))?
                    .into_inner();
                let adapter = response.adapter.unwrap_or(requested);
                let discovery = self.lora_discovery.get().ok_or_else(|| {
                    client::engine_shutdown("LoRA model-card discovery is not initialized")
                })?;
                if let Err(error) = discovery.attach(&adapter).await {
                    let _ = grpc_client
                        .unload_lora(pb::UnloadLoraRequest {
                            lora_name: adapter.lora_name.clone(),
                        })
                        .await;
                    return Err(error);
                }
                loaded.insert(name.to_string(), adapter.clone());
                Ok(lora_response(Some(adapter), Some(response.already_loaded)))
            }
            "unload_lora" => {
                let name = required_string(&body, "lora_name")?;
                let mut loaded = self.loaded_loras.lock().await;
                let Some(adapter) = loaded.get(name).cloned() else {
                    return Err(client::invalid_arg(format!(
                        "LoRA adapter `{name}` is not loaded"
                    )));
                };
                let response = grpc_client
                    .unload_lora(pb::UnloadLoraRequest {
                        lora_name: name.to_string(),
                    })
                    .await
                    .map_err(|status| client::status_to_dynamo("UnloadLora", status))?
                    .into_inner();
                let discovery = self.lora_discovery.get().ok_or_else(|| {
                    client::engine_shutdown("LoRA model-card discovery is not initialized")
                })?;
                if let Err(detach) = discovery.detach(name).await {
                    // Restore remote logical registration so a failed local
                    // detach does not split local and engine state.
                    match grpc_client
                        .load_lora(pb::LoadLoraRequest {
                            adapter: Some(adapter.clone()),
                        })
                        .await
                    {
                        Ok(_) => return Err(detach),
                        Err(rollback) => {
                            loaded.remove(name);
                            return Err(client::engine_shutdown(format!(
                                "local LoRA detach failed: {detach}; remote rollback also failed: {rollback}"
                            )));
                        }
                    }
                }
                loaded.remove(name);
                Ok(lora_response(response.adapter.or(Some(adapter)), None))
            }
            "list_loras" => {
                let response = grpc_client
                    .list_loras(pb::ListLorasRequest {})
                    .await
                    .map_err(|status| client::status_to_dynamo("ListLoras", status))?
                    .into_inner();
                let count = response.adapters.len();
                Ok(serde_json::json!({
                    "status": "ok",
                    "loras": response.adapters.into_iter().map(lora_json).collect::<Vec<_>>(),
                    "count": count,
                }))
            }
            _ => Ok(
                serde_json::json!({"status": "error", "message": format!("unsupported engine update: {update}")}),
            ),
        }
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        Ok(Some(serde_json::json!({
            "token_ids": [1],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": true},
            "sampling_options": {"temperature": 0.0}
        })))
    }

    async fn on_model_ready(
        &self,
        endpoint: dynamo_runtime::component::Endpoint,
        base_model: LocalModel,
        model_type: ModelType,
        worker_type: WorkerType,
        needs: Vec<Vec<WorkerType>>,
    ) -> Result<(), DynamoError> {
        self.lora_discovery
            .set(LoraDiscovery {
                endpoint,
                base_model,
                model_type,
                worker_type,
                needs,
                models: tokio::sync::Mutex::new(HashMap::new()),
            })
            .map_err(|_| client::engine_shutdown("LoRA discovery already initialized"))
    }
}

async fn load_loop(
    channel: tonic::transport::Channel,
    publisher: Arc<SnapshotPublisher>,
    ranks: Vec<u32>,
    interval: std::time::Duration,
    cancel: CancellationToken,
) {
    let mut grpc_client = pb::control_client::ControlClient::new(channel);
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {
                let Ok(Ok(response)) = tokio::time::timeout(
                    interval,
                    grpc_client.get_load(pb::GetLoadRequest { include_per_rank: true }),
                ).await else {
                    continue;
                };
                let load = response.into_inner();
                if load.ranks.is_empty() {
                    if ranks.len() == 1 && let Some(&rank) = ranks.first() {
                        publish_load(&publisher, rank, load.used_kv_blocks, load.total_kv_blocks);
                    }
                } else {
                    for rank in load.ranks {
                        let Some(dp_rank) = rank.data_parallel_rank else { continue; };
                        if !ranks.contains(&dp_rank) { continue; }
                        publish_load(
                            &publisher,
                            dp_rank,
                            rank.used_kv_blocks,
                            rank.total_kv_blocks,
                        );
                    }
                }
            }
        }
    }
}

fn publish_load(publisher: &SnapshotPublisher, rank: u32, used: Option<u64>, total: Option<u64>) {
    let (Some(used), Some(total)) = (used, total) else {
        return;
    };
    if total == 0 || used > total {
        return;
    }
    publisher.publish(
        rank,
        ComponentSnapshot {
            kv_used_blocks: used,
            kv_total_blocks: total,
            gpu_cache_usage: used as f32 / total as f32,
            kv_cache_hit_rate: None,
            dp_rank: rank,
        },
    );
}

struct AbortOnDrop {
    request_id: Option<String>,
    client: Option<client::Control>,
    active_requests: Arc<Mutex<HashSet<String>>>,
}

impl AbortOnDrop {
    fn new(
        request_id: String,
        client: client::Control,
        active_requests: Arc<Mutex<HashSet<String>>>,
        shutting_down: Arc<AtomicBool>,
    ) -> Option<Self> {
        let mut active = active_requests.lock();
        if shutting_down.load(Ordering::SeqCst) {
            return None;
        }
        active.insert(request_id.clone());
        drop(active);
        Some(Self {
            request_id: Some(request_id),
            client: Some(client),
            active_requests,
        })
    }

    fn complete(&mut self) {
        if let Some(request_id) = self.request_id.as_ref() {
            self.active_requests.lock().remove(request_id);
        }
        self.request_id = None;
        self.client = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        let (Some(request_id), Some(client)) = (self.request_id.take(), self.client.take()) else {
            return;
        };
        if !self.active_requests.lock().remove(&request_id) {
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(abort_request(client, request_id));
        }
    }
}

async fn abort_request(mut client: client::Control, request_id: String) {
    if let Err(error) = client
        .abort(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId(request_id.clone())),
        })
        .await
    {
        tracing::debug!(%error, %request_id, "OpenEngine Abort failed");
    }
}

pub(crate) fn load_is_quiescent(load: &pb::LoadInfo) -> bool {
    load.running_requests == Some(0)
        && load.queued_requests.unwrap_or(0) == 0
        && load.active_kv_sessions.unwrap_or(0) == 0
}

fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
    model: Option<&str>,
    expected_engine: Option<&str>,
    expected_schema_release: Option<&str>,
) -> Result<Discovery, DynamoError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| client::engine_shutdown(format!("bootstrap runtime: {error}")))?
        .block_on(async {
            let mut grpc_client = client::connect(endpoint, transport).await?;
            client::discover(
                &mut grpc_client,
                model,
                expected_engine,
                expected_schema_release,
                transport.deadline,
            )
            .await
        })
}

fn engine_role(discovery: &Discovery) -> Result<pb::EngineRole, DynamoError> {
    match pb::EngineRole::try_from(discovery.engine.engine_role) {
        Ok(
            role @ (pb::EngineRole::Aggregated | pb::EngineRole::Prefill | pb::EngineRole::Decode),
        ) => Ok(role),
        Ok(pb::EngineRole::Unspecified) | Err(_) => Err(client::invalid_arg(format!(
            "OpenEngine advertised unsupported role value {}",
            discovery.engine.engine_role
        ))),
    }
}

fn role_to_mode(role: pb::EngineRole) -> DisaggregationMode {
    match role {
        pb::EngineRole::Prefill => DisaggregationMode::Prefill,
        pb::EngineRole::Decode => DisaggregationMode::Decode,
        _ => DisaggregationMode::Aggregated,
    }
}

fn component_for_role(role: pb::EngineRole) -> &'static str {
    if role == pb::EngineRole::Prefill {
        "prefill"
    } else {
        "backend"
    }
}

fn health_request() -> pb::HealthRequest {
    pb::HealthRequest {
        include_inference_probe: false,
        model: String::new(),
        role: pb::EngineRole::Unspecified as i32,
    }
}

fn nonempty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

fn validate_model(discovery: &Discovery) -> Result<(), DynamoError> {
    if discovery.model.supports_token_ids_input != Some(true) {
        return Err(client::invalid_arg(
            "OpenEngine model must explicitly advertise token-id input required by the Dynamo worker pipeline",
        ));
    }
    if discovery.engine.schema_revision >= 3 && discovery.model.tokenizer.is_none() {
        return Err(client::invalid_arg(
            "OpenEngine schema revision 3 requires tokenizer discovery",
        ));
    }
    if let Some(tokenizer) = discovery.model.tokenizer.as_ref() {
        if tokenizer.source.is_empty() {
            return Err(client::invalid_arg(
                "OpenEngine tokenizer discovery omitted its source",
            ));
        }
        // Dynamo's Rust tokenizer reproduces the default/auto tokenizer.json
        // path. Python slow tokenizers and implementation-specific modes can
        // produce different IDs and must fail before worker registration.
        if !matches!(tokenizer.mode.as_str(), "auto" | "default") {
            return Err(client::invalid_arg(format!(
                "OpenEngine tokenizer mode `{}` is not supported by Dynamo's local tokenizer pipeline",
                tokenizer.mode
            )));
        }
    }
    validate_role_connector(discovery)?;
    Ok(())
}

fn validate_role_connector(discovery: &Discovery) -> Result<(), DynamoError> {
    let role = engine_role(discovery)?;
    if role == pb::EngineRole::Aggregated {
        return Ok(());
    }
    let connector = discovery.engine.kv_connector.as_ref().ok_or_else(|| {
        client::invalid_arg("disaggregated OpenEngine role omitted KV connector discovery")
    })?;
    if connector.enabled != Some(true)
        || connector.supports_abort_cleanup != Some(true)
        || connector.supports_drain != Some(true)
        || connector.transfer_backend.is_empty()
        || connector.handoff_profile.is_empty()
        || connector.supported_protocols.is_empty()
    {
        return Err(client::invalid_arg(
            "disaggregated OpenEngine role requires an enabled, drainable, abort-cleanable KV connector with a backend and protocol",
        ));
    }
    if role == pb::EngineRole::Prefill && connector.supports_remote_prefill != Some(true) {
        return Err(client::invalid_arg(
            "OpenEngine prefill role does not advertise remote-prefill KV support",
        ));
    }
    if role == pb::EngineRole::Decode && connector.supports_decode_pull != Some(true) {
        return Err(client::invalid_arg(
            "OpenEngine decode role does not advertise decode-pull KV support",
        ));
    }
    if role == pb::EngineRole::Prefill
        && connector.supports_client_bootstrap == Some(true)
        && client_bootstrap_endpoint(discovery).is_none()
    {
        return Err(client::invalid_arg(
            "client-bootstrap prefill connector omitted a routable local endpoint",
        ));
    }
    Ok(())
}

fn apply_client_bootstrap(
    discovery: &Discovery,
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    generated: &mut pb::GenerateRequest,
) -> Result<(), DynamoError> {
    let Some(info) = request.bootstrap_info.as_ref() else {
        return Ok(());
    };
    let connector =
        discovery.engine.kv_connector.as_ref().ok_or_else(|| {
            client::invalid_arg("bootstrap request requires KV connector discovery")
        })?;
    if connector.supports_client_bootstrap != Some(true) {
        return Err(client::invalid_arg(
            "OpenEngine KV connector does not advertise client bootstrap",
        ));
    }
    if !(mode.is_prefill() || mode.is_decode()) {
        return Err(client::invalid_arg(
            "client bootstrap is valid only for prefill/decode roles",
        ));
    }
    if info.bootstrap_host.is_empty() || info.bootstrap_port == 0 {
        return Err(client::invalid_arg(
            "Dynamo bootstrap endpoint must have a non-empty host and nonzero port",
        ));
    }
    let protocol = connector
        .local_endpoints
        .iter()
        .map(|endpoint| endpoint.protocol.as_str())
        .find(|protocol| {
            !protocol.is_empty()
                && connector
                    .supported_protocols
                    .iter()
                    .any(|supported| supported == protocol)
        })
        .or_else(|| connector.supported_protocols.first().map(String::as_str))
        .ok_or_else(|| client::invalid_arg("client-bootstrap connector omitted its protocol"))?;
    let rank = if mode.is_prefill() {
        request
            .routing
            .as_ref()
            .and_then(|routing| routing.prefill_dp_rank)
    } else {
        request.routing.as_ref().and_then(|routing| routing.dp_rank)
    }
    .unwrap_or_else(|| {
        discovery
            .engine
            .parallelism
            .as_ref()
            .and_then(|parallelism| parallelism.data_parallel_start_rank)
            .unwrap_or(0)
    });
    let bootstrap = pb::KvBootstrap {
        endpoint: Some(pb::KvEndpoint {
            host: info.bootstrap_host.clone(),
            port: u32::from(info.bootstrap_port),
            protocol: protocol.to_string(),
        }),
        room_id: info.bootstrap_room,
    };
    let kv = generated.kv.get_or_insert_with(pb::KvOptions::default);
    match kv.session.as_mut() {
        Some(session) => {
            if session.handoff_profile != connector.handoff_profile {
                return Err(client::invalid_arg(format!(
                    "prefill handoff profile `{}` does not match connector profile `{}`",
                    session.handoff_profile, connector.handoff_profile
                )));
            }
            if let Some(existing) = session.bootstrap.as_ref()
                && existing != &bootstrap
            {
                return Err(client::invalid_arg(
                    "prefill handoff bootstrap conflicts with Dynamo routing bootstrap",
                ));
            }
            session.bootstrap = Some(bootstrap);
        }
        None => {
            kv.session = Some(pb::KvSessionRef {
                session_id: request_id.to_string(),
                transfer_backend: connector.transfer_backend.clone(),
                endpoints: Vec::new(),
                dp_rank: rank,
                attributes_struct: None,
                handoff_profile: connector.handoff_profile.clone(),
                bootstrap: Some(bootstrap),
            });
        }
    }
    Ok(())
}

fn validate_decode_handoff(
    discovery: &Discovery,
    request: &pb::GenerateRequest,
    mode: DisaggregationMode,
) -> Result<(), DynamoError> {
    if !mode.is_decode() {
        return Ok(());
    }
    let session = request
        .kv
        .as_ref()
        .and_then(|kv| kv.session.as_ref())
        .ok_or_else(|| client::invalid_arg("decode request omitted its KV session"))?;
    let connector = discovery
        .engine
        .kv_connector
        .as_ref()
        .ok_or_else(|| client::invalid_arg("decode engine omitted KV connector discovery"))?;
    if session.transfer_backend != connector.transfer_backend {
        return Err(client::invalid_arg(format!(
            "handoff transfer backend `{}` is incompatible with decode backend `{}`",
            session.transfer_backend, connector.transfer_backend
        )));
    }
    for endpoint in &session.endpoints {
        if !connector
            .supported_protocols
            .iter()
            .any(|protocol| protocol == &endpoint.protocol)
        {
            return Err(client::invalid_arg(format!(
                "handoff protocol `{}` is not supported by the decode KV connector",
                endpoint.protocol
            )));
        }
    }
    validate_session_profile_and_bootstrap(connector, session, None)
}

fn validate_prefill_handoff(
    discovery: &Discovery,
    session: &pb::KvSessionRef,
    requested_bootstrap: Option<&pb::KvBootstrap>,
) -> Result<(), DynamoError> {
    let connector =
        discovery.engine.kv_connector.as_ref().ok_or_else(|| {
            client::engine_shutdown("prefill engine omitted KV connector discovery")
        })?;
    if session.transfer_backend != connector.transfer_backend {
        return Err(client::engine_shutdown(format!(
            "OpenEngine PrefillReady transfer backend `{}` does not match discovered backend `{}`",
            session.transfer_backend, connector.transfer_backend
        )));
    }
    for endpoint in &session.endpoints {
        if !connector
            .supported_protocols
            .iter()
            .any(|protocol| protocol == &endpoint.protocol)
        {
            return Err(client::engine_shutdown(format!(
                "OpenEngine PrefillReady protocol `{}` is not supported by the discovered KV connector",
                endpoint.protocol
            )));
        }
    }
    if !advertised_dp_ranks(discovery)?.contains(&session.dp_rank) {
        return Err(client::engine_shutdown(format!(
            "OpenEngine PrefillReady dp_rank {} is outside the advertised parallelism range",
            session.dp_rank
        )));
    }
    validate_session_profile_and_bootstrap(connector, session, requested_bootstrap)
        .map_err(|error| client::engine_shutdown(error.to_string()))
}

fn validate_session_profile_and_bootstrap(
    connector: &pb::KvConnectorInfo,
    session: &pb::KvSessionRef,
    requested_bootstrap: Option<&pb::KvBootstrap>,
) -> Result<(), DynamoError> {
    if session.handoff_profile != connector.handoff_profile {
        return Err(client::invalid_arg(format!(
            "KV handoff profile `{}` does not match connector profile `{}`",
            session.handoff_profile, connector.handoff_profile
        )));
    }
    if connector.supports_client_bootstrap == Some(true) {
        let bootstrap = session.bootstrap.as_ref().ok_or_else(|| {
            client::invalid_arg("client-bootstrap KV session omitted typed bootstrap")
        })?;
        let endpoint = bootstrap.endpoint.as_ref().ok_or_else(|| {
            client::invalid_arg("client-bootstrap KV session omitted bootstrap endpoint")
        })?;
        if endpoint.host.is_empty()
            || endpoint.port == 0
            || endpoint.port > u32::from(u16::MAX)
            || !connector
                .supported_protocols
                .iter()
                .any(|protocol| protocol == &endpoint.protocol)
        {
            return Err(client::invalid_arg(
                "client-bootstrap KV session has an invalid endpoint",
            ));
        }
        if let Some(expected) = requested_bootstrap
            && bootstrap != expected
        {
            return Err(client::invalid_arg(
                "OpenEngine PrefillReady returned a different bootstrap session than requested",
            ));
        }
    } else if session.bootstrap.is_some() {
        return Err(client::invalid_arg(
            "KV session supplied bootstrap for a connector that does not advertise it",
        ));
    }
    Ok(())
}

fn advertised_dp_ranks(discovery: &Discovery) -> Result<HashSet<u32>, DynamoError> {
    let parallelism = discovery.engine.parallelism.as_ref();
    let start = parallelism
        .and_then(|parallelism| parallelism.data_parallel_start_rank)
        .unwrap_or(0);
    let count = parallelism
        .and_then(|parallelism| parallelism.data_parallel_size)
        .unwrap_or(1)
        .max(1);
    let end = start.checked_add(count).ok_or_else(|| {
        client::invalid_arg("OpenEngine data-parallel rank range overflows uint32")
    })?;
    Ok((start..end).collect())
}

fn is_frontend_only_nvext(value: &serde_json::Value, cache_namespace: Option<&str>) -> bool {
    let Some(nvext) = value.as_object() else {
        return false;
    };
    nvext.iter().all(|(key, value)| match key.as_str() {
        // These fields select data that Dynamo's response postprocessor builds
        // from the normal backend output. They do not alter engine execution.
        "extra_fields" => value.as_array().is_some_and(|fields| {
            fields.iter().all(|field| {
                field.as_str().is_some_and(|field| {
                    matches!(
                        field,
                        "worker_id"
                            | "timing"
                            | "engine_data"
                            | "stop_reason"
                            | "completion_token_ids"
                            | "prompt_logprobs"
                    )
                })
            })
        }),
        // The preprocessor duplicates cache_salt into RoutingHints. Consume
        // this copy only when it agrees with the value sent over OpenEngine.
        "cache_salt" => value
            .as_str()
            .filter(|salt| !salt.is_empty())
            .is_some_and(|salt| cache_namespace == Some(salt)),
        // metadata_upload and token_in require backend-specific behavior and
        // remain rejected rather than being silently dropped.
        _ => false,
    })
}

fn validate_request_capabilities(
    discovery: &Discovery,
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Result<(), DynamoError> {
    let generation = discovery.model.generation.as_ref();
    let sampling = &request.sampling_options;
    let stopping = &request.stop_conditions;
    let output = &request.output_options;
    let n = u32::from(sampling.n.unwrap_or(1).max(1));
    if n > 1
        && generation
            .and_then(|capabilities| capabilities.max_num_sequences)
            .is_none_or(|maximum| n > maximum)
    {
        return Err(client::invalid_arg(format!(
            "OpenEngine model does not advertise support for {n} output sequences"
        )));
    }
    if let Some(max_tokens) = stopping.max_tokens
        && discovery
            .model
            .max_output_tokens
            .is_some_and(|maximum| max_tokens > maximum)
    {
        return Err(client::invalid_arg(format!(
            "requested max_tokens {max_tokens} exceeds OpenEngine maximum {}",
            discovery.model.max_output_tokens.unwrap_or_default()
        )));
    }
    if sampling
        .best_of
        .is_some_and(|best_of| best_of != sampling.n.unwrap_or(1))
        || sampling.use_beam_search == Some(true)
        || sampling.length_penalty.is_some()
    {
        return Err(client::invalid_arg(
            "OpenEngine does not support best_of or beam-search semantics",
        ));
    }
    if sampling.seed.is_some_and(|seed| seed < 0) {
        return Err(client::invalid_arg(
            "OpenEngine sampling seed must be non-negative",
        ));
    }
    if stopping
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
        || stopping.max_thinking_tokens.is_some()
    {
        return Err(client::invalid_arg(
            "OpenEngine cannot preserve visible stop-token or thinking-budget semantics",
        ));
    }
    if output.skip_special_tokens == Some(false)
        || output.formatted_prompt == Some(true)
        || output.return_tokens_as_token_ids == Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine cannot preserve the requested output formatting semantics",
        ));
    }
    if sampling.include_stop_str_in_output == Some(true)
        && generation.and_then(|value| value.supports_stop_in_output) != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support including stop matches in output",
        ));
    }
    if request.require_reasoning {
        return Err(client::invalid_arg(
            "OpenEngine does not support require_reasoning",
        ));
    }
    let supports_client_bootstrap = discovery
        .engine
        .kv_connector
        .as_ref()
        .and_then(|connector| connector.supports_client_bootstrap)
        == Some(true);
    if mode.is_prefill() || mode.is_decode() {
        if supports_client_bootstrap != request.bootstrap_info.is_some() {
            return Err(client::invalid_arg(if supports_client_bootstrap {
                "OpenEngine connector requires Dynamo client bootstrap"
            } else {
                "OpenEngine connector does not support Dynamo client bootstrap"
            }));
        }
    } else if request.bootstrap_info.is_some() {
        return Err(client::invalid_arg(
            "aggregated OpenEngine requests cannot carry Dynamo bootstrap",
        ));
    }
    if let Some(extra) = request.extra_args.as_ref() {
        let object = extra
            .as_object()
            .ok_or_else(|| client::invalid_arg("extra_args must be an object"))?;
        for (key, value) in object {
            let is_supported = matches!(
                key.as_str(),
                "messages" | "bypass_prefix_cache" | "disable_prefix_cache"
            ) || (key == "nvext"
                && is_frontend_only_nvext(
                    value,
                    request
                        .routing
                        .as_ref()
                        .and_then(|routing| routing.cache_namespace.as_deref()),
                ))
                || (key == "mm_hashes"
                    && value.as_array().is_some_and(|hashes| {
                        !hashes.is_empty()
                            && hashes.iter().all(|hash| {
                                hash.as_str().is_some_and(|hash| {
                                    hash.len() == 16
                                        && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                                })
                            })
                    })
                    && request
                        .multi_modal_data
                        .as_ref()
                        .and_then(|media| media.get("image_url"))
                        .is_some_and(|images| {
                            value
                                .as_array()
                                .is_some_and(|hashes| hashes.len() == images.len())
                        }))
                || (key == "formatted_prompt"
                    && request
                        .multi_modal_data
                        .as_ref()
                        .is_some_and(|media| media.values().any(|items| !items.is_empty()))
                    && value.as_str().is_some_and(|prompt| !prompt.is_empty()));
            if !is_supported {
                return Err(client::invalid_arg(format!(
                    "OpenEngine sidecar cannot preserve extra_args field `{key}`"
                )));
            }
        }
    }
    let routing = request.routing.as_ref();
    if routing.and_then(|routing| routing.priority).is_some()
        && generation.and_then(|value| value.supports_priority) != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support per-request priority",
        ));
    }
    if routing
        .and_then(|routing| routing.cache_namespace.as_deref())
        .is_some_and(|namespace| !namespace.is_empty())
        && generation.and_then(|value| value.supports_cache_salt) != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support cache-salt isolation",
        ));
    }
    if routing
        .and_then(|routing| routing.lora_name.as_deref())
        .is_some_and(|name| !name.is_empty())
        && discovery.model.supports_lora != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support LoRA selection",
        ));
    }
    validate_logprob_capability(
        "prompt",
        output.prompt_logprobs,
        generation.and_then(|value| value.prompt_logprobs.as_ref()),
    )?;
    validate_logprob_capability(
        "output",
        output.logprobs,
        generation.and_then(|value| value.output_logprobs.as_ref()),
    )?;
    validate_guided_capability(sampling.guided_decoding.as_ref(), generation)?;
    let requests_prefix_cache_bypass = request.extra_args.as_ref().is_some_and(|args| {
        args.get("bypass_prefix_cache")
            .and_then(serde_json::Value::as_bool)
            .or_else(|| {
                args.get("disable_prefix_cache")
                    .and_then(serde_json::Value::as_bool)
            })
            == Some(true)
    });
    if requests_prefix_cache_bypass
        && discovery
            .model
            .generation
            .as_ref()
            .and_then(|generation| generation.supports_prefix_cache_bypass)
            != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support per-request prefix-cache bypass",
        ));
    }
    let Some(media) = request.multi_modal_data.as_ref() else {
        if request.mm_processor_kwargs.is_some() {
            return Err(client::invalid_arg("media options require media inputs"));
        }
        return Ok(());
    };
    let Some(capabilities) = discovery.model.multimodal_capabilities.as_ref() else {
        return Err(client::invalid_arg(
            "OpenEngine model does not advertise multimodal support",
        ));
    };
    let supported = if mode.is_prefill() || mode.is_decode() {
        &capabilities.prefill_decode_modalities
    } else {
        &capabilities.aggregate_modalities
    };
    for (key, values) in media {
        let modality = match key.as_str() {
            "image_url" => pb::Modality::Image,
            "video_url" => pb::Modality::Video,
            "audio_url" => pb::Modality::Audio,
            _ => {
                return Err(client::invalid_arg(format!(
                    "unsupported media key `{key}`"
                )));
            }
        };
        if !supported.contains(&(modality as i32)) {
            return Err(client::invalid_arg(format!(
                "OpenEngine model does not support {modality:?} in {mode:?} mode"
            )));
        }
        for value in values {
            let source_type = match value {
                MultimodalData::Url(url) if url.as_str().starts_with("data:") => {
                    pb::MediaSourceType::DataUri
                }
                MultimodalData::RawUrl(url) if url.starts_with("data:") => {
                    pb::MediaSourceType::DataUri
                }
                MultimodalData::Url(_) | MultimodalData::RawUrl(_) => pb::MediaSourceType::Url,
                MultimodalData::Decoded(_) => {
                    return Err(client::invalid_arg(
                        "OpenEngine sidecar cannot dereference decoded/RDMA media; configure URL/data passthrough",
                    ));
                }
            };
            if !capabilities.source_types.contains(&(source_type as i32)) {
                return Err(client::invalid_arg(format!(
                    "OpenEngine model does not support {source_type:?} media sources"
                )));
            }
        }
    }
    if request.mm_processor_kwargs.is_some()
        && capabilities.supports_per_request_media_options != Some(true)
    {
        return Err(client::invalid_arg(
            "OpenEngine model does not support per-request media options",
        ));
    }
    Ok(())
}

fn validate_logprob_capability(
    kind: &str,
    requested: Option<u32>,
    capabilities: Option<&pb::LogprobCapabilities>,
) -> Result<(), DynamoError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    let supported = capabilities.is_some_and(|capabilities| {
        capabilities.supported == Some(true)
            && capabilities
                .candidate_selection_modes
                .contains(&(pb::CandidateTokenSelectionMode::TopN as i32))
            && capabilities
                .max_top_n
                .is_none_or(|maximum| requested <= maximum)
    });
    if !supported {
        return Err(client::invalid_arg(format!(
            "OpenEngine model does not support requested {kind} top-{requested} logprobs"
        )));
    }
    Ok(())
}

fn validate_guided_capability(
    requested: Option<&dynamo_backend_common::GuidedDecodingOptions>,
    generation: Option<&pb::GenerationCapabilities>,
) -> Result<(), DynamoError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if requested
        .backend
        .as_ref()
        .is_some_and(|backend| !backend.is_empty())
        || requested.whitespace_pattern.is_some()
    {
        return Err(client::invalid_arg(
            "OpenEngine does not advertise guided backend overrides or whitespace patterns",
        ));
    }
    let mode = if requested
        .json
        .as_ref()
        .is_some_and(serde_json::Value::is_null)
    {
        pb::GuidedDecodingMode::JsonObject
    } else if requested.json.is_some() {
        pb::GuidedDecodingMode::JsonSchema
    } else if requested.regex.is_some() {
        pb::GuidedDecodingMode::Regex
    } else if requested.grammar.is_some() {
        pb::GuidedDecodingMode::EbnfGrammar
    } else if requested.structural_tag.is_some() {
        pb::GuidedDecodingMode::StructuralTag
    } else if requested
        .choice
        .as_ref()
        .is_some_and(|choice| !choice.is_empty())
    {
        pb::GuidedDecodingMode::Choice
    } else {
        return Err(client::invalid_arg(
            "guided decoding request has no constraint",
        ));
    };
    let capabilities = generation.and_then(|value| value.guided_decoding.as_ref());
    if capabilities.is_none_or(|capabilities| {
        capabilities.supported != Some(true) || !capabilities.modes.contains(&(mode as i32))
    }) {
        return Err(client::invalid_arg(format!(
            "OpenEngine model does not support guided decoding mode {mode:?}"
        )));
    }
    Ok(())
}

fn build_engine_config(discovery: &Discovery) -> EngineConfig {
    let model = &discovery.model;
    let parallelism = discovery.engine.parallelism.unwrap_or_default();
    let capacity = discovery.engine.capacity.as_ref();
    let served_model_name =
        nonempty(&model.served_model_name).unwrap_or_else(|| model.model_id.clone());
    let mut aliases = model.served_model_aliases.clone();
    if model.model_id != served_model_name {
        aliases.push(model.model_id.clone());
    }
    aliases.retain(|alias| !alias.is_empty() && alias != &served_model_name);
    aliases.sort();
    aliases.dedup();
    let connector = discovery.engine.kv_connector.as_ref();
    let multimodal = model.multimodal_capabilities.as_ref();
    let tokenizer = model.tokenizer.as_ref();
    EngineConfig {
        model: model.model_id.clone(),
        served_model_name: Some(served_model_name),
        model_aliases: aliases,
        runtime_data: [
            (
                "openengine_engine".to_string(),
                serde_json::json!(discovery.engine.engine_name),
            ),
            (
                "openengine_schema_release".to_string(),
                serde_json::json!(discovery.engine.schema_release),
            ),
            (
                "openengine_tokenizer_source".to_string(),
                serde_json::json!(
                    tokenizer
                        .map(|value| value.source.as_str())
                        .unwrap_or_default()
                ),
            ),
            (
                "openengine_tokenizer_mode".to_string(),
                serde_json::json!(
                    tokenizer
                        .map(|value| value.mode.as_str())
                        .unwrap_or_default()
                ),
            ),
            (
                "openengine_handoff_profile".to_string(),
                serde_json::json!(
                    connector
                        .map(|value| value.handoff_profile.as_str())
                        .unwrap_or_default()
                ),
            ),
            (
                "openengine_routing_image_token_id".to_string(),
                serde_json::json!(multimodal.and_then(|value| value.routing_image_token_id)),
            ),
        ]
        .into_iter()
        .collect(),
        llm: Some(LlmRegistration {
            context_length: model.max_context_length,
            kv_cache_block_size: capacity.and_then(|value| value.kv_block_size),
            total_kv_blocks: capacity.and_then(|value| value.total_kv_blocks),
            max_num_seqs: capacity.and_then(|value| value.max_running_requests),
            max_num_batched_tokens: capacity.and_then(|value| value.max_batched_tokens),
            data_parallel_size: parallelism.data_parallel_size,
            data_parallel_start_rank: parallelism.data_parallel_start_rank,
            bootstrap_host: client_bootstrap_endpoint(discovery).map(|value| value.host.clone()),
            bootstrap_port: client_bootstrap_endpoint(discovery)
                .and_then(|value| u16::try_from(value.port).ok()),
        }),
    }
}

fn client_bootstrap_endpoint(discovery: &Discovery) -> Option<&pb::KvEndpoint> {
    if engine_role(discovery).ok()? != pb::EngineRole::Prefill {
        return None;
    }
    let connector = discovery.engine.kv_connector.as_ref()?;
    if connector.supports_client_bootstrap != Some(true) {
        return None;
    }
    connector.local_endpoints.iter().find(|endpoint| {
        !endpoint.host.is_empty()
            && endpoint.port > 0
            && endpoint.port <= u32::from(u16::MAX)
            && !endpoint.protocol.is_empty()
            && connector
                .supported_protocols
                .iter()
                .any(|protocol| protocol == &endpoint.protocol)
    })
}

fn prompt_logprobs_from_openengine(tokens: Vec<pb::TokenInfo>) -> PromptLogprobs {
    tokens
        .into_iter()
        .map(|token| {
            let mut candidates = HashMap::new();
            if let Some(logprob) = token.logprob {
                candidates.insert(
                    token.token_id,
                    PromptLogprobEntry {
                        logprob: logprob as f32,
                        rank: token.rank,
                        decoded_token: (!token.token.is_empty()).then_some(token.token),
                    },
                );
            }
            for candidate in token.candidates {
                candidates.insert(
                    candidate.token_id,
                    PromptLogprobEntry {
                        logprob: candidate.logprob as f32,
                        rank: candidate.rank,
                        decoded_token: (!candidate.token.is_empty()).then_some(candidate.token),
                    },
                );
            }
            (!candidates.is_empty()).then_some(candidates)
        })
        .collect()
}

fn attach_prompt_logprobs(output: &mut LLMEngineOutput, prompt_logprobs: Option<PromptLogprobs>) {
    if let Some(prompt_logprobs) = prompt_logprobs {
        output.engine_data = Some(serde_json::json!({"prompt_logprobs": prompt_logprobs}));
    }
}

fn finish_output(reason: pb::FinishReason, stop_match: Option<pb::StopMatch>) -> LLMEngineOutput {
    let mut output = match reason {
        pb::FinishReason::Length => LLMEngineOutput::length(),
        pb::FinishReason::Cancelled => LLMEngineOutput::cancelled(),
        _ => LLMEngineOutput::stop(),
    };
    output.stop_reason = stop_match.and_then(|stop_match| match stop_match.r#match? {
        pb::stop_match::Match::StopTokenId(token_id)
        | pb::stop_match::Match::EosTokenId(token_id) => {
            Some(dynamo_backend_common::StopReason::Int(i64::from(token_id)))
        }
        pb::stop_match::Match::StopText(text) => {
            Some(dynamo_backend_common::StopReason::String(text))
        }
    });
    output
}

fn openengine_usage(value: &pb::Usage) -> CompletionUsage {
    use dynamo_protocols::types::{CompletionTokensDetails, PromptTokensDetails};
    CompletionUsage {
        prompt_tokens: value.prompt_tokens,
        completion_tokens: value.completion_tokens,
        total_tokens: value.total_tokens,
        prompt_tokens_details: value.cached_prompt_tokens.map(|cached_tokens| {
            PromptTokensDetails {
                audio_tokens: None,
                cached_tokens: Some(cached_tokens),
            }
        }),
        completion_tokens_details: value.reasoning_tokens.map(|reasoning_tokens| {
            CompletionTokensDetails {
                reasoning_tokens: Some(reasoning_tokens),
                ..Default::default()
            }
        }),
    }
}

fn required_string<'a>(body: &'a serde_json::Value, key: &str) -> Result<&'a str, DynamoError> {
    body.get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| client::invalid_arg(format!("`{key}` is required")))
}

fn build_lora_downloader() -> Option<LoRADownloader> {
    if !dynamo_llm::lora::lora_serving_enabled() {
        return None;
    }
    let cache = LoRACache::from_env()
        .map_err(|error| tracing::warn!(%error, "failed to configure LoRA cache"))
        .ok()?;
    let mut sources: Vec<Arc<dyn LoRASource>> = vec![Arc::new(LocalLoRASource::new())];
    if let Ok(source) = S3LoRASource::from_env() {
        sources.push(Arc::new(source));
    }
    Some(LoRADownloader::new(sources, cache))
}

fn validate_lora_directory(path: &std::path::Path) -> Result<(), DynamoError> {
    if !path.is_dir() || !path.join("adapter_config.json").is_file() {
        return Err(client::invalid_arg(format!(
            "LoRA path `{}` is not a PEFT adapter directory (adapter_config.json missing)",
            path.display()
        )));
    }
    Ok(())
}

fn lora_json(value: pb::LoraAdapter) -> serde_json::Value {
    serde_json::json!({
        "lora_id": value.lora_id,
        "lora_name": value.lora_name,
        "source_path": value.source_path,
    })
}

fn lora_response(
    value: Option<pb::LoraAdapter>,
    already_loaded: Option<bool>,
) -> serde_json::Value {
    let adapter = value.map(lora_json).unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "status": "ok",
        "lora_id": adapter.get("lora_id").cloned(),
        "lora_name": adapter.get("lora_name").cloned(),
        "source_path": adapter.get("source_path").cloned(),
        "already_loaded": already_loaded,
    })
}

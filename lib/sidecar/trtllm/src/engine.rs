// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for TensorRT-LLM's OpenEngine (`openengine.v1`) gRPC server.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, usage,
};
use dynamo_sidecar_common::{
    GrpcEndpoint, GrpcTransportConfig, SidecarStartupError, startup_deadline,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, ModelLimits, TrtllmClient};
use crate::convert::{ResponseState, build_generate_request};
use crate::model::ConfiguredModel;

const ALREADY_STARTED: &str = "TensorRT-LLM sidecar has already started";

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

pub struct TrtllmSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    model: ConfiguredModel,
    /// Disaggregation role this worker plays. Selects the `context_only` /
    /// `kv.session` divergence in `convert`.
    mode: DisaggregationMode,
    client: OnceCell<TrtllmClient>,
    /// Engine limits resolved at `start` from `--context-length` and
    /// `Control.GetModelInfo`, so `generate` can derive a default `max_tokens`
    /// for requests that omit one.
    limits: OnceCell<ModelLimits>,
    cancel: CancellationToken,
}

impl TrtllmSidecarEngine {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        model: ConfiguredModel,
        mode: DisaggregationMode,
    ) -> Self {
        Self {
            endpoint,
            transport,
            model,
            mode,
            client: OnceCell::new(),
            limits: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::try_from_args(argv).map_err(SidecarStartupError::into_dynamo)
    }

    /// Parse injected arguments while retaining Clap's structured exit error.
    ///
    /// Embedded callers use this to distinguish help and version output from
    /// Dynamo startup failures without changing `from_args`'s error contract.
    pub fn try_from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), SidecarStartupError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)?;
        Self::from_parsed(args).map_err(Into::into)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(client::invalid_argument("model-path must not be empty"));
        }
        let mode = args.sidecar.common.disaggregation_mode;
        if mode.is_encode() {
            return Err(client::invalid_argument(
                "encode mode is not supported by the TensorRT-LLM sidecar",
            ));
        }
        if args.sidecar.common.route_to_encoder {
            return Err(client::invalid_argument(
                "route-to-encoder is not supported by the TensorRT-LLM sidecar",
            ));
        }

        let endpoint = args.sidecar.grpc_endpoint;
        let transport = args.sidecar.grpc.config();
        let model = ConfiguredModel {
            source: args.model_path,
            // Absent unless `--context-length` supplied one; `start` falls back
            // to the server's `Control.GetModelInfo` report.
            context_length: args.context_length,
        };
        let engine = Self::new(endpoint, transport, model.clone(), mode);
        let config = WorkerConfig {
            namespace: args.sidecar.common.namespace,
            // Every disaggregated role registers under its own component so
            // the frontend can target each separately; only an aggregated
            // worker uses the operator-configured one.
            component: if mode == DisaggregationMode::Aggregated {
                args.sidecar.common.component
            } else {
                mode.discovery_component().to_string()
            },
            endpoint: args.sidecar.common.endpoint,
            endpoint_types: args.sidecar.common.endpoint_types,
            custom_jinja_template: args.sidecar.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser: args.sidecar.common.dyn_tool_call_parser,
            reasoning_parser: args.sidecar.common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: args
                .sidecar
                .common
                .exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: mode,
            route_to_encoder: false,
            enable_rl: args.sidecar.common.enable_rl,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for TrtllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(client::engine_shutdown(ALREADY_STARTED));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.transport.connections.get(),
            "connecting to TensorRT-LLM gRPC"
        );
        let client = TrtllmClient::connect(&self.endpoint, self.transport).await?;
        let connection_count = client.connection_count();

        // `--context-length` wins over what the engine reports, and is the
        // only source when the engine reports nothing usable. The resolved
        // value backs both the registered window and the default-`max_tokens`
        // path in `convert::max_tokens`.
        let mut model = self.model.clone();
        let limits = match model.context_length {
            // Configured: the engine is consulted once, to cross-check the
            // value and to learn its output cap. It may not answer at all --
            // an older server has no Control service -- and that must not stop
            // a worker whose window the operator already supplied.
            Some(configured) => {
                let reported = match client.model_limits(&model.source).await {
                    Ok(reported) => reported,
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            configured_context_length = configured,
                            "Control.GetModelInfo failed; using the configured --context-length"
                        );
                        ModelLimits::default()
                    }
                };
                if let Some(engine_context_length) = reported.context_length
                    && engine_context_length != configured
                {
                    tracing::warn!(
                        configured_context_length = configured,
                        engine_context_length,
                        "--context-length disagrees with the context length TensorRT-LLM \
                         reported; using the configured --context-length"
                    );
                }
                ModelLimits {
                    context_length: Some(configured),
                    ..reported
                }
            }
            // Nothing configured: the engine is the only source, and it binds
            // its port before the model finishes loading, so wait for it.
            None => {
                match client
                    .wait_for_model_limits(
                        &model.source,
                        startup_deadline(self.transport.startup_deadline)?,
                        self.transport.retry_interval,
                    )
                    .await
                {
                    Ok(limits) => limits,
                    // Registering without a window is still useful: requests
                    // that carry their own `max_tokens` are served, and only
                    // the ones that omit it are rejected.
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "no context length is available; requests that omit max_tokens \
                             will be rejected. Supply --context-length."
                        );
                        ModelLimits::default()
                    }
                }
            }
        };
        model.context_length = limits.context_length;
        let _ = self.limits.set(limits);

        self.client
            .set(client)
            .map_err(|_| client::engine_shutdown(ALREADY_STARTED))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %model.source,
            context_length = ?limits.context_length,
            max_output_tokens = ?limits.max_output_tokens,
            "TensorRT-LLM gRPC is ready"
        );
        Ok(model.engine_config())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request = build_generate_request(
            &request,
            &request_id,
            &self.model.source,
            self.limits.get().copied(),
            self.mode,
        )?;
        let mut state = ResponseState::new(&request, self.mode);
        let cancel = self.cancel.clone();
        // A decode request that took a handoff has KV transferred into it, and
        // the transceiver releases those blocks when the engine finishes the
        // request -- not when the client goes away. Dropping the stream on
        // cancellation would strand the prefill worker's blocks. Defer only
        // until the first token proves the transfer landed; deferring past
        // that would let a cancelled request generate its whole budget with no
        // consumer. A decode-mode request without a handoff ran locally and
        // has nothing to strand.
        let defer_request_cancellation = self.mode.is_decode() && request.prefill_result.is_some();
        let stopped_ctx = ctx.inner_arc();
        // Hoisted: `stopped()` is an async-trait method, so re-creating it per
        // streamed chunk costs a boxed future and a waker registration on every
        // token.
        let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
        let shutdown = cancel.clone();
        let mut shutdown_cancellation = Box::pin(async move { shutdown.cancelled().await });

        // The same deferral applies here, not just to the streaming loop below.
        // `generate` sends the request and then awaits response headers, so
        // losing this race can drop a request the engine has already accepted
        // and begun pulling KV for; and an already-stopped context would skip
        // the dispatch entirely, leaving the prefill worker's blocks with no
        // decode leg to claim them. Both strand exactly what the deferral
        // exists to protect. Shutdown still wins -- the process is going away.
        let stream = tokio::select! {
            biased;
            _ = &mut request_cancellation, if !defer_request_cancellation => None,
            _ = &mut shutdown_cancellation => None,
            result = client.generate(proto_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            let mut transfer_settled = false;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut request_cancellation, if !defer_request_cancellation || transfer_settled => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = &mut shutdown_cancellation => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response) {
                                Ok(Some(output)) => {
                                    transfer_settled |= !output.token_ids.is_empty();
                                    let terminal = output.finish_reason.is_some();
                                    yield Ok(output);
                                    if terminal {
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            },
                            Ok(None) => {
                                yield Err(client::protocol_error(
                                    "Generate ended before a terminal response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("Generate", status));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else {
            return;
        };
        if let Err(error) = client.abort(ctx.id().to_string()).await {
            // Escaped: the message embeds the engine's gRPC status text, and
            // this site logs at the default level, so raw newlines from the
            // peer would let it forge what look like separate log records.
            tracing::warn!(
                request_id = ctx.id(),
                error = %error.to_string().escape_debug(),
                "TensorRT-LLM Control.Abort failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("TensorRT-LLM sidecar shutdown complete");
        Ok(())
    }
}

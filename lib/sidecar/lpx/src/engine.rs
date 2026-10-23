// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for the LPX engines' OpenEngine (`openengine.v1`) gRPC
//! servers.

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DynamoError, EngineConfig, GenerateContext, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, PreprocessedRequest, RuntimeConfig, WorkerConfig, usage,
};
use dynamo_sidecar_common::{
    EngineBootstrapResult, GrpcEndpoint, GrpcTransportConfig, SidecarStartupError, startup_deadline,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, ServeArgs, TopologyRole};
use crate::client::{self, OpenEngineClient, engine_shutdown, invalid_argument};
use crate::convert::{
    DispatchFailure, ResponseState, build_generate_request, build_handoff_cleanup_request,
    dispatch_failure, stream_status, trace_headers,
};
use crate::model::{DiscoveredEngine, engine_label, expected_role, role_name};
use crate::run::Topology;
use crate::topology::Role;

const ALREADY_STARTED: &str = "the LPX sidecar has already started";

/// What the sidecar executable runs, chosen from its arguments; [`crate::run`]
/// runs each.
pub enum Launch<S> {
    /// Serve the engine with the runtime settings and engine bootstrap that
    /// [`dynamo_sidecar_common::run`] takes.
    Serve(S),
    /// Serve the engine as [`Launch::Serve`] does, with a topology consumer
    /// that starts before the engine bootstrap waits for the engine: a
    /// conductor needs topology to become ready.
    ServeWithTopology(S, Topology),
    /// Only a topology role, with no engine: the publisher beside the Agent
    /// allocation's leader, or a consumer on a rank that serves no engine.
    Topology(Topology),
    /// Stand by as this rank of a multi-Pod engine, through
    /// [`crate::run_standby`]; only rank 0 serves.
    Standby(NonZeroU32),
}

type OutputStream = BoxStream<'static, Result<LLMEngineOutput, DynamoError>>;

/// Terminal output for a cancelled request, with the usage so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

fn single(item: Result<LLMEngineOutput, DynamoError>) -> OutputStream {
    Box::pin(futures::stream::once(async move { item }))
}

pub struct LpxSidecarEngine {
    endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
    discovered: DiscoveredEngine,
    /// Built during the bootstrap; `start` returns it unchanged.
    engine_config: EngineConfig,
    /// The configured connection pool, opened by `start` and named after the
    /// identified engine. The bootstrap uses only a one-connection probe.
    client: OnceCell<Arc<OpenEngineClient>>,
    cancel: CancellationToken,
}

impl LpxSidecarEngine {
    pub(crate) fn new(
        endpoint: GrpcEndpoint,
        transport: GrpcTransportConfig,
        discovered: DiscoveredEngine,
        engine_config: EngineConfig,
    ) -> Self {
        Self {
            endpoint,
            transport,
            discovered,
            engine_config,
            client: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    /// Parses the process arguments. A nonzero `--node-rank` selects standby,
    /// unless the sidecar is a topology consumer; only rank 0 serves the
    /// engine or publishes topology, and a publisher serves no engine. On
    /// rank 0, engine discovery runs once the sidecar runner has started its
    /// probes and connected the runtime. A CLI error exits the process.
    pub fn from_cli() -> Result<
        Launch<(
            RuntimeConfig,
            impl std::future::Future<Output = EngineBootstrapResult<Self>> + Send,
        )>,
        DynamoError,
    > {
        match Self::launch(<Args as clap::Parser>::parse()) {
            Err(SidecarStartupError::Cli(error)) => error.exit(),
            launch => launch.map_err(SidecarStartupError::into_dynamo),
        }
    }

    fn launch(
        args: Args,
    ) -> Result<
        Launch<(
            RuntimeConfig,
            impl std::future::Future<Output = EngineBootstrapResult<Self>> + Send,
        )>,
        SidecarStartupError,
    > {
        let rank = NonZeroU32::new(args.node_rank);
        let consumer = || {
            Topology::new(
                &args.common,
                Role::Consumer {
                    socket: args.topology.socket.clone(),
                },
            )
        };
        // Decided before any serving argument is checked, so neither a
        // standby nor a topology role alone depends on the serving
        // configuration.
        match (args.topology.role, rank) {
            (
                None | Some(TopologyRole::Publisher | TopologyRole::PublisherAndServe),
                Some(rank),
            ) => {
                return Ok(Launch::Standby(rank));
            }
            (Some(TopologyRole::Publisher), None) => {
                return Ok(Launch::Topology(Topology::new(
                    &args.common,
                    Role::Publisher {
                        agent_socket: args.topology.agent_socket.clone(),
                    },
                )));
            }
            (Some(TopologyRole::Consumer), Some(_)) => return Ok(Launch::Topology(consumer())),
            (None | Some(TopologyRole::Consumer | TopologyRole::PublisherAndServe), None) => {}
        }
        let topology = match args.topology.role {
            Some(TopologyRole::Consumer) => Some(consumer()),
            Some(TopologyRole::PublisherAndServe) => Some(Topology::new(
                &args.common,
                Role::Publisher {
                    agent_socket: args.topology.agent_socket.clone(),
                },
            )),
            _ => None,
        };
        let args = args.serve()?;
        validate_args(&args)?;
        let serve = (args.sidecar.common.runtime.clone(), discover(args));
        Ok(match topology {
            None => Launch::Serve(serve),
            Some(topology) => Launch::ServeWithTopology(serve, topology),
        })
    }

    /// Parses `argv` as the executable does, but only serves: a nonzero
    /// `--node-rank`, which the executable runs as a standby, and a topology
    /// role are refused. Discovery runs when the returned future is polled.
    pub fn try_from_args_async(
        argv: Vec<String>,
    ) -> Result<
        (
            RuntimeConfig,
            impl std::future::Future<Output = EngineBootstrapResult<Self>> + Send,
        ),
        SidecarStartupError,
    > {
        match Self::launch(<Args as clap::Parser>::try_parse_from(argv)?)? {
            Launch::Serve(serve) => Ok(serve),
            Launch::ServeWithTopology(..) | Launch::Topology(_) => Err(invalid_argument(
                "a topology role is supported only by the dynamo-lpx-sidecar executable",
            )
            .into()),
            Launch::Standby(rank) => Err(invalid_argument(format!(
                "node-rank {rank} stands by, which only the dynamo-lpx-sidecar executable supports"
            ))
            .into()),
        }
    }

    fn connected(&self) -> Result<&Arc<OpenEngineClient>, DynamoError> {
        self.client
            .get()
            .ok_or_else(|| engine_shutdown("the LPX sidecar is not started"))
    }
}

/// Rejects what this sidecar cannot serve before any runtime starts. The
/// engine is not known yet; its role is checked once it answers.
fn validate_args(args: &ServeArgs) -> Result<(), DynamoError> {
    if args.model_path.trim().is_empty() {
        return Err(invalid_argument("model-path must not be empty"));
    }
    let common = &args.sidecar.common;
    expected_role(common.disaggregation_mode)?;
    if common.route_to_encoder {
        return Err(invalid_argument(
            "route-to-encoder is not supported by the LPX sidecar",
        ));
    }
    Ok(())
}

/// Waits for the engine to be ready, identifies it, checks any
/// `--served-model-name` against the name it reports, and builds the
/// registration.
///
/// It holds one probe connection through the wait and drops it here; `start`
/// opens the configured pool, named after the identified engine.
async fn discover(args: ServeArgs) -> Result<(LpxSidecarEngine, WorkerConfig), DynamoError> {
    let endpoint = args.sidecar.grpc_endpoint.clone();
    let transport = args.sidecar.grpc.config();
    let mode = args.sidecar.common.disaggregation_mode;
    // One budget for connecting and for waiting until the engine is ready.
    let deadline = startup_deadline(transport.startup_deadline)?;
    tracing::info!(%endpoint, startup_deadline = ?transport.startup_deadline,
        "Discovering the LPX OpenEngine server");
    let probe_transport = GrpcTransportConfig {
        connections: NonZeroUsize::MIN,
        ..transport
    };
    let probe = OpenEngineClient::connect(
        client::UNIDENTIFIED_PEER,
        &endpoint,
        probe_transport,
        deadline,
    )
    .await?;
    let model_info = probe
        .wait_for_model_info(deadline, transport.retry_interval, client::RPC_TIMEOUT)
        .await?;
    let server_info = probe.server_info(deadline).await?;
    drop(probe);

    let discovered = DiscoveredEngine::check(server_info, model_info, mode)?;
    discovered.check_served_name(args.served_model_name.as_deref())?;
    let engine_config = discovered.engine_config();
    let worker_config = discovered.worker_config(args);
    let engine = LpxSidecarEngine::new(endpoint, transport, discovered, engine_config);
    Ok((engine, worker_config))
}

#[async_trait]
impl LLMEngine for LpxSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(engine_shutdown(ALREADY_STARTED));
        }
        // The engine answered GetModelInfo during the bootstrap, so this is a
        // reconnect to a server that is already serving.
        let discovered = &self.discovered;
        let deadline = startup_deadline(self.transport.startup_deadline)?;
        let client = OpenEngineClient::connect(
            engine_label(&discovered.engine_name),
            &self.endpoint,
            self.transport,
            deadline,
        )
        .await?;
        let connections = client.connection_count();
        self.client
            .set(Arc::new(client))
            .map_err(|_| engine_shutdown(ALREADY_STARTED))?;
        tracing::info!(
            endpoint = %self.endpoint,
            engine = &*discovered.engine_name,
            engine_version = %discovered.engine_version.escape_debug(),
            instance_id = %discovered.instance_id.escape_debug(),
            role = role_name(discovered.role),
            served_model_name = %discovered.served_model_name.escape_debug(),
            context_length = discovered.context_length,
            kv_block_size = ?discovered.kv_block_size,
            total_kv_blocks = ?discovered.total_kv_blocks,
            max_running_requests = ?discovered.max_running_requests,
            connections,
            "LPX OpenEngine server identified"
        );
        Ok(self.engine_config.clone())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<OutputStream, DynamoError> {
        let client = self.connected()?;
        let request_id = ctx.id().to_string();
        // A request this sidecar rejects is a deterministic 4xx: deliver it as
        // the first stream item so the frontend does not migrate it.
        let (proto_request, rejection) = match build_generate_request(
            &request,
            &request_id,
            &self.discovered,
        ) {
            Ok(proto_request) => (proto_request, None),
            Err(error) => {
                let Some(prefill) = &request.prefill_result else {
                    return Ok(single(Err(error)));
                };
                match build_handoff_cleanup_request(&request.token_ids, prefill, &request_id) {
                    Ok(cleanup) => (cleanup, Some(error)),
                    Err(cleanup_error) => {
                        tracing::warn!(
                            %request_id,
                            error = %cleanup_error.to_string().escape_debug(),
                            "Cannot dispatch decode handoff cleanup; prefill blocks stay held until their lease expires"
                        );
                        return Ok(single(Err(error)));
                    }
                }
            }
        };
        let mut state = ResponseState::new(&request, &self.discovered);
        // A decode leg, the one request with a prefill result, carries the
        // prefill worker's KV blocks, which only this leg's `Generate`
        // releases. The frontend dispatches it even after the client left, so
        // it is sent even when the request is already cancelled. The engine
        // then runs a detached decode request for one token, which pulls the
        // blocks, so the sidecar can drop the stream at once.
        let needs_handoff_dispatch = request.prefill_result.is_some();
        let stopped_ctx = ctx.inner_arc();
        // Hoisted so the loop below does not rebuild the boxed future per token.
        let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
        let shutdown = self.cancel.child_token();
        let mut shutdown_cancellation = Box::pin(async move { shutdown.cancelled().await });

        let dispatched = tokio::select! {
            biased;
            _ = &mut request_cancellation, if !needs_handoff_dispatch => None,
            // The process is going away; nothing would read the response.
            _ = &mut shutdown_cancellation => None,
            result = client.generate(proto_request, trace_headers()) => Some(result),
        };
        if let Some(error) = rejection {
            if let Some(Err(status)) = &dispatched {
                tracing::warn!(
                    %request_id,
                    engine = &*self.discovered.engine_name,
                    error = %status.to_string().escape_debug(),
                    "Decode handoff cleanup Generate failed"
                );
            }
            drop(dispatched);
            return Ok(single(Err(error)));
        }
        let mut stream = match dispatched {
            None => return Ok(single(Ok(cancelled(&state)))),
            Some(Ok(stream)) => stream,
            Some(Err(status)) => {
                return match dispatch_failure(status) {
                    DispatchFailure::Retryable(error) => Err(error),
                    DispatchFailure::Rejected(error) => Ok(single(Err(error))),
                };
            }
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    // Dropping the stream detaches the request in the hybrid
                    // engine and cancels it in the LPU-only engine. It is
                    // dropped before the terminal is yielded, so neither
                    // waits for the consumer to poll again.
                    _ = &mut request_cancellation => {
                        drop(stream);
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = &mut shutdown_cancellation => {
                        drop(stream);
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => match message {
                        Ok(Some(response)) => match state.convert(response) {
                            Ok(Some(output)) => {
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
                            yield Err(state.protocol_error(
                                "Generate ended before a terminal response",
                            ));
                            break;
                        }
                        Err(status) => {
                            yield Err(stream_status(status));
                            break;
                        }
                    },
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else {
            return;
        };
        if let Err(error) = client.abort(ctx.id().to_string()).await {
            // Escaped: the message carries the engine's status text.
            tracing::warn!(
                request_id = ctx.id(),
                engine = &*self.discovered.engine_name,
                error = %error.to_string().escape_debug(),
                "Control.Abort failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("LPX sidecar shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod topology_launch_tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn quasar_can_publish_topology_and_serve_only_on_the_allocation_leader() {
        for rank in [0, 1] {
            let args = Args::try_parse_from([
                "lpx",
                "--topology-role",
                "publisher-and-serve",
                "--node-rank",
                &rank.to_string(),
                "--grpc-endpoint",
                "unix:///run/openengine/engine.sock",
                "--model-path",
                "/model",
            ])
            .unwrap();
            let launch = LpxSidecarEngine::launch(args).unwrap();
            if rank == 0 {
                assert!(matches!(launch, Launch::ServeWithTopology(..)));
            } else {
                assert!(matches!(launch, Launch::Standby(_)));
            }
        }
    }
}

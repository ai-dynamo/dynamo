// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin sidecar client for SGLang's native `sglang.api.v1.SglangService`.

use std::collections::HashSet;
use std::future::Future;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig,
};
use tokio::time::error::Elapsed;
use tokio::time::{Instant, timeout_at};
use tonic::transport::Channel;

use crate::discovery::{self, Discovery, EngineInfo};
use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

pub type Client = SglangServiceClient<Channel>;

/// The gRPC service the sidecar speaks, published in runtime data.
pub(crate) const GRPC_SERVICE: &str = pb::sglang_service_server::SERVICE_NAME;

pub(crate) const WORKER_GROUP_KEY: &str = "sglang_worker_group_id";

/// The engine's place in the deployment and its KV-event publishers, as
/// GetServerInfo's `deployment_info` reports them. `dp_size` counts
/// attention-DP ranks, and the source list is authoritative even when empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeMetadata {
    pub node_rank: u32,
    pub nnodes: u32,
    pub dp_size: u32,
    pub dist_init_addr: Option<String>,
    pub kv_event_sources: Vec<LocalKvEventSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalKvEventSource {
    pub dp_rank: u32,
    pub endpoint: String,
    pub topic: String,
    pub block_size: u32,
}

impl NodeMetadata {
    /// Validate the reported topology and sources.
    pub(crate) fn new(
        node_rank: u32,
        nnodes: u32,
        dp_size: u32,
        dist_init_addr: Option<String>,
        kv_event_sources: Vec<LocalKvEventSource>,
    ) -> anyhow::Result<Self> {
        let metadata = Self {
            node_rank,
            nnodes,
            dp_size,
            dist_init_addr,
            kv_event_sources,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.nnodes > 0 && self.node_rank < self.nnodes,
            "invalid GetServerInfo node topology"
        );
        ensure!(self.dp_size > 0, "GetServerInfo dp_size must be positive");
        let deployment_dp_size = self.dp_size;
        ensure!(
            self.nnodes == 1
                || self
                    .dist_init_addr
                    .as_ref()
                    .is_some_and(|addr| !addr.trim().is_empty()),
            "multinode deployments require dist_init_addr for their worker group id"
        );
        let mut ranks = HashSet::new();
        let mut endpoints = HashSet::new();
        for source in &self.kv_event_sources {
            ensure!(
                source.dp_rank < deployment_dp_size,
                "KV source rank {} is outside deployment DP size {}",
                source.dp_rank,
                deployment_dp_size
            );
            ensure!(
                ranks.insert(source.dp_rank),
                "duplicate local KV source rank {}",
                source.dp_rank
            );
            ensure!(
                endpoints.insert(&source.endpoint),
                "duplicate local KV source endpoint {}",
                source.endpoint
            );
            ensure!(
                source.block_size > 0,
                "KV source block_size must be positive"
            );
            validate_endpoint(&source.endpoint)?;
        }
        Ok(())
    }

    /// Every source must publish at the registered block size.
    pub(crate) fn validate_registration(&self, block_size: Option<u32>) -> anyhow::Result<()> {
        for source in &self.kv_event_sources {
            ensure!(
                Some(source.block_size) == block_size,
                "KV source block_size does not match the registration"
            );
        }
        Ok(())
    }

    pub(crate) async fn worker_group_id(
        &self,
        deadline: Instant,
    ) -> anyhow::Result<Option<String>> {
        if self.nnodes == 1 {
            return Ok(None);
        }
        let raw = self
            .dist_init_addr
            .as_deref()
            .context("missing dist_init_addr")?
            .trim();
        let url = if raw.contains("://") {
            raw.to_owned()
        } else {
            format!("tcp://{raw}")
        };
        let address = url::Url::parse(&url).context("invalid dist_init_addr")?;
        ensure!(
            address.scheme() == "tcp",
            "dist_init_addr must be a TCP address"
        );
        validate_endpoint(&url)?;
        // Derive the group key from the shared rendezvous address, not from
        // the local source or gRPC address.
        resolve_rendezvous(move || address.socket_addrs(|| None), deadline)
            .await
            .and_then(worker_group_id_from_addresses)
            .map(Some)
    }
}

// getaddrinfo cannot be cancelled. Keep it off Tokio's workers and blocking
// pool so dropping startup or shutting down the runtime never waits for DNS.
// Each startup performs one lookup; an abandoned thread exits when DNS returns.
async fn resolve_rendezvous(
    resolve: impl FnOnce() -> std::io::Result<Vec<SocketAddr>> + Send + 'static,
    deadline: Instant,
) -> anyhow::Result<Vec<SocketAddr>> {
    anyhow::ensure!(
        Instant::now() < deadline,
        "rendezvous DNS startup deadline elapsed"
    );
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("sglang-rendezvous-dns".into())
        .spawn(move || {
            let _ = tx.send(resolve());
        })
        .context("failed to start rendezvous DNS resolver")?;
    timeout_at(deadline, rx)
        .await
        .context("timed out resolving SGLang rendezvous address")?
        .context("rendezvous DNS resolver stopped")?
        .context("failed to resolve SGLang rendezvous address")
}

fn worker_group_id_from_addresses(mut resolved: Vec<SocketAddr>) -> anyhow::Result<String> {
    // Nodes may receive the same DNS answers in different orders.
    resolved.sort_unstable();
    let address = resolved
        .first()
        .context("dist_init_addr resolved to no addresses")?;
    Ok(format!("dist_init:tcp://{address}"))
}

fn validate_endpoint(endpoint: &str) -> anyhow::Result<()> {
    if let Some(path) = endpoint.strip_prefix("ipc://") {
        ensure!(
            (path.starts_with('/') || path.starts_with('@'))
                && path.len() > 1
                && !path.contains('\0'),
            "IPC source must have an absolute or abstract socket path"
        );
        return Ok(());
    }
    let url = url::Url::parse(endpoint).context("invalid KV source endpoint")?;
    let host = url
        .host_str()
        .context("KV source endpoint requires a host")?;
    ensure!(
        url.scheme() == "tcp" && url.port().is_some_and(|port| port > 0),
        "KV source must use tcp://HOST:PORT or ipc://PATH"
    );
    ensure!(
        !matches!(host, "*" | "0.0.0.0" | "[::]" | "::"),
        "KV source endpoint must be dialable, not a wildcard bind address"
    );
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!("KV source TCP endpoint must contain only host and port");
    }
    Ok(())
}

/// Connect and discover the engine before the worker registers, which fixes
/// its role and identity.
///
/// `bootstrap`: true for synchronous constructors before logging setup;
/// false for deferred launcher discovery.
pub(crate) async fn bootstrap_discover(
    endpoint: &GrpcEndpoint,
    transport: &GrpcTransportConfig,
    bootstrap: bool,
) -> Result<EngineInfo, DynamoError> {
    let deadline = Instant::now() + transport.startup_deadline;
    let mut client = connect(endpoint, transport, deadline, bootstrap).await?;
    discovery::engine_info(discover(&mut client, deadline).await?)
}

/// `bootstrap`: true for synchronous constructors before logging setup;
/// false for deferred launcher discovery and `LLMEngine::start`.
pub async fn connect(
    uri: &GrpcEndpoint,
    cfg: &GrpcTransportConfig,
    deadline: Instant,
    bootstrap: bool,
) -> Result<Client, DynamoError> {
    let cfg = GrpcTransportConfig {
        connections: NonZeroUsize::MIN,
        ..*cfg
    };
    let pool = connect_pool(uri, cfg, deadline, bootstrap).await?;
    Ok(client_from_channel(pool.next_channel()))
}

/// `GrpcChannelPool` starts its own deadline from `cfg.startup_deadline`;
/// `deadline` also bounds it, because callers start their startup budget
/// before connecting.
async fn connect_pool(
    uri: &GrpcEndpoint,
    cfg: GrpcTransportConfig,
    deadline: Instant,
    bootstrap: bool,
) -> Result<GrpcChannelPool, DynamoError> {
    timeout_at(
        deadline,
        GrpcChannelPool::connect("SGLang", uri, cfg, bootstrap),
    )
    .await
    .map_err(|_| {
        cannot_connect(format!(
            "could not reach SGLang gRPC at {uri} within the startup deadline"
        ))
    })?
}

pub(crate) fn client_from_channel(channel: Channel) -> Client {
    SglangServiceClient::new(channel)
        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
}

/// Fixed-size pool of independent HTTP/2 connections. Generation calls are
/// round-robined so high concurrency does not funnel through one codec task.
pub struct Pool {
    clients: Vec<Client>,
    next: AtomicUsize,
}

impl Pool {
    pub async fn connect(
        uri: &GrpcEndpoint,
        cfg: &GrpcTransportConfig,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        // bootstrap=false: a pool is connected only from LLMEngine::start,
        // after the tracing subscriber is installed. See connect()'s own doc
        // comment.
        let pool = connect_pool(uri, *cfg, deadline, false).await?;
        let clients = (0..pool.len())
            .map(|_| client_from_channel(pool.next_channel()))
            .collect();
        Ok(Self {
            clients,
            next: AtomicUsize::new(0),
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn stream_client(&self) -> Client {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[index].clone()
    }

    pub fn control_client(&self) -> Client {
        self.clients[0].clone()
    }
}

/// GetServerInfo, GetModelInfo and ListModels, mapped to typed discovery.
/// An engine that answers any of them UNIMPLEMENTED does not serve the
/// contract: SGLang's Python server serves none of them, and a Rust server
/// that predates sgl-project/sglang#42118 lacks ListModels.
pub(crate) async fn discover(
    client: &mut Client,
    deadline: Instant,
) -> Result<Discovery, DynamoError> {
    let server_info = contract_rpc_with_deadline(
        "GetServerInfo",
        GRPC_SERVICE,
        deadline,
        client.get_server_info(pb::GetServerInfoRequest {}),
    )
    .await?
    .into_inner();
    let model = contract_rpc_with_deadline(
        "GetModelInfo",
        GRPC_SERVICE,
        deadline,
        client.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await?
    .into_inner();
    let models = contract_rpc_with_deadline(
        "ListModels",
        GRPC_SERVICE,
        deadline,
        client.list_models(pb::ListModelsRequest {}),
    )
    .await?
    .into_inner()
    .models;
    discovery::parse_discovery(model, server_info, models)
}

/// SGLang's Rust server answers UNAVAILABLE until its scheduler responds, and
/// the engine's readiness loop retries that error.
pub async fn health_check(client: &mut Client, deadline: Instant) -> Result<bool, DynamoError> {
    rpc_with_deadline(
        "HealthCheck",
        deadline,
        client.health_check(pb::HealthCheckRequest {}),
    )
    .await
    .map(|response| response.into_inner().healthy)
}

/// Returns whether an active generation matched `rid`.
pub async fn abort(client: &mut Client, rid: &str, timeout: Duration) -> Result<bool, DynamoError> {
    let request = pb::AbortRequest {
        rid: rid.to_string(),
        abort_all: false,
    };
    rpc_with_deadline("Abort", Instant::now() + timeout, client.abort(request))
        .await
        .map(|response| response.into_inner().success)
}

/// The Abort RPC on one connection. SGLang's Rust server has no HTTP
/// `/abort_request`; this RPC also cancels requests opened through native
/// HTTP `/generate`.
#[derive(Clone)]
pub(crate) struct AbortRpc(Client);

impl AbortRpc {
    pub(crate) fn new(client: Client) -> Self {
        Self(client)
    }

    /// Failures are logged, not returned.
    pub(crate) async fn abort(&self, request_id: &str, timeout: Duration) {
        match abort(&mut self.0.clone(), request_id, timeout).await {
            Ok(true) => {}
            Ok(false) => tracing::debug!(request_id, "SGLang Abort matched no active request"),
            Err(error) => tracing::debug!(request_id, %error, "SGLang Abort RPC failed"),
        }
    }

    /// An Abort RPC whose every call fails to connect.
    #[cfg(test)]
    pub(crate) fn unreachable() -> Self {
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        Self(client_from_channel(channel))
    }
}

pub(crate) async fn rpc_with_deadline<T, F>(
    rpc: &str,
    deadline: Instant,
    future: F,
) -> Result<T, DynamoError>
where
    F: Future<Output = Result<T, tonic::Status>>,
{
    deadline_result(rpc, timeout_at(deadline, future).await)
}

/// A discovery RPC. An engine that does not serve `service`, or serves an
/// older version of it, answers UNIMPLEMENTED.
pub(crate) async fn contract_rpc_with_deadline<T, F>(
    rpc: &str,
    service: &str,
    deadline: Instant,
    future: F,
) -> Result<T, DynamoError>
where
    F: Future<Output = Result<T, tonic::Status>>,
{
    match timeout_at(deadline, future).await {
        Ok(Err(status)) if status.code() == tonic::Code::Unimplemented => {
            Err(contract_mismatch(rpc, service))
        }
        result => deadline_result(rpc, result),
    }
}

fn deadline_result<T>(
    rpc: &str,
    result: Result<Result<T, tonic::Status>, Elapsed>,
) -> Result<T, DynamoError> {
    match result {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(status)) => Err(status_to_dynamo(rpc, status)),
        Err(_) => Err(connection_timeout(format!(
            "{rpc} exceeded the configured deadline"
        ))),
    }
}

fn contract_mismatch(rpc: &str, service: &str) -> DynamoError {
    invalid_arg(format!(
        "the SGLang engine does not serve {service}/{rpc} (UNIMPLEMENTED); \
         start SGLang with SGLANG_RUST_SERVER=1; the sidecar requires an SGLang build \
         with sgl-project/sglang#42118 and the api.v1 sidecar changes listed in \
         lib/sidecar/sglang/README.md"
    ))
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_arg(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

/// The frontend returns `message` to the client, so it takes only fixed request-validation text.
pub(crate) fn invalid_request(message: &'static str) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message)
        .public_message(message)
        .build()
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub(crate) fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
}

pub(crate) fn cancelled(message: impl Into<String>) -> DynamoError {
    backend(BackendError::Cancelled, message)
}

pub fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend(BackendError::Unknown, message)
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::OutOfRange => {
            BackendError::InvalidArgument
        }
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_backend_common::{BackendError, ErrorType};
    use tokio::net::TcpListener;
    use tokio::time::Instant;
    use tonic::transport::Endpoint;

    use super::{
        LocalKvEventSource, NodeMetadata, client_from_channel, discover, rpc_with_deadline,
        status_to_dynamo, worker_group_id_from_addresses,
    };

    fn source(dp_rank: u32, endpoint: &str) -> LocalKvEventSource {
        LocalKvEventSource {
            dp_rank,
            endpoint: endpoint.to_string(),
            topic: String::new(),
            block_size: 64,
        }
    }

    #[test]
    fn rejects_invalid_kv_sources() {
        let duplicate = vec![
            source(0, "tcp://127.0.0.1:5557"),
            source(0, "tcp://127.0.0.1:5558"),
        ];
        assert!(NodeMetadata::new(0, 1, 1, None, duplicate).is_err());
        let unbound = vec![source(0, "tcp://0.0.0.0:5557")];
        assert!(NodeMetadata::new(0, 1, 1, None, unbound).is_err());
        // Several nodes need dist_init_addr for their worker group id.
        assert!(NodeMetadata::new(0, 2, 1, None, Vec::new()).is_err());
    }

    #[tokio::test]
    async fn rendezvous_dns_deadline_bounds_a_blocked_resolver() {
        let (release, blocked) = std::sync::mpsc::channel::<()>();
        let error = super::resolve_rendezvous(
            move || {
                let _ = blocked.recv_timeout(std::time::Duration::from_secs(3));
                Ok(vec![])
            },
            Instant::now() + std::time::Duration::from_millis(25),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("timed out resolving"));
        drop(release);
    }

    #[test]
    fn cancelling_dns_does_not_hold_up_runtime_shutdown() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel::<()>();
        runtime.block_on(async {
            let lookup = super::resolve_rendezvous(
                move || {
                    let _ = started_tx.send(());
                    // Bound the injected delay even if runtime shutdown regresses.
                    let _ = blocked.recv_timeout(std::time::Duration::from_secs(3));
                    Ok(vec![])
                },
                Instant::now() + std::time::Duration::from_secs(10),
            );
            tokio::select! {
                result = lookup => panic!("resolver returned before cancellation: {result:?}"),
                _ = started_rx => {},
            }
        });
        let shutdown_started = std::time::Instant::now();
        drop(runtime);
        let elapsed = shutdown_started.elapsed();
        drop(release);
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "shutdown waited for DNS: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn rendezvous_dns_propagates_resolution_errors() {
        let error = super::resolve_rendezvous(
            || Err(std::io::Error::other("injected resolver failure")),
            Instant::now() + std::time::Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("injected resolver failure"));
    }

    #[test]
    fn worker_group_id_is_independent_of_dns_answer_order() {
        let addresses = vec![
            "[::1]:2345".parse().unwrap(),
            "127.0.0.2:2345".parse().unwrap(),
            "127.0.0.1:2345".parse().unwrap(),
        ];
        let reversed = addresses.iter().copied().rev().collect();
        let expected = "dist_init:tcp://127.0.0.1:2345";
        assert_eq!(worker_group_id_from_addresses(addresses).unwrap(), expected);
        assert_eq!(worker_group_id_from_addresses(reversed).unwrap(), expected);
        assert_eq!(
            worker_group_id_from_addresses(Vec::new())
                .unwrap_err()
                .to_string(),
            "dist_init_addr resolved to no addresses"
        );
    }

    #[test]
    fn rpc_status_mapping_preserves_error_kind_and_context() {
        for (code, kind) in [
            (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
            (tonic::Code::NotFound, BackendError::InvalidArgument),
            (tonic::Code::OutOfRange, BackendError::InvalidArgument),
            (tonic::Code::Unavailable, BackendError::CannotConnect),
            (tonic::Code::Cancelled, BackendError::Cancelled),
            (
                tonic::Code::DeadlineExceeded,
                BackendError::ConnectionTimeout,
            ),
            (tonic::Code::Internal, BackendError::Unknown),
        ] {
            let error =
                status_to_dynamo("GetModelInfo", tonic::Status::new(code, "native failure"));
            assert_eq!(error.error_type(), ErrorType::Backend(kind));
            assert!(error.to_string().contains("GetModelInfo: native failure"));
            assert!(error.to_string().contains(&format!("{code:?}")));
        }
    }

    #[tokio::test]
    async fn rpc_deadline_preserves_success_status_and_timeout() {
        let deadline = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            rpc_with_deadline("HealthCheck", deadline, async { Ok(17) })
                .await
                .unwrap(),
            17
        );
        let error = rpc_with_deadline::<(), _>("HealthCheck", deadline, async {
            Err(tonic::Status::unavailable("offline"))
        })
        .await
        .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );

        let error =
            rpc_with_deadline::<(), _>("HealthCheck", Instant::now(), std::future::pending())
                .await
                .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::ConnectionTimeout)
        );
        assert!(
            error
                .to_string()
                .contains("HealthCheck exceeded the configured deadline")
        );
    }

    #[tokio::test]
    async fn unimplemented_rpc_reports_the_contract_mismatch() {
        let error = super::contract_rpc_with_deadline::<(), _>(
            "GetServerInfo",
            "sglang.api.v1.SglangService",
            Instant::now() + Duration::from_secs(60),
            async { Err(tonic::Status::unimplemented("")) },
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("SGLANG_RUST_SERVER=1"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn discovery_deadline_bounds_a_half_open_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let peer = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        let mut client = client_from_channel(channel);
        let started = Instant::now();
        let result = discover(&mut client, started + Duration::from_millis(100)).await;
        peer.abort();

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}

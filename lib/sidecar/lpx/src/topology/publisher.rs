// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The publisher, beside the Agent allocation's leader: it registers the
//! allocation's topology endpoint and relays each request to the Agent.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow, bail};
use async_trait::async_trait;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::stream;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Status};

use super::proto::{GetSnapshotRequest, GetSnapshotResponse, Identity};
use super::{
    AgentClient, COMPONENT, RelayRequest, RelayResponse, check_expected, check_snapshot,
    endpoint_name,
};

/// How long the publisher waits for the Agent's answer to one request.
const AGENT_TIMEOUT: Duration = Duration::from_secs(5);
/// How often the publisher asks an Agent that has no snapshot yet.
const STARTUP_RETRY: Duration = Duration::from_secs(1);
/// How often an unchanged startup failure is logged again.
const RETRY_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Waits for the Agent's first valid snapshot, registers the topology
/// endpoint of its identity in `namespace`, and relays each request on it to
/// the Agent until `shutdown`.
///
/// It asks every second while the Agent answers UNAVAILABLE or cannot be
/// reached, with no deadline. It fails on an invalid first snapshot and on
/// any other answer, which only a broken Agent gives.
pub(crate) async fn publish(
    drt: &DistributedRuntime,
    namespace: &str,
    agent: AgentClient,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let Some(identity) = first_identity(&agent, &shutdown).await? else {
        return Ok(());
    };
    let endpoint = drt
        .namespace(namespace)?
        .component(COMPONENT)?
        .endpoint(endpoint_name(&identity));
    let relay = Arc::new(Relay {
        agent,
        identity: identity.clone(),
    });
    // No model card and no health-check payload: the endpoint is not an
    // inference worker, and its health is the Agent's to report. Registration
    // is not abandoned midway, which could leave it behind.
    let started = endpoint
        .endpoint_builder()
        .handler(Ingress::for_engine(relay)?)
        .graceful_shutdown(true)
        .start_with_registration()
        .await?;
    tracing::info!(
        namespace,
        component = COMPONENT,
        endpoint = %endpoint.name(),
        deployment = %identity.deployment.escape_debug(),
        allocation = %identity.allocation.escape_debug(),
        build = %identity.build.escape_debug(),
        "Publishing the LPX allocation's topology"
    );
    shutdown.cancelled().await;
    started.shutdown().await
}

/// The identity of the Agent's first valid snapshot, or `None` on shutdown.
async fn first_identity(
    agent: &AgentClient,
    shutdown: &CancellationToken,
) -> anyhow::Result<Option<Identity>> {
    let mut logged: Option<(String, Instant)> = None;
    loop {
        let answer = tokio::select! {
            _ = shutdown.cancelled() => return Ok(None),
            answer = agent.get_snapshot(GetSnapshotRequest::default(), AGENT_TIMEOUT) => answer,
        };
        match answer {
            Ok(response) => {
                let snapshot = response
                    .snapshot
                    .as_ref()
                    .context("the Agent's topology service answered OK without a snapshot")?;
                let identity = check_snapshot(snapshot).map_err(|error| {
                    anyhow!("the Agent's first topology snapshot is invalid: {error}")
                })?;
                return Ok(Some(identity.clone()));
            }
            Err(status) if status.code() == Code::Unavailable => {
                let message = status.message().to_string();
                let now = Instant::now();
                if logged.as_ref().is_none_or(|(last, at)| {
                    *last != message || now.duration_since(*at) >= RETRY_LOG_INTERVAL
                }) {
                    tracing::info!(
                        error = %message.escape_debug(),
                        "Waiting for the Agent's first LPX topology snapshot"
                    );
                    logged = Some((message, now));
                }
            }
            Err(status) => bail!(
                "the Agent's topology service answered {:?}: {}",
                status.code(),
                status.message().escape_debug()
            ),
        }
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(None),
            _ = tokio::time::sleep(STARTUP_RETRY) => {}
        }
    }
}

/// Relays each request to the Agent, live.
struct Relay {
    agent: AgentClient,
    /// The identity the endpoint is registered for.
    identity: Identity,
}

impl Relay {
    async fn answer(&self, request: &RelayRequest) -> Result<GetSnapshotResponse, Status> {
        let request = request.decode().map_err(|error| {
            Status::invalid_argument(format!("malformed topology request: {error}"))
        })?;
        let expected = request.expected.clone();
        let response = self.agent.get_snapshot(request, AGENT_TIMEOUT).await?;
        let snapshot = response.snapshot.as_ref().ok_or_else(|| {
            Status::failed_precondition("the Agent answered OK without a snapshot")
        })?;
        let identity = check_snapshot(snapshot).map_err(|error| {
            Status::failed_precondition(format!("the Agent's snapshot is invalid: {error}"))
        })?;
        check_expected(identity, Some(&self.identity))
            .and_then(|()| check_expected(identity, expected.as_ref()))
            .map_err(|error| {
                Status::failed_precondition(format!(
                    "the Agent's snapshot is not the registered allocation's: {error}"
                ))
            })?;
        Ok(response)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<RelayRequest>, ManyOut<Annotated<RelayResponse>>, anyhow::Error>
    for Relay
{
    async fn generate(
        &self,
        input: SingleIn<RelayRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<RelayResponse>>> {
        let (request, context) = input.into_parts();
        let response = RelayResponse::encode(self.answer(&request).await);
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { Annotated::from_data(response) })),
            context.context(),
        ))
    }
}

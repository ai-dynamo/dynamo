// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The consumer, in a conductor pod: a local `TopologyService` that answers
//! each request with a live answer from the authority registered for the
//! identity the request expects.

// The protocol's errors are gRPC statuses, which tonic does not box.
#![allow(clippy::result_large_err)]

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use anyhow::Context as _;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::pipeline::{PushRouter, RouterMode, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use futures::future::join_all;
use tokio::time::{Instant, timeout_at};
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataMap;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

use super::proto::topology_service_server::{TopologyService, TopologyServiceServer};
use super::proto::{GetSnapshotRequest, GetSnapshotResponse, Identity};
use super::{
    COMPONENT, RelayRequest, RelayResponse, allocation_prefix, check_expected, check_snapshot,
    endpoint_name, require_expected, socket,
};

/// The longest one request waits for the authority, whatever deadline the
/// caller set.
const MAX_ATTEMPT: Duration = Duration::from_secs(10);
/// How often a call checks whether its discovery watch has caught up with
/// the authority it listed.
const INSTANCE_POLL: Duration = Duration::from_millis(10);

type Router = PushRouter<RelayRequest, Annotated<RelayResponse>>;

/// The time one request has: it ends at `deadline`, `budget` after it
/// started.
#[derive(Clone, Copy)]
struct Attempt {
    deadline: Instant,
    budget: Duration,
}

impl Attempt {
    fn new(budget: Duration) -> Self {
        Self {
            deadline: Instant::now() + budget,
            budget,
        }
    }

    /// `future`'s result, or UNAVAILABLE once the attempt is over.
    async fn bound<T>(self, future: impl Future<Output = Result<T, Status>>) -> Result<T, Status> {
        timeout_at(self.deadline, future).await.unwrap_or_else(|_| {
            Err(Status::unavailable(format!(
                "the topology authority did not answer within {:?}",
                self.budget
            )))
        })
    }
}

/// What the topology component's registrations hold for an expected
/// identity.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Registered {
    /// No authority of the expected allocation.
    Nothing,
    /// Authorities of the expected allocation for other builds only, by
    /// their build scopes.
    OtherBuilds(Vec<String>),
    /// The instances registered under the expected identity's endpoint.
    Authorities(Vec<u64>),
}

/// Sorts the topology component's `(endpoint, instance)` registrations by
/// what they hold for `expected`. Endpoint names start with the allocation's
/// prefix whatever build they serve, so another build of the expected
/// allocation is told apart from no authority at all.
pub(crate) fn registered<'a>(
    expected: &Identity,
    listed: impl IntoIterator<Item = (&'a str, u64)>,
) -> Registered {
    let endpoint = endpoint_name(expected);
    let prefix = allocation_prefix(expected);
    let mut authorities = BTreeSet::new();
    let mut other_builds = BTreeSet::new();
    for (name, instance_id) in listed {
        if name == endpoint {
            authorities.insert(instance_id);
        } else if let Some(build_scope) = name.strip_prefix(&prefix) {
            other_builds.insert(build_scope.to_string());
        }
    }
    if !authorities.is_empty() {
        Registered::Authorities(authorities.into_iter().collect())
    } else if !other_builds.is_empty() {
        Registered::OtherBuilds(other_builds.into_iter().collect())
    } else {
        Registered::Nothing
    }
}

/// Picks the answer of the authorities registered for `expected`, which
/// were all asked: exactly one OK answer wins, and is checked. A leader that
/// is shutting down or a registration a crashed publisher left behind
/// therefore never hides the authority that answers. More than one OK
/// answer is ambiguous. Without an OK answer, a FAILED_PRECONDITION is final
/// and anything else is UNAVAILABLE, with every authority's reason.
pub(crate) fn choose(
    expected: &Identity,
    answers: Vec<(u64, Result<GetSnapshotResponse, Status>)>,
) -> Result<GetSnapshotResponse, Status> {
    let mut answered = Vec::new();
    let mut failures = Vec::new();
    for (instance_id, answer) in answers {
        match answer {
            Ok(response) => answered.push(response),
            Err(status) => failures.push((instance_id, status)),
        }
    }
    if answered.len() > 1 {
        return Err(Status::unavailable(format!(
            "{} topology authorities registered for {} answered OK; refusing an ambiguous answer",
            answered.len(),
            describe(expected)
        )));
    }
    if let Some(response) = answered.pop() {
        return check_answer(expected, response);
    }
    let refusals: Vec<&str> = failures
        .iter()
        .filter(|(_, status)| status.code() == Code::FailedPrecondition)
        .map(|(_, status)| status.message())
        .collect();
    if !refusals.is_empty() {
        return Err(Status::failed_precondition(refusals.join("; ")));
    }
    let reasons: Vec<String> = failures
        .iter()
        .map(|(instance_id, status)| match status.code() {
            Code::Unavailable => format!("authority {instance_id}: {}", status.message()),
            code => format!(
                "authority {instance_id} answered {code:?}: {}",
                status.message()
            ),
        })
        .collect();
    Err(Status::unavailable(format!(
        "no topology authority registered for {} answered OK: {}",
        describe(expected),
        reasons.join("; ")
    )))
}

/// Checks that the chosen answer holds a valid snapshot of `expected`.
fn check_answer(
    expected: &Identity,
    response: GetSnapshotResponse,
) -> Result<GetSnapshotResponse, Status> {
    let snapshot = response.snapshot.as_ref().ok_or_else(|| {
        Status::failed_precondition("the topology authority answered OK without a snapshot")
    })?;
    let identity = check_snapshot(snapshot).map_err(|error| {
        Status::failed_precondition(format!(
            "the topology authority's snapshot is invalid: {error}"
        ))
    })?;
    check_expected(identity, Some(expected)).map_err(|error| {
        Status::failed_precondition(format!(
            "the topology authority answered for another allocation: {error}"
        ))
    })?;
    Ok(response)
}

/// The local topology service.
pub(crate) struct Consumer {
    drt: DistributedRuntime,
    namespace: String,
    /// Ends requests still in flight, and their discovery watches, on
    /// shutdown.
    cancel: CancellationToken,
}

impl Consumer {
    pub(crate) fn new(
        drt: DistributedRuntime,
        namespace: String,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            drt,
            namespace,
            cancel,
        }
    }

    /// Lists the topology component's registrations now, asks every
    /// authority registered for `expected` at once, and picks the answer.
    async fn fetch(
        &self,
        expected: &Identity,
        request: &GetSnapshotRequest,
        attempt: Attempt,
    ) -> Result<GetSnapshotResponse, Status> {
        let listed = attempt
            .bound(async {
                self.drt
                    .discovery()
                    .list(DiscoveryQuery::ComponentEndpoints {
                        namespace: self.namespace.clone(),
                        component: COMPONENT.to_string(),
                    })
                    .await
                    .map_err(|error| {
                        Status::unavailable(format!(
                            "cannot list LPX topology authorities: {error:#}"
                        ))
                    })
            })
            .await?;
        let registrations = listed.iter().filter_map(|instance| match instance {
            DiscoveryInstance::Endpoint(instance) => {
                Some((instance.endpoint.as_str(), instance.instance_id))
            }
            _ => None,
        });
        match registered(expected, registrations) {
            Registered::Nothing => Err(Status::unavailable(format!(
                "no topology authority is registered for {}",
                describe(expected)
            ))),
            Registered::OtherBuilds(build_scopes) => Err(Status::failed_precondition(format!(
                "the allocation's authority serves another build, not the expected build `{}`: \
                 deployment `{}`, allocation `{}` has authorities only under {}, none under {}",
                expected.build.escape_debug(),
                expected.deployment.escape_debug(),
                expected.allocation.escape_debug(),
                build_scopes
                    .iter()
                    .map(|scope| format!("{}{scope}", allocation_prefix(expected)))
                    .collect::<Vec<_>>()
                    .join(", "),
                endpoint_name(expected)
            ))),
            Registered::Authorities(authorities) => {
                let answers = self
                    .ask_all(&endpoint_name(expected), &authorities, request, attempt)
                    .await?;
                choose(expected, answers)
            }
        }
    }

    /// Sends `request` to each of `authorities` on `endpoint` directly, all
    /// at once, and collects their answers.
    async fn ask_all(
        &self,
        endpoint: &str,
        authorities: &[u64],
        request: &GetSnapshotRequest,
        attempt: Attempt,
    ) -> Result<Vec<(u64, Result<GetSnapshotResponse, Status>)>, Status> {
        let endpoint = self
            .drt
            .namespace(&self.namespace)
            .and_then(|namespace| namespace.component(COMPONENT))
            .map_err(|error| Status::internal(format!("invalid topology endpoint: {error:#}")))?
            .endpoint(endpoint);
        // The client's discovery watch ends with this request.
        let watch = self.cancel.child_token();
        let _watch = watch.clone().drop_guard();
        let router = attempt
            .bound(async {
                let client = endpoint
                    .client_with_cancellation(watch)
                    .await
                    .map_err(unreachable)?;
                Router::from_client_no_fault_detection(client, RouterMode::Direct)
                    .await
                    .map_err(unreachable)
            })
            .await?;
        let router = &router;
        Ok(join_all(authorities.iter().map(|&instance_id| async move {
            let answer = ask(router, instance_id, request, attempt).await;
            (instance_id, answer)
        }))
        .await)
    }
}

fn unreachable(error: anyhow::Error) -> Status {
    Status::unavailable(format!("the topology authority is unreachable: {error:#}"))
}

/// Sends `request` to the authority `instance_id` alone and reads its answer
/// stream to the end.
async fn ask(
    router: &Router,
    instance_id: u64,
    request: &GetSnapshotRequest,
    attempt: Attempt,
) -> Result<GetSnapshotResponse, Status> {
    let mut answer = attempt
        .bound(async {
            // The listing can be ahead of the client's own discovery watch.
            while !router.client.instance_ids().contains(&instance_id) {
                tokio::time::sleep(INSTANCE_POLL).await;
            }
            router
                .direct(SingleIn::new(RelayRequest::encode(request)), instance_id)
                .await
                .map_err(unreachable)
        })
        .await?;
    let first = attempt.bound(async { Ok(answer.next().await) }).await?;
    // A stream dropped before its end is a cancelled request to the router,
    // which logs and traces it as one.
    let _ = timeout_at(attempt.deadline, async {
        while answer.next().await.is_some() {}
    })
    .await;
    match first.map(Annotated::into_data) {
        Some(Ok(Some(response))) => response.decode(),
        Some(Err(error)) => Err(Status::unavailable(format!(
            "the topology authority failed: {error}"
        ))),
        Some(Ok(None)) | None => Err(Status::unavailable(
            "the topology authority ended its answer without a response",
        )),
    }
}

/// Names an identity in messages, with its endpoint.
fn describe(identity: &Identity) -> String {
    format!(
        "deployment `{}`, allocation `{}`, build `{}` (endpoint {})",
        identity.deployment.escape_debug(),
        identity.allocation.escape_debug(),
        identity.build.escape_debug(),
        endpoint_name(identity)
    )
}

/// The deadline the caller set in `grpc-timeout`, if any.
fn request_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let value = metadata.get("grpc-timeout")?.to_str().ok()?;
    let (amount, unit) = value.split_at(value.len().checked_sub(1)?);
    if amount.is_empty() || amount.len() > 8 {
        return None;
    }
    let amount: u64 = amount.parse().ok()?;
    Some(match unit {
        "H" => Duration::from_secs(amount * 3600),
        "M" => Duration::from_secs(amount * 60),
        "S" => Duration::from_secs(amount),
        "m" => Duration::from_millis(amount),
        "u" => Duration::from_micros(amount),
        "n" => Duration::from_nanos(amount),
        _ => return None,
    })
}

#[tonic::async_trait]
impl TopologyService for Consumer {
    async fn get_snapshot(
        &self,
        request: Request<GetSnapshotRequest>,
    ) -> Result<Response<GetSnapshotResponse>, Status> {
        let budget = request_timeout(request.metadata())
            .map_or(MAX_ATTEMPT, |timeout| timeout.min(MAX_ATTEMPT));
        let request = request.into_inner();
        let expected = require_expected(&request).map_err(|error| {
            Status::invalid_argument(format!("malformed topology request: {error}"))
        })?;
        let endpoint = endpoint_name(expected);
        let result = tokio::select! {
            _ = self.cancel.cancelled() => {
                Err(Status::unavailable("the LPX sidecar is shutting down"))
            }
            result = self.fetch(expected, &request, Attempt::new(budget)) => result,
        };
        match &result {
            Ok(_) => tracing::info!(%endpoint, "Relayed a live LPX topology snapshot"),
            Err(status) if status.code() == Code::Unavailable => tracing::debug!(
                %endpoint,
                error = %status.message().escape_debug(),
                "No LPX topology snapshot yet"
            ),
            Err(status) => tracing::warn!(
                %endpoint,
                code = ?status.code(),
                error = %status.message().escape_debug(),
                "LPX topology request failed"
            ),
        }
        result.map(Response::new)
    }
}

/// Serves `consumer` on the Unix socket at `path` until `shutdown`, and
/// removes the socket when it stops.
pub(crate) async fn serve(
    consumer: Consumer,
    path: &Path,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (listener, file) = socket::bind(path).await?;
    tracing::info!(socket = %path.display(), "Serving the LPX topology API");
    let served = Server::builder()
        .add_service(TopologyServiceServer::new(consumer))
        .serve_with_incoming_shutdown(
            UnixListenerStream::new(listener),
            shutdown.cancelled_owned(),
        )
        .await;
    drop(file);
    served.context("the LPX topology API failed")?;
    tracing::info!(socket = %path.display(), "LPX topology API stopped");
    Ok(())
}

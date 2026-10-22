// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An LPX Agent allocation leader's `TopologyService` on a Unix socket, and a
//! conductor's view of the sidecar's local topology API.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_lpx_sidecar::topology::proto::topology_service_client::TopologyServiceClient;
use dynamo_lpx_sidecar::topology::proto::topology_service_server::{
    TopologyService, TopologyServiceServer,
};
use dynamo_lpx_sidecar::topology::proto::{
    Endpoint, Execution, GetSnapshotRequest, GetSnapshotResponse, Identity, Partition, Protocol,
    Snapshot, partition::Placement,
};
use dynamo_sidecar_testkit::bounded;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Response, Status};

/// The identity of allocation `index` of one deployment and build.
pub fn identity(index: u32) -> Identity {
    Identity {
        deployment: "0d4c7a9e-5b1f-4c38-9a62-3f1e2d7c8b90".to_string(),
        allocation: format!("llama-0-afd-{index}"),
        build: "main/llama3_2-1b-lpx-lp20/7a8a706a".to_string(),
    }
}

/// The endpoint the authority of [`identity`]`(0)` registers: `snapshot-`
/// and the first 16 hex digits of
/// `printf 'lpx.topology.v1\0%s\0%s' DEPLOYMENT ALLOCATION | sha256sum`,
/// then `-` and those of
/// `printf 'lpx.topology.v1\0%s\0%s\0%s' DEPLOYMENT ALLOCATION BUILD | sha256sum`.
pub const ENDPOINT_0: &str = "snapshot-776f7e41bb573162-7e63d7f9aae8ebe8";

/// A complete snapshot of `identity`: one single-host partition whose control
/// endpoint is on `port`.
pub fn snapshot(identity: Identity, port: u32) -> Snapshot {
    let host = format!(
        "{}-agt-0.llama-0.dynamo.svc.cluster.local",
        identity.allocation
    );
    Snapshot {
        identity: Some(identity),
        partitions: vec![Partition {
            partition_id: 1,
            placement: Some(Placement::Execution(Execution {
                node_count: 1,
                control: Some(Endpoint {
                    host,
                    port,
                    protocol: Protocol::HydraControl as i32,
                }),
                state_ingress: None,
            })),
        }],
    }
}

/// What the Agent answers.
#[derive(Clone)]
enum Answer {
    Snapshot(Snapshot),
    /// A snapshot regardless of what the request expects, as a broken Agent
    /// might answer.
    Unchecked(Snapshot),
    /// No snapshot, such as UNAVAILABLE while the Agent shuts down.
    Status(Code, String),
}

#[derive(Clone)]
struct State {
    answer: Arc<Mutex<Answer>>,
    calls: Arc<Mutex<usize>>,
}

#[tonic::async_trait]
impl TopologyService for State {
    async fn get_snapshot(
        &self,
        request: Request<GetSnapshotRequest>,
    ) -> Result<Response<GetSnapshotResponse>, Status> {
        *self.calls.lock().unwrap() += 1;
        let answer = self.answer.lock().unwrap().clone();
        let snapshot = match answer {
            // The Agent's contract: every non-empty expected field must
            // match.
            Answer::Snapshot(snapshot) => {
                let actual = snapshot.identity.clone().unwrap_or_default();
                let expected = request.into_inner().expected.unwrap_or_default();
                for (expected, actual) in [
                    (expected.deployment, actual.deployment),
                    (expected.allocation, actual.allocation),
                    (expected.build, actual.build),
                ] {
                    if !expected.is_empty() && expected != actual {
                        return Err(Status::failed_precondition(format!(
                            "the Agent serves {actual}, not {expected}"
                        )));
                    }
                }
                snapshot
            }
            Answer::Unchecked(snapshot) => snapshot,
            Answer::Status(code, message) => return Err(Status::new(code, message)),
        };
        Ok(Response::new(GetSnapshotResponse {
            snapshot: Some(snapshot),
        }))
    }
}

/// The Agent allocation leader's `TopologyService`.
pub struct Agent {
    pub socket: PathBuf,
    state: State,
    shutdown: CancellationToken,
    task: JoinHandle<()>,
}

impl Agent {
    pub async fn start(socket: PathBuf, snapshot: Snapshot) -> Self {
        let state = State {
            answer: Arc::new(Mutex::new(Answer::Snapshot(snapshot))),
            calls: Arc::default(),
        };
        let listener = UnixListener::bind(&socket).unwrap();
        let shutdown = CancellationToken::new();
        let task = tokio::spawn({
            let service = TopologyServiceServer::new(state.clone());
            let shutdown = shutdown.clone();
            async move {
                tonic::transport::Server::builder()
                    .add_service(service)
                    .serve_with_incoming_shutdown(
                        UnixListenerStream::new(listener),
                        shutdown.cancelled_owned(),
                    )
                    .await
                    .unwrap();
            }
        });
        Self {
            socket,
            state,
            shutdown,
            task,
        }
    }

    pub fn answer(&self, snapshot: Snapshot) {
        *self.state.answer.lock().unwrap() = Answer::Snapshot(snapshot);
    }

    pub fn answer_unchecked(&self, snapshot: Snapshot) {
        *self.state.answer.lock().unwrap() = Answer::Unchecked(snapshot);
    }

    pub fn answer_status(&self, code: Code, message: &str) {
        *self.state.answer.lock().unwrap() = Answer::Status(code, message.to_string());
    }

    /// How many requests the Agent has answered.
    pub fn calls(&self) -> usize {
        *self.state.calls.lock().unwrap()
    }

    pub async fn shutdown(mut self) {
        self.shutdown.cancel();
        bounded("Agent topology service shutdown", &mut self.task)
            .await
            .unwrap();
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.task.abort();
    }
}

/// Asks the sidecar's local topology API on `socket` once, as a conductor
/// does at startup.
pub async fn ask(socket: &Path, expected: Option<Identity>) -> Result<Snapshot, Status> {
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("unix://{}", socket.display())).unwrap();
    let channel = endpoint
        .connect()
        .await
        .map_err(|error| Status::unavailable(format!("cannot connect: {error}")))?;
    let mut client = TopologyServiceClient::new(channel);
    let mut request = Request::new(GetSnapshotRequest { expected });
    request.set_timeout(Duration::from_secs(10));
    let response = client.get_snapshot(request).await?;
    Ok(response
        .into_inner()
        .snapshot
        .expect("an OK answer carries a snapshot"))
}

/// Waits until something listens on `socket`.
pub async fn serving(socket: &Path) {
    bounded("the local topology API", async {
        while tokio::net::UnixStream::connect(socket).await.is_err() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
}

/// Asks until the answer is something other than UNAVAILABLE, as a conductor
/// retries until its topology-acquisition deadline.
pub async fn acquire(socket: &Path, expected: Identity) -> Result<Snapshot, Status> {
    bounded("a topology answer other than UNAVAILABLE", async {
        loop {
            match ask(socket, Some(expected.clone())).await {
                Err(status) if status.code() == Code::Unavailable => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                answer => return answer,
            }
        }
    })
    .await
}

/// Asserts that `answer` failed with `code` and a message containing
/// `needle`.
pub fn assert_status(answer: Result<Snapshot, Status>, code: Code, needle: &str) {
    let status = answer.expect_err("an error status");
    assert_eq!(status.code(), code, "{status:?}");
    assert!(
        status.message().contains(needle),
        "{status:?} lacks {needle:?}"
    );
}

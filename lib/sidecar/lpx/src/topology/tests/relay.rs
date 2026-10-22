// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The publisher and the consumer relaying a fake Agent's answers through one
//! in-process runtime.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::stream;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Endpoint as Channel, Server};
use tonic::{Code, Request, Response, Status};

use super::*;
use crate::topology::consumer::Consumer;
use crate::topology::proto::GetSnapshotRequest;
use crate::topology::proto::topology_service_client::TopologyServiceClient;
use crate::topology::proto::topology_service_server::{TopologyService, TopologyServiceServer};
use crate::topology::{COMPONENT, RelayRequest, RelayResponse, Role, run as run_role};

const NAMESPACE: &str = "lpx-topology-test";
/// The endpoint of [`identity`]: `snapshot-` and the first 16 hex digits of
/// `printf 'lpx.topology.v1\0%s\0%s' DEPLOYMENT ALLOCATION | sha256sum`,
/// then `-` and those of
/// `printf 'lpx.topology.v1\0%s\0%s\0%s' DEPLOYMENT ALLOCATION BUILD | sha256sum`.
const ENDPOINT: &str = "snapshot-776f7e41bb573162-7e63d7f9aae8ebe8";

/// Runs `test` on the runtime every test here shares. The request plane's TCP
/// server is process-wide, and it stops with the runtime that started it.
fn run(test: impl Future<Output = ()>) {
    static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("the shared test runtime")
    });
    RUNTIME.block_on(test);
}

async fn within<T>(what: &str, future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(15), future)
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
}

/// What the fake Agent answers.
#[derive(Clone)]
enum Answer {
    /// A snapshot, held to the Agent's contract: every non-empty field of the
    /// request's `expected` must equal the snapshot's, else
    /// FAILED_PRECONDITION.
    Snapshot(Snapshot),
    /// A snapshot regardless of `expected`, as a broken Agent might answer.
    Unchecked(Snapshot),
    /// OK without a snapshot, which no Agent may answer.
    Empty,
    Status(Code, &'static str),
    /// No answer at all.
    Hang,
}

#[derive(Clone)]
struct AgentState {
    answer: Arc<Mutex<Answer>>,
    requests: Arc<Mutex<Vec<GetSnapshotRequest>>>,
}

#[tonic::async_trait]
impl TopologyService for AgentState {
    async fn get_snapshot(
        &self,
        request: Request<GetSnapshotRequest>,
    ) -> Result<Response<GetSnapshotResponse>, Status> {
        let request = request.into_inner();
        self.requests.lock().unwrap().push(request.clone());
        let answer = self.answer.lock().unwrap().clone();
        let snapshot = match answer {
            Answer::Snapshot(snapshot) => {
                let actual = snapshot.identity.clone().unwrap_or_default();
                let expected = request.expected.unwrap_or_default();
                for (name, expected, actual) in [
                    ("deployment", expected.deployment, actual.deployment),
                    ("allocation", expected.allocation, actual.allocation),
                    ("build", expected.build, actual.build),
                ] {
                    if !expected.is_empty() && expected != actual {
                        return Err(Status::failed_precondition(format!(
                            "the Agent's {name} is {actual}, not {expected}"
                        )));
                    }
                }
                Some(snapshot)
            }
            Answer::Unchecked(snapshot) => Some(snapshot),
            Answer::Empty => None,
            Answer::Status(code, message) => return Err(Status::new(code, message)),
            Answer::Hang => std::future::pending().await,
        };
        Ok(Response::new(GetSnapshotResponse { snapshot }))
    }
}

/// The Agent's `TopologyService` on a Unix socket.
struct FakeAgent {
    state: AgentState,
    shutdown: CancellationToken,
    task: JoinHandle<()>,
}

impl FakeAgent {
    async fn start(socket: &Path, answer: Answer) -> Self {
        let state = AgentState {
            answer: Arc::new(Mutex::new(answer)),
            requests: Arc::default(),
        };
        let listener = UnixListener::bind(socket).expect("bind the Agent socket");
        let shutdown = CancellationToken::new();
        let task = tokio::spawn({
            let service = TopologyServiceServer::new(state.clone());
            let shutdown = shutdown.clone();
            async move {
                Server::builder()
                    .add_service(service)
                    .serve_with_incoming_shutdown(
                        UnixListenerStream::new(listener),
                        shutdown.cancelled_owned(),
                    )
                    .await
                    .expect("the Agent serves");
            }
        });
        Self {
            state,
            shutdown,
            task,
        }
    }

    fn answer(&self, answer: Answer) {
        *self.state.answer.lock().unwrap() = answer;
    }

    fn requests(&self) -> Vec<GetSnapshotRequest> {
        self.state.requests.lock().unwrap().clone()
    }

    async fn stop(self) {
        self.shutdown.cancel();
        within("the Agent to stop", self.task).await.unwrap();
    }
}

/// An authority on the request plane that answers OK with `answer` and checks
/// nothing, as a publisher of another version or a broken one might.
#[derive(Clone)]
struct StubAuthority {
    answer: Arc<Mutex<GetSnapshotResponse>>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl AsyncEngine<SingleIn<RelayRequest>, ManyOut<Annotated<RelayResponse>>, anyhow::Error>
    for StubAuthority
{
    async fn generate(
        &self,
        input: SingleIn<RelayRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<RelayResponse>>> {
        let (_, context) = input.into_parts();
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = RelayResponse::encode(Ok(self.answer.lock().unwrap().clone()));
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { Annotated::from_data(response) })),
            context.context(),
        ))
    }
}

/// A runtime, a fake Agent, and a consumer serving on a Unix socket.
struct Fixture {
    dir: tempfile::TempDir,
    drt: DistributedRuntime,
    consumer: CancellationToken,
    consumer_task: JoinHandle<anyhow::Result<()>>,
}

impl Fixture {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let drt = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let consumer = CancellationToken::new();
        let consumer_task = tokio::spawn({
            let role = Role::Consumer {
                socket: dir.path().join("topology.sock"),
            };
            let drt = drt.clone();
            let shutdown = consumer.clone();
            async move { run_role(&role, &drt, NAMESPACE, shutdown).await }
        });
        Self {
            dir,
            drt,
            consumer,
            consumer_task,
        }
    }

    fn agent_socket(&self) -> PathBuf {
        self.dir.path().join("agent.sock")
    }

    fn publish(&self) -> (CancellationToken, JoinHandle<anyhow::Result<()>>) {
        let shutdown = CancellationToken::new();
        let role = Role::Publisher {
            agent_socket: self.agent_socket(),
        };
        let task = tokio::spawn({
            let drt = self.drt.clone();
            let shutdown = shutdown.clone();
            async move { run_role(&role, &drt, NAMESPACE, shutdown).await }
        });
        (shutdown, task)
    }

    /// The topology endpoints registered in the namespace.
    async fn registered(&self) -> Vec<String> {
        self.drt
            .discovery()
            .list(DiscoveryQuery::ComponentEndpoints {
                namespace: NAMESPACE.to_string(),
                component: COMPONENT.to_string(),
            })
            .await
            .unwrap()
            .into_iter()
            .map(|instance| match instance {
                DiscoveryInstance::Endpoint(instance) => instance.endpoint,
                other => panic!("unexpected registration {other:?}"),
            })
            .collect()
    }

    async fn wait_registered(&self) {
        within("the topology endpoint", async {
            while self.registered().await != [ENDPOINT] {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
    }

    /// Asks the consumer's local API, as a conductor does.
    async fn ask(&self, expected: Option<Identity>) -> Result<Snapshot, Status> {
        let socket = self.dir.path().join("topology.sock");
        let channel = within("the consumer socket", async {
            loop {
                let endpoint =
                    Channel::from_shared(format!("unix://{}", socket.display())).unwrap();
                if let Ok(channel) = endpoint.connect().await {
                    break channel;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        let response = within(
            "the consumer's answer",
            TopologyServiceClient::new(channel).get_snapshot(GetSnapshotRequest { expected }),
        )
        .await?;
        Ok(response
            .into_inner()
            .snapshot
            .expect("OK carries a snapshot"))
    }

    async fn stop(self) {
        self.consumer.cancel();
        within("the consumer to stop", self.consumer_task)
            .await
            .unwrap()
            .expect("the consumer stops cleanly");
        assert!(
            !self.dir.path().join("topology.sock").exists(),
            "the consumer removes its socket"
        );
        self.drt.shutdown();
    }
}

fn assert_status(result: Result<Snapshot, Status>, code: Code, needles: &[&str]) {
    let status = result.expect_err("an error status");
    assert_eq!(status.code(), code, "{status:?}");
    for needle in needles {
        assert!(
            status.message().contains(needle),
            "{status:?} lacks {needle:?}"
        );
    }
}

/// The publisher waits for an Agent that is not listening yet and then has no
/// snapshot, registers only once it has one, and relays every later request
/// to the Agent live.
#[test]
fn the_publisher_registers_after_the_first_snapshot_and_relays_live_answers() {
    run(async {
        let fixture = Fixture::start().await;
        let (shutdown, publisher) = fixture.publish();
        // No Agent is listening yet.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let agent = FakeAgent::start(
            &fixture.agent_socket(),
            Answer::Status(Code::Unavailable, "placement is not resolved yet"),
        )
        .await;
        within("the publisher to retry", async {
            while agent.requests().len() < 2 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(fixture.registered().await.is_empty());
        assert_status(
            fixture.ask(Some(identity())).await,
            Code::Unavailable,
            &["no topology authority"],
        );

        agent.answer(Answer::Snapshot(snapshot()));
        fixture.wait_registered().await;
        let asked = agent.requests().len();
        assert_eq!(fixture.ask(Some(identity())).await.unwrap(), snapshot());
        let requests = agent.requests();
        assert_eq!(requests.len(), asked + 1, "one Agent call per request");
        assert_eq!(
            requests.last().unwrap().expected,
            Some(identity()),
            "the conductor's request reaches the Agent"
        );

        // Nothing is cached: the next request sees the Agent's new answer.
        let mut moved = snapshot();
        moved.partitions[3] = execution(300, 1, endpoint("10.0.3.18", 19877, CONTROL), None);
        agent.answer(Answer::Snapshot(moved.clone()));
        assert_eq!(fixture.ask(Some(identity())).await.unwrap(), moved);
        assert_eq!(agent.requests().len(), asked + 2);

        agent.answer(Answer::Status(Code::Unavailable, "the Agent is restarting"));
        assert_status(
            fixture.ask(Some(identity())).await,
            Code::Unavailable,
            &["the Agent is restarting"],
        );
        // Without an OK answer, only a refusal is final.
        agent.answer(Answer::Status(Code::Internal, "the Agent failed"));
        assert_status(
            fixture.ask(Some(identity())).await,
            Code::Unavailable,
            &["the Agent failed"],
        );
        agent.stop().await;
        let unreachable = format!(
            "the Agent's topology socket {} is unreachable",
            fixture.agent_socket().display()
        );
        assert_status(
            fixture.ask(Some(identity())).await,
            Code::Unavailable,
            &[&unreachable, "Connection refused"],
        );

        shutdown.cancel();
        within("the publisher to stop", publisher)
            .await
            .unwrap()
            .expect("the publisher stops cleanly");
        within("the topology endpoint's withdrawal", async {
            while !fixture.registered().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        fixture.stop().await;
    });
}

/// The answers of a broken authority fail without a retry.
#[test]
fn answers_that_break_the_contract_are_failed_precondition() {
    run(async {
        let fixture = Fixture::start().await;
        let agent = FakeAgent::start(&fixture.agent_socket(), Answer::Snapshot(snapshot())).await;
        let (shutdown, publisher) = fixture.publish();
        fixture.wait_registered().await;

        let other_build = Identity {
            build: "main/llama3_2-1b-lpx-lp20/0000".to_string(),
            ..identity()
        };
        // The endpoint names alone show that the allocation's authority
        // serves another build, so that request never reaches the Agent.
        let asked = agent.requests().len();
        assert_status(
            fixture.ask(Some(other_build.clone())).await,
            Code::FailedPrecondition,
            &["serves another build", "main/llama3_2-1b-lpx-lp20/0000"],
        );
        assert_eq!(agent.requests().len(), asked);
        // Nothing is registered for another allocation yet.
        assert_status(
            fixture
                .ask(Some(Identity {
                    allocation: "llama-0-afd-1".to_string(),
                    ..identity()
                }))
                .await,
            Code::Unavailable,
            &["no topology authority"],
        );
        let duplicate = Snapshot {
            partitions: [snapshot().partitions, vec![member(4, 1)]].concat(),
            ..snapshot()
        };
        let changed = Snapshot {
            identity: Some(other_build),
            ..snapshot()
        };
        for (answer, needles) in [
            (Answer::Unchecked(duplicate), &["partition 4"][..]),
            (Answer::Unchecked(changed.clone()), &["build"][..]),
            (Answer::Snapshot(changed), &["build"][..]),
            (Answer::Empty, &["snapshot"][..]),
        ] {
            agent.answer(answer);
            let asked = agent.requests().len();
            assert_status(
                fixture.ask(Some(identity())).await,
                Code::FailedPrecondition,
                needles,
            );
            assert_eq!(agent.requests().len(), asked + 1);
        }

        shutdown.cancel();
        within("the publisher to stop", publisher)
            .await
            .unwrap()
            .unwrap();
        agent.stop().await;
        fixture.stop().await;
    });
}

/// The consumer checks the authority's answer itself, so a snapshot that
/// breaks the contract fails even when the authority skipped its checks.
#[test]
fn the_consumer_checks_what_the_authority_answers() {
    run(async {
        let fixture = Fixture::start().await;
        let authority = StubAuthority {
            answer: Arc::new(Mutex::new(response())),
            calls: Arc::default(),
        };
        let started = fixture
            .drt
            .namespace(NAMESPACE)
            .unwrap()
            .component(COMPONENT)
            .unwrap()
            .endpoint(ENDPOINT)
            .endpoint_builder()
            .handler(Ingress::for_engine(Arc::new(authority.clone())).unwrap())
            .graceful_shutdown(true)
            .start_with_registration()
            .await
            .unwrap();
        fixture.wait_registered().await;
        assert_eq!(fixture.ask(Some(identity())).await.unwrap(), snapshot());

        let duplicate = Snapshot {
            partitions: [snapshot().partitions, vec![member(4, 1)]].concat(),
            ..snapshot()
        };
        let other_allocation = Snapshot {
            identity: Some(Identity {
                allocation: "llama-0-afd-1".to_string(),
                ..identity()
            }),
            ..snapshot()
        };
        for (snapshot, needles) in [
            (
                None,
                &["the topology authority answered OK without a snapshot"][..],
            ),
            (
                Some(duplicate),
                &[
                    "the topology authority's snapshot is invalid",
                    "partition 4",
                ][..],
            ),
            (
                Some(other_allocation),
                &[
                    "the topology authority answered for another allocation",
                    "llama-0-afd-1",
                ][..],
            ),
        ] {
            *authority.answer.lock().unwrap() = GetSnapshotResponse { snapshot };
            let calls = authority.calls.load(Ordering::SeqCst);
            assert_status(
                fixture.ask(Some(identity())).await,
                Code::FailedPrecondition,
                needles,
            );
            assert_eq!(authority.calls.load(Ordering::SeqCst), calls + 1);
        }

        started.shutdown().await.unwrap();
        fixture.stop().await;
    });
}

/// The consumer finds the authority by the identity it expects, so a request
/// without all of it is malformed and never reaches discovery or the Agent.
#[test]
fn the_consumer_requires_a_complete_expected_identity() {
    run(async {
        let fixture = Fixture::start().await;
        let agent = FakeAgent::start(&fixture.agent_socket(), Answer::Snapshot(snapshot())).await;
        let (shutdown, publisher) = fixture.publish();
        fixture.wait_registered().await;
        let asked = agent.requests().len();
        for (expected, needle) in [
            (None, "expected"),
            (
                Some(Identity {
                    build: String::new(),
                    ..identity()
                }),
                "build",
            ),
            (
                Some(Identity {
                    deployment: "with\nnewline".to_string(),
                    ..identity()
                }),
                "deployment",
            ),
        ] {
            assert_status(
                fixture.ask(expected).await,
                Code::InvalidArgument,
                &[needle],
            );
        }
        assert_eq!(agent.requests().len(), asked);
        shutdown.cancel();
        within("the publisher to stop", publisher)
            .await
            .unwrap()
            .unwrap();
        agent.stop().await;
        fixture.stop().await;
    });
}

/// An attempt ends with UNAVAILABLE by the deadline the conductor set, even
/// when the Agent never answers.
#[test]
fn an_attempt_ends_by_the_request_deadline() {
    run(async {
        let fixture = Fixture::start().await;
        let agent = FakeAgent::start(&fixture.agent_socket(), Answer::Snapshot(snapshot())).await;
        let (shutdown, publisher) = fixture.publish();
        fixture.wait_registered().await;
        agent.answer(Answer::Hang);

        let consumer = Consumer::new(
            fixture.drt.clone(),
            NAMESPACE.to_string(),
            CancellationToken::new(),
        );
        let mut request = Request::new(GetSnapshotRequest {
            expected: Some(identity()),
        });
        request
            .metadata_mut()
            .insert("grpc-timeout", "300m".parse().unwrap());
        let started = Instant::now();
        let status = within("the bounded attempt", consumer.get_snapshot(request))
            .await
            .expect_err("the Agent never answers");
        let elapsed = started.elapsed();
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert!(
            elapsed >= Duration::from_millis(300) && elapsed < Duration::from_secs(4),
            "{elapsed:?}"
        );

        shutdown.cancel();
        within("the publisher to stop", publisher)
            .await
            .unwrap()
            .unwrap();
        agent.stop().await;
        fixture.stop().await;
    });
}

/// The publisher fails, registering nothing, when the Agent's first answer is
/// an invalid snapshot or a status other than UNAVAILABLE.
#[test]
fn the_publisher_fails_on_a_broken_first_answer() {
    run(async {
        let invalid = Snapshot {
            partitions: Vec::new(),
            ..snapshot()
        };
        for (answer, needle) in [
            (Answer::Snapshot(invalid), "partitions"),
            (Answer::Empty, "snapshot"),
            (
                Answer::Status(Code::InvalidArgument, "the Agent refuses"),
                "the Agent refuses",
            ),
        ] {
            let fixture = Fixture::start().await;
            let agent = FakeAgent::start(&fixture.agent_socket(), answer).await;
            let (_shutdown, publisher) = fixture.publish();
            let error = within("the publisher to fail", publisher)
                .await
                .unwrap()
                .expect_err("a broken first answer");
            assert!(format!("{error:#}").contains(needle), "{error:#}");
            assert!(fixture.registered().await.is_empty());
            agent.stop().await;
            fixture.stop().await;
        }
    });
}

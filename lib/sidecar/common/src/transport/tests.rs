// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::transport::connect_pool_with;
use crate::{GrpcChannelPool, GrpcEndpoint, GrpcTransportConfig};
use dynamo_backend_common::{BackendError, ErrorType};
use tokio::net::UnixListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::codegen::http;
use tonic::transport::Channel;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tower::util::MapRequestLayer;

fn config() -> GrpcTransportConfig {
    GrpcTransportConfig {
        connections: NonZeroUsize::new(3).unwrap(),
        connect_attempt_timeout: Duration::from_millis(80),
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_millis(100),
    }
}

#[tokio::test(start_paused = true)]
async fn failed_connection_retries_then_pool_contains_every_slot() {
    let attempts = Cell::new(0);
    let started = Instant::now();
    let channels = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
        assert_eq!(timeout, Duration::from_millis(80));
        let attempt = attempts.get();
        attempts.set(attempt + 1);
        async move {
            if attempt == 0 {
                Err(io::Error::other("initial failure"))
            } else {
                Ok(slot)
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(channels, vec![1, 2, 3]);
    assert_eq!(attempts.get(), 4);
    assert_eq!(started.elapsed(), Duration::from_millis(10));
}

#[tokio::test(start_paused = true)]
async fn later_pool_slots_share_the_original_deadline() {
    let observed = std::cell::RefCell::new(Vec::new());
    let started = Instant::now();
    let error = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
        observed.borrow_mut().push((slot, timeout));
        async move {
            if slot == 1 {
                sleep(Duration::from_millis(70)).await;
                Ok(slot)
            } else {
                std::future::pending::<Result<usize, io::Error>>().await
            }
        }
    })
    .await
    .unwrap_err();
    assert_eq!(started.elapsed(), Duration::from_millis(100));
    assert_eq!(
        *observed.borrow(),
        vec![
            (1, Duration::from_millis(80)),
            (2, Duration::from_millis(30)),
            (3, Duration::from_millis(30))
        ]
    );
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::CannotConnect)
    );
    assert!(error.to_string().contains("pool slot 2"));
    assert!(error.to_string().contains("exceeded the startup deadline"));
}

#[tokio::test(start_paused = true)]
async fn retry_wait_is_capped_and_last_failure_is_preserved() {
    let mut transport = config();
    transport.retry_interval = Duration::from_millis(60);
    let attempts = Cell::new(0);
    let started = Instant::now();
    let error = connect_pool_with("test", "in-memory", transport, false, |_, _| {
        let attempt = attempts.get();
        attempts.set(attempt + 1);
        async move {
            Err::<(), _>(io::Error::other(if attempt == 0 {
                "initial connection failure"
            } else {
                "latest connection failure"
            }))
        }
    })
    .await
    .unwrap_err();
    assert_eq!(started.elapsed(), Duration::from_millis(100));
    assert_eq!(attempts.get(), 2);
    assert!(error.to_string().contains("latest connection failure"));
    assert!(!error.to_string().contains("initial connection failure"));
    assert!(error.to_string().contains("after 2 attempts"));
}

/// A gRPC server on a Unix socket, as an engine in the sidecar's pod serves
/// one. It reports its health as `SERVING`.
struct UnixServer {
    /// The `:authority` of every request it has served.
    authorities: Arc<Mutex<Vec<String>>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl UnixServer {
    fn serve(path: &Path) -> Self {
        let incoming = UnixListenerStream::new(UnixListener::bind(path).expect("bind socket"));
        let (_, health) = tonic_health::server::health_reporter();
        let authorities = Arc::new(Mutex::new(Vec::new()));
        let seen = authorities.clone();
        let record_authority = move |request: http::Request<tonic::body::Body>| {
            let authority = request.uri().authority().map(ToString::to_string);
            seen.lock().unwrap().extend(authority);
            request
        };
        let (shutdown, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .layer(MapRequestLayer::new(record_authority))
                .add_service(health)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("serve on the Unix socket");
        });
        Self {
            authorities,
            shutdown: Some(shutdown),
            task,
        }
    }

    /// Stops serving and closes every connection, as an exiting engine does.
    /// The socket file stays behind, as it does when a process exits.
    async fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        timeout(Duration::from_secs(5), &mut self.task)
            .await
            .expect("the server stops")
            .expect("the server task");
    }
}

async fn health(channel: Channel) -> Result<i32, String> {
    HealthClient::new(channel)
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .map(|response| response.into_inner().status)
        .map_err(|status| status.to_string())
}

/// The engine creates its socket only once it has started, so the sidecar
/// keeps retrying until the socket exists, talks gRPC over every pooled
/// connection, and reconnects on the existing channel after a restart.
#[tokio::test]
async fn a_unix_socket_endpoint_waits_for_its_socket_then_serves_rpcs() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("engine.sock");
    let endpoint = GrpcEndpoint::parse(&format!("unix://{}", path.display()), "--test-endpoint")
        .expect("valid Unix socket endpoint");
    let transport = GrpcTransportConfig {
        connections: NonZeroUsize::new(2).unwrap(),
        connect_attempt_timeout: Duration::from_millis(500),
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_secs(5),
    };
    let server_path = path.clone();
    let server = tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        UnixServer::serve(&server_path)
    });

    assert!(
        !path.exists(),
        "the socket must not exist before connecting"
    );
    let pool = GrpcChannelPool::connect("test", &endpoint, transport, false)
        .await
        .expect("connect once the socket exists");
    assert_eq!(pool.len(), 2);
    for _ in 0..pool.len() {
        assert_eq!(
            health(pool.next_channel())
                .await
                .expect("health over the socket"),
            ServingStatus::Serving as i32
        );
    }
    let server = server.await.expect("server");
    assert_eq!(
        *server.authorities.lock().unwrap(),
        ["localhost", "localhost"],
        "a Unix socket has no host, so requests name `localhost` as gRPC does"
    );
    server.stop().await;

    let channel = pool.next_channel();
    assert!(
        health(channel.clone()).await.is_err(),
        "an engine that has exited does not answer"
    );
    std::fs::remove_file(&path).expect("remove the stale socket");
    let server = UnixServer::serve(&path);
    timeout(Duration::from_secs(5), async {
        while health(channel.clone()).await.is_err() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the channel reconnects to the restarted engine");
    server.stop().await;
}

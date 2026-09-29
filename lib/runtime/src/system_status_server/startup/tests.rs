// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{config::HealthStatus, distributed::DistributedConfig};

fn http_env() -> [(&'static str, Option<&'static str>); 4] {
    [
        ("DYN_SYSTEM_HOST", Some("127.0.0.1")),
        ("DYN_SYSTEM_PORT", Some("0")),
        ("DYN_HEALTH_CHECK_ENABLED", Some("false")),
        ("DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", None),
    ]
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap()
}

async fn status(client: &reqwest::Client, base: &str, path: &str) -> u16 {
    client
        .get(format!("{base}{path}"))
        .send()
        .await
        .unwrap()
        .status()
        .as_u16()
}

async fn wait_closed(address: std::net::SocketAddr) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if tokio::net::TcpListener::bind(address).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("listener released");
}

#[tokio::test]
async fn runtime_only_probes_and_routes_share_the_advertised_listener() {
    temp_env::async_with_vars(http_env(), async {
        // Port zero used to bind twice and fail the second server-info registration.
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            DistributedConfig::process_local(),
            SystemProbePolicy::RuntimeOnly,
        )
        .await
        .unwrap();
        let info = drt.system_status_server_info().unwrap();
        assert_ne!(info.port(), 0);
        let base = format!("http://{}", info.socket_addr);
        let client = client();
        drt.system_health()
            .lock()
            .set_health_status(HealthStatus::NotReady);
        assert_eq!(status(&client, &base, "/live").await, 200);
        assert_eq!(status(&client, &base, "/health").await, 200);
        assert_eq!(status(&client, &base, "/metrics").await, 200);
        assert_eq!(status(&client, &base, "/missing").await, 404);
        // Keep DRT and its server-info handle alive: explicit shutdown must suffice.
        runtime.shutdown();
        wait_closed(info.socket_addr).await;
    })
    .await;
}

#[tokio::test]
async fn default_worker_probes_keep_configured_health_and_response() {
    temp_env::async_with_vars(http_env(), async {
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_LIVE_PATH", Some("/custom-live")),
                ("DYN_SYSTEM_HEALTH_PATH", Some("/custom-health")),
            ],
            async {
                let runtime = Runtime::from_current().unwrap();
                let drt =
                    DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                        .await
                        .unwrap();
                let info = drt.system_status_server_info().unwrap();
                let base = format!("http://{}", info.socket_addr);
                let client = client();
                for (health, code) in [(HealthStatus::NotReady, 503), (HealthStatus::Ready, 200)] {
                    drt.system_health().lock().set_health_status(health);
                    for path in ["/custom-live", "/custom-health"] {
                        let response = client.get(format!("{base}{path}")).send().await.unwrap();
                        assert_eq!(response.status().as_u16(), code);
                        let body: serde_json::Value = response.json().await.unwrap();
                        assert!(body.get("uptime").is_some());
                        assert!(body.get("endpoints").is_some());
                    }
                }
                runtime.shutdown();
                wait_closed(info.socket_addr).await;
            },
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn runtime_shutdown_withdraws_readiness_without_stopping_liveness() {
    temp_env::async_with_vars(http_env(), async {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(),
            DistributedConfig::process_local(),
            SystemProbePolicy::RuntimeOnly,
        )
        .await
        .unwrap();
        let info = drt.system_status_server_info().unwrap();
        let base = format!("http://{}", info.socket_addr);
        let client = client();
        assert_eq!(status(&client, &base, "/health").await, 200);
        let guard = runtime.graceful_shutdown_tracker().register_task();
        runtime.shutdown();
        assert_eq!(status(&client, &base, "/health").await, 503);
        assert_eq!(status(&client, &base, "/live").await, 200);
        assert_eq!(status(&client, &base, "/metrics").await, 200);
        assert!(!runtime.primary_token().is_cancelled());
        drop(guard);
        wait_closed(info.socket_addr).await;
    })
    .await;
}

// DYN_SYSTEM_PORT is an i16. Reserve a dynamically selected port in its range.
fn reserve_system_port() -> std::net::TcpListener {
    for _ in 0..1000 {
        if let Ok(listener) =
            std::net::TcpListener::bind(("127.0.0.1", fastrand::u16(10000..32768)))
        {
            return listener;
        }
    }
    panic!("reserve system port");
}

#[tokio::test]
async fn pending_construction_serves_probes_and_releases_listener_on_cancel_or_error() {
    temp_env::async_with_vars(http_env(), async {
        for finish in ["drop", "shutdown", "failure"] {
            let reserved = reserve_system_port();
            let address = reserved.local_addr().unwrap();
            let port = address.port().to_string();
            temp_env::async_with_vars([("DYN_SYSTEM_PORT", Some(port.as_str()))], async {
                let runtime = Runtime::from_current().unwrap();
                // NATS accepts TCP but never sends INFO, holding DRT construction open.
                let peer = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let distributed = DistributedConfig {
                    nats_config: Some(crate::transports::nats::ClientOptions::builder()
                        .server(format!("nats://{}", peer.local_addr().unwrap())).build().unwrap()),
                    ..DistributedConfig::process_local()
                };
                drop(reserved);
                let mut construction = Box::pin(DistributedRuntime::new_with_probe_policy(
                    runtime.clone(), distributed, SystemProbePolicy::RuntimeOnly,
                ));
                let (mut connection, _) = tokio::select! {
                    result = &mut construction => panic!("constructed before NATS INFO: {result:?}"),
                    peer = tokio::time::timeout(Duration::from_secs(5), peer.accept()) => peer.unwrap().unwrap(),
                };
                let base = format!("http://{address}");
                let client = client();
                assert_eq!(status(&client, &base, "/live").await, 200);
                assert_eq!(status(&client, &base, "/health").await, 503);
                assert_eq!(status(&client, &base, "/metrics").await, 503);
                match finish {
                    "drop" => drop(construction),
                    "shutdown" => {
                        runtime.shutdown();
                        assert!(tokio::time::timeout(Duration::from_secs(5), construction).await.unwrap().is_err());
                    }
                    "failure" => {
                        use tokio::io::AsyncWriteExt;
                        connection.write_all(b"INFO invalid-json\r\n").await.unwrap();
                        assert!(tokio::time::timeout(Duration::from_secs(5), construction).await.unwrap().is_err());
                    }
                    _ => unreachable!(),
                }
                wait_closed(address).await;
                runtime.shutdown();
            }).await;
        }
    }).await;
}

#[tokio::test]
async fn disabled_http_and_bind_failure_preserve_policy_contracts() {
    temp_env::async_with_vars(http_env(), async {
        temp_env::async_with_vars([("DYN_SYSTEM_PORT", Some("-1"))], async {
            for policy in [SystemProbePolicy::Worker, SystemProbePolicy::RuntimeOnly] {
                let runtime = Runtime::from_current().unwrap();
                let drt = DistributedRuntime::new_with_probe_policy(
                    runtime.clone(),
                    DistributedConfig::process_local(),
                    policy,
                )
                .await
                .unwrap();
                assert!(drt.system_status_server_info().is_none());
                runtime.shutdown();
            }
        })
        .await;
        let occupied = reserve_system_port();
        let port = occupied.local_addr().unwrap().port().to_string();
        temp_env::async_with_vars([("DYN_SYSTEM_PORT", Some(port.as_str()))], async {
            let runtime = Runtime::from_current().unwrap();
            assert!(
                DistributedRuntime::new_with_probe_policy(
                    runtime.clone(),
                    DistributedConfig::process_local(),
                    SystemProbePolicy::RuntimeOnly
                )
                .await
                .is_err()
            );
            let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
            assert!(drt.system_status_server_info().is_none());
            runtime.shutdown();
        })
        .await;
    })
    .await;
}
#[tokio::test]
async fn dependency_outage_only_fails_readiness_and_can_recover() {
    temp_env::async_with_vars(http_env(), async {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::sync::Notify;

        // A minimal NATS peer lets us close and restore the real client connection
        // without an external daemon or inference requests.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let disconnect = Arc::new(Notify::new());
        let reconnect = Arc::new(Notify::new());
        let peer_disconnect = disconnect.clone();
        let peer_reconnect = reconnect.clone();
        let peer = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
            for attempt in 0..2 {
                if attempt > 0 {
                    peer_reconnect.notified().await;
                }
                let (socket, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = socket.into_split();
                writer.write_all(b"INFO {\"server_id\":\"probe-test\",\"version\":\"2.10.0\",\"proto\":1,\"max_payload\":1048576}\r\n").await.unwrap();
                let connection = async {
                    let mut lines = BufReader::new(reader).lines();
                    while let Some(line) = lines.next_line().await.unwrap() {
                        if line == "PING" {
                            writer.write_all(b"PONG\r\n").await.unwrap();
                        }
                    }
                };
                tokio::select! {
                    _ = peer_disconnect.notified() => {},
                    _ = connection => {},
                }
            }
        }));
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedConfig {
            nats_config: Some(
                crate::transports::nats::ClientOptions::builder()
                    .server(format!("nats://{address}"))
                    .build()
                    .unwrap(),
            ),
            ..DistributedConfig::process_local()
        };
        let drt = DistributedRuntime::new_with_probe_policy(
            runtime.clone(), distributed, SystemProbePolicy::RuntimeOnly,
        ).await.unwrap();
        let base = format!("http://{}", drt.system_status_server_info().unwrap().socket_addr);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        assert_eq!(status(&client, &base, "/health").await, 200);
        disconnect.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            while status(&client, &base, "/health").await != 503 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(status(&client, &base, "/live").await, 200);
        reconnect.notify_one();
        tokio::time::timeout(Duration::from_secs(10), async {
            while status(&client, &base, "/health").await != 200 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        runtime.shutdown();
        drop(peer);
        }).await;
}

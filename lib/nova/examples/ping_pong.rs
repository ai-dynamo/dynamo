// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ping-pong benchmark for Nova active message system.
//!
//! This benchmark measures round-trip time (RTT) for unary messages between two Nova instances
//! running in the same process on separate single-threaded tokio runtimes.
//!
//! Supports multiple transport types: TCP, HTTP, and UCX (when available).

use anyhow::Result;
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use dynamo_nova::am::{Nova, NovaHandler};
use dynamo_nova_backend::{Transport, tcp::TcpTransportBuilder};

#[cfg(feature = "http")]
use dynamo_nova_backend::http::HttpTransportBuilder;

#[cfg(feature = "grpc")]
use dynamo_nova_backend::grpc::GrpcTransportBuilder;

#[cfg(feature = "ucx")]
use dynamo_nova_backend::ucx::UcxTransportBuilder;

#[cfg(feature = "nats")]
use dynamo_nova_backend::nats::NatsTransportBuilder;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Helper to get a random available port
fn get_random_port() -> u16 {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Transport type selection
#[derive(Debug, Clone, Copy, ValueEnum)]
enum TransportType {
    /// TCP transport
    Tcp,
    /// HTTP transport
    #[cfg(feature = "http")]
    Http,
    /// gRPC transport
    #[cfg(feature = "grpc")]
    Grpc,
    /// UCX transport
    #[cfg(feature = "ucx")]
    Ucx,
    /// NATS transport
    #[cfg(feature = "nats")]
    Nats,
}

/// CLI arguments for ping-pong benchmark
#[derive(Parser, Debug)]
#[command(name = "ping_pong")]
#[command(about = "Benchmark Nova unary message RTT performance")]
struct Args {
    /// Number of ping-pong iterations
    #[arg(long, default_value = "1000")]
    rounds: u32,

    /// Transport type to use
    #[arg(long, default_value = "tcp")]
    transport: TransportType,

    /// NATS server URL (only used when transport=nats)
    #[cfg(feature = "nats")]
    #[arg(long, default_value = "nats://127.0.0.1:4222")]
    nats_url: String,
}

/// Create a transport based on the selected type
fn create_transport(
    transport_type: TransportType,
    #[cfg(feature = "nats")] nats_url: &str,
) -> Result<Arc<dyn Transport>> {
    match transport_type {
        TransportType::Tcp => {
            let addr = format!("127.0.0.1:{}", get_random_port())
                .parse::<SocketAddr>()
                .unwrap();
            Ok(Arc::new(
                TcpTransportBuilder::new().bind_addr(addr).build()?,
            ))
        }
        #[cfg(feature = "http")]
        TransportType::Http => {
            let addr = format!("127.0.0.1:{}", get_random_port())
                .parse::<SocketAddr>()
                .unwrap();
            Ok(Arc::new(
                HttpTransportBuilder::new().bind_addr(addr).build()?,
            ))
        }
        #[cfg(feature = "grpc")]
        TransportType::Grpc => {
            let addr = format!("127.0.0.1:{}", get_random_port())
                .parse::<SocketAddr>()
                .unwrap();
            Ok(Arc::new(
                GrpcTransportBuilder::new().bind_addr(addr).build()?,
            ))
        }
        #[cfg(feature = "ucx")]
        TransportType::Ucx => Ok(Arc::new(UcxTransportBuilder::new().build()?)),
        #[cfg(feature = "nats")]
        TransportType::Nats => Ok(Arc::new(
            NatsTransportBuilder::new().nats_url(nats_url).build()?,
        )),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Using {:?} transport", args.transport);

    // Create two separate single-threaded tokio runtimes
    let runtime_server = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let runtime_client = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Channel to pass PeerInfo from server to client
    let (peer_info_tx, peer_info_rx) = std::sync::mpsc::channel();

    let transport_type = args.transport;

    #[cfg(feature = "nats")]
    let nats_url_server = args.nats_url.clone();

    #[cfg(feature = "nats")]
    let nats_url_client = args.nats_url.clone();

    // Spawn server thread
    let server_handle = std::thread::spawn(move || {
        runtime_server.block_on(async {
            // Create server Nova instance with selected transport
            let transport = create_transport(
                transport_type,
                #[cfg(feature = "nats")]
                &nats_url_server,
            )
            .unwrap();
            let nova = Nova::new(vec![transport], vec![]).await.unwrap();

            // Give transport a moment to bind and start accepting connections
            sleep(Duration::from_millis(100)).await;

            // Register ping handler that returns empty bytes
            let ping_handler =
                NovaHandler::unary_handler("ping", |_ctx| Ok(Some(Bytes::new()))).build();

            nova.register_handler(ping_handler).unwrap();

            // Get PeerInfo and send to client
            let peer_info = nova.peer_info();
            peer_info_tx.send(peer_info).unwrap();

            println!("Server started");
            println!("Server instance ID: {:?}", nova.instance_id());

            // Keep runtime alive to handle requests
            std::future::pending::<()>().await;
        });
    });

    // Wait for server to send PeerInfo
    let server_peer_info = peer_info_rx.recv().unwrap();

    // Spawn client thread
    let client_handle = std::thread::spawn(move || {
        runtime_client.block_on(async {
            // Create client Nova instance with selected transport
            let transport = create_transport(
                transport_type,
                #[cfg(feature = "nats")]
                &nats_url_client,
            )
            .unwrap();
            let nova = Nova::new(vec![transport], vec![]).await.unwrap();

            // Give transport a moment to bind
            sleep(Duration::from_millis(100)).await;

            println!("Client started");
            println!("Client instance ID: {:?}", nova.instance_id());
            println!("Connecting to server: {:?}", server_peer_info.instance_id());

            // Register peer
            nova.register_peer(server_peer_info.clone()).unwrap();

            // Wait for connection to establish
            sleep(Duration::from_millis(500)).await;

            println!("Performing warmup ping...");

            // Warmup ping
            let warmup_result = nova
                .unary("ping")?
                .raw_payload(Bytes::new())
                .instance(server_peer_info.instance_id())
                .send()
                .await;

            match warmup_result {
                Ok(_) => println!("Warmup complete"),
                Err(e) => {
                    println!("Warmup failed: {}", e);
                    return Ok(());
                }
            }

            sleep(Duration::from_millis(50)).await;

            println!(
                "Starting ping-pong measurements for {} rounds...",
                args.rounds
            );

            let mut rtts = Vec::new();

            for i in 1..=args.rounds {
                let start = Instant::now();

                let result = nova
                    .unary("ping")?
                    .raw_payload(Bytes::new())
                    .instance(server_peer_info.instance_id())
                    .send()
                    .await;

                match result {
                    Ok(_) => {
                        let rtt = start.elapsed();
                        rtts.push(rtt);

                        if i % 100 == 0 {
                            println!("Round {}: RTT = {:?}", i, rtt);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Round {} failed: {}", i, e);
                    }
                }
            }

            if !rtts.is_empty() {
                let total: Duration = rtts.iter().sum();
                let avg = total / rtts.len() as u32;
                let min = rtts.iter().min().unwrap();
                let max = rtts.iter().max().unwrap();

                println!("\n=== RTT Statistics ===");
                println!("Rounds completed: {}/{}", rtts.len(), args.rounds);
                println!("Average RTT: {:?}", avg);
                println!("Min RTT: {:?}", min);
                println!("Max RTT: {:?}", max);

                let variance: f64 = rtts
                    .iter()
                    .map(|rtt| {
                        let diff = rtt.as_nanos() as f64 - avg.as_nanos() as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    / rtts.len() as f64;

                let std_dev = Duration::from_nanos(variance.sqrt() as u64);
                println!("Std deviation: {:?}", std_dev);
            } else {
                println!("No successful ping-pong rounds completed");
            }

            Ok(())
        })
    });

    // Wait for client to complete
    let client_result: Result<()> = client_handle.join().unwrap();
    client_result?;

    // Server will run indefinitely, but we can drop it here
    // In a real scenario, you might want to add shutdown logic
    drop(server_handle);

    Ok(())
}

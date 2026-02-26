// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ping-pong benchmark for Messenger active message system.
//!
//! This benchmark measures round-trip time (RTT) for unary messages between two Messenger instances
//! running in the same process on separate single-threaded tokio runtimes.

use anyhow::Result;
use bytes::Bytes;
use clap::Parser;
use velo_backend::tcp::TcpTransportBuilder;
use velo_messenger::{Handler, Messenger};

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Create a TcpTransport bound to an OS-assigned port (no TOCTOU race).
fn new_transport() -> Arc<velo_backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

/// CLI arguments for ping-pong benchmark
#[derive(Parser, Debug)]
#[command(name = "ping_pong")]
#[command(about = "Benchmark Messenger unary message RTT performance")]
struct Args {
    /// Number of ping-pong iterations
    #[arg(long, default_value = "1000")]
    rounds: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Using TCP transport");

    // Create two separate single-threaded tokio runtimes
    let runtime_server = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let runtime_client = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Channel to pass PeerInfo from server to client
    let (peer_info_tx, peer_info_rx) = std::sync::mpsc::channel();

    // Spawn server thread
    let server_handle = std::thread::spawn(move || {
        runtime_server.block_on(async {
            // Create server Messenger instance with TCP transport
            let transport = new_transport();
            let messenger = Messenger::new(vec![transport], None).await.unwrap();

            // Give transport a moment to bind and start accepting connections
            sleep(Duration::from_millis(100)).await;

            // Register ping handler that returns empty bytes
            let ping_handler =
                Handler::unary_handler("ping", |_ctx| Ok(Some(Bytes::new()))).build();

            messenger.register_handler(ping_handler).unwrap();

            // Get PeerInfo and send to client
            let peer_info = messenger.peer_info();
            peer_info_tx.send(peer_info).unwrap();

            println!("Server started");
            println!("Server instance ID: {:?}", messenger.instance_id());

            // Keep runtime alive to handle requests
            std::future::pending::<()>().await;
        });
    });

    // Wait for server to send PeerInfo
    let server_peer_info = peer_info_rx.recv().unwrap();

    // Spawn client thread
    let client_handle = std::thread::spawn(move || {
        runtime_client.block_on(async {
            // Create client Messenger instance with TCP transport
            let transport = new_transport();
            let messenger = Messenger::new(vec![transport], None).await.unwrap();

            // Give transport a moment to bind
            sleep(Duration::from_millis(100)).await;

            println!("Client started");
            println!("Client instance ID: {:?}", messenger.instance_id());
            println!("Connecting to server: {:?}", server_peer_info.instance_id());

            // Register peer
            messenger.register_peer(server_peer_info.clone()).unwrap();

            // Wait for connection to establish
            sleep(Duration::from_millis(500)).await;

            println!("Performing warmup ping...");

            // Warmup ping
            let warmup_result = messenger
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

                let result = messenger
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
    drop(server_handle);

    Ok(())
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    env,
    net::{IpAddr, SocketAddr},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use async_stream::stream;
use async_trait::async_trait;
use dynamo_llm::{
    endpoint_type::EndpointType, http::service::service_v2::HttpService,
    types::RealtimeBidirectionalEngine,
};
use dynamo_protocols::types::realtime::{RealtimeClientEvent, RealtimeServerEvent};
use dynamo_runtime::{
    CancellationToken,
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ResponseStream},
    pipeline::{Error, ManyIn, ManyOut},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{
        TcpListener,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    sync::{Mutex, MutexGuard, Notify},
};
use voice_agent_realtime::server_event_finishes_turn;

const MODEL_NAME: &str = "echo";
const NAMESPACE: &str = "voice-agent";

#[derive(Debug)]
struct Args {
    host: String,
    port: u16,
    backend_host: String,
    backend_port: u16,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut host = env::var("DYN_REALTIME_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let mut port = env::var("DYN_REALTIME_PORT")
            .ok()
            .map(|raw| raw.parse().context("invalid DYN_REALTIME_PORT"))
            .transpose()?
            .unwrap_or(8080);
        let mut backend_host =
            env::var("DYN_REALTIME_BACKEND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let mut backend_port = env::var("DYN_REALTIME_BACKEND_PORT")
            .ok()
            .map(|raw| raw.parse().context("invalid DYN_REALTIME_BACKEND_PORT"))
            .transpose()?
            .unwrap_or(8081);

        let mut argv = env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--host" => host = argv.next().context("--host requires a value")?,
                "--port" => {
                    port = argv
                        .next()
                        .context("--port requires a value")?
                        .parse()
                        .context("--port must be a valid u16")?;
                }
                "--backend-host" => {
                    backend_host = argv.next().context("--backend-host requires a value")?;
                }
                "--backend-port" => {
                    backend_port = argv
                        .next()
                        .context("--backend-port requires a value")?
                        .parse()
                        .context("--backend-port must be a valid u16")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }

        IpAddr::from_str(&host).with_context(|| format!("--host must be an IP address: {host}"))?;
        IpAddr::from_str(&backend_host)
            .with_context(|| format!("--backend-host must be an IP address: {backend_host}"))?;

        Ok(Self {
            host,
            port,
            backend_host,
            backend_port,
        })
    }

    fn backend_addr(&self) -> Result<SocketAddr> {
        format!("{}:{}", self.backend_host, self.backend_port)
            .parse()
            .context("invalid backend bridge address")
    }
}

fn print_help() {
    println!(
        "Run a local Dynamo realtime frontend\n\n\
         Usage: cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
         --bin voice-agent-realtime-frontend -- [OPTIONS]\n\n\
         Options:\n\
           --host <IP>           HTTP listen address [default: 127.0.0.1]\n\
           --port <PORT>         HTTP listen port [default: 8080]\n\
           --backend-host <IP>   Backend bridge listen address [default: 127.0.0.1]\n\
           --backend-port <PORT> Backend bridge listen port [default: 8081]\n\
           -h, --help            Print help"
    );
}

struct BackendConnection {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
}

#[derive(Default)]
struct BackendBridge {
    connection: Mutex<Option<BackendConnection>>,
    notify: Notify,
}

impl BackendBridge {
    async fn accept_loop(
        self: Arc<Self>,
        addr: SocketAddr,
        cancel: CancellationToken,
    ) -> Result<()> {
        let listener = TcpListener::bind(addr)
            .await
            .with_context(|| format!("failed to bind backend bridge on {addr}"))?;
        println!("Backend bridge listening on tcp://{addr}");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                accepted = listener.accept() => {
                    let (stream, peer) = accepted.context("backend bridge accept failed")?;
                    let (read_half, write_half) = stream.into_split();
                    *self.connection.lock().await = Some(BackendConnection {
                        reader: BufReader::new(read_half),
                        writer: write_half,
                    });
                    self.notify.notify_waiters();
                    tracing::info!(%peer, "realtime backend connected");
                }
            }
        }
    }

    async fn wait_for_connection<'a>(
        &'a self,
        ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Option<MutexGuard<'a, Option<BackendConnection>>> {
        loop {
            let guard = self.connection.lock().await;
            if guard.is_some() {
                return Some(guard);
            }
            drop(guard);

            if ctx.is_stopped() {
                return None;
            }

            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }
    }
}

struct BackendProxyEngine {
    bridge: Arc<BackendBridge>,
}

#[async_trait]
impl AsyncEngine<ManyIn<RealtimeClientEvent>, ManyOut<Annotated<RealtimeServerEvent>>, Error>
    for BackendProxyEngine
{
    async fn generate(
        &self,
        mut incoming: ManyIn<RealtimeClientEvent>,
    ) -> Result<ManyOut<Annotated<RealtimeServerEvent>>, Error> {
        let ctx = incoming.context();
        let ctx_for_stream = ctx.clone();
        let bridge = self.bridge.clone();

        let output = stream! {
            let ctx = ctx_for_stream;

            while let Some(event) = incoming.next().await {
                if ctx.is_stopped() {
                    break;
                }

                let Some(mut guard) = bridge.wait_for_connection(&ctx).await else {
                    break;
                };
                let Some(conn) = guard.as_mut() else {
                    continue;
                };

                match serde_json::to_string(&event) {
                    Ok(payload) => {
                        if let Err(err) = conn.writer.write_all(payload.as_bytes()).await {
                            tracing::warn!(%err, "failed to write request to realtime backend");
                            *guard = None;
                            continue;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(%err, "failed to serialize realtime request");
                        continue;
                    }
                }
                if let Err(err) = conn.writer.write_all(b"\n").await {
                    tracing::warn!(%err, "failed to finish request frame to realtime backend");
                    *guard = None;
                    continue;
                }
                if let Err(err) = conn.writer.flush().await {
                    tracing::warn!(%err, "failed to flush request to realtime backend");
                    *guard = None;
                    continue;
                }

                let mut line = String::new();
                loop {
                    if ctx.is_stopped() {
                        break;
                    }

                    line.clear();
                    match conn.reader.read_line(&mut line).await {
                        Ok(0) => {
                            tracing::warn!("realtime backend disconnected");
                            *guard = None;
                            break;
                        }
                        Ok(_) => {}
                        Err(err) => {
                            tracing::warn!(%err, "failed to read response from realtime backend");
                            *guard = None;
                            break;
                        }
                    }

                    let chunk: Annotated<RealtimeServerEvent> =
                        match serde_json::from_str(line.trim_end()) {
                            Ok(chunk) => chunk,
                            Err(err) => {
                                tracing::warn!(%err, "realtime backend sent malformed response");
                                continue;
                            }
                        };
                    let finished = server_event_finishes_turn(&chunk);
                    yield chunk;
                    if finished {
                        break;
                    }
                }
            }
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG").unwrap_or_else(|_| "info,dynamo_llm=info".to_string()),
        )
        .init();

    let args = Args::parse()?;
    let backend_addr = args.backend_addr()?;
    let bridge = Arc::new(BackendBridge::default());

    let service = HttpService::builder()
        .host(args.host.clone())
        .port(args.port)
        .build()
        .context("failed to build HTTP service")?;

    service.enable_model_endpoint(EndpointType::Realtime, true);
    let engine: RealtimeBidirectionalEngine = Arc::new(BackendProxyEngine {
        bridge: bridge.clone(),
    });
    service
        .model_manager()
        .add_realtime_model(MODEL_NAME, NAMESPACE, engine)
        .context("failed to register realtime model")?;

    let cancel = CancellationToken::new();
    let http_handle = service.spawn(cancel.clone()).await;
    let bridge_handle = tokio::spawn(bridge.accept_loop(backend_addr, cancel.clone()));

    println!(
        "Dynamo realtime frontend listening on http://{}:{}",
        args.host, args.port
    );
    println!("Registered realtime model: {MODEL_NAME}");
    println!("Waiting for realtime backend on tcp://{backend_addr}");
    println!(
        "Connect a client with: python examples/voice_agent/client.py --url ws://{}:{}/v1/realtime --model {MODEL_NAME}",
        args.host, args.port
    );
    println!("Press Ctrl-C to stop.");

    tokio::select! {
        result = http_handle => {
            result.context("HTTP service task panicked")??;
        }
        result = bridge_handle => {
            result.context("backend bridge task panicked")??;
        }
        result = tokio::signal::ctrl_c() => {
            result.context("failed to listen for Ctrl-C")?;
            cancel.cancel();
        }
    }

    Ok(())
}

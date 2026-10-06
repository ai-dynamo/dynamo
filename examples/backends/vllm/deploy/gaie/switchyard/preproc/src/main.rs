// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod error;
mod json;
mod proto;
mod request;
mod router;
mod server;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use axum::{Router as HttpRouter, extract::State, http::StatusCode, routing::get};
use tokio::sync::{Semaphore, watch};
use tonic::transport::Server;

use crate::{
    proto::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer,
    router::Router,
    server::{MAX_BODY, MAX_CONCURRENT, MAX_IN_FLIGHT, Preproc},
};

async fn readiness(State(ready): State<Arc<AtomicBool>>) -> StatusCode {
    if ready.load(Ordering::Relaxed) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let router = Arc::new(Router::load(
        std::env::var("ROUTES_CONFIG").unwrap_or_else(|_| "config/routes.toml".into()),
    )?);
    let capacity = Arc::new(Semaphore::new(MAX_CONCURRENT));
    let ready = Arc::new(AtomicBool::new(true));
    let (shutdown, receiver) = watch::channel(false);
    let http = HttpRouter::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/readyz", get(readiness))
        .with_state(ready.clone());
    let listener = tokio::net::TcpListener::bind(
        std::env::var("ADMIN_ADDR").unwrap_or_else(|_| "0.0.0.0:9003".into()),
    )
    .await?;
    let mut http_shutdown = receiver.clone();
    let admin_task = tokio::spawn(async move {
        axum::serve(listener, http)
            .with_graceful_shutdown(async move {
                let _ = http_shutdown.changed().await;
            })
            .await
    });
    let processor = ExternalProcessorServer::new(Preproc {
        router,
        capacity,
        streams: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
    })
    .max_decoding_message_size(MAX_BODY + 65536)
    .max_encoding_message_size(MAX_BODY + 65536);
    let addr = std::env::var("GRPC_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9002".into())
        .parse()?;
    tracing::info!(%addr, "configured decision-only preprocessor listening");
    let grpc = Server::builder()
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .add_service(processor)
        .serve_with_shutdown(addr, async move {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            ready.store(false, Ordering::Relaxed);
            let _ = shutdown.send(true);
        });
    grpc.await?;
    admin_task.await??;
    Ok(())
}

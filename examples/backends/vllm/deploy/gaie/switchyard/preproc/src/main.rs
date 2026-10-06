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

mod config;
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

#[derive(Clone)]
struct Admin {
    ready: Arc<AtomicBool>,
    router: Arc<Router>,
    capacity: Arc<Semaphore>,
}

async fn readiness(State(admin): State<Admin>) -> StatusCode {
    if admin.ready.load(Ordering::Relaxed) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn metrics(State(admin): State<Admin>) -> ([(http::HeaderName, &'static str); 1], String) {
    let mut body = String::from("# TYPE switchyard_routing_decisions_total counter\n");
    let mut count = 0;
    for (target, counter) in &admin.router.selections {
        let selected = counter.load(Ordering::Relaxed);
        count += selected;
        body.push_str(&format!(
            "switchyard_routing_decisions_total{{target=\"{target}\"}} {selected}\n"
        ));
    }
    body.push_str(&format!(
        "# TYPE switchyard_preproc_errors_total counter\nswitchyard_preproc_errors_total {}\n# TYPE switchyard_decision_seconds summary\nswitchyard_decision_seconds_count {}\nswitchyard_decision_seconds_sum {}\n# TYPE switchyard_preproc_active_streams gauge\nswitchyard_preproc_active_streams {}\n",
        admin.router.errors.load(Ordering::Relaxed),
        count,
        admin.router.decision_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        MAX_CONCURRENT - admin.capacity.available_permits()
    ));
    (
        [(http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
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
        std::env::var("POOL_BINDINGS_CONFIG")
            .unwrap_or_else(|_| "config/pool-bindings.toml".into()),
    )?);
    let capacity = Arc::new(Semaphore::new(MAX_CONCURRENT));
    let ready = Arc::new(AtomicBool::new(true));
    let (shutdown, receiver) = watch::channel(false);
    let admin = Admin {
        ready: ready.clone(),
        router: router.clone(),
        capacity: capacity.clone(),
    };
    let http = HttpRouter::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/readyz", get(readiness))
        .route("/metrics", get(metrics))
        .with_state(admin);
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

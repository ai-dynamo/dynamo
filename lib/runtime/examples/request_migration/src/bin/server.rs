// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait,
        network::{Ingress, STREAM_ERR_MSG},
    },
    protocols::annotated::Annotated,
    stream,
};
use request_migration::DEFAULT_NAMESPACE;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn main() -> anyhow::Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

enum FailureMode {
    Never,        // Never fail (normal operation)
    FirstRequest, // Fail only on first request
    Always,       // Always fail every request
}

struct RequestHandler {
    // Flag to track first request (used when failure_mode = FirstRequest)
    first_request: AtomicBool,
    // Server name to identify which server is responding
    server_name: String,
    // Failure behavior
    failure_mode: FailureMode,
}

impl RequestHandler {
    fn new() -> Arc<Self> {
        // Read server name from environment variable, default to "server"
        let server_name = std::env::var("SERVER_NAME").unwrap_or_else(|_| "server".to_string());

        // Read failure mode from environment variable
        let failure_mode = match std::env::var("FAILURE_MODE").as_deref() {
            Ok("never") => FailureMode::Never,
            Ok("first") => FailureMode::FirstRequest,
            Ok("always") => FailureMode::Always,
            _ => FailureMode::FirstRequest, // Default: fail on first request
        };

        let mode_str = match failure_mode {
            FailureMode::Never => "never",
            FailureMode::FirstRequest => "first request only",
            FailureMode::Always => "always",
        };

        tracing::info!(
            "Server initialized: name={}, failure_mode={}",
            server_name,
            mode_str
        );

        Arc::new(Self {
            first_request: AtomicBool::new(true),
            server_name,
            failure_mode,
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(
        &self,
        input: SingleIn<String>,
    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        let mut chars: Vec<Annotated<String>> = data
            .chars()
            .map(|c| {
                let formatted = format!("from server: {}, data: {}", self.server_name, c);
                Annotated::from_data(formatted)
            })
            .collect();

        // Determine if we should inject an error based on failure mode
        let should_fail = match self.failure_mode {
            FailureMode::Never => false,
            FailureMode::FirstRequest => {
                // Check if this is the first request
                self.first_request.swap(false, Ordering::SeqCst)
            }
            FailureMode::Always => true,
        };

        // Simulate error after 2 characters if should_fail is true
        if should_fail && chars.len() > 2 {
            tracing::info!(
                "Simulating stream error after 2 characters (server: {})",
                self.server_name
            );
            // Keep only first 2 characters, then add error
            chars.truncate(2);
            chars.push(Annotated::from_error(STREAM_ERR_MSG.to_string()));
        }

        let stream = stream::iter(chars);

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(runtime: DistributedRuntime) -> anyhow::Result<()> {
    // attach an ingress to an engine
    let ingress = Ingress::for_engine(RequestHandler::new())?;

    let component = runtime.namespace(DEFAULT_NAMESPACE)?.component("backend")?;
    component
        .endpoint("generate")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}

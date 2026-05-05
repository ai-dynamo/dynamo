// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidi streaming example — server side.
//!
//! Registers an `AsyncEngine<ManyIn<String>, ManyOut<Annotated<String>>, Error>`
//! that:
//! 1. Reads the optional `init: String` payload from the request and
//!    prepends it to each echoed line (demonstrates init-payload plumbing
//!    via `BIDI_INIT_KEY`).
//! 2. Uppercases each input item and emits it.
//! 3. After the input stream ends (peer sent `BidiFrame::Done`), emits a
//!    short trailing summary — this exercises the half-close: the server
//!    is free to keep producing after the client has stopped sending.
//!
//! Wired through `EndpointConfigBuilder::bidi_engine` which wraps the
//! engine in a `BidiIngress` and registers it on the velo bidi handler.

use std::sync::Arc;

use bidi_streaming::{COMPONENT_NAME, DEFAULT_NAMESPACE, ENDPOINT_NAME};
use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, BIDI_INIT_KEY, Error, ManyIn, ManyOut,
        ResponseStream, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;

fn main() -> anyhow::Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct UppercaseHandler;

#[async_trait]
impl AsyncEngine<ManyIn<String>, ManyOut<Annotated<String>>, Error> for UppercaseHandler {
    async fn generate(
        &self,
        request: ManyIn<String>,
    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
        // Pull the init payload out of the request registry. The framework
        // stuffed it there during the bidi handshake.
        let prefix: String = request
            .get::<String>(BIDI_INIT_KEY)
            .map(|arc| arc.as_str().to_string())
            .unwrap_or_default();

        let (holder, ctx_unit) = request.into_parts();
        let ctx = ctx_unit.context();
        let mut input = holder
            .take()
            .ok_or_else(|| anyhow::anyhow!("input stream was already taken"))?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Annotated<String>>(16);
        tokio::spawn(async move {
            let mut count = 0usize;
            while let Some(line) = input.next().await {
                let out = if prefix.is_empty() {
                    line.to_uppercase()
                } else {
                    format!("[{prefix}] {}", line.to_uppercase())
                };
                if tx.send(Annotated::from_data(out)).await.is_err() {
                    return;
                }
                count += 1;
            }

            // Half-close demo: input has ended (peer sent Done), but our
            // half of the bidi stream stays open until we drop `tx`. Emit a
            // summary line.
            let _ = tx
                .send(Annotated::from_data(format!(
                    "(handler saw {count} item(s))"
                )))
                .await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

async fn backend(runtime: DistributedRuntime) -> anyhow::Result<()> {
    let component = runtime.namespace(DEFAULT_NAMESPACE)?.component(COMPONENT_NAME)?;
    let engine: Arc<
        dyn AsyncEngine<ManyIn<String>, ManyOut<Annotated<String>>, Error>,
    > = Arc::new(UppercaseHandler);

    component
        .endpoint(ENDPOINT_NAME)
        .endpoint_builder()
        .bidi_engine::<String, Annotated<String>, String>(engine)?
        .start()
        .await
}

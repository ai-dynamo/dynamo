// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidi streaming example — client side.
//!
//! Builds a small `Stream<String>` of input items, opens a bidi session
//! against the server's `uppercase` endpoint, prints each response item,
//! and exits when the server finalizes.
//!
//! The bidi protocol specifics (handshake, dual velo anchors, half-close)
//! are hidden behind `PushRouter::bidi_generate_with`; the user's view is
//! `Stream<T>` in, `Stream<U>` out.

use bidi_streaming::{COMPONENT_NAME, DEFAULT_NAMESPACE, ENDPOINT_NAME};
use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker,
    engine::DataStream,
    logging,
    pipeline::{AsyncRequestStream, Context, ManyIn, PushRouter},
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

    let client = distributed
        .namespace(DEFAULT_NAMESPACE)?
        .component(COMPONENT_NAME)?
        .endpoint(ENDPOINT_NAME)
        .client()
        .await?;
    client.wait_for_instances().await?;

    // PushRouter is shared between unary and bidi; the only requirement is
    // that T/U implement Serialize + DeserializeOwned and that U
    // implements MaybeError (Annotated<U> does).
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, Default::default()).await?;

    // Build the input stream. In real apps this could be wired to any
    // async source (websocket, stdin, a channel, etc.).
    let inputs = vec![
        "hello".to_string(),
        "bidi".to_string(),
        "streaming".to_string(),
        "world".to_string(),
    ];
    let input_count = inputs.len();
    let input_stream: DataStream<String> = Box::pin(tokio_stream::iter(inputs));
    let many_in: ManyIn<String> = Context::new(AsyncRequestStream::new(input_stream));

    // The optional `init: I` payload travels with the bidi handshake. Here
    // we send a small prefix that the server stamps onto every echoed line.
    let mut response = router
        .bidi_generate_with::<String>(many_in, "demo".to_string())
        .await?;

    println!("client: streaming {input_count} input items...");
    while let Some(item) = response.next().await {
        if let Some(data) = item.data {
            println!("server -> {data}");
        }
    }
    println!("client: response stream ended cleanly");

    runtime.shutdown();
    Ok(())
}

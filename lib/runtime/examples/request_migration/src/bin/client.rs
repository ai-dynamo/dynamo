// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging,
    pipeline::{PushRouter, RouterMode, network::STREAM_ERR_MSG},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
    stream::StreamExt,
};
use request_migration::DEFAULT_NAMESPACE;

fn main() -> anyhow::Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let client = distributed
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .endpoint("generate")
        .client()
        .await?;
    client.wait_for_instances().await?;
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, RouterMode::Random).await?;

    const MAX_RETRIES: usize = 3;

    for _ in 0..=10 {
        for attempt in 1..=MAX_RETRIES {
            println!("\n=== Attempt {} ===", attempt);

            let mut stream = router.random("hello world".to_string().into()).await?;
            let mut had_error = false;

            while let Some(resp) = stream.next().await {
                // Check if the response contains an error
                if let Some(err) = resp.err() {
                    let err_msg = format!("{:?}", err);
                    if err_msg.contains(STREAM_ERR_MSG) {
                        println!("Error: {}", STREAM_ERR_MSG);
                        println!("Retrying request...");
                        had_error = true;
                        break; // Break to retry
                    }
                }

                println!("{:?}", resp);
            }

            if !had_error {
                println!("\n=== Success! ===");
                break;
            }

            if attempt == MAX_RETRIES {
                anyhow::bail!("Failed after {} attempts", MAX_RETRIES);
            }
        }
    }

    runtime.shutdown();

    Ok(())
}

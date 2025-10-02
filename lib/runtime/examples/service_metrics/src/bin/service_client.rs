// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;
use service_metrics::DEFAULT_NAMESPACE;

use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker, logging, pipeline::PushRouter,
    protocols::annotated::Annotated, utils::Duration,
};

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let namespace = distributed.namespace(DEFAULT_NAMESPACE)?;
    let component = namespace.component("backend")?;

    let client = component.endpoint("generate").client().await?;

    client.wait_for_instances().await?;
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, Default::default()).await?;

    let mut stream = router.random("hello world".to_string().into()).await?;

    while let Some(resp) = stream.next().await {
        println!("{:?}", resp);
    }

    // This is just an illustration to invoke the server's stats_registry(<action>), where
    // the action currently increments the `service_requests_total` metric. You can validate
    // the result by running `curl http://localhost:8000/metrics`
    let service_set = component.scrape_stats(Duration::from_millis(100)).await?;
    println!("{:?}", service_set);

    runtime.shutdown();

    Ok(())
}

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;
use system_metrics::{DEFAULT_COMPONENT, DEFAULT_ENDPOINT, DEFAULT_NAMESPACE};

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
    let component = namespace.component(DEFAULT_COMPONENT)?;

    let client = component.endpoint(DEFAULT_ENDPOINT).client().await?;

    client.wait_for_instances().await?;
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, Default::default()).await?;

    let mut stream = router.random("hello world".to_string().into()).await?;

    while let Some(resp) = stream.next().await {
        println!("{:?}", resp);
    }

    let service_set = component.scrape_stats(Duration::from_millis(100)).await?;
    println!("{:?}", service_set);

    runtime.shutdown();

    Ok(())
}

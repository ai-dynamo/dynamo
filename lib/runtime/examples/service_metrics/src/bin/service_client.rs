// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::StreamExt;
use service_metrics::DEFAULT_NAMESPACE;

use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging, pipeline::PushRouter,
    protocols::annotated::Annotated,
};

fn main() -> anyhow::Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> anyhow::Result<()> {
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

    runtime.shutdown();
    Ok(())
}

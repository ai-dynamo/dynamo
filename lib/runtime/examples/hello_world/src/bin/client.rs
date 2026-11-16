// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging,
    pipeline::{PushRouter, RouterMode},
    protocols::annotated::Annotated,
    stream::StreamExt,
};
use hello_world::DEFAULT_NAMESPACE;

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
    let instance_id = client.instance_ids()[0];
    let router =
        PushRouter::<String, Annotated<String>>::from_client(client, RouterMode::Direct).await?;

    let mut stream = router
        .direct("hello world".to_string().into(), instance_id)
        .await?;

    while let Some(resp) = stream.next().await {
        println!("{:?}", resp);
    }

    runtime.shutdown();

    Ok(())
}

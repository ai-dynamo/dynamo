// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use futures::StreamExt;
use service_metrics::DEFAULT_NAMESPACE;

use dynamo_runtime::{
    logging, pipeline::PushRouter, protocols::annotated::Annotated, utils::Duration,
    DistributedRuntime, Result, Runtime, Worker,
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

    client.wait_for_endpoints().await?;
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

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

// TODO(#400):
// Instead of passing in a block_size, we should get this data from the backend component's config.
// What changes need to be made:
// 1. Take as an argument the name of the backend component.
// 2. Update the backend component to produce a config in a standard location.
// 3. Update the KvRouter to read the config from the backend component.

use std::sync::Arc;

use clap::Parser;

use dynamo_llm::kv_router::{
    protocols::WorkerSelectionResult,
    scheduler::{DefaultWorkerSelector, KvSchedulerError, SchedulingRequest},
    scoring::ProcessedEndpoints,
    KvRouter, WorkerSelector,
};
use dynamo_runtime::{
    logging, pipeline::network::Ingress, DistributedRuntime, Result, Runtime, Worker,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Namespace for the distributed component
    #[arg(long)]
    namespace: String,

    /// Component name for the service
    #[arg(long, default_value = "kv_aware_router")]
    component: String,

    /// Block size for the router
    #[arg(long)]
    block_size: usize,
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse();
    let runtime = DistributedRuntime::from_settings(runtime).await?;

    let component = runtime
        .namespace(&args.namespace)?
        .component(&args.component)?;

    let selector = Box::new(CustomWorkerSelector::default());

    let router = KvRouter::new(component.clone(), args.block_size, Some(selector)).await?;
    let router = Ingress::for_engine(Arc::new(router))?;

    component
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        .handler(router)
        .start()
        .await
}

#[derive(Default)]
pub struct CustomWorkerSelector(DefaultWorkerSelector);

impl WorkerSelector for CustomWorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: usize,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        // customize logic here
        // F12 into [DefaultWorkerSelector] to see the original logic
        self.0.select_worker(workers, request, block_size)
    }
}

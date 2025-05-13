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

use clap::Parser;
use std::sync::Arc;

use dynamo_llm::{
    http::service::{
        discovery::{LLMRouterMode, ModelWatcher},
        service_v2::HttpService,
    },
    model_type::ModelType,
};
use dynamo_runtime::{
    logging, transports::etcd::PrefixWatcher, DistributedRuntime, Result, Runtime, Worker,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Host for the HTTP service
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port number for the HTTP service
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Namespace for the distributed component
    #[arg(long, default_value = "public")]
    namespace: String,

    /// Component name for the service
    #[arg(long, default_value = "http")]
    component: String,
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let args = Args::parse();

    let http_service = HttpService::builder()
        .port(args.port)
        .host(args.host)
        .build()?;
    let manager = http_service.model_manager().clone();

    // todo - use the IntoComponent trait to register the component
    // todo - start a service
    // todo - we want the service to create an entry and register component definition
    // todo - the component definition should be the type of component and it's config
    // in this example we will have an HttpServiceComponentDefinition object which will be
    // written to etcd
    // the cli when operating on an `http` component will validate the namespace.component is
    // registered with HttpServiceComponentDefinition
    let component = distributed
        .namespace(&args.namespace)?
        .component(&args.component)?;
    let etcd_root = component.etcd_path();

    // TODO: A single watcher already watches all model types and does the right thing.
    // The paths need change here and in llmctl to not include the model_type

    // Create watchers for `Chat` and `Completion` model types
    for model_type in [ModelType::Chat, ModelType::Completion] {
        let etcd_path = format!("{}/models/{}/", etcd_root, model_type.as_str());

        let watch_obj = Arc::new(
            ModelWatcher::new(
                component.clone(),
                manager.clone(),
                &etcd_path,
                LLMRouterMode::Random,
            )
            .await?,
        );

        if let Some(etcd_client) = distributed.etcd_client() {
            let models_watcher: PrefixWatcher =
                etcd_client.kv_get_and_watch_prefix(etcd_path).await?;

            let (_prefix, _watcher, receiver) = models_watcher.dissolve();
            tokio::spawn(watch_obj.watch(receiver));
        }
    }

    // Run the service
    http_service.run(runtime.child_token()).await
}

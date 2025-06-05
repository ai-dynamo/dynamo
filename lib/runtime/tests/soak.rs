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

#[cfg(feature = "integration")]
mod integration {

    pub const DEFAULT_NAMESPACE: &str = "dynamo";

    use dynamo_runtime::{
        logging,
        pipeline::{
            async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
            ResponseStream, SingleIn,
        },
        protocols::annotated::Annotated,
        DistributedRuntime, ErrorContext, Result, Runtime, Worker,
    };
    use futures::StreamExt;
    use std::{sync::Arc, time::Duration};
    use tokio::time::Instant;

    #[test]
    fn main() -> Result<()> {
        logging::init();
        let worker = Worker::from_settings()?;
        worker.execute(app)
    }

    async fn app(runtime: Runtime) -> Result<()> {
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
        let server = tokio::spawn(backend(distributed.clone()));
        let client = tokio::spawn(client(distributed.clone()));

        client.await??;
        distributed.shutdown();
        server.await??;

        Ok(())
    }

    struct RequestHandler {}

    impl RequestHandler {
        fn new() -> Arc<Self> {
            Arc::new(Self {})
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
        async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
            let (data, ctx) = input.into_parts();

            let chars = data
                .chars()
                .map(|c| Annotated::from_data(c.to_string()))
                .collect::<Vec<_>>();

            let stream = async_stream::stream! {
                for c in chars {
                    yield c;
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            };

            Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
        }
    }

    async fn backend(runtime: DistributedRuntime) -> Result<()> {
        // attach an ingress to an engine
        let ingress = Ingress::for_engine(RequestHandler::new())?;

        // // make the ingress discoverable via a component service
        // // we must first create a service, then we can attach one more more endpoints
        runtime
            .namespace(DEFAULT_NAMESPACE)?
            .component("backend")?
            .service_builder()
            .create()
            .await?
            .endpoint("generate")
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await
    }

    async fn client(runtime: DistributedRuntime) -> Result<()> {
        // get the run duration from env
        let run_duration = std::env::var("DYN_SOAK_RUN_DURATION").unwrap_or("1m".to_string());
        let run_duration =
            humantime::parse_duration(&run_duration).unwrap_or(Duration::from_secs(60));

        let batch_load = std::env::var("DYN_SOAK_BATCH_LOAD").unwrap_or("10000".to_string());
        let batch_load: usize = batch_load.parse().unwrap_or(10000);

        let client = runtime
            .namespace(DEFAULT_NAMESPACE)?
            .component("backend")?
            .endpoint("generate")
            .client::<String, Annotated<String>>()
            .await?;

        client.wait_for_instances().await?;
        let client = Arc::new(client);

        let start = Instant::now();
        let mut count = 0;

        loop {
            let mut tasks = Vec::new();
            for _ in 0..batch_load {
                let client = client.clone();
                tasks.push(tokio::spawn(async move {
                    let mut stream = tokio::time::timeout(
                        Duration::from_secs(30),
                        client.random("hello world".to_string().into()),
                    )
                    .await
                    .context("request timed out")??;

                    while let Some(_resp) =
                        tokio::time::timeout(Duration::from_secs(30), stream.next())
                            .await
                            .context("stream timed out")?
                    {}
                    Ok::<(), Error>(())
                }));
            }

            for task in tasks.into_iter() {
                task.await??;
            }

            let elapsed = start.elapsed();
            count += batch_load;
            println!("elapsed: {:?}; count: {}", elapsed, count);

            if elapsed > run_duration {
                println!("done");
                break;
            }
        }

        Ok(())
    }
}

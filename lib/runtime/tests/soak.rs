// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// cargo test --test soak integration::main --features integration
//!
//! It will send a batch of requests to the runtime and measure the throughput.
//!
//! It will also measure the latency of the requests.
//!
//! A reasonable soak test configuration to start off is 1 minute duration with 10000 batch load:
//! export DYN_QUEUED_UP_PROCESSING=true
//! export DYN_SOAK_BATCH_LOAD=10000
//! export DYN_SOAK_RUN_DURATION=60s
//! cargo test --test soak integration::main --features integration -- --nocapture
#[cfg(feature = "integration")]
mod integration {

    pub const DEFAULT_NAMESPACE: &str = "dynamo";

    use dynamo_runtime::{
        DistributedRuntime, Runtime, Worker, logging,
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
            SingleIn, async_trait, network::Ingress,
        },
        protocols::annotated::Annotated,
        stream,
    };

    use anyhow::{Context, Result};
    use futures::StreamExt;
    use std::{
        sync::Arc,
        sync::atomic::{AtomicU64, Ordering},
        time::Duration,
    };
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
        let handler = server.await??;

        // Print final backend counter value
        let final_count = handler.backend_counter.load(Ordering::Relaxed);
        println!(
            "Final RequestHandler backend_counter: {} requests processed",
            final_count
        );

        Ok(())
    }

    struct RequestHandler {
        backend_counter: AtomicU64,
        queued_up_processing: bool,
    }

    impl RequestHandler {
        fn new(queued_up_processing: bool) -> Arc<Self> {
            Arc::new(Self {
                backend_counter: AtomicU64::new(0),
                queued_up_processing,
            })
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
        async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
            let (data, ctx) = input.into_parts();

            // Increment backend counter
            self.backend_counter.fetch_add(1, Ordering::Relaxed);

            let chars = data
                .chars()
                .map(|c| Annotated::from_data(c.to_string()))
                .collect::<Vec<_>>();

            if self.queued_up_processing {
                // queued up processing - delayed response to saturate the queue
                let async_stream = async_stream::stream! {
                    for c in chars {
                        yield c;
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                };
                Ok(ResponseStream::new(Box::pin(async_stream), ctx.context()))
            } else {
                // normal processing - immediate response
                let iter_stream = stream::iter(chars);
                Ok(ResponseStream::new(Box::pin(iter_stream), ctx.context()))
            }
        }
    }

    async fn backend(runtime: DistributedRuntime) -> Result<Arc<RequestHandler>> {
        // get the queued up processing setting from env (not delayed)
        let queued_up_processing =
            std::env::var("DYN_QUEUED_UP_PROCESSING").unwrap_or("false".to_string());
        let queued_up_processing: bool = queued_up_processing.parse().unwrap_or(false);

        // attach an ingress to an engine
        let handler = RequestHandler::new(queued_up_processing);
        let ingress = Ingress::for_engine(handler.clone())?;

        // // make the ingress discoverable via a component service
        // // we must first create a service, then we can attach one more more endpoints
        let mut component = runtime.namespace(DEFAULT_NAMESPACE)?.component("backend")?;
        component.add_stats_service().await?;
        component
            .endpoint("generate")
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await?;

        Ok(handler)
    }

    async fn client(runtime: DistributedRuntime) -> Result<()> {
        // get the run duration from env
        let run_duration = std::env::var("DYN_SOAK_RUN_DURATION").unwrap_or("3s".to_string());
        let run_duration =
            humantime::parse_duration(&run_duration).unwrap_or(Duration::from_secs(3));

        let batch_load = std::env::var("DYN_SOAK_BATCH_LOAD").unwrap_or("100".to_string());
        let batch_load: usize = batch_load.parse().unwrap_or(100);

        let client = runtime
            .namespace(DEFAULT_NAMESPACE)?
            .component("backend")?
            .endpoint("generate")
            .client()
            .await?;

        client.wait_for_instances().await?;
        let router =
            PushRouter::<String, Annotated<String>>::from_client(client, Default::default())
                .await?;
        let router = Arc::new(router);

        let start = Instant::now();
        let mut count = 0;

        loop {
            let mut tasks = Vec::new();
            for _ in 0..batch_load {
                let router = router.clone();
                tasks.push(tokio::spawn(async move {
                    let mut stream = tokio::time::timeout(
                        Duration::from_secs(5),
                        router.random("hello world".to_string().into()),
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
            if count % 1000 == 0 {
                println!("elapsed: {:?}; count: {}", elapsed, count);
            }

            if elapsed > run_duration {
                println!("done");
                break;
            }
        }

        Ok(())
    }
}

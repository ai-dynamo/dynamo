// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

use dynamo_llm::discovery::{ModelWatcher, MODEL_ROOT_PATH};
use dynamo_llm::http::service::rate_limiter::{RateLimiterConfig};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::{
    logging, pipeline::RouterMode, transports::etcd::PrefixWatcher, DistributedRuntime, Result, error,
    Runtime, Worker,
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

    /// Enable rate limiting
    #[arg(long, default_value = "false")]
    enable_rate_limiting: bool,

    /// Time to first token threshold in milliseconds
    #[arg(
        long, 
        default_value = "1000.0", 
        help = "Desired time to first token threshold in milliseconds"
    )]
    ttft_threshold_ms: f64,

    /// Inter-token latency threshold in milliseconds
    #[arg(
        long, 
        default_value = "30.0", 
        help = "Desired inter-token latency threshold in milliseconds"
    )]
    itl_threshold_ms: f64,

    /// Time constant for the rate limiter in seconds
    #[arg(
        long, 
        default_value = "15.0", 
        help = "Time constant for the exponential moving average calculation in the rate limiter, in seconds"
    )]
    time_constant_secs: f64,

    /// Per model rate limiting
    #[arg(
        long, 
        default_value = "false", 
        help = "Track rate limits per model separately, instead of globally"
    )]
    per_model_rate_limiting: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_current()?;
    worker.execute_async(app).await
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let args = Args::parse();

    validate_args(&args)?;

    let mut http_service_builder = HttpService::builder().port(args.port).host(args.host);

    if args.enable_rate_limiting {
        let rate_limiter = RateLimiterConfig::new(
            args.ttft_threshold_ms / 1_000.0,
            args.itl_threshold_ms / 1_000.0,
            args.time_constant_secs,
            args.per_model_rate_limiting,
        )?;
        http_service_builder = http_service_builder.with_rate_limiter(rate_limiter);
    }

    let http_service = http_service_builder.build()?;
    let manager = http_service.state().manager_clone();

    // todo - use the IntoComponent trait to register the component
    // todo - start a service
    // todo - we want the service to create an entry and register component definition
    // todo - the component definition should be the type of component and it's config
    // in this example we will have an HttpServiceComponentDefinition object which will be
    // written to etcd
    // the cli when operating on an `http` component will validate the namespace.component is
    // registered with HttpServiceComponentDefinition

    let watch_obj = ModelWatcher::new(distributed.clone(), manager, RouterMode::Random, None);

    if let Some(etcd_client) = distributed.etcd_client() {
        let models_watcher: PrefixWatcher =
            etcd_client.kv_get_and_watch_prefix(MODEL_ROOT_PATH).await?;

        let (_prefix, _watcher, receiver) = models_watcher.dissolve();
        tokio::spawn(async move {
            watch_obj.watch(receiver).await;
        });
    }

    // Run the service
    http_service.run(runtime.child_token()).await
}

fn validate_args(args: &Args) -> Result<()> {
    if args.enable_rate_limiting {
        if args.ttft_threshold_ms <= 0.0 {
            return Err(error!("Time to first token threshold must be greater than 0"));
        }

        if args.itl_threshold_ms <= 0.0 {
            return Err(error!("Inter-token latency threshold must be greater than 0"));
        }

        if args.time_constant_secs <= 0.0 {
            return Err(error!("Time constant must be greater than 0"));
        }
    }

    Ok(())
}